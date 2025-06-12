


from urllib.request import Request
from urllib.error import URLError, HTTPError

from src.agents.execution.utils.config_loader import load_global_config, get_config_section
from src.agents.execution.execution_memory import ExecutionMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Web Brouwser")
printer = PrettyPrinter

class WebBrouwser:
    def __init__(self):
        self.config = load_global_config()
        self.web_config = get_config_section('web_browser')

        logger.info(f"Web Brouwser succesfully initialized")

    def _enforce_rate_limit(self):
        """Implement token bucket rate limiting algorithm"""
        with self.rate_lock:
            now = time.time()
            if len(self.request_times) >= self.rate_limit:
                elapsed = now - self.request_times[0]
                if elapsed < 1.0:
                    time.sleep(1.0 - elapsed)
                self.request_times.popleft()
            self.request_times.append(time.time())

    def _make_request(self, url, method='GET', headers=None, data=None, use_cache=True):
        """Core request handler with all enhanced features"""
        # Check cache first
        cache_key = self._cache_key(url, data if method == 'GET' else None)
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Rate limiting
        self._enforce_rate_limit()
        
        # Prepare request
        headers = headers or {}
        headers.setdefault('User-Agent', self.user_agent)
        
        if data and not isinstance(data, bytes):
            data = urlencode(data).encode('utf-8')
        
        req = Request(url, data=data, headers=headers, method=method.upper())
        
        # Retry logic
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.opener.open(req, timeout=self.timeout)
                content = response.read()
                result = {
                    'status': response.status,
                    'headers': dict(response.headers),
                    'content': content,
                    'url': response.url
                }
                
                # Cache successful responses
                if use_cache and response.status == 200:
                    self.cache[cache_key] = result
                return result
                
            except (HTTPError, URLError) as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_delays[min(attempt, len(self.retry_delays)-1)])
                    continue
                raise ConnectionError(f"Request failed after {self.max_retries} attempts: {str(e)}")

    def browse_web(self, url, parse=True, parser='html', use_cache=True, **kwargs):
        """
        Enhanced web browsing with multiple parser options
        
        Args:
            url: Target URL
            parse: Whether to parse content
            parser: Parser type ('html' or 'json')
            use_cache: Utilize caching system
            kwargs: Allow flexible parameters for different search providers
            
        Returns:
            Parsed content or raw response
        """
        result = self._make_request(url, use_cache=use_cache)
        
        if not parse:
            return result
        
        if parser not in self.parsers:
            raise ValueError(f"Unsupported parser: {parser}. Available: {list(self.parsers.keys())}")
        
        try:
            content = result['content'].decode('utf-8', errors='replace')
            if parser == 'html':
                parser_instance = self.parsers[parser]()
                parser_instance.feed(content)

                if 'max_results' in kwargs:
                    return {
                        'metadata': parser_instance.structure['metadata'],
                        'results': parser_instance.structure['links'][:kwargs['max_results']]
                    }

                return parser_instance.structure
            else:
                return self.parsers[parser](content)
        except Exception as e:
            raise ValueError(f"Parsing failed with {parser} parser: {str(e)}")

    def call_api(self, endpoint, method='GET', headers=None, data=None, params=None, use_cache=True):
        """
        Enhanced API interaction with intelligent caching
        
        Args:
            endpoint: Target API URL
            method: HTTP method
            headers: Additional headers
            data: Request body
            params: Query parameters
            use_cache: Utilize caching system
            
        Returns:
            Parsed response (automatic JSON detection)
        """
        result = self._make_request(
            endpoint,
            method=method,
            headers=headers,
            data=data,
            params=params,
            use_cache=use_cache
        )
        
        try:
            return json.loads(result['content'].decode('utf-8'))
        except json.JSONDecodeError:
            return result['content']

    def _build_scholarly_url(self, query: str) -> str:
        """Build validated academic search URL based on configurable endpoints"""
        # Academic search endpoints (RFC 3986-compliant URI construction)
        endpoints = {
            'crossref': 'https://api.crossref.org/works?query=',
            'semantic_scholar': 'https://api.semanticscholar.org/graph/v1/paper/search?query=',
            'arxiv': 'http://export.arxiv.org/api/query?search_query='
        }
    
        # Select endpoint based on safety configuration
        selected = next((e for e in endpoints if urlparse(endpoints[e]).netloc in self.safety.allowlisted_domains), None)

        if not selected:
            raise SecurityError("No allowlisted academic endpoints available")
    
        return f"{endpoints[selected]}{urllib.parse.quote(query)}&rows=10"
