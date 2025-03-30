"""
Enhanced Execution Agent with Advanced Web Interaction Capabilities
Key Academic References:
- Cookie Management: Barth (2011) "HTTP State Management Mechanism" (RFC 6265)
- Caching: Fielding et al. (1999) "Hypertext Transfer Protocol - HTTP/1.1" (RFC 2616)
- Retry Strategies: Thaler & Ravishankar (1998) "Using Name-Based Mappings to Increase Hit Rates"
- Rate Limiting: Floyd & Jacobson (1993) "Random Early Detection Gateways"
"""

import os
import json
import time
import shelve
import hashlib
from urllib.request import Request, urlopen, build_opener, HTTPCookieProcessor
from urllib.parse import urlparse, urlencode
from urllib.error import URLError, HTTPError
from http.cookiejar import CookieJar
from html.parser import HTMLParser
from collections import deque
from threading import Lock

class EnhancedHTMLParser(HTMLParser):
    """Extended HTML parser with support for common semantic elements"""
    def __init__(self):
        super().__init__()
        self.structure = {
            'title': '',
            'links': [],
            'metadata': {},
            'headings': {f'h{i}': [] for i in range(1, 7)},
            'scripts': [],
            'images': []
        }
        self.current_tag = None
    
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        attrs = dict(attrs)
        
        if tag == 'title':
            pass
        elif tag == 'a' and 'href' in attrs:
            self.structure['links'].append(attrs['href'])
        elif tag in self.structure['headings']:
            pass
        elif tag == 'meta' and ('name' in attrs or 'property' in attrs):
            key = attrs.get('name') or attrs.get('property')
            self.structure['metadata'][key] = attrs.get('content', '')
        elif tag == 'script' and 'src' in attrs:
            self.structure['scripts'].append(attrs['src'])
        elif tag == 'img' and 'src' in attrs:
            self.structure['images'].append(attrs['src'])
    
    def handle_data(self, data):
        if self.current_tag == 'title':
            self.structure['title'] += data
        elif self.current_tag in self.structure['headings']:
            self.structure['headings'][self.current_tag].append(data.strip())
    
    def handle_endtag(self, tag):
        self.current_tag = None

class ExecutionAgent:
    def __init__(self, config=None):
        """
        Initialize with comprehensive configuration
        
        Config options:
            timeout: Request timeout (default 10)
            user_agent: User agent string
            cache_dir: Directory for persistent cache (default None)
            max_retries: Maximum request retries (default 3)
            rate_limit: Requests per second (default 5)
            cookie_policy: Cookie acceptance policy
        """
        config = config or {}
        self.timeout = config.get('timeout', 10)
        self.user_agent = config.get('user_agent', "EnhancedExecutionAgent/2.0")
        
        # Cookie management
        self.cookie_jar = CookieJar()
        self.cookie_processor = HTTPCookieProcessor(self.cookie_jar)
        self.opener = build_opener(self.cookie_processor)
        
        # Caching system
        self.cache_dir = config.get('cache_dir')
        self.cache = self._init_cache()
        
        # Rate limiting
        self.rate_limit = config.get('rate_limit', 5)
        self.request_times = deque(maxlen=self.rate_limit)
        self.rate_lock = Lock()
        
        # Retry configuration
        self.max_retries = config.get('max_retries', 3)
        self.retry_delays = [0.5, 1, 2]  # Exponential backoff
        
        # Alternative parsers registry
        self.parsers = {
            'html': EnhancedHTMLParser,
            'json': json.loads
        }

    def _init_cache(self):
        """Initialize persistent cache if configured"""
        if not self.cache_dir:
            return {}
        
        os.makedirs(self.cache_dir, exist_ok=True)
        return shelve.open(os.path.join(self.cache_dir, 'agent_cache'))

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

    def _cache_key(self, url, params=None):
        """Generate consistent cache key using SHA-256 hashing"""
        key_data = url + (urlencode(params) if params else '')
        return hashlib.sha256(key_data.encode()).hexdigest()

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
                    time.sleep(self.retry_delays[min(attempt, len(self.retry_delays)-1])
                    continue
                raise ConnectionError(f"Request failed after {self.max_retries} attempts: {str(e)}")

    def browse_web(self, url, parse=True, parser='html', use_cache=True):
        """
        Enhanced web browsing with multiple parser options
        
        Args:
            url: Target URL
            parse: Whether to parse content
            parser: Parser type ('html' or 'json')
            use_cache: Utilize caching system
            
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
                return parser_instance.structure
            else:
                return self.parsers[parser](content)
        except Exception as e:
            raise ValueError(f"Parsing failed with {parser} parser: {str(e)}")

    def handle_file(self, file_path, mode='r', content=None, encoding='utf-8'):
        """
        Robust file operations with atomic writes and validation.
        
        Args:
            file_path: Path to target file
            mode: 'r' (read), 'w' (write), 'a' (append)
            content: Content for write operations
            encoding: Text encoding
            
        Returns:
            File content for read mode, None otherwise
        """
        if mode not in ('r', 'w', 'a'):
            raise ValueError("Mode must be 'r', 'w', or 'a'")
            
        if mode == 'r':
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"No such file: {file_path}")
            if not os.path.isfile(file_path):
                raise IsADirectoryError(f"Path is directory: {file_path}")
                
            with open(file_path, mode, encoding=encoding) as f:
                return f.read()
                
        else:  # Write/append modes
            if content is None:
                raise ValueError("Content required for write operations")
                
            # Atomic write using temporary file
            tmp_path = f"{file_path}.tmp"
            try:
                with open(tmp_path, 'w', encoding=encoding) as f:
                    f.write(content)
                os.replace(tmp_path, file_path)
            except Exception as e:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise e

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

    def clear_cache(self, expired_after=None):
        """Clear cache entries, optionally older than specified timestamp"""
        if not self.cache_dir:
            return
            
        if expired_after is None:
            self.cache.clear()
            return
            
        now = time.time()
        to_delete = []
        for key, entry in self.cache.items():
            if now - entry.get('timestamp', 0) > expired_after:
                to_delete.append(key)
        
        for key in to_delete:
            del self.cache[key]

    def save_cookies(self, path):
        """Persist cookies to disk"""
        self.cookie_jar.save(path, ignore_discard=True)

    def load_cookies(self, path):
        """Load cookies from disk"""
        self.cookie_jar.load(path, ignore_discard=True)

    def register_parser(self, name, parser_func):
        """Add custom parser to the parser registry"""
        self.parsers[name] = parser_func

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'cache') and not isinstance(self.cache, dict):
            self.cache.close()
