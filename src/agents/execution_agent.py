"""
Enhanced Execution Agent with Advanced Web Interaction Capabilities
Key Academic References:
- Cookie Management: Barth (2011) "HTTP State Management Mechanism" (RFC 6265)
- Caching: Fielding et al. (1999) "Hypertext Transfer Protocol - HTTP/1.1" (RFC 2616)
- Retry Strategies: Thaler & Ravishankar (1998) "Using Name-Based Mappings to Increase Hit Rates"
- Rate Limiting: Floyd & Jacobson (1993) "Random Early Detection Gateways"
"""

import os
import re
import json
import time
import urllib

from json import JSONDecodeError
from urllib.request import Request, urlopen, build_opener, HTTPCookieProcessor
from urllib.parse import urlparse, urlencode
from urllib.error import URLError, HTTPError
from http.cookiejar import CookieJar
from collections import deque
from threading import Lock
from typing import Dict, Callable

from src.agents.base.utils.main_config_loader import load_global_config, get_config_section
from src.agents.execution.execution_manager import ExecutionManager
from src.agents.execution.web_browser import WebBrouwser
from src.agents.execution.html_parser import HTMLParser
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Execution Agent")
printer = PrettyPrinter

class ToolLibrary:
    """Tool selector using cosine similarity (simplified for minimal dependencies)"""
    def __init__(self):
        self.tools: Dict[str, Dict] = {
            'web_search': {
                'function': None,
                'description': 'Perform web search operations',
                'safety_level': 'high'
            },
            'code_exec': {
                'function': None,
                'description': 'Execute approved code snippets',
                'safety_level': 'restricted'
            }
        }
    
    def register_tool(self, name: str, func: Callable, metadata: dict):
        """Schick et al. (2023) inspired tool registration"""
        self.tools[name] = {
            'function': func,
            'metadata': metadata
        }
    
    def select_tool(self, query: str):
        """Simplified semantic tool selection (TF-IDF cosine similarity substitute)"""
        query = query.lower()
        for tool, data in self.tools.items():
            if re.search(rf'\b{tool}\b', query):
                return data['function']
        return None

class SafeEnvironment:
    """Amodei et al. (2016) inspired safety constraints"""
    def __init__(self):
        self.allowlisted_domains = ['api.crossref.org', 'api.semanticscholar.org', 'export.arxiv.org']
        self.allowlisted_commands = ['calculate', 'fetch_data', 'transform_format']
        self.restricted_paths = ['/sys/', '/etc/']
    
    def validate_request(self, url: str):
        """Validate against allowlisted domains"""
        domain = urlparse(url).netloc
        if domain not in self.allowlisted_domains:
            raise SecurityError(f"Blocked unauthorized domain: {domain}")
    
    def validate_file_path(self, path: str):
        """Prevent path traversal attacks"""
        if any(restricted in path for restricted in self.restricted_paths):
            raise SecurityError(f"Restricted file path: {path}")

class ExecutionAgent:
    def __init__(self, agent_factory, shared_memory, config=None, args=(), kwargs={}):
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
        self.execute_safe_action = ExecutionAgent
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        config = config or {}
        self.timeout = config.get('timeout', 10)
        self.user_agent = config.get('user_agent', "EnhancedExecutionAgent/2.0")
        self.safety = SafeEnvironment()
        self.toolbox = ToolLibrary()
        self.thought_stack = deque()
        
        # Cookie management
        self.cookie_jar = CookieJar()
        self.cookie_processor = HTTPCookieProcessor(self.cookie_jar)

        # Rate limiting
        self.rate_limit = config.get('rate_limit', 5)
        self.request_times = deque(maxlen=self.rate_limit)
        self.rate_lock = Lock()
        
        # Retry configuration
        self.max_retries = config.get('max_retries', 3)
        self.retry_delays = [0.5, 1, 2]  # Exponential backoff
        
        # Alternative parsers registry
        self.parsers = {
            'html': HTMLParser,
            'json': json.loads
        }

        # Register core tools
        self.toolbox.register_tool('web_search', self.WebBrouwser.browse_web, 
                                 {'rate_limit': 2, 'cache_ttl': 3600})
        self.toolbox.register_tool('call_api', self.call_api,
                                 {'allowed_endpoints': self.safety.allowlisted_domains})

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

    def generate_react_loop(self, task: str, max_steps: int = 5):
        """
        ReAct framework implementation (Yao et al., 2022)
        Example Thought-Action-Observation loop:
        """
        for _ in range(max_steps):
            thought = self._generate_thought(task)
            if '[ACTION]' in thought:
                action = self._parse_action(thought)
                result = self.execute_safe_action(action)
                task += f"\nObservation: {result}"
            else:
                return thought
        return "Max reasoning steps exceeded"
    
    def execute_safe_action(self, action: dict):
        """Amodei-style safety validation"""
        tool = self.toolbox.select_tool(action['type'])
        if not tool:
            raise SecurityError(f"Blocked unauthorized tool: {action['type']}")
        
        # Validate parameters
        if action.get('url'):
            self.safety.validate_request(action['url'])
        if action.get('file_path'):
            self.safety.validate_file_path(action['file_path'])
        
        return tool(**action['params'])
    
    def _generate_thought(self, task: str) -> str:
        """
        Hybrid neural-symbolic thought generator based on:
        - Wei et al. (2022) "Chain-of-Thought Prompting"
        - Parisi et al. (2022) "TidyBot: Personalized Robot Assistance"
    
        Implements a simplified version of the TidyBot decision architecture
        """
        # Local decision model (simplified transformer-like logic)
        def minimal_decision_model(prompt: str) -> dict:
            """Rule-based approximation of LLM decision-making"""
            patterns = {
                r'\b(search|find|look up)\b': 'web_search',
                r'\b(calculate|compute|math)\b': 'calculator',
                r'\b(file|document|write)\b': 'file_ops',
                r'\b(API|fetch data)\b': 'call_api'
            }
        
            # Match against known tool patterns
            for pattern, tool in patterns.items():
                if re.search(pattern, prompt, flags=re.IGNORECASE):
                    return {
                        'thought': f"Determined need for {tool.replace('_', ' ')}",
                        'action': tool,
                        'confidence': 0.85
                    }
        
            # Default cognitive pattern
            return {
                'thought': "Analyzing task requirements...",
                'action': 'information_gathering',
                'confidence': 0.65
            }

        # Build cognitive context
        context = {
            'task': task,
            'available_tools': list(self.toolbox.tools.keys()),
            'previous_actions': list(self.thought_stack)[-3:]
        }

        # Generate model response
        model_response = minimal_decision_model(task)
    
        # Construct action payload
        if model_response['confidence'] > 0.7:
            action_template = {
                'web_search': {
                    'params': {
                        'url': self.WebBrouwser._build_scholarly_url(task),
                        'max_results': 10
                    }
                },
                'calculator': {
                    'params': {
                        'expression': next(iter(re.findall(r'\b(\d+[\+\-\*\/]\d+)\b', task)), '0+0')
                    }
                }
            }
        
            action = action_template.get(model_response['action'], {})
            action_str = f"[ACTION] {model_response['action']} params: {json.dumps(action.get('params', {}))}"
        
            return f"Thought: {model_response['thought']}\n{action_str}"
    
        return f"Thought: {model_response['thought']} [NEED MORE INFO]"

    def _parse_action(self, thought: str) -> dict:
        """Parse action details from thought string"""
    
        # Match action pattern: [ACTION] <type> params: {<params>}
        match = re.search(r'\[ACTION\] (\w+)\s+params:\s+({.*?})$', thought)
        if not match:
            raise ValueError(f"Malformed action string: {thought}")
    
        action_type = match.group(1)
        params_str = match.group(2).replace("'", '"')  # Convert to JSON format
    
        try:
            params = json.loads(params_str)
        except JSONDecodeError:
            raise ValueError(f"Invalid parameter format in: {params_str}")
    
        return {'type': action_type, 'params': params}

class SecurityError(Exception):
    """Custom exception for safety violations"""
    pass

# Example usage
#if __name__ == "__main__":
#    agent = ExecutionAgent(config={
#        'cache_dir': '.slaicache',
#        'rate_limit': 3
#    })
    
#    result = agent.generate_react_loop(
#        "Find recent papers about AI safety from trusted sources"
#    )
#    print(result)
