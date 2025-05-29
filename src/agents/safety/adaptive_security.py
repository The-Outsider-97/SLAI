
import re
import math
import hashlib
import json, yaml

from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional

from src.agents.safety.utils.config_loader import load_global_config, get_config_section
from src.agents.safety.utils.neural_network import NeuralNetwork
from logs.logger import get_logger

logger = get_logger("Adaptive Security System")

PHISHING_MODEL_PATH = "src/agents/safety/models/phishing_model.json"

class AdaptiveSecurity:
    """
    Comprehensive cybersecurity system focused on phishing detection and threat prevention.
    Combines neural networks with heuristic rules for multi-layered protection.
    """
    
    def __init__(self):
        self.config = load_global_config()
        self.adaptive_config = get_config_section('adaptive_security')
        self.nn_layer_config = self.config.get('layers', [])
        self.request_tracker = defaultdict(lambda: deque(maxlen=100))  # Tracks requests per IP
        self.safe_package_hashes = self._load_trusted_hashes()
        
        # Security thresholds (configurable)
        self.email_nn = self._initialize_neural_network(
            model_type='email',
            num_inputs=11,
            model_path="src/agents/safety/models/email_phishing_model.json"
        )
        self.url_nn = self._initialize_neural_network(
            model_type='url',
            num_inputs=8,
            model_path="src/agents/safety/models/url_phishing_model.json"
        )
        self.rate_limit = self.config.get('rate_limit', 30)  # Requests per minute
        self.input_size_limit = self.config.get('input_size_limit', 1024)  # KB
        self.phishing_threshold = self.config.get('phishing_threshold', 0.85)

    def _initialize_neural_network(self, model_type: str, num_inputs: int, model_path: str) -> NeuralNetwork:
        try:
            logger.info(f"Attempting to load pre-trained {model_type} model from: {model_path}")
            return NeuralNetwork.load_model(model_path)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e: # Catch specific errors
            logger.warning(f"No pre-trained {model_type} model found at {model_path} or error loading: {e}. Initializing new model.")
            # Fallback to creating a new model
            # Ensure self.nn_layer_config is appropriate for this new model or have default per type
            
            # You might want to define default layer configs per model type if they differ significantly
            # For now, using the general self.nn_layer_config from the main config
            if not self.nn_layer_config:
                 logger.warning(f"No 'layers' configuration found in safety_config for new {model_type} model. Using a default.")
                 # Define a very simple default if 'layers' is missing
                 if num_inputs == 11: # email
                    fallback_layer_config = [{'neurons': 16, 'activation': 'relu'}, {'neurons': 1, 'activation': 'sigmoid'}]
                 elif num_inputs == 8: # url
                    fallback_layer_config = [{'neurons': 12, 'activation': 'relu'}, {'neurons': 1, 'activation': 'sigmoid'}]
                 else:
                    fallback_layer_config = [{'neurons': num_inputs * 2, 'activation': 'relu'}, {'neurons': 1, 'activation': 'sigmoid'}]
            else:
                fallback_layer_config = self.nn_layer_config


            return NeuralNetwork(
                num_inputs=num_inputs,
                layer_config=fallback_layer_config,
                loss_function_name='cross_entropy',
                optimizer_name='adam',
                problem_type='binary_classification',
                config=self.safety_config
            )

    def _load_trusted_hashes(self) -> Dict[str, str]:
        """Load trusted package hashes from file"""
        try:
            with open("src/agents/safety/templates/trusted_hashes.json") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("No trusted hashes file found. Starting empty.")
            return {}

    def analyze_email(self, email: Dict) -> Dict:
        """
        Analyze email for phishing characteristics.
        Returns: {
            "phishing_score": float,
            "is_phishing": bool,
            "threat_type": Optional[str],
            "features": List[float]
        }
        """
        features = self._extract_email_features(email)
        return self._analyze_features(features, "email", self.email_nn)

    def analyze_url(self, url: str) -> Dict:
        """
        Analyze URL for phishing characteristics.
        Returns: {
            "phishing_score": float,
            "is_phishing": bool,
            "threat_type": Optional[str],
            "features": List[float]
        }
        """
        features = self._extract_url_features(url)
        return self._analyze_features(features, "url", self.url_nn)

    def _analyze_features(self, features: List[float], source_type: str, model: NeuralNetwork) -> Dict:
        """Common analysis method for all feature types"""
        if self._check_input_overload():
            return {"error": "Request rate limit exceeded"}

        phishing_score = model.predict(features)[0]
        threat_type = self._determine_threat_type(features, source_type)

        return {
            "phishing_score": phishing_score,
            "is_phishing": phishing_score >= self.phishing_threshold,
            "threat_type": threat_type,
            "features": features
        }

    def _extract_email_features(self, email: Dict) -> List[float]:
        """Convert email characteristics to numerical features"""
        return [
            # Header analysis
            len(email.get('from', '')),
            len(email.get('subject', '')),
            self._contains_suspicious_keywords(email.get('subject', '')),
            len(email.get('links', [])),
            
            # Content analysis
            self._contains_urgent_language(email.get('body', '')),
            self._contains_attachment(email),
            self._domain_mismatch_score(email),
            
            # URL features
            self._avg_url_length(email.get('links', [])),
            self._ssl_cert_score(email.get('links', [])),
            
            # Behavioral features
            self._unusual_sender_score(email.get('from')),
            self._unusual_time_score(email.get('timestamp'))
        ]

    def _extract_url_features(self, url: str) -> List[float]:
        """Convert URL characteristics to numerical features"""
        return [
            len(url),
            self._url_entropy(url),
            self._num_subdomains(url),
            self._contains_ip_address(url),
            self._https_used(url),
            self._url_redirect_count(url),
            self._domain_age(url),
            self._special_char_count(url)
        ]

    def check_supply_chain(self, file_path: str) -> Dict:
        """Check file against known good hashes"""
        file_hash = self._calculate_file_hash(file_path)
        return {
            "file_hash": file_hash,
            "is_trusted": file_hash in self.safe_package_hashes.values(),
            "risk_level": "high" if file_hash not in self.safe_package_hashes.values() else "low"
        }

    def _check_input_overload(self) -> bool:
        """Detect potential DDoS/overload attempts"""
        client_ip = self._get_client_ip()
        self.request_tracker[client_ip].append(time.time())
        
        if len(self.request_tracker[client_ip]) > self.rate_limit:
            logger.warning(f"Rate limit exceeded for IP {client_ip}")
            return True
        return False

    def _determine_threat_type(self, features: List[float], source_type: str) -> Optional[str]:
        """Classify threat type based on feature patterns"""
        # Placeholder for demonstration - would use actual pattern matching
        if source_type == "email":
            if features[2] > 0.8:  # Suspicious keywords
                return "Phishing: Social Engineering"
            if features[6] > 0.7:  # Domain mismatch
                return "Phishing: Impersonation"
        elif source_type == "url":
            if features[3] > 0.9:  # Contains IP address
                return "Phishing: Suspicious URL Structure"
        return None

    # Security Helper Methods
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(4096):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _contains_suspicious_keywords(self, text: str) -> float:
        """Check for phishing-related keywords"""
        keywords = {'urgent', 'password', 'verify', 'account', 'suspended'}
        return len(keywords & set(text.lower().split())) / len(keywords)

    def _contains_urgent_language(self, email_body: str) -> float:
        """Calculate urgency score of email content (0-1 scale)"""
        urgent_keywords = {
            'urgent', 'immediately', 'action required', 'verify now',
            'account suspended', 'security alert', 'password reset'
        }
        words = re.findall(r'\b\w+\b', email_body.lower())
        matches = [1 for word in words if word in urgent_keywords]
        return min(len(matches)/3, 1.0)  # Cap at 1.0 for 3+ matches

    def _contains_attachment(self, email: Dict) -> float:
        """Check for file attachments (binary check)"""
        return 1.0 if len(email.get('attachments', [])) > 0 else 0.0

    def _domain_mismatch_score(self, email: Dict) -> float:
        """Compare display name domain with actual sender domain"""
        from_header = email.get('from', '')
        match = re.match(r'.*@([\w.-]+)', from_header)
        if not match:
            return 0.0  # Invalid format
        
        actual_domain = match.group(1).lower()
        display_name = re.match(r'\s*"?([^<"]+)"?\s*<', from_header)
        
        if not display_name:
            return 0.0  # No display name to compare
            
        display_domains = re.findall(r'\b[\w-]+\.\w{2,}\b', display_name.group(1).lower())
        if not display_domains:
            return 0.0
            
        return 0.0 if any(d in actual_domain for d in display_domains) else 1.0

    def _avg_url_length(self, links: List[str]) -> float:
        """Calculate average length of URLs in email"""
        if not links:
            return 0.0
        total = sum(len(link) for link in links)
        return total / len(links)

    def _ssl_cert_score(self, links: List[str]) -> float:
        """Calculate HTTPS adoption rate among links"""
        if not links:
            return 1.0  # No links = perfect score
        secure = sum(1 for link in links if link.startswith('https://'))
        return secure / len(links)

    def _unusual_sender_score(self, sender: str) -> float:
        """Detect unknown/uncommon senders (placeholder implementation)"""
        trusted_domains = {'example.com', 'trusted.org', 'company.net'}
        domain = sender.split('@')[-1] if '@' in sender else ''
        return 0.0 if domain in trusted_domains else 1.0

    def _unusual_time_score(self, timestamp: float) -> float:
        """Detect off-hour emails (9PM-6AM local time)"""
        if not timestamp:
            return 0.0
        dt = datetime.fromtimestamp(timestamp)
        return 1.0 if 21 <= dt.hour or dt.hour < 6 else 0.0

    def _url_entropy(self, url: str) -> float:
        """Calculate Shannon entropy of URL characters"""
        url = url.lower().strip()
        if not url:
            return 0.0
            
        freq = defaultdict(int)
        for char in url:
            freq[char] += 1
            
        entropy = 0.0
        total = len(url)
        for count in freq.values():
            p = count / total
            entropy -= p * math.log2(p)
            
        return entropy / 8  # Normalize to 0-1 range (max entropy for 256 chars is 8)

    def _num_subdomains(self, url: str) -> float:
        """Count the number of subdomains in the URL"""
        try:
            # Extract domain part (remove protocol and path)
            domain = re.sub(r'^https?://', '', url).split('/')[0]
            parts = domain.split('.')
            # Remove empty parts and www prefix
            parts = [p for p in parts if p != 'www' and p != '']
            return max(len(parts) - 2, 0)  # Subtract main domain (domain + TLD)
        except:
            return 0.0

    def _contains_ip_address(self, url: str) -> float:
        """Check if URL contains an IP address (v4 or v6)"""
        # IPv4 pattern
        ipv4_pattern = r'\b(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
        # Basic IPv6 pattern (simplified check)
        ipv6_pattern = r'\[?[0-9a-fA-F:]+\]?'
        return 1.0 if re.search(ipv4_pattern, url) or re.search(ipv6_pattern, url) else 0.0

    def _https_used(self, url: str) -> float:
        """Check if URL uses HTTPS protocol"""
        return 1.0 if url.startswith('https://') else 0.0

    def _url_redirect_count(self, url: str) -> float:
        """Estimate redirect likelihood through URL structure"""
        redirect_params = ['redirect', 'url', 'goto', 'dest', 'return']
        query = url.split('?')[-1] if '?' in url else ''
        params = query.split('&')
        redirect_count = sum(1 for param in params if any(p in param.lower() for p in redirect_params))
        # Normalize count to 0-1 range (cap at 5 redirects)
        return min(redirect_count / 5, 1.0)

    def _domain_age(self, url: str) -> float:
        """Calculate domain age using WHOIS data"""
        domain = self._extract_domain(url)
        if not domain:
            return 0.8  # Default to suspicious if domain extraction fails
        try:
            import whois
            from whois.exceptions import WhoisCommandFailed, WhoisException
            w = whois.whois(domain)
            
            domain = self._extract_domain(url)
            if not domain:
                return 0.8  # Default to suspicious if domain extraction fails

            # Handle multiple creation dates
            creation_date = w.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
                
            if not creation_date:
                return 0.8  # Suspicious if no creation date
                
            # Calculate age in days
            age_days = (datetime.now() - creation_date).days
            
            # Normalize to 0-1 scale (0=suspicious, 1=trusted)
            if age_days < 30:    # < 1 month
                return 0.2
            elif age_days < 365:  # < 1 year
                return 0.5
            else:                 # > 1 year
                return 0.0

        except ImportError:
            logger.error("python-whois package not installed!")
            return self._domain_age_fallback(domain)
        except (WhoisCommandFailed, WhoisException) as e:
            logger.warning(f"WHOIS lookup failed for {domain}: {e}")
            return 0.5
        except Exception as e:
            logger.error(f"Unexpected error during WHOIS lookup: {e}")
            return 0.5
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        # Remove protocol and path
        domain = re.sub(r'^https?://', '', url).split('/')[0]
        # Remove port number if present
        domain = domain.split(':')[0]
        return domain

    def _domain_age_fallback(self, domain: str) -> float:
        """Fallback method when whois is not available"""
        tld = domain.split('.')[-1].lower() if domain else ''
        established_tlds = {'com', 'org', 'net', 'edu', 'gov'}
        return 0.2 if tld in established_tlds else 0.8

    def _special_char_count(self, url: str) -> float:
        """Count percentage of special characters in URL"""
        total_chars = len(url)
        if total_chars == 0:
            return 0.0
        # Allow alphanumerics and common safe characters
        special_chars = re.findall(r'[^a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=]', url)
        return len(special_chars) / total_chars

    # Network Security Methods
    def monitor_traffic(self, packet: Dict):
        """Analyze network traffic patterns"""
        if self._detect_anomalous_payload(packet):
            logger.warning(f"Anomalous payload detected from {packet['source_ip']}")
            self._block_ip(packet['source_ip'])

    def _detect_anomalous_payload(self, packet: Dict) -> bool:
        """Check for suspicious payload characteristics"""
        return (
            len(packet.get('payload', '')) > self.input_size_limit or
            self._contains_binary_patterns(packet.get('payload', ''))
        )

    def _contains_binary_patterns(self, data: str) -> bool:
        """Detect potential binary exploits"""
        return bool(re.search(r'[\x00-\x08\x0e-\x1f]', data))

    # Utility Methods
    def _get_client_ip(self) -> str:
        """Get client IP from request context with multi-framework support"""
        try:
            # Flask
            from flask import request as flask_request
            try:
                return flask_request.remote_addr
            except RuntimeError as e:
                # Handle "Working outside of request context" error
                if "Working outside of request context" in str(e):
                    logger.debug("Flask request context not available")
                else:
                    raise
        except ImportError:
            pass
        
        # Fallback for non-web contexts
        return "127.0.0.1"

    def _block_ip(self, ip: str):
        """Block malicious IP"""
        logger.info(f"Blocking potentially malicious IP: {ip}")

    def train_phishing_model(self, model_type: str, training_data: List[Tuple[List[float], List[float]]]):
        """Retrain a specific neural network with new data"""
        
        target_nn: Optional[NeuralNetwork] = None
        model_save_path: Optional[str] = None

        if model_type == 'email':
            target_nn = self.email_nn
            model_save_path = "src/agents/safety/models/email_phishing_model.json"
        elif model_type == 'url':
            target_nn = self.url_nn
            model_save_path = "src/agents/safety/models/url_phishing_model.json"
        else:
            logger.error(f"Unknown model type for training: {model_type}")
            return

        if target_nn and model_save_path:
            logger.info(f"Starting training for {model_type} model...")
            target_nn.train(
                training_data,
                epochs=self.config.get('training_epochs', 50),
                initial_learning_rate=self.config.get('training_lr', 0.001),
                batch_size=self.config.get('training_batch_size', 32),
                validation_data=[], # Add validation data if available
                verbose=True,
                save_best_model_path=model_save_path # Save best during training
            )
            # Optionally, save the final model state regardless of "best" if needed
            # target_nn.save_model(model_save_path) 
            logger.info(f"Training complete for {model_type} model. Saved to {model_save_path}")
        else:
            logger.error(f"Could not find neural network or save path for model type: {model_type}")

# Add to FLASK
# In web framework view (Flask example)
# @app.route('/check-url', methods=['POST'])
# def check_url():
#     url = request.json['url']
#     security_system = AdaptiveSecurity()
#     result = security_system.analyze_url(
#         url,
#         client_ip=security_system._get_client_ip(request)
#     )
#     return jsonify(result)

# In non-web context
# security_system = AdaptiveSecurity()
# result = security_system.analyze_url("http://example.com")

if __name__ == "__main__":
    import time
    config = load_global_config()
    security_system = AdaptiveSecurity()
    security_system._get_client_ip = lambda: "192.168.1.100"
    
    # Example email analysis
    sample_email = {
        "from": "support@fakebank.com",
        "subject": "Urgent: Your Account Needs Verification!",
        "body": "Click here to verify your account: http://malicious.link",
        "links": ["http://malicious.link"],
        "timestamp": time.time()
    }
    email_analysis = security_system.analyze_email(sample_email)
    print(f"Email Analysis: {email_analysis}")

    # Example URL analysis
    url_analysis = security_system.analyze_url("http://phishing-site.com/login")
    print(f"URL Analysis: {url_analysis}")

    # Example supply chain check
    package_check = security_system.check_supply_chain("src/agents/safety/important_library.dll")
    print(f"Supply Chain Check: {package_check}")
