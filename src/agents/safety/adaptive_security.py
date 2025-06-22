
import re
import time
import math
import whois
import hashlib
import ipaddress
import json, yaml

from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, request
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Union

from src.agents.safety.utils.config_loader import load_global_config, get_config_section
from src.agents.safety.utils.neural_network import NeuralNetwork
from src.agents.safety.utils.safety_features import SafetyFeatures
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Adaptive Security System")
printer = PrettyPrinter

class AdaptiveSecurity:
    """
    Comprehensive cybersecurity system focused on phishing detection and threat prevention.
    Combines neural networks with heuristic rules for multi-layered protection.
    """
    # Pre-compiled regex patterns for efficient matching
    _IPV4_PATTERN = re.compile(
        r'\b(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.'
        r'(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.'
        r'(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.'
        r'(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    )
    _IPV6_PATTERN = re.compile(
        r'\b(?:[A-F0-9]{1,4}:){7}[A-F0-9]{1,4}\b|'                  # Standard
        r'\b(?:[A-F0-9]{1,4}:){1,7}:\b|'                            # Compressed
        r'\b(?:[A-F0-9]{1,4}:){1,6}:[A-F0-9]{1,4}\b|'               # Mixed
        r'\b::(?:[A-F0-9]{1,4}:){0,5}[A-F0-9]{1,4}\b|'              # Leading compression
        r'\b(?:[A-F0-9]{1,4}:){1,5}::\b',                           # Trailing compression
        re.IGNORECASE
    )
    _IP_WITH_PORT = re.compile(r':\d{1,5}\b')
    _IP_IN_URL = re.compile(r'https?://([\w\.\-]+)')

    def __init__(self):
        self.config = load_global_config()
        self.adaptive_config = get_config_section('adaptive_security')
        self.nn_layer_config = self.adaptive_config.get('layers', [])
        self.rate_limit = self.adaptive_config.get('rate_limit')  # Requests per minute
        self.input_size_limit = self.adaptive_config.get('input_size_limit')  # KB
        self.phishing_threshold = self.adaptive_config.get('phishing_threshold')
        self.email_model_path = self.adaptive_config.get('email_model_path')
        self.url_model_path = self.adaptive_config.get('url_model_path')

        self.safety_features = SafetyFeatures()

        self.request_tracker = defaultdict(lambda: deque(maxlen=100))  # Tracks requests per IP
        self.safe_package_hashes = self._load_trusted_hashes()
        
        # Security thresholds (configurable)
        self.email_nn = self._initialize_neural_network(
            model_type='email',
            num_inputs=11,
            model_path=self.email_model_path
        )
        self.url_nn = self._initialize_neural_network(
            model_type='url',
            num_inputs=8,
            model_path=self.url_model_path
        )
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
        printer.status("ADAPT", "Analyzing email", "info")

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
        printer.status("ADAPT", "Analyzing url", "info")

        features = self._extract_url_features(url)
        return self._analyze_features(features, "url", self.url_nn)

    def _analyze_features(self, features: List[float], source_type: str, model: NeuralNetwork) -> Dict:
        """Common analysis method for all feature types"""
        printer.status("ADAPT", "Analyzing features", "info")

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
        printer.status("ADAPT", "Extracting email features", "info")

        return [
            # Header analysis
            len(email.get('from', '')),
            len(email.get('subject', '')),
            self.safety_features._contains_suspicious_keywords(email.get('subject', '')),
            len(email.get('links', [])),
            
            # Content analysis
            self.safety_features._contains_urgent_language(email.get('body', '')),
            self.safety_features._contains_attachment(email),
            self.safety_features._domain_mismatch_score(email),
            
            # URL features
            self.safety_features._avg_url_length(email.get('links', [])),
            self.safety_features._ssl_cert_score(email.get('links', [])),
            
            # Behavioral features
            self.safety_features._unusual_sender_score(email.get('from')),
            self.safety_features._unusual_time_score(email.get('timestamp'))
        ]

    def _extract_url_features(self, url: str) -> List[float]:
        """Convert URL characteristics to numerical features"""
        printer.status("ADAPT", "Extracting url features", "info")
        
        # Get IP presence as boolean then convert to float
        contains_ip = self._contains_ip_address(url, validate=False)
        
        return [
            float(len(url)),
            float(self.safety_features._url_entropy(url)),
            float(self.safety_features._num_subdomains(url)),
            float(contains_ip),  # Convert bool to float (1.0 or 0.0)
            float(self.safety_features._https_used(url)),
            float(self.safety_features._url_redirect_count(url)),
            float(self._domain_age(url)),
            float(self.safety_features._special_char_count(url))
        ]

    def _domain_age(self, url: str) -> float:
        """Calculate domain age using WHOIS data"""
        printer.status("FEATURES", "Calculating domain age", "info")

        domain = self._extract_domain(url)
        if not domain:
            return 0.8  # Default to suspicious if domain extraction fails
        try:

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
        except Exception as e:
            logger.warning(f"WHOIS lookup failed for {domain}: {e}")
            return 0.5
        except Exception as e:
            logger.error(f"Unexpected error during WHOIS lookup: {e}")
            return 0.5

    def _domain_age_fallback(domain: str) -> float:
        """Fallback method when whois is not available"""
        printer.status("FEATURES", "Fallback...", "info")

        tld = domain.split('.')[-1].lower() if domain else ''
        established_tlds = {'com', 'org', 'net', 'edu', 'gov'}
        return 0.2 if tld in established_tlds else 0.8

    def check_supply_chain(self, file_path: str) -> Dict:
        """Check file against known good hashes"""
        printer.status("ADAPT", "Checking supply chain", "info")

        file_hash = self._calculate_file_hash(file_path)
        return {
            "file_hash": file_hash,
            "is_trusted": file_hash in self.safe_package_hashes.values(),
            "risk_level": "high" if file_hash not in self.safe_package_hashes.values() else "low"
        }

    def _check_input_overload(self) -> bool:
        """Detect potential DDoS/overload attempts"""
        printer.status("ADAPT", "Checking input overload", "info")

        client_ip = self._get_client_ip()
        self.request_tracker[client_ip].append(time.time())
        
        if len(self.request_tracker[client_ip]) > self.rate_limit:
            logger.warning(f"Rate limit exceeded for IP {client_ip}")
            return True
        return False

    def _determine_threat_type(self, features: List[float], source_type: str) -> Optional[str]:
        """Classify threat type based on feature patterns with enhanced detection rules"""
        printer.status("ADAPT", "Determine threat type", "info")
    
        if source_type == "email":
            # Feature indices reference:
            # 0: from_length, 1: subject_length, 2: suspicious_keywords, 
            # 3: link_count, 4: urgent_language, 5: attachment_present,
            # 6: domain_mismatch, 7: avg_url_length, 8: ssl_score, 
            # 9: unusual_sender, 10: unusual_time
            
            # Impersonation attacks
            if features[6] > 0.85:  # High domain mismatch
                if features[9] > 0.75:  # Unusual sender
                    return "Phishing: Targeted Impersonation Attack"
                return "Phishing: Domain Impersonation"
            
            # Malware delivery
            if features[5] > 0.9 and features[3] == 0:  # Attachment with no links
                return "Malware: Suspicious Attachment Delivery"
                
            # Credential harvesting
            if features[3] > 3 and features[7] > 60:  # Multiple long URLs
                if features[8] < 0.4:  # Low SSL security
                    return "Phishing: Credential Harvesting (Low Security)"
                return "Phishing: Credential Harvesting"
            
            # Financial scams
            if features[4] > 0.9 and features[2] > 0.85:  # Urgent + keywords
                if "bank" in self.last_email_content or "paypal" in self.last_email_content:
                    return "Financial Scam: Payment Service Impersonation"
                return "Financial Scam: Urgent Action Required"
            
            # Social engineering
            if features[2] > 0.8 or features[4] > 0.8:
                return "Phishing: Social Engineering"
            
            # Temporal attacks
            if features[10] > 0.95:  # Highly unusual time
                return "Phishing: Temporal Anomaly Attack"
    
        elif source_type == "url":
            # Feature indices reference:
            # 0: url_length, 1: entropy, 2: subdomain_count,
            # 3: contains_ip, 4: https_used, 5: redirect_count,
            # 6: domain_age, 7: special_char_count
            
            # IP-based threats
            if features[3] > 0.9:  # Contains IP address
                if features[4] < 0.5:  # No HTTPS
                    return "Phishing: Direct IP Access (Insecure)"
                return "Phishing: Suspicious IP URL"
            
            # Obfuscation techniques
            if features[1] > 6.0 or features[7] > 15:  # High entropy/special chars
                if features[2] > 4:  # Multiple subdomains
                    return "Phishing: Multi-layer Obfuscation"
                return "Phishing: URL Obfuscation"
            
            # New domain threats
            if features[6] > 0.85:  # Very new domain
                if features[5] > 2:  # Multiple redirects
                    return "Phishing: New Domain Redirect Chain"
                return "Phishing: New Domain Registration"
            
            # Redirect-based threats
            if features[5] > 3:  # Excessive redirects
                return "Phishing: Multi-hop Redirect"
            
            # HTTPS spoofing
            if features[4] > 0.9 and features[1] > 5.5:  # HTTPS with high entropy
                return "Phishing: HTTPS Spoofing"
            
            # Subdomain attacks
            if features[2] > 5:  # Excessive subdomains
                return "Phishing: Subdomain Spam"
    
        # Advanced threat patterns
        if self._detect_advanced_threat_pattern(features, source_type):
            return "Advanced Persistent Threat (APT)"
            
        return None
    
    def _detect_advanced_threat_pattern(self, features: List[float], source_type: str) -> bool:
        """Detect sophisticated threat patterns using combinatorial logic"""
        if source_type == "email":
            # Combination of medium-risk features
            risk_score = (
                0.3 * features[6] +  # Domain mismatch
                0.2 * features[9] +  # Unusual sender
                0.25 * features[4] +  # Urgent language
                0.15 * features[2] +  # Suspicious keywords
                0.1 * features[10]   # Unusual time
            )
            return risk_score > 0.65 and features[3] > 1
            
        elif source_type == "url":
            # Combinatorial logic
            return (
                features[3] > 0.7 and  # IP-like
                features[6] > 0.6 and  # New domain
                features[1] > 4.5 and  # High entropy
                features[5] > 1       # Redirects
            )
        return False

    # Security Helper Methods
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file"""
        printer.status("ADAPT", "Calculating file hash", "info")

        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(4096):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _contains_ip_address(self, text: str, *,
                            validate: bool = True,
                            check_urls: bool = True,
                            allow_private: bool = False,
                            allow_reserved: bool = False) -> Union[bool, Tuple[bool, Optional[str]]]:
        """
        Enhanced IP address detection with validation and flexible options
        
        Args:
            text: Input string to check
            validate: Verify if IP is valid and routable
            check_urls: Extract IPs/hostnames from URLs
            allow_private: Consider private IPs as valid
            allow_reserved: Consider reserved IPs as valid
            
        Returns:
            bool: True if IP found (basic mode)
            Tuple: (bool, reason) in validation mode
        """
        printer.status("ADAPT", "Detecting IP address", "info")

        if not text:
            return False if not validate else (False, "Empty input")

        # Check for IPs in URLs if enabled
        if check_urls:
            url_match = self._IP_IN_URL.search(text)
            if url_match:
                host = url_match.group(1)
                if self._is_ip_like(host):
                    text += f" {host}"  # Add extracted host for IP checking

        # Check for IPv4 matches
        ipv4_matches = self._IPV4_PATTERN.findall(text)
        if ipv4_matches:
            for match in ipv4_matches:
                if isinstance(match, tuple):  # Regex groups
                    ip_str = '.'.join(match)
                else:
                    ip_str = match
                
                if validate:
                    valid, reason = self._validate_ip(ip_str, 4, allow_private, allow_reserved)
                    if valid:
                        return (True, reason) if validate else True
                else:
                    return True

        # Check for IPv6 matches
        ipv6_matches = self._IPV6_PATTERN.findall(text)
        for ip_str in ipv6_matches:
            # Clean IPv6 string (remove brackets, ports, etc)
            clean_ip = self._clean_ipv6(ip_str)
            
            if validate:
                valid, reason = self._validate_ip(clean_ip, 6, allow_private, allow_reserved)
                if valid:
                    return (True, reason) if validate else True
            else:
                return True

        return False if not validate else (False, "No valid IP found")

    def _is_ip_like(self, host: str) -> bool:
        """Check if a string looks like an IP address"""
        printer.status("ADAPT", "Checking if a string looks like an IP address", "info")

        return ('.' in host and any(char.isdigit() for char in host)) or (':' in host)

    def _clean_ipv6(self, ip_str: str) -> str:
        """Clean and normalize IPv6 address string"""
        printer.status("ADAPT", "Cleaning and normalizing IPv6 address string", "info")

        # Remove port if present
        if self._IP_WITH_PORT.search(ip_str):
            ip_str = ip_str.rsplit(':', 1)[0]
        
        # Remove brackets
        ip_str = ip_str.strip('[]')
        
        # Normalize compression
        if '::' in ip_str:
            parts = ip_str.split('::')
            left = parts[0].count(':') if parts[0] else 0
            right = parts[1].count(':') if len(parts) > 1 and parts[1] else 0
            missing = 8 - (left + right)
            ip_str = ip_str.replace('::', ':' + '0:' * missing, 1)
        
        return ip_str

    def _validate_ip(self, ip_str: str, version: int, 
                    allow_private: bool, allow_reserved: bool) -> Tuple[bool, str]:
        """Validate IP address with security considerations"""
        printer.status("ADAPT", "Validating IP address with security considerations", "info")

        try:
            ip = ipaddress.ip_address(ip_str)
            
            # Check if version matches
            if ip.version != version:
                return False, f"IP version mismatch ({ip.version} vs {version})"
            
            # Check for private addresses
            if ip.is_private and not allow_private:
                return True, "Private IP found"
            
            # Check for reserved addresses
            if ip.is_reserved and not allow_reserved:
                return True, "Reserved IP found"
            
            # Check for special addresses
            if ip.is_loopback or ip.is_link_local or ip.is_multicast:
                return True, "Special-use IP found"
            
            # Check global address if required
            if ip.is_global:
                return True, "Public routable IP found"
                
            return False, "Invalid IP address"
            
        except ValueError:
            return False, "Invalid IP format"
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        printer.status("ADAPT", "Extracting domain", "info")

        # Remove protocol and path
        domain = re.sub(r'^https?://', '', url).split('/')[0]
        # Remove port number if present
        domain = domain.split(':')[0]
        return domain

    # Network Security Methods
    def monitor_traffic(self, packet: Dict):
        """Analyze network traffic patterns"""
        printer.status("ADAPT", "Analyzing network traffic patterns", "info")

        if self._detect_anomalous_payload(packet):
            logger.warning(f"Anomalous payload detected from {packet['source_ip']}")
            self._block_ip(packet['source_ip'])

    def _detect_anomalous_payload(self, packet: Dict) -> bool:
        """Check for suspicious payload characteristics"""
        printer.status("ADAPT", "Checking for suspicious payload characteristics", "info")

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
            model_save_path = self.email_model_path
        elif model_type == 'url':
            target_nn = self.url_nn
            model_save_path = self.url_model_path
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


app = Flask(__name__)

@app.route('/check-url', methods=['POST'])
def check_url():
    url = request.json['url']
    security_system = AdaptiveSecurity()
    result = security_system.analyze_url(
        url,
        client_ip=security_system._get_client_ip(request)
    )
    return jsonify(result)

if __name__ == "__main__":
    print("\n=== Running Adaptive Cyber Security Test ===\n")
    printer.status("Init", "Adaptive Cyber Security initialized", "success")

    config = load_global_config()
    security_system = AdaptiveSecurity()
    print(security_system)
    print("\n* * * * * Phase 2 * * * * *\n")
    
    security_system._get_client_ip = lambda: "192.168.1.100"
    # Basic detection
    print(security_system._contains_ip_address("Server: 192.168.1.1"))  # True
    
    # Validated check
    valid, reason = security_system._contains_ip_address(
        "Visit https://[2001:db8::1]/admin",
        validate=True,
        check_urls=True
    )
    print(valid)   # True
    print(reason)  # "Public routable IP found"
    
    # Private IP check
    valid, reason = security_system._contains_ip_address(
        "Internal service at 10.0.0.5:3000",
        validate=True,
        allow_private=False
    )
    print(valid)   # True
    print(reason)  # "Private IP found"

    # Example email analysis
    sample_email = {
        "from": "support@fakebank.com",
        "subject": "Urgent: Your Account Needs Verification! Please contact us as soon a possible.",
        "body": "Click here to verify your account: http://malicious.link",
        "links": ["http://malicious.link"],
        "timestamp": time.time()
    }
    email_analysis = security_system.analyze_email(sample_email)
    printer.pretty(f"Email Analysis:", email_analysis, "success")

    # Example URL analysis
    url_analysis = security_system.analyze_url("http://phishing-site.com/login")
    printer.pretty(f"URL Analysis:", url_analysis, "success")

    # Example supply chain check
    package_check = security_system.check_supply_chain("src/agents/safety/important_library.dll")
    printer.pretty(f"Supply Chain Check:", package_check, "success")

    print("\n=== Successfully Ran Adaptive Cyber Security ===\n")
    app.run(debug=True)
