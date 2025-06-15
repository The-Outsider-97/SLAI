
from collections import defaultdict
import math
import re

from typing import Dict, List
from datetime import datetime
from dataclasses import dataclass

from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Safety Features")
printer = PrettyPrinter

class SafetyFeatures:

    _trusted_domains = None
    _malicious_domains = None
    _domain_reputation_cache = {}
    _DOMAIN_CACHE_TTL = 86400  # 24 hours in seconds

# ==============================
# 1. Extracting email features
# ==============================

    @staticmethod
    def _contains_suspicious_keywords(text: str) -> float:
        """Check for phishing-related keywords"""
        printer.status("FEATURES", "Check for phishing-related keywords", "info")

        keywords = {'urgent', 'password', 'verify', 'account', 'suspended'}
        return len(keywords & set(text.lower().split())) / len(keywords)

    @staticmethod
    def _contains_urgent_language(email_body: str) -> float:
        """Calculate urgency score of email content (0-1 scale)"""
        printer.status("FEATURES", "Calculating urgency score", "info")

        urgent_keywords = {
            'urgent', 'immediately', 'action required', 'verify now',
            'account suspended', 'security alert', 'password reset'
        }
        words = re.findall(r'\b\w+\b', email_body.lower())
        matches = [1 for word in words if word in urgent_keywords]
        return min(len(matches)/3, 1.0)  # Cap at 1.0 for 3+ matches

    @staticmethod
    def _contains_attachment(email: Dict) -> float:
        """Check for file attachments (binary check)"""
        printer.status("FEATURES", "Checking file attachments", "info")

        return 1.0 if len(email.get('attachments', [])) > 0 else 0.0

    @staticmethod
    def _domain_mismatch_score(email: Dict) -> float:
        """Compare display name domain with actual sender domain"""
        printer.status("FEATURES", "Comparing scores", "info")

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

    @staticmethod
    def _avg_url_length(links: List[str]) -> float:
        """Calculate average length of URLs in email"""
        printer.status("FEATURES", "Calculating average length", "info")

        if not links:
            return 0.0
        total = sum(len(link) for link in links)
        return total / len(links)

    @staticmethod
    def _ssl_cert_score(links: List[str]) -> float:
        """Calculate HTTPS adoption rate among links"""
        printer.status("FEATURES", "Calculating HTTPS adoption", "info")

        if not links:
            return 1.0  # No links = perfect score
        secure = sum(1 for link in links if link.startswith('https://'))
        return secure / len(links)

    @classmethod
    def _load_domain_lists(cls):
        """Load trusted and malicious domains from files with caching"""
        if cls._trusted_domains is not None and cls._malicious_domains is not None:
            return

        # Default trusted domains (fallback if file not found)
        default_trusted = {'example.com', 'trusted.org', 'company.net', 'yourdomain.com'}
        cls._trusted_domains = set()
        cls._malicious_domains = set()

        # Load trusted domains
        try:
            with open('config/trusted_domains.txt', 'r') as f:
                for line in f:
                    domain = line.strip().lower()
                    if domain and not domain.startswith('#'):
                        cls._trusted_domains.add(domain)
            logger.info(f"Loaded {len(cls._trusted_domains)} trusted domains from file")
        except FileNotFoundError:
            logger.warning("Trusted domains file not found, using defaults")
            cls._trusted_domains = default_trusted

        # Load malicious domains
        try:
            with open('config/malicious_domains.txt', 'r') as f:
                for line in f:
                    domain = line.strip().lower()
                    if domain and not domain.startswith('#'):
                        cls._malicious_domains.add(domain)
            logger.info(f"Loaded {len(cls._malicious_domains)} malicious domains from file")
        except FileNotFoundError:
            logger.warning("Malicious domains file not found, using empty set")

        # Ensure no overlap between trusted and malicious
        cls._trusted_domains -= cls._malicious_domains

    @staticmethod
    def _get_domain_reputation(domain: str) -> float:
        """Check domain reputation using external service (simulated)"""
        # In a real implementation, this would call an API like:
        # - VirusTotal Domain Report
        # - Cisco Talos Intelligence
        # - Google Safe Browsing
        
        # Simulated reputation scores:
        reputation_map = {
            'trusted.org': 0.9,
            'reputable.com': 0.8,
            'new-domain.net': 0.4,
            'shady-site.cc': 0.1
        }
        
        # Return cached value if available
        if domain in SafetyFeatures._domain_reputation_cache:
            entry = SafetyFeatures._domain_reputation_cache[domain]
            if datetime.now().timestamp() - entry['timestamp'] < SafetyFeatures._DOMAIN_CACHE_TTL:
                return entry['score']
        
        # Simulate API lookup
        score = reputation_map.get(domain, 0.5)  # Default to neutral
        
        # Update cache
        SafetyFeatures._domain_reputation_cache[domain] = {
            'score': score,
            'timestamp': datetime.now().timestamp()
        }
        
        return score

    @staticmethod
    def _unusual_sender_score(sender: str) -> float:
        """Enhanced sender reputation scoring with multiple factors"""
        printer.status("FEATURES", "Analyzing sender reputation", "info")
        
        # Initialize domain lists if needed
        SafetyFeatures._load_domain_lists()
        
        # Extract domain from sender address
        if '@' not in sender:
            logger.warning(f"Invalid sender format: {sender}")
            return 1.0  # Treat invalid format as suspicious
            
        domain = sender.split('@')[-1].lower()
        
        # 1. Check against malicious domains list
        if domain in SafetyFeatures._malicious_domains:
            logger.warning(f"Malicious domain detected: {domain}")
            return 1.0  # Definitely malicious
        
        # 2. Check against trusted domains list
        if domain in SafetyFeatures._trusted_domains:
            return 0.0  # Fully trusted
            
        # 3. Domain age heuristic (newer domains are more suspicious)
        domain_age_score = 0.0
        try:
            # In real implementation, use WHOIS lookup
            # Placeholder: TLD-based heuristic
            new_tlds = {'xyz', 'top', 'icu', 'cyou', 'rest', 'shop'}
            tld = domain.split('.')[-1]
            domain_age_score = 0.7 if tld in new_tlds else 0.3
        except:
            domain_age_score = 0.5
        
        # 4. Domain reputation check
        reputation_score = SafetyFeatures._get_domain_reputation(domain)
        
        # 5. String entropy analysis (high entropy = more suspicious)
        entropy = SafetyFeatures._url_entropy(domain)
        entropy_score = min(entropy * 2, 1.0)  # Scale entropy to 0-1 range
        
        # 6. Subdomain count (excessive subdomains = suspicious)
        subdomain_count = SafetyFeatures._num_subdomains(domain)
        subdomain_score = min(subdomain_count / 5, 1.0)  # 5+ subdomains = max score
        
        # Combine scores with weighting
        weights = {
            'age': 0.3,
            'reputation': 0.4,
            'entropy': 0.2,
            'subdomains': 0.1
        }
        
        combined_score = (
            weights['age'] * domain_age_score +
            weights['reputation'] * (1 - reputation_score) +  # Invert reputation
            weights['entropy'] * entropy_score +
            weights['subdomains'] * subdomain_score
        )
        
        # Clamp final score between 0-1
        final_score = max(0.0, min(combined_score, 1.0))
        
        logger.info(
            f"Sender reputation analysis: {domain} | "
            f"Age: {domain_age_score:.2f}, Rep: {reputation_score:.2f}, "
            f"Entropy: {entropy_score:.2f}, Subdomains: {subdomain_score:.2f} | "
            f"Final: {final_score:.2f}"
        )
        
        return final_score

    @staticmethod
    def _unusual_time_score(timestamp: float) -> float:
        """Detect off-hour emails (9PM-6AM local time)"""
        printer.status("FEATURES", "Detecting off-hour emails", "info")

        if not timestamp:
            return 0.0
        dt = datetime.fromtimestamp(timestamp)
        return 1.0 if 21 <= dt.hour or dt.hour < 6 else 0.0

# ==============================
# 2. Extracting url features
# ==============================

    @staticmethod
    def _url_entropy(url: str) -> float:
        """Calculate Shannon entropy of URL characters"""
        printer.status("FEATURES", "Calculating Shannon entropy", "info")

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

    @staticmethod
    def _num_subdomains(url: str) -> float:
        """Count the number of subdomains in the URL"""
        printer.status("FEATURES", "Counting subdomains in the URL", "info")

        try:
            # Extract domain part (remove protocol and path)
            domain = re.sub(r'^https?://', '', url).split('/')[0]
            parts = domain.split('.')
            # Remove empty parts and www prefix
            parts = [p for p in parts if p != 'www' and p != '']
            return max(len(parts) - 2, 0)  # Subtract main domain (domain + TLD)
        except:
            return 0.0

    @staticmethod
    def _https_used(url: str) -> float:
        """Check if URL uses HTTPS protocol"""
        printer.status("FEATURES", "Checking if URL uses HTTPS protocol", "info")

        return 1.0 if url.startswith('https://') else 0.0

    @staticmethod
    def _url_redirect_count(url: str) -> float:
        """Estimate redirect likelihood through URL structure"""
        redirect_params = ['redirect', 'url', 'goto', 'dest', 'return']
        query = url.split('?')[-1] if '?' in url else ''
        params = query.split('&')
        redirect_count = sum(1 for param in params if any(p in param.lower() for p in redirect_params))
        # Normalize count to 0-1 range (cap at 5 redirects)
        return min(redirect_count / 5, 1.0)

    @staticmethod
    def _special_char_count(url: str) -> float:
        """Count percentage of special characters in URL"""
        printer.status("FEATURES", "Counting...", "info")

        total_chars = len(url)
        if total_chars == 0:
            return 0.0
        # Allow alphanumerics and common safe characters
        special_chars = re.findall(r'[^a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=]', url)
        return len(special_chars) / total_chars
