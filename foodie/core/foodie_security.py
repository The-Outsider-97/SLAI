
import secrets
import hashlib
import json
import time
import uuid
import re, os

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from foodie.utils.error_handler import (AuthenticationFailure, PermissionDeniedError, DataPrivacyError,
                                        SecurityBreachError, DataValidationError, ConfigurationError,
                                        RateLimitExceededError, SuspiciousActivityError)
from foodie.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Foodie Security")
printer = PrettyPrinter

class FoodieSecurity:
    """
    Handles security-related functionalities for the Foodie delivery platform,
    including authentication, authorization, and security logging.
    """
    def __init__(self):
        try:
            self.config = load_global_config()
            self.security_config = get_config_section('foodie_security')

            # Validate essential configuration
            required_configs = [
                'session_db_path', 'reset_token_db_path', 'rate_limit_db_path',
                'password_rules', 'session_timeout_minutes'
            ]
            for config_key in required_configs:
                if config_key not in self.security_config:
                    raise ConfigurationError(
                        component='foodie_security',
                        param=config_key
                    )

            self.password_rules = self.security_config.get('password_rules', {})
            self.session_timeout = timedelta(minutes=self.security_config.get('session_timeout_minutes'))
            self.max_login_attempts = self.security_config.get('max_login_attempts')
            self.allowed_roles = self.security_config.get('allowed_roles', [])
            self.role_permissions = self.security_config.get('role_permissions', {})
            self.rate_limit_window = self.security_config.get('rate_limit_window')

            # Session database setup
            self.session_db_path = Path(self.security_config.get('session_db_path'))
            self.session_db_path.parent.mkdir(parents=True, exist_ok=True)
            self.sessions = self._load_sessions()

            # Password reset token database setup
            self.reset_token_db_path = Path(self.security_config.get('reset_token_db_path'))
            self.reset_token_db_path.parent.mkdir(parents=True, exist_ok=True)
            self.reset_tokens = self._load_reset_tokens()

            # Rate limiting database setup
            self.rate_limit_db_path = Path(self.security_config.get('rate_limit_db_path'))
            self.rate_limit_db_path.parent.mkdir(parents=True, exist_ok=True)
            self.rate_limits = self._load_rate_limits()

            logger.info("Foodie Security module initialized.")

        except Exception as e:
            logger.critical(f"Critical error initializing FoodieSecurity: {str(e)}")
            raise

    def _load_sessions(self) -> Dict:
        """Load sessions from persistent storage"""
        try:
            if self.session_db_path.exists():
                with open(self.session_db_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading sessions: {str(e)}")
            raise SecurityBreachError(
                resource='session_database',
                access_type='data_retrieval'
            ) from e

    def _load_reset_tokens(self) -> Dict:
        """Load password reset tokens from persistent storage"""
        try:
            if self.reset_token_db_path.exists():
                with open(self.reset_token_db_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading reset tokens: {str(e)}")
            raise SecurityBreachError(
                resource='reset_token_database',
                access_type='data_retrieval'
            ) from e

    def _load_rate_limits(self) -> Dict:
        """Load rate limits from persistent storage"""
        try:
            if self.rate_limit_db_path.exists():
                with open(self.rate_limit_db_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading rate limits: {str(e)}")
            raise SecurityBreachError(
                resource='rate_limit_database',
                access_type='data_retrieval'
            ) from e
        
    def register_safety_callback(self, callback: callable):
        """Register safety agent callback"""
        self.safety_callback = callback

    def _save_sessions(self) -> None:
        """Save sessions to persistent storage"""
        try:
            with open(self.session_db_path, 'w') as f:
                json.dump(self.sessions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving sessions: {str(e)}")
            raise SecurityBreachError(
                resource='session_database',
                access_type='data_storage'
            ) from e

    def _clean_expired_sessions(self) -> None:
        """Remove expired sessions from memory and storage"""
        try:
            now = datetime.utcnow()
            expired = []
            
            for token, session in self.sessions.items():
                expires_at = datetime.fromisoformat(session['expires_at'])
                if expires_at < now:
                    expired.append(token)
            
            for token in expired:
                del self.sessions[token]
            
            if expired:
                logger.info(f"Cleaned {len(expired)} expired sessions")
                self._save_sessions()
        except Exception as e:
            logger.error(f"Error cleaning expired sessions: {str(e)}")
            raise SecurityBreachError(
                resource='session_database',
                access_type='data_cleanup'
            ) from e

    def _save_reset_tokens(self) -> None:
        """Save password reset tokens to persistent storage"""
        try:
            with open(self.reset_token_db_path, 'w') as f:
                json.dump(self.reset_tokens, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving reset tokens: {str(e)}")
            raise SecurityBreachError(
                resource='reset_token_database',
                access_type='data_storage'
            ) from e

    def _clean_expired_reset_tokens(self) -> None:
        """Remove expired reset tokens from memory and storage"""
        try:
            now = datetime.utcnow()
            expired = []
            
            for token, token_data in self.reset_tokens.items():
                expires_at = datetime.fromisoformat(token_data['expires_at'])
                if expires_at < now:
                    expired.append(token)
            
            for token in expired:
                del self.reset_tokens[token]
            
            if expired:
                logger.info(f"Cleaned {len(expired)} expired reset tokens")
                self._save_reset_tokens()
        except Exception as e:
            logger.error(f"Error cleaning expired reset tokens: {str(e)}")
            raise SecurityBreachError(
                resource='reset_token_database',
                access_type='data_cleanup'
            ) from e

    def _save_rate_limits(self) -> None:
        """Save rate limits to persistent storage"""
        try:
            with open(self.rate_limit_db_path, 'w') as f:
                json.dump(self.rate_limits, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving rate limits: {str(e)}")
            raise SecurityBreachError(
                resource='rate_limit_database',
                access_type='data_storage'
            ) from e

    def _clean_old_rate_limits(self) -> None:
        """Remove old rate limit entries from memory and storage"""
        try:
            now = time.time()
            to_remove = []
            
            for key, entry in self.rate_limits.items():
                if now - entry['last_attempt'] > self.rate_limit_window:
                    to_remove.append(key)
            
            for key in to_remove:
                del self.rate_limits[key]
            
            if to_remove:
                self._save_rate_limits()
        except Exception as e:
            logger.error(f"Error cleaning old rate limits: {str(e)}")
            raise SecurityBreachError(
                resource='rate_limit_database',
                access_type='data_cleanup'
            ) from e

    def hash_password(self, password: str) -> Tuple[str, str]:
        """
        Securely hashes a password with a randomly generated salt.
        
        Args:
            password: Plain text password
            
        Returns:
            Tuple: (salt, hashed_password)
        """
        # Add safety validation
        if hasattr(self, 'safety_callback'):
            safety_report = self.safety_callback(
                password, 
                context={"operation": "password_handling"}
            )
            if not safety_report.get("is_safe", False):
                raise SecurityBreachError("Password security validation failed")
        try:
            salt = secrets.token_hex(16)
            salted_password = password.encode() + salt.encode()
            hashed = hashlib.pbkdf2_hmac(
                'sha256',
                salted_password,
                b'',
                100000  # Number of iterations
            )
            return salt, hashed.hex()
        except Exception as e:
            logger.error(f"Password hashing failed: {str(e)}")
            raise SecurityBreachError(
                resource='password_processing',
                access_type='hashing'
            ) from e

    def verify_password(self, password: str, salt: str, stored_hash: str) -> bool:
        """
        Verifies a password against a stored hash.
        
        Args:
            password: Plain text password to verify
            salt: Salt used in original hashing
            stored_hash: Previously stored password hash
            
        Returns:
            bool: True if password matches, False otherwise
        """
        try:
            salted_password = password.encode() + salt.encode()
            hashed = hashlib.pbkdf2_hmac(
                'sha256',
                salted_password,
                b'',
                100000
            )
            return hashed.hex() == stored_hash
        except Exception as e:
            logger.error(f"Password verification failed: {str(e)}")
            raise SecurityBreachError(
                resource='password_processing',
                access_type='verification'
            ) from e

    def validate_password_strength(self, password: str) -> None:
        """
        Validates password against security rules.
        
        Args:
            password: Password to validate
            
        Raises:
            ValueError: If password doesn't meet requirements
        """
        try:
            errors = []
            
            if len(password) < self.password_rules.get('min_length'):
                errors.append(f"Password must be at least {self.password_rules['min_length']} characters")
                
            if self.password_rules.get('require_uppercase') and not re.search(r'[A-Z]', password):
                errors.append("Password must contain at least one uppercase letter")
                
            if self.password_rules.get('require_lowercase') and not re.search(r'[a-z]', password):
                errors.append("Password must contain at least one lowercase letter")
                
            if self.password_rules.get('require_digit') and not re.search(r'[0-9]', password):
                errors.append("Password must contain at least one digit")
                
            if self.password_rules.get('require_special') and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
                errors.append("Password must contain at least one special character")
                
            if errors:
                raise DataValidationError(
                    message="; ".join(errors),
                    field="password"
                )
        except Exception as e:
            logger.error(f"Password strength validation failed: {str(e)}")
            raise

    def generate_session_token(self, user_id: str, roles: List[str]) -> str:
        """
        Generates a secure session token and stores it in the session database.
        
        Args:
            user_id: ID of the authenticated user
            roles: List of user roles
            
        Returns:
            str: Generated session token
        """
        try:
            # Clean expired sessions before creating a new one
            self._clean_expired_sessions()
            
            # Generate secure token
            token = secrets.token_urlsafe(64)
            expires_at = datetime.utcnow() + self.session_timeout
            
            # Create session data
            session_data = {
                "user_id": user_id,
                "roles": roles,
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": expires_at.isoformat(),
                "last_accessed": datetime.utcnow().isoformat()
            }
            
            # Store session
            self.sessions[token] = session_data
            self._save_sessions()
            
            logger.info(f"Generated session token for user: {user_id}")
            return token
        except Exception as e:
            logger.error(f"Session token generation failed for user {user_id}: {str(e)}")
            raise SecurityBreachError(
                resource='session_management',
                access_type='token_generation'
            ) from e

    def validate_session_token(self, token: str) -> Dict:
        """
        Validates a session token and returns user information.
        
        Args:
            token: Session token to validate
            
        Returns:
            Dict: User information if valid
            
        Raises:
            ValueError: If token is invalid or expired
        """
        try:
            # Clean expired sessions first
            self._clean_expired_sessions()
            
            # Check token exists
            if token not in self.sessions:
                raise AuthenticationFailure(
                    entity='session_token',
                    auth_method='token_validation'
                )
            
            session = self.sessions[token]
            expires_at = datetime.fromisoformat(session['expires_at'])
            
            # Check expiration
            if expires_at < datetime.utcnow():
                del self.sessions[token]
                self._save_sessions()
                raise AuthenticationFailure(
                    entity='session_token',
                    auth_method='token_validation',
                    message="Session token has expired"
                )
            
            # Update last accessed time
            session['last_accessed'] = datetime.utcnow().isoformat()
            self.sessions[token] = session
            self._save_sessions()
            
            return {
                "user_id": session["user_id"],
                "roles": session["roles"],
                "expires_at": session["expires_at"]
            }
        except AuthenticationFailure:
            raise
        except Exception as e:
            logger.error(f"Session token validation failed: {str(e)}")
            raise SecurityBreachError(
                resource='session_management',
                access_type='token_validation'
            ) from e

    def revoke_session_token(self, token: str) -> None:
        """
        Revokes/removes a session token from the database
        
        Args:
            token: Session token to revoke
        """
        try:
            if token in self.sessions:
                del self.sessions[token]
                self._save_sessions()
                logger.info(f"Revoked session token: {token}")
        except Exception as e:
            logger.error(f"Session revocation failed for token {token}: {str(e)}")
            raise SecurityBreachError(
                resource='session_management',
                access_type='token_revocation'
            ) from e

    def get_user_sessions(self, user_id: str) -> List[Dict]:
        """
        Retrieves all active sessions for a user
        
        Args:
            user_id: ID of the user
            
        Returns:
            List: Active session data
        """
        try:
            self._clean_expired_sessions()
            return [session for session in self.sessions.values() 
                    if session["user_id"] == user_id]
        except Exception as e:
            logger.error(f"Failed to get sessions for user {user_id}: {str(e)}")
            raise SecurityBreachError(
                resource='session_management',
                access_type='data_retrieval'
            ) from e

    def revoke_all_user_sessions(self, user_id: str) -> None:
        """
        Revokes all sessions for a specific user
        
        Args:
            user_id: ID of the user
        """
        try:
            tokens_to_revoke = []
            for token, session in self.sessions.items():
                if session["user_id"] == user_id:
                    tokens_to_revoke.append(token)
            
            for token in tokens_to_revoke:
                del self.sessions[token]
            
            if tokens_to_revoke:
                self._save_sessions()
                logger.info(f"Revoked {len(tokens_to_revoke)} sessions for user: {user_id}")
        except Exception as e:
            logger.error(f"Failed to revoke sessions for user {user_id}: {str(e)}")
            raise SecurityBreachError(
                resource='session_management',
                access_type='bulk_revocation'
            ) from e

    def check_permission(self, user_roles: List[str], required_permission: str) -> bool:
        """
        Checks if user has required permission based on their roles.
        
        Args:
            user_roles: List of user roles
            required_permission: Permission to check
            
        Returns:
            bool: True if user has permission, False otherwise
        """
        try:
            for role in user_roles:
                permissions = self.role_permissions.get(role, [])
                if required_permission in permissions:
                    return True
            return False
        except Exception as e:
            logger.error(f"Permission check failed: {str(e)}")
            raise PermissionDeniedError(
                operation='permission_verification',
                required_role='system_admin'
            ) from e

    def log_security_event(self, event_type: str, user_id: str, details: Dict) -> None:
        """
        Logs security-related events.
        
        Args:
            event_type: Type of event (login_attempt, password_change, etc.)
            user_id: ID of relevant user
            details: Additional event details
        """
        try:
            # Check for sensitive data that shouldn't be logged
            sensitive_fields = ['password', 'credit_card', 'cvv']
            for field in sensitive_fields:
                if field in details.get('additional_info', {}):
                    raise DataPrivacyError(
                        data_type='sensitive_field',
                        regulation='internal_policy'
                    )
            
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "user_id": user_id,
                "ip_address": details.get("ip_address", "unknown"),
                "user_agent": details.get("user_agent", "unknown"),
                "success": details.get("success", False),
                "additional_info": details.get("additional_info", {})
            }
            logger.warning(f"SECURITY EVENT: {json.dumps(log_entry)}")
        except DataPrivacyError:
            raise
        except Exception as e:
            logger.error(f"Security event logging failed: {str(e)}")

    def rate_limit_check(self, user_id: str, action: str) -> bool:
        """
        Checks if a user has exceeded rate limits for security-sensitive actions.
        
        Args:
            user_id: ID of the user
            action: Type of action (login, password_reset, etc.)
            
        Returns:
            bool: True if action is allowed, False if rate limited
        """
        try:
            # Clean old entries first
            self._clean_old_rate_limits()
            
            # Create rate limit key
            key = f"{user_id}_{action}"
            now = time.time()
            
            # Initialize if new entry
            if key not in self.rate_limits:
                self.rate_limits[key] = {
                    'count': 1,
                    'first_attempt': now,
                    'last_attempt': now
                }
                self._save_rate_limits()
                return True
            
            # Check if user is rate limited
            entry = self.rate_limits[key]
            max_attempts = self.max_login_attempts
            
            # Reset count if outside time window
            if now - entry['first_attempt'] > self.rate_limit_window:
                entry['count'] = 1
                entry['first_attempt'] = now
            else:
                entry['count'] += 1
            
            # Update last attempt time
            entry['last_attempt'] = now
            self.rate_limits[key] = entry
            self._save_rate_limits()
            
            # Check if exceeded max attempts
            if entry['count'] > max_attempts:
                # Detect suspicious patterns
                if entry['count'] > max_attempts * 2:
                    self.log_security_event(
                        "suspicious_rate_activity",
                        user_id,
                        {
                            "action": action,
                            "attempt_count": entry['count'],
                            "time_window": self.rate_limit_window
                        }
                    )
                    raise SuspiciousActivityError(
                        activity_type='excessive_rate_attempts',
                        severity='high'
                    )
                
                raise RateLimitExceededError(
                    endpoint=f"{action}_action",
                    limit=max_attempts
                )
            
            return True
        except (RateLimitExceededError, SuspiciousActivityError):
            raise
        except Exception as e:
            logger.error(f"Rate limit check failed: {str(e)}")
            raise SecurityBreachError(
                resource='rate_limiting',
                access_type='check_processing'
            ) from e

    def generate_password_reset_token(self, user_id: str) -> str:
        """
        Generates a secure password reset token and stores it in the database.
        
        Args:
            user_id: ID of the user
            
        Returns:
            str: Generated reset token
        """
        try:
            # Clean expired tokens first
            self._clean_expired_reset_tokens()
            
            # Generate secure token
            token = secrets.token_urlsafe(32)
            expires_at = datetime.utcnow() + timedelta(minutes=30)
            
            # Create token data
            token_data = {
                "user_id": user_id,
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": expires_at.isoformat(),
                "used": False,
                "ip_address": "unknown",  # Will be set when token is used
                "user_agent": "unknown"   # Will be set when token is used
            }
            
            # Store token
            self.reset_tokens[token] = token_data
            self._save_reset_tokens()
            
            logger.info(f"Generated password reset token for user: {user_id}")
            return token
        except Exception as e:
            logger.error(f"Password reset token generation failed for user {user_id}: {str(e)}")
            raise SecurityBreachError(
                resource='password_reset',
                access_type='token_generation'
            ) from e

    def verify_password_reset_token(self, token: str, user_id: str, request_details: Dict = None) -> bool:
        """
        Verifies a password reset token and marks it as used if valid.
        
        Args:
            token: Token to verify
            user_id: ID of the user
            request_details: Additional request details (ip, user agent)
            
        Returns:
            bool: True if token is valid, False otherwise
        """
        try:
            # Clean expired tokens first
            self._clean_expired_reset_tokens()
            
            # Check token exists
            if token not in self.reset_tokens:
                raise AuthenticationFailure(
                    entity='password_reset_token',
                    auth_method='token_validation',
                    message="Invalid reset token"
                )
            
            token_data = self.reset_tokens[token]
            
            # Check user match
            if token_data['user_id'] != user_id:
                raise AuthenticationFailure(
                    entity='password_reset_token',
                    auth_method='token_validation',
                    message="Token-user mismatch"
                )
                
            # Check expiration
            expires_at = datetime.fromisoformat(token_data['expires_at'])
            if expires_at < datetime.utcnow():
                raise AuthenticationFailure(
                    entity='password_reset_token',
                    auth_method='token_validation',
                    message="Token expired"
                )
                
            # Check if already used
            if token_data['used']:
                raise AuthenticationFailure(
                    entity='password_reset_token',
                    auth_method='token_validation',
                    message="Token already used"
                )
                
            # Update token with request details
            if request_details:
                token_data['ip_address'] = request_details.get('ip_address', 'unknown')
                token_data['user_agent'] = request_details.get('user_agent', 'unknown')
                
            # Mark token as used
            token_data['used'] = True
            token_data['used_at'] = datetime.utcnow().isoformat()
            self.reset_tokens[token] = token_data
            self._save_reset_tokens()
            
            return True
        except AuthenticationFailure:
            raise
        except Exception as e:
            logger.error(f"Password reset token verification failed: {str(e)}")
            raise SecurityBreachError(
                resource='password_reset',
                access_type='token_verification'
            ) from e

    def get_active_reset_tokens(self, user_id: str) -> List[Dict]:
        """
        Retrieves all active (unused and unexpired) reset tokens for a user
        
        Args:
            user_id: ID of the user
            
        Returns:
            List: Active reset token data
        """
        try:
            self._clean_expired_reset_tokens()
            now = datetime.utcnow()
            
            active_tokens = []
            for token, token_data in self.reset_tokens.items():
                if token_data['user_id'] == user_id and not token_data['used']:
                    expires_at = datetime.fromisoformat(token_data['expires_at'])
                    if expires_at > now:
                        active_tokens.append({
                            "token": token,
                            "created_at": token_data['created_at'],
                            "expires_at": token_data['expires_at']
                        })
            
            return active_tokens
        except Exception as e:
            logger.error(f"Failed to get active reset tokens for user {user_id}: {str(e)}")
            raise SecurityBreachError(
                resource='password_reset',
                access_type='token_retrieval'
            ) from e
        
    def validate_action(self, action_params: Dict, context: Dict) -> Dict:
        """Default safety validation logic"""
        operation = action_params.get("operation")
        content = action_params.get("content", "")
        
        # Basic safety checks
        if operation == "add_document":
            if any(bad_word in content.lower() for bad_word in ["malicious", "hack"]):
                return {"approved": False, "reason": "Malicious content detected"}
            return {"approved": True}
        
        # Default approval for other operations
        return {"approved": True}

# Example usage
if __name__ == "__main__":
    printer.status("MAIN", "Testing FoodieSecurity Methods", "info")
    try:
        security = FoodieSecurity()
        
        # 1. Test rate limiting
        for i in range(1, 7):
            try:
                allowed = security.rate_limit_check("user_123", "login")
                printer.pretty(f"LOGIN ATTEMPT {i}", f"Allowed: {allowed}", "info" if allowed else "warning")
            except RateLimitExceededError as e:
                printer.pretty(f"LOGIN ATTEMPT {i}", f"Blocked: {str(e)}", "error")
            except SuspiciousActivityError as e:
                printer.pretty(f"LOGIN ATTEMPT {i}", f"Suspicious: {str(e)}", "critical")
        
        # 2. Test password reset tokens
        try:
            reset_token = security.generate_password_reset_token("user_123")
            printer.pretty("PASSWORD RESET TOKEN", reset_token, "success")
        except SecurityBreachError as e:
            printer.pretty("TOKEN GENERATION ERROR", str(e), "error")
        
        # 3. Get active tokens
        try:
            active_tokens = security.get_active_reset_tokens("user_123")
            printer.pretty("ACTIVE RESET TOKENS", active_tokens, "success")
        except SecurityBreachError as e:
            printer.pretty("TOKEN RETRIEVAL ERROR", str(e), "error")
        
        # 4. Verify token
        try:
            is_valid = security.verify_password_reset_token(
                reset_token, 
                "user_123",
                {
                    "ip_address": "192.168.1.1",
                    "user_agent": "Mozilla/5.0"
                }
            )
            printer.pretty("RESET TOKEN VALIDATION", f"Token valid: {is_valid}", "success")
        except AuthenticationFailure as e:
            printer.pretty("TOKEN VALIDATION ERROR", str(e), "error")
        
        # 5. Try to verify again (should fail)
        try:
            is_valid = security.verify_password_reset_token(reset_token, "user_123")
            printer.pretty("REUSED TOKEN VALIDATION", f"Token valid: {is_valid}", "success" if not is_valid else "error")
        except AuthenticationFailure as e:
            printer.pretty("REUSED TOKEN ERROR", str(e), "error")
        
        # 6. Get active tokens after use
        try:
            active_tokens = security.get_active_reset_tokens("user_123")
            printer.pretty("ACTIVE RESET TOKENS AFTER USE", f"Count: {len(active_tokens)}", "success")
        except SecurityBreachError as e:
            printer.pretty("TOKEN RETRIEVAL ERROR", str(e), "error")
        
        # 7. Test session token validation
        try:
            token = security.generate_session_token("user_123", ["customer"])
            session = security.validate_session_token(token)
            printer.pretty("SESSION VALIDATION", session, "success")
            
            # Test invalid token
            security.validate_session_token("invalid_token")
        except AuthenticationFailure as e:
            printer.pretty("SESSION VALIDATION ERROR", str(e), "error")
        except SecurityBreachError as e:
            printer.pretty("SESSION ERROR", str(e), "error")
        
        # 8. Test password strength validation
        try:
            security.validate_password_strength("weak")
        except DataValidationError as e:
            printer.pretty("PASSWORD VALIDATION ERROR", str(e), "error")
    
    except ConfigurationError as e:
        printer.pretty("CONFIGURATION ERROR", str(e), "critical")
    except Exception as e:
        printer.pretty("UNEXPECTED ERROR", str(e), "critical")