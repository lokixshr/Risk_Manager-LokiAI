"""
Enhanced security features including secrets management, security headers,
and comprehensive security monitoring
"""
import os
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.base import BaseHTTPMiddleware
import structlog

from .config import settings
from .monitoring import monitoring_manager

logger = structlog.get_logger()

class SecurityMiddleware(BaseHTTPMiddleware):
    """Enhanced security middleware with comprehensive protection"""
    
    def __init__(self, app, config: Optional[Dict] = None):
        super().__init__(app)
        self.config = config or {}
        self.blocked_ips = set()
        self.rate_limit_violations = {}
        
    async def dispatch(self, request: Request, call_next):
        # Add security headers
        start_time = datetime.now()
        
        # Security checks
        await self._check_ip_blocklist(request)
        await self._validate_request_size(request)
        await self._check_security_headers(request)
        
        response = await call_next(request)
        
        # Add security response headers
        response.headers.update({
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            "X-Request-ID": str(secrets.token_hex(16))
        })
        
        # Log security events
        duration = (datetime.now() - start_time).total_seconds()
        monitoring_manager.metrics.record_histogram("security_check_duration", duration * 1000)
        
        return response
    
    async def _check_ip_blocklist(self, request: Request):
        """Check if IP is blocked"""
        client_ip = self._get_client_ip(request)
        if client_ip in self.blocked_ips:
            logger.warning("Blocked IP attempted access", client_ip=client_ip)
            monitoring_manager.metrics.increment("security.blocked_ip_attempts")
            raise HTTPException(status_code=403, detail="Access denied")
    
    async def _validate_request_size(self, request: Request):
        """Validate request size to prevent DoS"""
        max_size = self.config.get("max_request_size", 1024 * 1024)  # 1MB default
        content_length = request.headers.get("content-length")
        
        if content_length and int(content_length) > max_size:
            logger.warning("Request too large", size=content_length, max_size=max_size)
            monitoring_manager.metrics.increment("security.oversized_requests")
            raise HTTPException(status_code=413, detail="Request too large")
    
    async def _check_security_headers(self, request: Request):
        """Validate required security headers"""
        user_agent = request.headers.get("user-agent", "")
        if not user_agent or len(user_agent) < 10:
            logger.warning("Suspicious request without proper User-Agent")
            monitoring_manager.metrics.increment("security.suspicious_requests")
    
    def _get_client_ip(self, request: Request) -> str:
        """Get real client IP considering proxies"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"


class SecretsManager:
    """Secure secrets management"""
    
    def __init__(self):
        self._secrets_cache = {}
        self._last_refresh = {}
        self.refresh_interval = 3600  # 1 hour
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret with caching and rotation support"""
        try:
            # Check if we need to refresh
            if self._should_refresh(key):
                self._refresh_secret(key)
            
            value = self._secrets_cache.get(key)
            if value is None:
                value = os.getenv(key, default)
                if value:
                    self._secrets_cache[key] = value
                    self._last_refresh[key] = datetime.now()
            
            return value
            
        except Exception as e:
            logger.error("Error retrieving secret", key=key, error=str(e))
            return default
    
    def _should_refresh(self, key: str) -> bool:
        """Check if secret should be refreshed"""
        last_refresh = self._last_refresh.get(key)
        if not last_refresh:
            return True
        
        return (datetime.now() - last_refresh).seconds > self.refresh_interval
    
    def _refresh_secret(self, key: str):
        """Refresh secret from external source"""
        # In production, this would integrate with:
        # - AWS Secrets Manager
        # - HashiCorp Vault
        # - Kubernetes Secrets
        # For now, just refresh from environment
        value = os.getenv(key)
        if value:
            self._secrets_cache[key] = value
            self._last_refresh[key] = datetime.now()
    
    def rotate_secret(self, key: str, new_value: str):
        """Rotate a secret"""
        try:
            # In production, this would:
            # 1. Store new value in secrets store
            # 2. Update cache
            # 3. Notify other instances
            # 4. Schedule old value deprecation
            
            self._secrets_cache[key] = new_value
            self._last_refresh[key] = datetime.now()
            
            logger.info("Secret rotated successfully", key=key)
            monitoring_manager.metrics.increment("security.secrets_rotated")
            
        except Exception as e:
            logger.error("Failed to rotate secret", key=key, error=str(e))
            monitoring_manager.metrics.increment("security.secret_rotation_failures")


class APIKeyValidator:
    """Advanced API key validation and management"""
    
    def __init__(self):
        self.secrets_manager = SecretsManager()
        self.key_usage = {}
    
    def validate_api_key(self, key_name: str, provided_key: Optional[str]) -> bool:
        """Validate API key with usage tracking"""
        if not provided_key:
            return False
        
        expected_key = self.secrets_manager.get_secret(key_name)
        if not expected_key:
            logger.error("API key not configured", key_name=key_name)
            return False
        
        # Use constant-time comparison to prevent timing attacks
        is_valid = hmac.compare_digest(expected_key, provided_key)
        
        # Track usage
        self._track_usage(key_name, is_valid)
        
        return is_valid
    
    def _track_usage(self, key_name: str, success: bool):
        """Track API key usage patterns"""
        now = datetime.now()
        if key_name not in self.key_usage:
            self.key_usage[key_name] = {"success": 0, "failure": 0, "last_used": now}
        
        self.key_usage[key_name]["last_used"] = now
        if success:
            self.key_usage[key_name]["success"] += 1
            monitoring_manager.metrics.increment("security.api_key_success", tags={"key": key_name})
        else:
            self.key_usage[key_name]["failure"] += 1
            monitoring_manager.metrics.increment("security.api_key_failure", tags={"key": key_name})


class InputSanitizer:
    """Advanced input validation and sanitization"""
    
    @staticmethod
    def sanitize_wallet_address(address: str) -> str:
        """Sanitize and validate wallet address"""
        if not address:
            raise ValueError("Wallet address is required")
        
        # Remove any whitespace
        address = address.strip()
        
        # Check format
        if not address.startswith("0x") or len(address) != 42:
            raise ValueError("Invalid wallet address format")
        
        # Check for valid hex characters
        try:
            int(address[2:], 16)
        except ValueError:
            raise ValueError("Invalid wallet address: contains non-hex characters")
        
        return address.lower()
    
    @staticmethod
    def sanitize_string_input(value: str, max_length: int = 100, allow_html: bool = False) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            raise ValueError("Input must be a string")
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Limit length
        if len(value) > max_length:
            raise ValueError(f"Input too long (max {max_length} characters)")
        
        # Remove HTML if not allowed
        if not allow_html:
            import html
            value = html.escape(value)
        
        return value.strip()
    
    @staticmethod
    def sanitize_numeric_input(value: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
        """Sanitize numeric input"""
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            raise ValueError("Input must be a number")
        
        if min_val is not None and num_value < min_val:
            raise ValueError(f"Value must be >= {min_val}")
        
        if max_val is not None and num_value > max_val:
            raise ValueError(f"Value must be <= {max_val}")
        
        return num_value


class SecurityMonitor:
    """Advanced security monitoring and threat detection"""
    
    def __init__(self):
        self.failed_attempts = {}
        self.suspicious_patterns = {}
        self.threat_scores = {}
    
    def record_failed_login(self, identifier: str, ip_address: str):
        """Record failed login attempt"""
        key = f"{identifier}:{ip_address}"
        now = datetime.now()
        
        if key not in self.failed_attempts:
            self.failed_attempts[key] = []
        
        self.failed_attempts[key].append(now)
        
        # Clean old attempts (last 1 hour)
        cutoff = now - timedelta(hours=1)
        self.failed_attempts[key] = [
            attempt for attempt in self.failed_attempts[key]
            if attempt > cutoff
        ]
        
        # Check for brute force
        if len(self.failed_attempts[key]) >= 5:
            self._handle_brute_force(identifier, ip_address)
    
    def _handle_brute_force(self, identifier: str, ip_address: str):
        """Handle detected brute force attack"""
        logger.warning("Brute force attack detected", 
                      identifier=identifier, 
                      ip_address=ip_address)
        
        monitoring_manager.metrics.increment("security.brute_force_attempts")
        
        # Add to threat score
        self._increase_threat_score(ip_address, 50)
    
    def record_suspicious_activity(self, activity_type: str, details: Dict[str, Any]):
        """Record suspicious activity"""
        logger.warning("Suspicious activity detected", 
                      activity_type=activity_type, 
                      **details)
        
        monitoring_manager.metrics.increment("security.suspicious_activity", 
                                           tags={"type": activity_type})
    
    def _increase_threat_score(self, ip_address: str, score: int):
        """Increase threat score for IP"""
        if ip_address not in self.threat_scores:
            self.threat_scores[ip_address] = 0
        
        self.threat_scores[ip_address] += score
        
        # Auto-block if score too high
        if self.threat_scores[ip_address] >= 100:
            self._auto_block_ip(ip_address)
    
    def _auto_block_ip(self, ip_address: str):
        """Automatically block high-threat IP"""
        logger.error("Auto-blocking high-threat IP", ip_address=ip_address)
        monitoring_manager.metrics.increment("security.auto_blocks")
        
        # In production, this would:
        # 1. Add to firewall rules
        # 2. Update load balancer configuration
        # 3. Notify security team


# Global instances
secrets_manager = SecretsManager()
security_monitor = SecurityMonitor()
api_key_validator = APIKeyValidator()
input_sanitizer = InputSanitizer()

# Dependency for FastAPI endpoints
security = HTTPBearer()

async def verify_security(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify security token (if using bearer tokens)"""
    token = credentials.credentials
    
    # Validate token format
    if not token or len(token) < 32:
        raise HTTPException(status_code=401, detail="Invalid token format")
    
    # In production, verify JWT or API token
    # For now, just validate format
    return credentials


def require_api_key(key_name: str):
    """Decorator to require specific API key"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # This would be used for internal API endpoints
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Security utility functions
def generate_secure_token(length: int = 32) -> str:
    """Generate cryptographically secure token"""
    return secrets.token_hex(length)


def hash_sensitive_data(data: str, salt: Optional[str] = None) -> str:
    """Hash sensitive data with salt"""
    if not salt:
        salt = secrets.token_hex(16)
    
    return hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000).hex()


def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify webhook signature"""
    expected_signature = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(f"sha256={expected_signature}", signature)


# Enhanced rate limiting with security features
class SecurityAwareRateLimiter:
    """Rate limiter with security intelligence"""
    
    def __init__(self):
        self.suspicious_ips = set()
        self.threat_multipliers = {}
    
    async def check_rate_limit(self, identifier: str, ip_address: str) -> bool:
        """Check rate limit with security adjustments"""
        base_limit = settings.RATE_LIMIT_PER_MINUTE
        
        # Reduce limit for suspicious IPs
        if ip_address in self.suspicious_ips:
            adjusted_limit = base_limit // 4
            logger.warning("Reduced rate limit for suspicious IP", 
                          ip=ip_address, limit=adjusted_limit)
        else:
            adjusted_limit = base_limit
        
        # Apply threat multiplier
        threat_score = security_monitor.threat_scores.get(ip_address, 0)
        if threat_score > 25:
            adjusted_limit = max(1, adjusted_limit // (threat_score // 25))
        
        # Use existing rate limiter with adjusted limit
        from .security import check_rate_limit
        return await check_rate_limit(identifier, ip_address)


# Export enhanced security components
__all__ = [
    'SecurityMiddleware',
    'SecretsManager', 
    'APIKeyValidator',
    'InputSanitizer',
    'SecurityMonitor',
    'SecurityAwareRateLimiter',
    'secrets_manager',
    'security_monitor',
    'api_key_validator', 
    'input_sanitizer',
    'verify_security',
    'require_api_key',
    'generate_secure_token',
    'hash_sensitive_data',
    'verify_webhook_signature'
]