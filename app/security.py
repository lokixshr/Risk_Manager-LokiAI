import re
import asyncio
import httpx
from datetime import datetime, timedelta
from typing import Optional, Dict
import structlog
from .config import settings
from .database import get_redis

logger = structlog.get_logger()

# Wallet address validation regex
WALLET_ADDRESS_PATTERN = re.compile(r'^0x[a-fA-F0-9]{40}$')

def verify_wallet_address(wallet_address: str) -> bool:
    """Verify if wallet address is valid Ethereum format"""
    if not wallet_address:
        return False
    return bool(WALLET_ADDRESS_PATTERN.match(wallet_address))

class RateLimiter:
    """Redis-based rate limiter for API requests"""
    
    def __init__(self):
        self.window_seconds = 60  # 1 minute window
        self.max_requests = settings.RATE_LIMIT_PER_MINUTE
    
    async def check_rate_limit(self, wallet_address: str, client_ip: str) -> bool:
        """Check if request is within rate limits"""
        redis_client = get_redis()
        
        if not redis_client:
            # If Redis is not available, allow request but log warning
            logger.warning("Redis not available for rate limiting", 
                         wallet=wallet_address, ip=client_ip)
            return True
        
        current_time = datetime.utcnow()
        window_start = current_time - timedelta(seconds=self.window_seconds)
        
        # Rate limit key combining wallet and IP
        rate_limit_key = f"rate_limit:{wallet_address}:{client_ip}"
        
        try:
            # Use sliding window rate limiting
            async with redis_client.pipeline(transaction=True) as pipe:
                # Remove expired entries
                await pipe.zremrangebyscore(
                    rate_limit_key, 
                    0, 
                    window_start.timestamp()
                )
                
                # Count current requests in window
                current_count = await pipe.zcard(rate_limit_key)
                
                if current_count >= self.max_requests:
                    logger.warning("Rate limit exceeded", 
                                 wallet=wallet_address, 
                                 ip=client_ip, 
                                 current_count=current_count,
                                 max_requests=self.max_requests)
                    return False
                
                # Add current request to window
                await pipe.zadd(
                    rate_limit_key, 
                    {str(current_time.timestamp()): current_time.timestamp()}
                )
                
                # Set expiration for cleanup
                await pipe.expire(rate_limit_key, self.window_seconds + 10)
                
                await pipe.execute()
                
            return True
            
        except Exception as e:
            logger.error("Error checking rate limit", error=str(e), 
                        wallet=wallet_address, ip=client_ip)
            # On error, allow request (fail open)
            return True
    
    async def get_rate_limit_status(self, wallet_address: str, client_ip: str) -> Dict:
        """Get current rate limit status for a wallet/IP combination"""
        redis_client = get_redis()
        
        if not redis_client:
            return {
                "requests_remaining": self.max_requests,
                "reset_time": datetime.utcnow() + timedelta(seconds=self.window_seconds),
                "window_seconds": self.window_seconds
            }
        
        current_time = datetime.utcnow()
        window_start = current_time - timedelta(seconds=self.window_seconds)
        rate_limit_key = f"rate_limit:{wallet_address}:{client_ip}"
        
        try:
            # Clean up expired entries and count current requests
            async with redis_client.pipeline(transaction=True) as pipe:
                await pipe.zremrangebyscore(
                    rate_limit_key, 
                    0, 
                    window_start.timestamp()
                )
                current_count = await pipe.zcard(rate_limit_key)
                await pipe.execute()
            
            requests_remaining = max(0, self.max_requests - current_count)
            reset_time = current_time + timedelta(seconds=self.window_seconds)
            
            return {
                "requests_remaining": requests_remaining,
                "reset_time": reset_time,
                "window_seconds": self.window_seconds,
                "current_count": current_count,
                "max_requests": self.max_requests
            }
            
        except Exception as e:
            logger.error("Error getting rate limit status", error=str(e))
            return {
                "requests_remaining": self.max_requests,
                "reset_time": current_time + timedelta(seconds=self.window_seconds),
                "window_seconds": self.window_seconds
            }

# Global rate limiter instance
rate_limiter = RateLimiter()

async def check_rate_limit(wallet_address: str, client_ip: str) -> bool:
    """Check rate limit for wallet and IP combination"""
    return await rate_limiter.check_rate_limit(wallet_address, client_ip)

async def get_rate_limit_status(wallet_address: str, client_ip: str) -> Dict:
    """Get rate limit status for wallet and IP combination"""
    return await rate_limiter.get_rate_limit_status(wallet_address, client_ip)

class LokiLogger:
    """Logger that sends structured logs to LokiAI system"""
    
    def __init__(self):
        self.loki_url = settings.LOKI_URL
        self.client = None
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=10.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    async def log(self, event_type: str, data: Dict, level: str = "info"):
        """Send structured log to LokiAI"""
        if not self.client:
            self.client = httpx.AsyncClient(timeout=10.0)
        
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": "risk_manager",
            "event_type": event_type,
            "level": level,
            "data": data
        }
        
        try:
            response = await self.client.post(
                f"{self.loki_url}/api/logs",
                json=log_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code >= 400:
                logger.warning(f"Failed to log to LokiAI: {response.status_code}")
                
        except Exception as e:
            # Don't fail the request if logging fails
            logger.warning(f"Error logging to LokiAI: {str(e)}")

# Global Loki logger instance
loki_logger = LokiLogger()

async def log_to_loki(event_type: str, data: Dict, level: str = "info"):
    """Send log to LokiAI system"""
    try:
        async with loki_logger:
            await loki_logger.log(event_type, data, level)
    except Exception as e:
        logger.warning(f"Failed to log to LokiAI: {str(e)}")

class SecurityMiddleware:
    """Security middleware for request validation and logging"""
    
    @staticmethod
    async def validate_request_headers(headers: Dict[str, str]) -> bool:
        """Validate required security headers"""
        wallet_address = headers.get("x-wallet-address")
        
        if not wallet_address:
            return False
        
        if not verify_wallet_address(wallet_address):
            return False
        
        return True
    
    @staticmethod
    async def log_security_event(event_type: str, wallet_address: str, 
                                client_ip: str, details: Dict = None):
        """Log security-related events"""
        event_data = {
            "wallet_address": wallet_address,
            "client_ip": client_ip,
            "user_agent": details.get("user_agent") if details else None,
            "endpoint": details.get("endpoint") if details else None,
            "details": details or {}
        }
        
        await log_to_loki(f"security_{event_type}", event_data, "warning")

# Request tracking for analytics
class RequestTracker:
    """Track request patterns for analytics and security"""
    
    def __init__(self):
        self.redis_key_prefix = "request_tracker"
    
    async def track_request(self, wallet_address: str, client_ip: str, 
                          endpoint: str, user_agent: Optional[str] = None):
        """Track request for analytics"""
        redis_client = get_redis()
        
        if not redis_client:
            return
        
        try:
            current_time = datetime.utcnow()
            
            # Track requests per wallet per hour
            hour_key = f"{self.redis_key_prefix}:wallet:{wallet_address}:{current_time.hour}"
            await redis_client.incr(hour_key)
            await redis_client.expire(hour_key, 7200)  # 2 hours
            
            # Track unique IPs per wallet
            ip_key = f"{self.redis_key_prefix}:ips:{wallet_address}"
            await redis_client.sadd(ip_key, client_ip)
            await redis_client.expire(ip_key, 86400)  # 24 hours
            
            # Track endpoint usage
            endpoint_key = f"{self.redis_key_prefix}:endpoints:{endpoint}:{current_time.hour}"
            await redis_client.incr(endpoint_key)
            await redis_client.expire(endpoint_key, 7200)  # 2 hours
            
        except Exception as e:
            logger.error("Error tracking request", error=str(e))
    
    async def get_wallet_stats(self, wallet_address: str) -> Dict:
        """Get request statistics for a wallet"""
        redis_client = get_redis()
        
        if not redis_client:
            return {"requests_last_hour": 0, "unique_ips": 0}
        
        try:
            current_hour = datetime.utcnow().hour
            hour_key = f"{self.redis_key_prefix}:wallet:{wallet_address}:{current_hour}"
            ip_key = f"{self.redis_key_prefix}:ips:{wallet_address}"
            
            requests_last_hour = await redis_client.get(hour_key) or 0
            unique_ips = await redis_client.scard(ip_key)
            
            return {
                "requests_last_hour": int(requests_last_hour),
                "unique_ips": unique_ips
            }
            
        except Exception as e:
            logger.error("Error getting wallet stats", error=str(e))
            return {"requests_last_hour": 0, "unique_ips": 0}

# Global request tracker
request_tracker = RequestTracker()

# Security helper functions
async def detect_suspicious_activity(wallet_address: str, client_ip: str) -> bool:
    """Detect potentially suspicious request patterns"""
    stats = await request_tracker.get_wallet_stats(wallet_address)
    
    # Flag if too many requests from single wallet
    if stats["requests_last_hour"] > 100:
        await log_to_loki("suspicious_activity", {
            "wallet_address": wallet_address,
            "client_ip": client_ip,
            "reason": "high_request_volume",
            "requests_last_hour": stats["requests_last_hour"]
        }, "warning")
        return True
    
    # Flag if too many different IPs for single wallet
    if stats["unique_ips"] > 10:
        await log_to_loki("suspicious_activity", {
            "wallet_address": wallet_address,
            "client_ip": client_ip,
            "reason": "multiple_ips",
            "unique_ips": stats["unique_ips"]
        }, "warning")
        return True
    
    return False

def sanitize_input(input_str: str, max_length: int = 100) -> str:
    """Sanitize user input to prevent injection attacks"""
    if not input_str:
        return ""
    
    # Remove any potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', input_str)
    
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized.strip()