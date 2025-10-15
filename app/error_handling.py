"""
Advanced error handling system with circuit breakers, retry logic, and graceful degradation
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
import structlog
from contextlib import asynccontextmanager

from tenacity import (
    retry, stop_after_attempt, wait_exponential, 
    retry_if_exception_type, before_sleep_log
)

logger = structlog.get_logger()


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Number of failures before opening
    timeout: int = 60  # Seconds to wait before trying again
    expected_exception: type = Exception
    name: str = "default"


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    state_changed_time: datetime = field(default_factory=datetime.utcnow)
    total_calls: int = 0


class CircuitBreaker:
    """Async circuit breaker implementation"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            self.stats.total_calls += 1
            
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.config.name} moved to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker {self.config.name} is OPEN"
                    )
            
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
            
        except self.config.expected_exception as e:
            await self._on_failure(e)
            raise
    
    async def _on_success(self):
        """Handle successful call"""
        async with self._lock:
            self.stats.failure_count = 0
            self.stats.success_count += 1
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.stats.state_changed_time = datetime.utcnow()
                logger.info(f"Circuit breaker {self.config.name} CLOSED after recovery")
    
    async def _on_failure(self, exception: Exception):
        """Handle failed call"""
        async with self._lock:
            self.stats.failure_count += 1
            self.stats.last_failure_time = datetime.utcnow()
            
            if (self.stats.failure_count >= self.config.failure_threshold and 
                self.state == CircuitState.CLOSED):
                self.state = CircuitState.OPEN
                self.stats.state_changed_time = datetime.utcnow()
                logger.error(
                    f"Circuit breaker {self.config.name} OPENED after {self.stats.failure_count} failures",
                    exception=str(exception)
                )
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.stats.state_changed_time = datetime.utcnow()
                logger.error(
                    f"Circuit breaker {self.config.name} returned to OPEN after test failure",
                    exception=str(exception)
                )
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        return (datetime.utcnow() - self.stats.state_changed_time).total_seconds() >= self.config.timeout
    
    def get_stats(self) -> Dict:
        """Get circuit breaker statistics"""
        return {
            "name": self.config.name,
            "state": self.state.value,
            "failure_count": self.stats.failure_count,
            "success_count": self.stats.success_count,
            "total_calls": self.stats.total_calls,
            "last_failure_time": self.stats.last_failure_time.isoformat() if self.stats.last_failure_time else None,
            "uptime_seconds": (datetime.utcnow() - self.stats.state_changed_time).total_seconds()
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreakerManager:
    """Manages multiple circuit breakers"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def register_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Register a new circuit breaker"""
        config.name = name
        circuit_breaker = CircuitBreaker(config)
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """Get stats for all circuit breakers"""
        return {name: cb.get_stats() for name, cb in self.circuit_breakers.items()}


# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()


# Decorator for automatic circuit breaker protection
def with_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to add circuit breaker protection to async functions"""
    def decorator(func: Callable):
        if config:
            circuit_breaker = circuit_breaker_manager.register_circuit_breaker(name, config)
        else:
            circuit_breaker = circuit_breaker_manager.get_circuit_breaker(name)
            if not circuit_breaker:
                default_config = CircuitBreakerConfig(name=name)
                circuit_breaker = circuit_breaker_manager.register_circuit_breaker(name, default_config)
        
        async def wrapper(*args, **kwargs):
            return await circuit_breaker.call(func, *args, **kwargs)
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator


# Enhanced retry decorators with structured logging
def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
):
    """Retry decorator with exponential backoff"""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=base_delay, max=max_delay),
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(logger, structlog.stdlib.ERROR)
    )


class ErrorCollector:
    """Collects and analyzes errors for better observability"""
    
    def __init__(self):
        self.errors: List[Dict] = []
        self.error_counts: Dict[str, int] = {}
        self.max_errors = 1000  # Keep last 1000 errors
    
    def record_error(self, error: Exception, context: Dict[str, Any] = None):
        """Record an error with context"""
        error_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": type(error).__name__,
            "message": str(error),
            "context": context or {},
            "traceback": None  # Could add traceback if needed
        }
        
        self.errors.append(error_info)
        
        # Maintain size limit
        if len(self.errors) > self.max_errors:
            self.errors = self.errors[-self.max_errors:]
        
        # Update error counts
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Log the error
        logger.error(
            "Error recorded",
            error_type=error_type,
            error_message=str(error),
            context=context
        )
    
    def get_error_summary(self, hours: int = 24) -> Dict:
        """Get error summary for the last N hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_errors = [
            error for error in self.errors
            if datetime.fromisoformat(error["timestamp"]) > cutoff_time
        ]
        
        error_types = {}
        for error in recent_errors:
            error_type = error["type"]
            if error_type not in error_types:
                error_types[error_type] = {"count": 0, "examples": []}
            
            error_types[error_type]["count"] += 1
            if len(error_types[error_type]["examples"]) < 3:
                error_types[error_type]["examples"].append({
                    "message": error["message"],
                    "timestamp": error["timestamp"],
                    "context": error["context"]
                })
        
        return {
            "time_window_hours": hours,
            "total_errors": len(recent_errors),
            "error_types": error_types,
            "most_common_errors": sorted(
                error_types.items(), 
                key=lambda x: x[1]["count"], 
                reverse=True
            )[:5]
        }


# Global error collector
error_collector = ErrorCollector()


class GracefulDegradationManager:
    """Manages graceful degradation strategies"""
    
    def __init__(self):
        self.fallback_strategies: Dict[str, Callable] = {}
        self.degradation_levels: Dict[str, int] = {}  # 0 = full service, 1-3 = degraded levels
    
    def register_fallback(self, service_name: str, fallback_func: Callable):
        """Register a fallback function for a service"""
        self.fallback_strategies[service_name] = fallback_func
        logger.info(f"Registered fallback strategy for {service_name}")
    
    def set_degradation_level(self, service_name: str, level: int):
        """Set degradation level for a service (0-3)"""
        self.degradation_levels[service_name] = level
        logger.info(f"Set degradation level for {service_name} to {level}")
    
    def get_degradation_level(self, service_name: str) -> int:
        """Get current degradation level for a service"""
        return self.degradation_levels.get(service_name, 0)
    
    async def execute_with_fallback(self, service_name: str, primary_func: Callable, *args, **kwargs):
        """Execute function with fallback if primary fails"""
        try:
            return await primary_func(*args, **kwargs)
        except Exception as e:
            error_collector.record_error(e, {"service": service_name})
            
            fallback = self.fallback_strategies.get(service_name)
            if fallback:
                logger.warning(f"Primary function failed for {service_name}, using fallback")
                try:
                    return await fallback(*args, **kwargs)
                except Exception as fallback_error:
                    error_collector.record_error(fallback_error, {
                        "service": service_name, 
                        "fallback": True
                    })
                    raise fallback_error
            else:
                logger.error(f"No fallback available for {service_name}")
                raise e


# Global degradation manager
degradation_manager = GracefulDegradationManager()


@asynccontextmanager
async def error_handling_context(
    service_name: str,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    fallback_func: Optional[Callable] = None
):
    """Context manager that provides comprehensive error handling"""
    
    # Set up circuit breaker if config provided
    if circuit_breaker_config:
        circuit_breaker_manager.register_circuit_breaker(service_name, circuit_breaker_config)
    
    # Register fallback if provided
    if fallback_func:
        degradation_manager.register_fallback(service_name, fallback_func)
    
    try:
        yield
    except Exception as e:
        error_collector.record_error(e, {"service": service_name})
        raise


# Health check functions
async def check_circuit_breaker_health() -> Dict:
    """Check health of all circuit breakers"""
    stats = circuit_breaker_manager.get_all_stats()
    
    healthy_count = sum(1 for cb_stats in stats.values() if cb_stats["state"] == "closed")
    total_count = len(stats)
    
    return {
        "status": "healthy" if healthy_count == total_count else "degraded",
        "healthy_breakers": healthy_count,
        "total_breakers": total_count,
        "circuit_breakers": stats
    }


async def check_error_rates() -> Dict:
    """Check current error rates"""
    error_summary = error_collector.get_error_summary(hours=1)
    total_errors = error_summary["total_errors"]
    
    # Consider high error rate as more than 10 errors in the last hour
    status = "healthy" if total_errors < 10 else "unhealthy"
    
    return {
        "status": status,
        "errors_last_hour": total_errors,
        "error_summary": error_summary
    }


# Exception classes for different error scenarios
class ExternalAPIError(Exception):
    """Raised when external API calls fail"""
    pass


class DatabaseError(Exception):
    """Raised when database operations fail"""
    pass


class RiskCalculationError(Exception):
    """Raised when risk calculations fail"""
    pass


class RateLimitError(Exception):
    """Raised when rate limits are exceeded"""
    pass


class ValidationError(Exception):
    """Raised when input validation fails"""
    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass


# Initialize default circuit breakers for common services
def initialize_default_circuit_breakers():
    """Initialize circuit breakers for common services"""
    
    # External API circuit breakers
    api_configs = [
        ("coingecko_api", CircuitBreakerConfig(failure_threshold=3, timeout=30, expected_exception=ExternalAPIError)),
        ("alchemy_api", CircuitBreakerConfig(failure_threshold=5, timeout=60, expected_exception=ExternalAPIError)),
        ("etherscan_api", CircuitBreakerConfig(failure_threshold=3, timeout=30, expected_exception=ExternalAPIError)),
        ("thegraph_api", CircuitBreakerConfig(failure_threshold=3, timeout=45, expected_exception=ExternalAPIError)),
    ]
    
    # Database circuit breakers
    db_configs = [
        ("mongodb", CircuitBreakerConfig(failure_threshold=3, timeout=30, expected_exception=DatabaseError)),
        ("redis", CircuitBreakerConfig(failure_threshold=5, timeout=15, expected_exception=DatabaseError)),
    ]
    
    # Risk calculation circuit breaker
    risk_configs = [
        ("risk_calculation", CircuitBreakerConfig(failure_threshold=10, timeout=60, expected_exception=RiskCalculationError)),
    ]
    
    all_configs = api_configs + db_configs + risk_configs
    
    for name, config in all_configs:
        circuit_breaker_manager.register_circuit_breaker(name, config)
        logger.info(f"Initialized circuit breaker for {name}")


# Call initialization
initialize_default_circuit_breakers()


# Utility functions for common error handling patterns
async def safe_external_api_call(
    api_name: str,
    func: Callable,
    *args,
    fallback_value: Any = None,
    **kwargs
) -> Any:
    """Safely call external API with circuit breaker and fallback"""
    circuit_breaker = circuit_breaker_manager.get_circuit_breaker(f"{api_name}_api")
    
    if circuit_breaker:
        try:
            return await circuit_breaker.call(func, *args, **kwargs)
        except (CircuitBreakerOpenError, ExternalAPIError):
            if fallback_value is not None:
                logger.warning(f"Using fallback value for {api_name} API")
                return fallback_value
            raise
    else:
        return await func(*args, **kwargs)


async def safe_database_operation(
    operation_name: str,
    func: Callable,
    *args,
    fallback_func: Optional[Callable] = None,
    **kwargs
) -> Any:
    """Safely perform database operation with error handling"""
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        error_collector.record_error(e, {"operation": operation_name, "database": True})
        
        if fallback_func:
            logger.warning(f"Database operation {operation_name} failed, trying fallback")
            return await fallback_func(*args, **kwargs)
        
        raise DatabaseError(f"Database operation {operation_name} failed: {str(e)}") from e