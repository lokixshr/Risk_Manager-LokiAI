"""
Production monitoring, metrics collection, and observability system
"""
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from contextlib import asynccontextmanager
import structlog
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor

from .config import settings
from .error_handling import error_collector, circuit_breaker_manager

logger = structlog.get_logger()


@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = "count"


@dataclass
class HealthCheckResult:
    """Health check result"""
    service: str
    status: str  # healthy, degraded, unhealthy
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MetricsCollector:
    """Collects and stores application metrics"""
    
    def __init__(self, max_points_per_metric: int = 1000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def increment(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        with self._lock:
            self.counters[name] += value
            self._record_point(name, self.counters[name], tags or {}, "count")
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric"""
        with self._lock:
            self.gauges[name] = value
            self._record_point(name, value, tags or {}, "gauge")
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram value"""
        with self._lock:
            self.histograms[name].append(value)
            # Keep only last 1000 values
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
            self._record_point(name, value, tags or {}, "histogram")
    
    def _record_point(self, name: str, value: float, tags: Dict[str, str], unit: str):
        """Record a metric point"""
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags,
            unit=unit
        )
        self.metrics[name].append(point)
    
    def get_metric_summary(self, name: str, hours: int = 24) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_points = [
            point for point in self.metrics[name]
            if point.timestamp > cutoff_time
        ]
        
        if not recent_points:
            return {"error": "No data available"}
        
        values = [point.value for point in recent_points]
        
        return {
            "name": name,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else 0,
            "time_window_hours": hours
        }
    
    def get_all_current_metrics(self) -> Dict[str, Any]:
        """Get current values for all metrics"""
        with self._lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histogram_counts": {name: len(values) for name, values in self.histograms.items()}
            }


class SystemMetricsCollector:
    """Collects system-level metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self._process = psutil.Process()
    
    def collect_system_metrics(self):
        """Collect system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.set_gauge("system.cpu.usage_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics.set_gauge("system.memory.usage_percent", memory.percent)
            self.metrics.set_gauge("system.memory.available_mb", memory.available / (1024 * 1024))
            
            # Process metrics
            process_memory = self._process.memory_info()
            self.metrics.set_gauge("process.memory.rss_mb", process_memory.rss / (1024 * 1024))
            self.metrics.set_gauge("process.memory.vms_mb", process_memory.vms / (1024 * 1024))
            self.metrics.set_gauge("process.cpu.percent", self._process.cpu_percent())
            
            # Thread count
            self.metrics.set_gauge("process.threads.count", self._process.num_threads())
            
            # File descriptors (Unix only)
            try:
                self.metrics.set_gauge("process.fds.count", self._process.num_fds())
            except AttributeError:
                pass  # Windows doesn't have this
                
        except Exception as e:
            logger.error("Error collecting system metrics", error=str(e))


class HealthChecker:
    """Performs health checks on various system components"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.health_checks: Dict[str, Callable] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
    
    def register_health_check(self, name: str, check_func: Callable):
        """Register a health check function"""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    async def run_health_check(self, name: str) -> HealthCheckResult:
        """Run a single health check"""
        if name not in self.health_checks:
            return HealthCheckResult(
                service=name,
                status="unknown",
                details={"error": "Health check not found"}
            )
        
        start_time = time.time()
        try:
            check_func = self.health_checks[name]
            
            # Handle both sync and async health check functions
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    result = await loop.run_in_executor(executor, check_func)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Standardize result format
            if isinstance(result, dict):
                status = result.get("status", "healthy")
                details = {k: v for k, v in result.items() if k != "status"}
            elif isinstance(result, bool):
                status = "healthy" if result else "unhealthy"
                details = {}
            else:
                status = "healthy"
                details = {"result": str(result)}
            
            health_result = HealthCheckResult(
                service=name,
                status=status,
                latency_ms=latency_ms,
                details=details
            )
            
            self.last_results[name] = health_result
            
            # Record metrics
            self.metrics.set_gauge(f"health_check.{name}.latency_ms", latency_ms)
            self.metrics.set_gauge(f"health_check.{name}.status", 1 if status == "healthy" else 0)
            
            return health_result
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            
            health_result = HealthCheckResult(
                service=name,
                status="unhealthy",
                latency_ms=latency_ms,
                details={"error": str(e)}
            )
            
            self.last_results[name] = health_result
            
            # Record metrics
            self.metrics.set_gauge(f"health_check.{name}.latency_ms", latency_ms)
            self.metrics.set_gauge(f"health_check.{name}.status", 0)
            
            logger.error(f"Health check failed: {name}", error=str(e))
            return health_result
    
    async def run_all_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        tasks = [
            self.run_health_check(name) 
            for name in self.health_checks.keys()
        ]
        
        if not tasks:
            return {}
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        health_results = {}
        for i, result in enumerate(results):
            name = list(self.health_checks.keys())[i]
            if isinstance(result, Exception):
                health_results[name] = HealthCheckResult(
                    service=name,
                    status="unhealthy",
                    details={"error": str(result)}
                )
            else:
                health_results[name] = result
        
        return health_results
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        if not self.last_results:
            return {"status": "unknown", "services": {}}
        
        healthy_count = sum(1 for result in self.last_results.values() if result.status == "healthy")
        degraded_count = sum(1 for result in self.last_results.values() if result.status == "degraded")
        unhealthy_count = sum(1 for result in self.last_results.values() if result.status == "unhealthy")
        
        total_services = len(self.last_results)
        
        if unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return {
            "status": overall_status,
            "healthy_services": healthy_count,
            "degraded_services": degraded_count,
            "unhealthy_services": unhealthy_count,
            "total_services": total_services,
            "services": {name: result.status for name, result in self.last_results.items()}
        }


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.alert_rules: List[Dict] = []
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_handlers: List[Callable] = []
    
    def add_alert_rule(self, rule: Dict):
        """Add an alert rule
        
        Example rule:
        {
            "name": "high_error_rate",
            "condition": lambda metrics: metrics.get_metric_summary("api_errors", hours=1)["avg"] > 10,
            "message": "High error rate detected",
            "severity": "critical"
        }
        """
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule['name']}")
    
    def register_alert_handler(self, handler: Callable):
        """Register an alert handler function"""
        self.alert_handlers.append(handler)
    
    async def check_alerts(self):
        """Check all alert rules and trigger alerts if needed"""
        for rule in self.alert_rules:
            try:
                if rule["condition"](self.metrics):
                    alert = {
                        "name": rule["name"],
                        "message": rule["message"],
                        "severity": rule.get("severity", "warning"),
                        "timestamp": datetime.utcnow().isoformat(),
                        "metadata": rule.get("metadata", {})
                    }
                    
                    await self._trigger_alert(alert)
                    
            except Exception as e:
                logger.error(f"Error checking alert rule {rule['name']}", error=str(e))
    
    async def _trigger_alert(self, alert: Dict):
        """Trigger an alert"""
        self.alert_history.append(alert)
        self.metrics.increment("alerts.triggered", tags={"severity": alert["severity"]})
        
        logger.error("Alert triggered", **alert)
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error("Alert handler failed", handler=str(handler), error=str(e))


class RequestTracker:
    """Tracks API request metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
    
    @asynccontextmanager
    async def track_request(self, endpoint: str, method: str = "GET"):
        """Context manager to track request metrics"""
        start_time = time.time()
        
        # Increment request counter
        self.metrics.increment("api.requests.total", tags={"endpoint": endpoint, "method": method})
        
        try:
            yield
            
            # Record successful request
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_histogram("api.request.duration_ms", duration_ms, 
                                        tags={"endpoint": endpoint, "method": method})
            self.metrics.increment("api.requests.success", tags={"endpoint": endpoint, "method": method})
            
        except Exception as e:
            # Record failed request
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_histogram("api.request.duration_ms", duration_ms,
                                        tags={"endpoint": endpoint, "method": method, "status": "error"})
            self.metrics.increment("api.requests.error", tags={"endpoint": endpoint, "method": method})
            raise


class MonitoringManager:
    """Main monitoring manager that coordinates all monitoring components"""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.system_metrics = SystemMetricsCollector(self.metrics)
        self.health_checker = HealthChecker(self.metrics)
        self.alert_manager = AlertManager(self.metrics)
        self.request_tracker = RequestTracker(self.metrics)
        
        self._monitoring_task = None
        self._is_running = False
        
        # Initialize default health checks
        self._register_default_health_checks()
        self._register_default_alert_rules()
    
    def _register_default_health_checks(self):
        """Register default health checks"""
        
        async def database_health():
            """Check database health"""
            try:
                from .database import db_manager
                result = await db_manager.health_check()
                
                mongodb_status = result.get("mongodb", {}).get("status", "disconnected")
                redis_status = result.get("redis", {}).get("status", "disconnected")
                
                if mongodb_status == "connected" and redis_status == "connected":
                    return {"status": "healthy", "details": result}
                elif mongodb_status == "connected":
                    return {"status": "degraded", "details": result}
                else:
                    return {"status": "unhealthy", "details": result}
                    
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}
        
        def system_resources():
            """Check system resources"""
            try:
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                if memory.percent > 90 or cpu_percent > 90:
                    return {"status": "unhealthy", "memory_percent": memory.percent, "cpu_percent": cpu_percent}
                elif memory.percent > 80 or cpu_percent > 80:
                    return {"status": "degraded", "memory_percent": memory.percent, "cpu_percent": cpu_percent}
                else:
                    return {"status": "healthy", "memory_percent": memory.percent, "cpu_percent": cpu_percent}
                    
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}
        
        async def external_apis_health():
            """Check external APIs health"""
            try:
                from .error_handling import circuit_breaker_manager
                circuit_stats = circuit_breaker_manager.get_all_stats()
                
                api_breakers = {k: v for k, v in circuit_stats.items() if "_api" in k}
                
                if not api_breakers:
                    return {"status": "healthy", "details": "No API circuit breakers"}
                
                open_breakers = [name for name, stats in api_breakers.items() if stats["state"] == "open"]
                
                if len(open_breakers) > len(api_breakers) // 2:  # More than half are open
                    return {"status": "unhealthy", "open_breakers": open_breakers}
                elif open_breakers:
                    return {"status": "degraded", "open_breakers": open_breakers}
                else:
                    return {"status": "healthy", "api_breakers": len(api_breakers)}
                    
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}
        
        # Register health checks
        self.health_checker.register_health_check("database", database_health)
        self.health_checker.register_health_check("system_resources", system_resources)
        self.health_checker.register_health_check("external_apis", external_apis_health)
    
    def _register_default_alert_rules(self):
        """Register default alert rules"""
        
        # High error rate alert
        self.alert_manager.add_alert_rule({
            "name": "high_error_rate",
            "condition": lambda metrics: error_collector.get_error_summary(hours=1)["total_errors"] > 20,
            "message": "High error rate detected: >20 errors in the last hour",
            "severity": "critical"
        })
        
        # High memory usage alert
        self.alert_manager.add_alert_rule({
            "name": "high_memory_usage",
            "condition": lambda metrics: psutil.virtual_memory().percent > 85,
            "message": "High memory usage detected",
            "severity": "warning"
        })
        
        # High CPU usage alert
        self.alert_manager.add_alert_rule({
            "name": "high_cpu_usage",
            "condition": lambda metrics: psutil.cpu_percent(interval=0.1) > 85,
            "message": "High CPU usage detected",
            "severity": "warning"
        })
        
        # Multiple circuit breakers open
        self.alert_manager.add_alert_rule({
            "name": "multiple_circuit_breakers_open",
            "condition": lambda metrics: len([
                cb for cb in circuit_breaker_manager.get_all_stats().values()
                if cb["state"] == "open"
            ]) >= 3,
            "message": "Multiple circuit breakers are open",
            "severity": "critical"
        })
    
    async def start_monitoring(self, interval_seconds: int = 60):
        """Start the monitoring loop"""
        if self._is_running:
            logger.warning("Monitoring is already running")
            return
        
        self._is_running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop(interval_seconds))
        logger.info(f"Started monitoring with {interval_seconds}s interval")
    
    async def stop_monitoring(self):
        """Stop the monitoring loop"""
        self._is_running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped monitoring")
    
    async def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop"""
        while self._is_running:
            try:
                # Collect system metrics
                self.system_metrics.collect_system_metrics()
                
                # Run health checks
                await self.health_checker.run_all_health_checks()
                
                # Check alerts
                await self.alert_manager.check_alerts()
                
                # Log monitoring cycle
                logger.debug("Monitoring cycle completed")
                
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
            
            await asyncio.sleep(interval_seconds)
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_health": self.health_checker.get_overall_health(),
            "system_metrics": self.metrics.get_all_current_metrics(),
            "circuit_breakers": circuit_breaker_manager.get_all_stats(),
            "error_summary": error_collector.get_error_summary(hours=24),
            "recent_alerts": list(self.alert_manager.alert_history)[-10:] if self.alert_manager.alert_history else []
        }
    
    async def health_check_endpoint(self) -> Dict[str, Any]:
        """Health check endpoint for load balancers"""
        health_results = await self.health_checker.run_all_health_checks()
        overall_health = self.health_checker.get_overall_health()
        
        status_code = 200
        if overall_health["status"] == "unhealthy":
            status_code = 503
        elif overall_health["status"] == "degraded":
            status_code = 200  # Still accepting traffic but degraded
        
        return {
            "status": overall_health["status"],
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "service": "risk_manager",
            "health_checks": {name: result.status for name, result in health_results.items()},
            "status_code": status_code
        }


# Global monitoring instance
monitoring_manager = MonitoringManager()


# Convenience functions and decorators
def track_function_calls(func_name: str):
    """Decorator to track function calls"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            monitoring_manager.metrics.increment(f"function_calls.{func_name}")
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                monitoring_manager.metrics.record_histogram(f"function_duration.{func_name}", duration_ms)
                return result
            except Exception as e:
                monitoring_manager.metrics.increment(f"function_errors.{func_name}")
                raise
        
        def sync_wrapper(*args, **kwargs):
            monitoring_manager.metrics.increment(f"function_calls.{func_name}")
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                monitoring_manager.metrics.record_histogram(f"function_duration.{func_name}", duration_ms)
                return result
            except Exception as e:
                monitoring_manager.metrics.increment(f"function_errors.{func_name}")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


@asynccontextmanager
async def monitor_operation(operation_name: str):
    """Context manager to monitor an operation"""
    async with monitoring_manager.request_tracker.track_request(operation_name, "OPERATION"):
        yield