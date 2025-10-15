from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime

from .config import settings
from .database import db_manager
from .routes import router
from .background_tasks import background_task_manager
from .security import log_to_loki

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Application lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    startup_start_time = time.time()
    logger.info("Starting Risk Manager Agent")
    
    try:
        # Connect to databases
        await db_manager.connect()
        logger.info("Database connections established")
        
        # Start background tasks
        await background_task_manager.start()
        logger.info("Background tasks started")
        
        # Get initial wallet count
        from .database import get_collection
        from .config import Collections
        
        metrics_collection = get_collection(Collections.METRICS)
        tracked_wallets = await metrics_collection.distinct("wallet_address")
        wallet_count = len(tracked_wallets)
        
        startup_duration = time.time() - startup_start_time
        
        logger.info("✅ Risk Manager Agent ready", 
                   tracking_wallets=wallet_count,
                   startup_time_seconds=round(startup_duration, 2))
        
        # Log startup to LokiAI
        await log_to_loki("risk_manager_started", {
            "tracked_wallets": wallet_count,
            "startup_time_seconds": startup_duration,
            "version": "1.0.0"
        })
        
        yield
        
    except Exception as e:
        logger.error("Failed to start Risk Manager Agent", error=str(e))
        raise
    
    # Shutdown
    logger.info("Shutting down Risk Manager Agent")
    
    try:
        # Stop background tasks
        await background_task_manager.stop()
        logger.info("Background tasks stopped")
        
        # Disconnect from databases
        await db_manager.disconnect()
        logger.info("Database connections closed")
        
        # Log shutdown
        await log_to_loki("risk_manager_stopped", {
            "shutdown_at": datetime.utcnow().isoformat()
        })
        
        logger.info("Risk Manager Agent shutdown complete")
        
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))

# Create FastAPI app
app = FastAPI(
    title="Risk Manager Agent",
    description="Production-level FastAPI microservice for DeFi portfolio risk monitoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info("Request received", 
               method=request.method,
               url=str(request.url),
               client_ip=request.client.host if request.client else "unknown")
    
    try:
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info("Request completed",
                   method=request.method,
                   url=str(request.url),
                   status_code=response.status_code,
                   process_time=round(process_time, 3))
        
        # Add process time header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        
        logger.error("Request failed",
                    method=request.method,
                    url=str(request.url),
                    error=str(e),
                    process_time=round(process_time, 3))
        
        # Log error to LokiAI
        await log_to_loki("request_error", {
            "method": request.method,
            "url": str(request.url),
            "error": str(e),
            "client_ip": request.client.host if request.client else "unknown"
        }, "error")
        
        raise

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    
    logger.error("Unhandled exception",
                method=request.method,
                url=str(request.url),
                error=str(exc),
                error_type=type(exc).__name__)
    
    # Log to LokiAI
    await log_to_loki("unhandled_exception", {
        "method": request.method,
        "url": str(request.url),
        "error": str(exc),
        "error_type": type(exc).__name__,
        "client_ip": request.client.host if request.client else "unknown"
    }, "error")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": id(request)
        }
    )

# HTTP exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler for HTTP exceptions"""
    
    logger.warning("HTTP exception",
                  method=request.method,
                  url=str(request.url),
                  status_code=exc.status_code,
                  detail=exc.detail)
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Include routes
app.include_router(router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Risk Manager Agent",
        "version": "1.0.0",
        "description": "DeFi portfolio risk monitoring microservice",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "health": "/api/risk/health",
            "status": "/api/risk/status",
            "summary": "/api/risk/summary?wallet={wallet_address}",
            "analyze": "/api/risk/analyze",
            "alerts": "/api/risk/alerts",
            "docs": "/docs"
        },
        "authentication": "Requires x-wallet-address header",
        "rate_limit": f"{settings.RATE_LIMIT_PER_MINUTE} requests per minute per wallet"
    }

# Additional metadata endpoint
@app.get("/api/info")
async def get_service_info():
    """Get detailed service information"""
    from .database import db_manager
    
    try:
        # Get database health
        db_health = await db_manager.health_check()
        
        return {
            "service": {
                "name": "Risk Manager Agent",
                "version": "1.0.0",
                "description": "Real-time DeFi portfolio risk monitoring",
                "environment": settings.ENV,
                "port": settings.RISK_MANAGER_PORT
            },
            "features": {
                "protocols": ["Aave V2", "Aave V3", "Compound", "Curve"],
                "risk_metrics": [
                    "Health ratio monitoring",
                    "Liquidation risk analysis",
                    "Volatility & VaR calculation",
                    "Concentration risk scoring",
                    "Real-time alerts"
                ],
                "data_sources": [
                    "The Graph subgraphs",
                    "CoinGecko API",
                    "Alchemy RPC",
                    "Etherscan API",
                    "DefiLlama API"
                ]
            },
            "database": db_health,
            "configuration": {
                "background_task_interval": settings.BACKGROUND_TASK_INTERVAL,
                "rate_limit_per_minute": settings.RATE_LIMIT_PER_MINUTE,
                "safety_threshold": settings.SAFETY_THRESHOLD,
                "liquidation_threshold": settings.LIQUIDATION_THRESHOLD,
                "volatility_window_days": settings.VOLATILITY_WINDOW_DAYS
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error getting service info", error=str(e))
        return JSONResponse(
            status_code=500,
            content={"error": "Unable to retrieve service information"}
        )

if __name__ == "__main__":
    import uvicorn
    
    # Configure uvicorn logging
    logging_config = uvicorn.config.LOGGING_CONFIG
    logging_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logger.info("Starting Risk Manager Agent server",
                host="0.0.0.0",
                port=settings.RISK_MANAGER_PORT,
                environment=settings.ENV)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.RISK_MANAGER_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=settings.ENV == "development",
        access_log=True
    )