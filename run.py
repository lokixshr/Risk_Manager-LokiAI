#!/usr/bin/env python3
"""
Risk Manager Agent Startup Script

This script starts the Risk Manager Agent FastAPI microservice with proper
configuration and error handling.

Usage:
    python run.py [--port PORT] [--host HOST] [--env ENV]

Environment Variables:
    RISK_MANAGER_PORT: Port to run the service on (default: 8001)
    ENV: Environment (development/production)
    LOG_LEVEL: Logging level (DEBUG/INFO/WARNING/ERROR)
"""

import argparse
import sys
import os
import asyncio
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import after path setup
try:
    import uvicorn
    from app.main import app
    from app.config import settings
    from app.ml_model import ml_predictor
    import structlog
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)

# Configure logging
logger = structlog.get_logger()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Risk Manager Agent - DeFi Portfolio Risk Monitoring"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=settings.RISK_MANAGER_PORT,
        help=f"Port to run the service on (default: {settings.RISK_MANAGER_PORT})"
    )
    
    parser.add_argument(
        "--host", "-H",
        type=str,
        default="0.0.0.0",
        help="Host to bind the service to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--env", "-e",
        type=str,
        choices=["development", "production"],
        default=settings.ENV,
        help=f"Environment mode (default: {settings.ENV})"
    )
    
    parser.add_argument(
        "--reload", "-r",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=settings.LOG_LEVEL,
        help=f"Log level (default: {settings.LOG_LEVEL})"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    return parser.parse_args()

async def initialize_ml_model():
    """Initialize ML model if available"""
    try:
        logger.info("Initializing ML model...")
        success = await ml_predictor.load_or_train_model()
        if success:
            logger.info("✅ ML model initialized successfully")
        else:
            logger.warning("⚠️ ML model initialization failed - predictions will be disabled")
    except Exception as e:
        logger.error("Error initializing ML model", error=str(e))

def validate_environment():
    """Validate environment setup"""
    errors = []
    
    # Check required environment variables
    required_vars = [
        "MONGODB_URI",
        "COINGECKO_API_KEY",
        "ALCHEMY_API_KEY",
        "ETHERSCAN_API_KEY"
    ]
    
    for var in required_vars:
        if not os.getenv(var):
            errors.append(f"Missing required environment variable: {var}")
    
    # Check MongoDB URI format
    mongo_uri = os.getenv("MONGODB_URI")
    if mongo_uri and not mongo_uri.startswith("mongodb"):
        errors.append("Invalid MONGODB_URI format")
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    if errors:
        print("❌ Environment validation failed:")
        for error in errors:
            print(f"   - {error}")
        print("\nPlease check your .env file and ensure all required variables are set.")
        return False
    
    return True

def print_startup_banner():
    """Print startup banner with service information"""
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           🚀 Risk Manager Agent                              ║
║                   DeFi Portfolio Risk Monitoring Service                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Version: 1.0.0                                                              ║
║ Port: {settings.RISK_MANAGER_PORT:<10} Environment: {settings.ENV:<20}                       ║
║ Database: MongoDB Atlas                                                      ║
║ Cache: Redis {'(enabled)' if settings.ENABLE_REDIS else '(disabled)':<10}                                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Supported Protocols: Aave V2/V3, Compound, Curve                           ║
║ Risk Metrics: Health Ratio, VaR, Volatility, Concentration                  ║
║ Background Tasks: Every 10 minutes                                          ║
║ Rate Limiting: {settings.RATE_LIMIT_PER_MINUTE} req/min per wallet                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ API Endpoints:                                                               ║
║ • GET  /api/risk/status       - System status                               ║
║ • GET  /api/risk/summary      - Wallet risk summary                         ║
║ • POST /api/risk/analyze      - Manual risk analysis                        ║
║ • GET  /api/risk/alerts       - Risk alerts                                 ║
║ • GET  /docs                  - API documentation                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

async def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Print banner
    print_startup_banner()
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Initialize ML model in background
    asyncio.create_task(initialize_ml_model())
    
    # Configure uvicorn
    uvicorn_config = {
        "app": "app.main:app",
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level.lower(),
        "access_log": True,
        "reload": args.reload or args.env == "development",
        "workers": args.workers if args.env == "production" else 1,
    }
    
    try:
        logger.info("Starting Risk Manager Agent", 
                   host=args.host, 
                   port=args.port, 
                   env=args.env,
                   workers=args.workers)
        
        # Start the server
        uvicorn.run(**uvicorn_config)
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error("Failed to start server", error=str(e))
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Risk Manager Agent stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)