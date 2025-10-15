"""
Risk Manager Agent - DeFi Portfolio Risk Monitoring Microservice

This package provides a production-level FastAPI microservice for monitoring
DeFi portfolio positions across major protocols to detect and prevent
liquidation, over-exposure, and volatility risks.

Key Features:
- Real-time risk monitoring across Aave, Compound, Curve
- Live data from CoinGecko, Alchemy, Etherscan, DefiLlama APIs
- Health ratio calculation and liquidation risk assessment
- Volatility & VaR analysis with historical price data
- Concentration and exposure risk scoring
- Automated alert generation and management
- Background task processing every 10 minutes
- Optional XGBoost ML model for liquidation prediction
- Redis-based rate limiting and caching
- MongoDB Atlas for data storage
- Structured logging to LokiAI

Author: LokiAI Development Team
Version: 1.0.0
License: Proprietary
"""

__version__ = "1.0.0"
__author__ = "LokiAI Development Team"
__email__ = "dev@lokiai.com"
__license__ = "Proprietary"

from .main import app
from .config import settings

__all__ = ["app", "settings"]