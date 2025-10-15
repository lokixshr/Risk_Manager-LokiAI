from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Environment
    ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    
    # MongoDB Configuration
    MONGODB_URI: str
    MONGO_DB_NAME: str = "loki_agents"
    
    # Redis Configuration
    ENABLE_REDIS: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # API Keys
    ALCHEMY_API_KEY: str
    ALCHEMY_URL: str
    ETHERSCAN_API_KEY: str
    COINGECKO_API_KEY: str
    COINGECKO_BASE_URL: str = "https://api.coingecko.com/api/v3"
    THEGRAPH_API_KEY: str
    CURVE_API_BASE: str = "https://api.curve.fi/api"
    DEFILLAMA_BASE_URL: str = "https://api.llama.fi"
    
    # Risk Manager Configuration
    RISK_MANAGER_PORT: int = 8001
    RATE_LIMIT_PER_MINUTE: int = 60
    BACKGROUND_TASK_INTERVAL: int = 600
    LOKI_URL: str = "http://localhost:8000"
    
    # Risk Calculation Parameters
    LIQUIDATION_THRESHOLD: float = 1.0
    SAFETY_THRESHOLD: float = 1.2
    VAR_CONFIDENCE: float = 0.95
    VOLATILITY_WINDOW_DAYS: int = 7
    
    # ML Model Configuration
    MODEL_RETRAIN_HOURS: int = 24
    MIN_TRAINING_SAMPLES: int = 100
    
    # Application settings
    REFRESH_INTERVAL_SECONDS: int = 600
    DEFAULT_ALLOCATION_USD: float = 1000
    HISTORY_MAX_ENTRIES: int = 5000
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Allow extra fields in .env

# Global settings instance
settings = Settings()

# MongoDB Collection Names
class Collections:
    POSITIONS = "risk_manager_positions"
    ALERTS = "risk_manager_alerts"
    METRICS = "risk_manager_metrics"
    TRAINING_DATA = "risk_manager_training_data"

# Risk Severity Levels
class RiskSeverity:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Alert Types
class AlertType:
    LIQUIDATION_RISK = "liquidation_risk"
    HIGH_VOLATILITY = "high_volatility"
    OVER_EXPOSURE = "over_exposure"
    LOW_LIQUIDITY = "low_liquidity"
    GAS_SPIKE = "gas_spike"

# Supported Protocols
class SupportedProtocols:
    AAVE_V2 = "aave_v2"
    AAVE_V3 = "aave_v3"
    COMPOUND = "compound"
    CURVE = "curve"
    UNISWAP_V2 = "uniswap_v2"
    UNISWAP_V3 = "uniswap_v3"

# Network Configuration
NETWORK_CONFIG = {
    "ethereum": {
        "chain_id": 1,
        "rpc_url": settings.ALCHEMY_URL,
        "block_explorer": "https://etherscan.io",
        "subgraph_endpoints": {
            "aave_v2": "https://api.thegraph.com/subgraphs/name/aave/protocol-v2",
            "aave_v3": "https://api.thegraph.com/subgraphs/name/aave/protocol-v3",
            "compound": "https://api.thegraph.com/subgraphs/name/graphprotocol/compound-v2",
            "uniswap_v2": "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2",
            "uniswap_v3": "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
        }
    }
}