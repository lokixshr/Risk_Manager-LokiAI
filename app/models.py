from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
from .config import RiskSeverity, AlertType, SupportedProtocols

# Base Models
class TimestampedModel(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

# Asset Models
class AssetPrice(BaseModel):
    symbol: str
    price_usd: float
    market_cap: Optional[float] = None
    volume_24h: Optional[float] = None
    price_change_24h: Optional[float] = None
    volatility_7d: Optional[float] = None
    last_updated: datetime

class AssetBalance(BaseModel):
    token_address: str
    symbol: str
    balance: float
    balance_usd: float
    decimals: int = 18

# Position Models
class LendingPosition(BaseModel):
    protocol: str
    token_address: str
    symbol: str
    supplied_amount: float = 0.0
    supplied_usd: float = 0.0
    borrowed_amount: float = 0.0
    borrowed_usd: float = 0.0
    collateral_factor: float = 0.0
    liquidation_threshold: float = 0.0
    apy_supply: float = 0.0
    apy_borrow: float = 0.0

class PositionSnapshot(TimestampedModel):
    wallet_address: str
    protocol: str
    chain_id: int = 1
    
    # Position details
    positions: List[LendingPosition] = []
    total_supplied_usd: float = 0.0
    total_borrowed_usd: float = 0.0
    total_collateral_usd: float = 0.0
    
    # Risk metrics
    health_ratio: Optional[float] = None
    liquidation_price: Optional[float] = None
    available_borrow_usd: float = 0.0
    
    # Additional metadata
    block_number: Optional[int] = None
    transaction_hash: Optional[str] = None

# Risk Models
class VolatilityMetrics(BaseModel):
    asset_symbol: str
    daily_volatility: float
    weekly_volatility: float
    monthly_volatility: float
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    max_drawdown: float
    sharpe_ratio: Optional[float] = None

class ExposureMetrics(BaseModel):
    wallet_address: str
    protocol: str
    
    # Concentration risk
    concentration_score: float  # 0-1, higher = more concentrated
    largest_position_pct: float
    
    # Liquidity risk
    liquidity_score: float  # 0-1, higher = more liquid
    avg_daily_volume_usd: float
    
    # Volatility exposure
    weighted_volatility: float
    correlation_risk: float
    
    # Overall exposure score
    exposure_score: float
    risk_level: str  # low, medium, high, critical

class RiskAlert(TimestampedModel):
    id: Optional[str] = Field(default=None, alias="_id")
    wallet_address: str
    protocol: str
    alert_type: str
    severity: str
    
    # Alert details
    title: str
    description: str
    recommendation: str
    
    # Risk data
    current_value: float
    threshold_value: float
    risk_score: float
    
    # Status
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    
    # Additional context
    metadata: Dict[str, Any] = {}

class RiskSummary(BaseModel):
    wallet_address: str
    last_updated: datetime
    
    # Overall risk metrics
    overall_risk_score: float  # 0-100
    risk_level: str
    health_ratio: Optional[float] = None
    
    # Position metrics
    total_portfolio_value_usd: float
    total_debt_usd: float
    net_worth_usd: float
    
    # Risk breakdown
    liquidation_risk: float  # 0-100
    volatility_risk: float   # 0-100
    concentration_risk: float # 0-100
    liquidity_risk: float    # 0-100
    
    # Predictions
    predicted_liquidation_prob_24h: Optional[float] = None
    predicted_liquidation_prob_7d: Optional[float] = None
    
    # Protocol breakdown
    protocols: List[str] = []
    protocol_exposure: Dict[str, float] = {}
    
    # Active alerts
    active_alerts_count: int = 0
    critical_alerts_count: int = 0

# API Request/Response Models
class WalletAnalysisRequest(BaseModel):
    wallet_address: str
    force_refresh: bool = False
    protocols: Optional[List[str]] = None
    
    @validator('wallet_address')
    def validate_wallet_address(cls, v):
        if not v.startswith('0x') or len(v) != 42:
            raise ValueError('Invalid Ethereum wallet address format')
        return v.lower()

class SystemStatus(BaseModel):
    status: str = "operational"
    version: str = "1.0.0"
    uptime_seconds: int
    tracked_wallets: int
    active_alerts: int
    
    # Service health
    database_status: Dict[str, Any]
    external_apis_status: Dict[str, str]
    
    # Performance metrics
    avg_response_time_ms: float
    requests_per_minute: int
    error_rate_percent: float

class AlertsResponse(BaseModel):
    total_count: int
    alerts: List[RiskAlert]
    pagination: Optional[Dict[str, Any]] = None

# ML Training Data Models
class TrainingDataPoint(TimestampedModel):
    wallet_address: str
    protocol: str
    
    # Features for ML model
    health_ratio: float
    total_supplied_usd: float
    total_borrowed_usd: float
    volatility_score: float
    liquidity_score: float
    concentration_score: float
    gas_price_gwei: float
    
    # Market features
    market_fear_greed_index: Optional[float] = None
    btc_price_change_24h: Optional[float] = None
    eth_price_change_24h: Optional[float] = None
    
    # Target variable
    liquidated_within_24h: bool = False
    liquidated_within_7d: bool = False
    liquidation_occurred_at: Optional[datetime] = None

# Gas and Network Models
class GasMetrics(BaseModel):
    chain_id: int
    standard_gas_price: float  # gwei
    fast_gas_price: float     # gwei
    instant_gas_price: float  # gwei
    
    # Gas price impact on liquidation
    liquidation_gas_cost_usd: float
    safe_gas_threshold: float

# Protocol-specific models
class AavePositionData(BaseModel):
    user_address: str
    total_collateral_eth: float
    total_debt_eth: float
    available_borrow_eth: float
    current_liquidation_threshold: float
    health_factor: float
    
class CompoundPositionData(BaseModel):
    user_address: str
    total_collateral_value_usd: float
    total_borrow_value_usd: float
    account_liquidity: float
    shortfall: float
    
class CurvePoolData(BaseModel):
    pool_address: str
    pool_name: str
    virtual_price: float
    total_supply: float
    liquidity_usd: float
    daily_volume_usd: float

# Configuration Models
class RiskThresholds(BaseModel):
    liquidation_warning: float = 1.2
    liquidation_critical: float = 1.1
    volatility_high: float = 0.05  # 5% daily volatility
    volatility_critical: float = 0.10  # 10% daily volatility
    concentration_warning: float = 0.5  # 50% of portfolio in single asset
    concentration_critical: float = 0.8  # 80% of portfolio in single asset

class MonitoringConfig(BaseModel):
    wallet_address: str
    enabled: bool = True
    alert_channels: List[str] = ["database"]  # database, webhook, email
    custom_thresholds: Optional[RiskThresholds] = None
    protocols_to_monitor: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)