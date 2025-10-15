import pytest
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import Dict, Any

import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Set test environment
os.environ["ENV"] = "test"
os.environ["LOG_LEVEL"] = "ERROR"  # Reduce noise in tests
os.environ["MONGODB_URI"] = "mongodb://test:test@localhost:27017/test_db"
os.environ["ENABLE_REDIS"] = "false"  # Disable Redis for tests
os.environ["COINGECKO_API_KEY"] = "test_key"
os.environ["ALCHEMY_API_KEY"] = "test_key"
os.environ["ETHERSCAN_API_KEY"] = "test_key"
os.environ["THEGRAPH_API_KEY"] = "test_key"

from app.main import app
from app.database import db_manager
from app.models import RiskSummary, RiskAlert, AavePositionData, CompoundPositionData

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)

@pytest_asyncio.fixture
async def async_client():
    """Create an async test client"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def sample_wallet_address():
    """Sample Ethereum wallet address for testing"""
    return "0x742b4c0d8fd9b2b29e70dc3e08f4e98a78b3a2b5"

@pytest.fixture
def sample_headers(sample_wallet_address):
    """Sample headers for API requests"""
    return {"x-wallet-address": sample_wallet_address}

@pytest.fixture
def sample_risk_summary(sample_wallet_address):
    """Sample risk summary for testing"""
    return RiskSummary(
        wallet_address=sample_wallet_address,
        last_updated=datetime.utcnow(),
        overall_risk_score=45.2,
        risk_level="medium",
        health_ratio=1.5,
        total_portfolio_value_usd=10000.0,
        total_debt_usd=6000.0,
        net_worth_usd=4000.0,
        liquidation_risk=25.0,
        volatility_risk=40.0,
        concentration_risk=50.0,
        liquidity_risk=30.0,
        protocols=["aave_v3", "compound"],
        protocol_exposure={"aave_v3": 0.7, "compound": 0.3},
        active_alerts_count=2,
        critical_alerts_count=0
    )

@pytest.fixture
def sample_risk_alert(sample_wallet_address):
    """Sample risk alert for testing"""
    return RiskAlert(
        wallet_address=sample_wallet_address,
        protocol="aave_v3",
        alert_type="liquidation_risk",
        severity="high",
        title="High Liquidation Risk",
        description="Health ratio is approaching liquidation threshold",
        recommendation="Consider adding more collateral or repaying debt",
        current_value=1.2,
        threshold_value=1.1,
        risk_score=75.0,
        timestamp=datetime.utcnow()
    )

@pytest.fixture
def sample_aave_position():
    """Sample Aave position data"""
    return AavePositionData(
        user_address="0x742b4c0d8fd9b2b29e70dc3e08f4e98a78b3a2b5",
        total_collateral_eth=5.0,
        total_debt_eth=3.0,
        available_borrow_eth=1.0,
        current_liquidation_threshold=0.8,
        health_factor=1.33
    )

@pytest.fixture
def sample_compound_position():
    """Sample Compound position data"""
    return CompoundPositionData(
        user_address="0x742b4c0d8fd9b2b29e70dc3e08f4e98a78b3a2b5",
        total_collateral_value_usd=10000.0,
        total_borrow_value_usd=6000.0,
        account_liquidity=1000.0,
        shortfall=0.0
    )

@pytest.fixture
def mock_price_data():
    """Mock historical price data"""
    return [
        {"timestamp": datetime.utcnow(), "price": 100.0},
        {"timestamp": datetime.utcnow(), "price": 102.0},
        {"timestamp": datetime.utcnow(), "price": 98.0},
        {"timestamp": datetime.utcnow(), "price": 105.0},
        {"timestamp": datetime.utcnow(), "price": 97.0},
        {"timestamp": datetime.utcnow(), "price": 103.0},
        {"timestamp": datetime.utcnow(), "price": 101.0}
    ]

@pytest.fixture
def mock_database():
    """Mock database collections"""
    mock_db = MagicMock()
    
    # Mock collections
    mock_positions = AsyncMock()
    mock_alerts = AsyncMock()
    mock_metrics = AsyncMock()
    
    mock_db.__getitem__ = lambda self, key: {
        "risk_manager_positions": mock_positions,
        "risk_manager_alerts": mock_alerts,
        "risk_manager_metrics": mock_metrics
    }.get(key)
    
    return mock_db

@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    mock_redis = AsyncMock()
    mock_redis.ping.return_value = True
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.incr.return_value = 1
    mock_redis.expire.return_value = True
    return mock_redis

# Mock external API responses
@pytest.fixture
def mock_coingecko_response():
    """Mock CoinGecko API response"""
    return {
        "0xa0b86991c431c24b0b13715bf94de7b15c7b96000": {
            "usd": 1.0,
            "usd_market_cap": 50000000000,
            "usd_24h_vol": 2000000000,
            "usd_24h_change": 0.1
        }
    }

@pytest.fixture
def mock_aave_subgraph_response():
    """Mock Aave subgraph response"""
    return {
        "data": {
            "user": {
                "reserves": [
                    {
                        "currentATokenBalance": "1000000000000000000000",
                        "currentStableDebt": "0",
                        "currentVariableDebt": "500000000000000000000",
                        "reserve": {
                            "symbol": "USDC",
                            "decimals": 6,
                            "liquidationThreshold": "8500",
                            "price": {
                                "priceInEth": "0.0005"
                            }
                        }
                    }
                ]
            }
        }
    }

@pytest.fixture
def mock_compound_response():
    """Mock Compound API response"""
    return {
        "accounts": [
            {
                "total_collateral_value_in_eth": {"value": "5.0"},
                "total_borrow_value_in_eth": {"value": "3.0"},
                "liquidity": {"value": "1.0"},
                "shortfall": {"value": "0.0"}
            }
        ]
    }

@pytest.fixture
def mock_etherscan_gas_response():
    """Mock Etherscan gas price response"""
    return {
        "result": {
            "SafeGasPrice": "20",
            "ProposeGasPrice": "25",
            "FastGasPrice": "30"
        }
    }

# Database setup and teardown
@pytest_asyncio.fixture
async def setup_test_db():
    """Setup test database"""
    # Mock database connection for tests
    with patch('app.database.db_manager.connect') as mock_connect, \
         patch('app.database.db_manager.disconnect') as mock_disconnect:
        mock_connect.return_value = None
        mock_disconnect.return_value = None
        yield