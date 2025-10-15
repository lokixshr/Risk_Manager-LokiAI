import pytest
import asyncio
from httpx import AsyncClient
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import json

from app.main import app
from app.models import RiskSummary, RiskAlert, SystemStatus
from app.config import settings


@pytest.mark.asyncio
class TestRiskAPIEndpoints:
    """Integration tests for Risk Manager API endpoints"""

    @pytest.fixture
    async def async_client(self):
        """Create async client for testing"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client

    @pytest.fixture
    def valid_wallet_address(self):
        """Valid Ethereum wallet address for testing"""
        return "0x742b4c0d8fd9b2b29e70dc3e08f4e98a78b3a2b5"

    @pytest.fixture
    def valid_headers(self, valid_wallet_address):
        """Valid request headers"""
        return {"x-wallet-address": valid_wallet_address}

    @pytest.fixture
    def mock_database_operations(self):
        """Mock all database operations"""
        with patch('app.database.get_collection') as mock_get_collection:
            mock_collection = AsyncMock()
            mock_collection.distinct.return_value = ["wallet1", "wallet2", "wallet3"]
            mock_collection.count_documents.return_value = 5
            mock_collection.insert_one.return_value = MagicMock(inserted_id="mock_id")
            mock_collection.find_one.return_value = None
            mock_collection.update_one.return_value = MagicMock(matched_count=1)
            mock_collection.find.return_value.sort.return_value.skip.return_value.limit.return_value.to_list = AsyncMock(return_value=[])
            mock_get_collection.return_value = mock_collection
            yield mock_collection

    @pytest.fixture
    def mock_risk_calculator(self):
        """Mock risk calculator operations"""
        with patch('app.routes.risk_calculator') as mock_calc:
            mock_summary = RiskSummary(
                wallet_address="0x742b4c0d8fd9b2b29e70dc3e08f4e98a78b3a2b5",
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
            mock_calc.aggregate_risk_summary.return_value = mock_summary
            mock_calc.generate_alerts.return_value = []
            yield mock_calc

    @pytest.fixture
    def mock_rate_limiting(self):
        """Mock rate limiting"""
        with patch('app.routes.check_rate_limit') as mock_rate_limit:
            mock_rate_limit.return_value = True
            yield mock_rate_limit

    class TestSystemStatusEndpoint:
        """Test /api/risk/status endpoint"""

        @pytest.mark.asyncio
        async def test_system_status_success(self, async_client, mock_database_operations):
            """Test successful system status retrieval"""
            
            with patch('app.routes.db_manager') as mock_db_manager, \
                 patch('app.routes.api_manager') as mock_api_manager:
                
                # Mock database health check
                mock_db_manager.health_check.return_value = {
                    "mongodb": {"status": "connected", "latency_ms": 15.2},
                    "redis": {"status": "connected", "latency_ms": 5.1}
                }
                
                # Mock API manager health check
                mock_api_manager.health_check.return_value = {
                    "coingecko": "healthy",
                    "alchemy": "healthy",
                    "etherscan": "healthy"
                }
                
                response = await async_client.get("/api/risk/status")
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["status"] == "operational"
                assert data["version"] == "1.0.0"
                assert "uptime_seconds" in data
                assert "tracked_wallets" in data
                assert "active_alerts" in data
                assert "database_status" in data
                assert "external_apis_status" in data

        @pytest.mark.asyncio
        async def test_system_status_database_failure(self, async_client):
            """Test system status when database fails"""
            
            with patch('app.routes.db_manager') as mock_db_manager:
                mock_db_manager.health_check.side_effect = Exception("Database error")
                
                response = await async_client.get("/api/risk/status")
                
                assert response.status_code == 500

    class TestRiskSummaryEndpoint:
        """Test /api/risk/summary endpoint"""

        @pytest.mark.asyncio
        async def test_risk_summary_success(self, async_client, valid_wallet_address, 
                                          valid_headers, mock_database_operations, 
                                          mock_risk_calculator, mock_rate_limiting):
            """Test successful risk summary retrieval"""
            
            response = await async_client.get(
                f"/api/risk/summary?wallet={valid_wallet_address}",
                headers=valid_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["wallet_address"] == valid_wallet_address
            assert data["overall_risk_score"] == 45.2
            assert data["risk_level"] == "medium"
            assert data["health_ratio"] == 1.5
            assert data["total_portfolio_value_usd"] == 10000.0
            assert data["total_debt_usd"] == 6000.0
            assert data["net_worth_usd"] == 4000.0

        @pytest.mark.asyncio
        async def test_risk_summary_missing_header(self, async_client, valid_wallet_address):
            """Test risk summary without required header"""
            
            response = await async_client.get(
                f"/api/risk/summary?wallet={valid_wallet_address}"
            )
            
            assert response.status_code == 401
            assert "Missing x-wallet-address header" in response.json()["detail"]

        @pytest.mark.asyncio
        async def test_risk_summary_invalid_wallet_format(self, async_client):
            """Test risk summary with invalid wallet address format"""
            
            invalid_headers = {"x-wallet-address": "invalid_wallet"}
            
            response = await async_client.get(
                "/api/risk/summary?wallet=invalid_wallet",
                headers=invalid_headers
            )
            
            assert response.status_code == 400
            assert "Invalid wallet address format" in response.json()["detail"]

        @pytest.mark.asyncio
        async def test_risk_summary_wallet_mismatch(self, async_client, valid_headers):
            """Test risk summary with mismatched wallet addresses"""
            
            different_wallet = "0x1234567890123456789012345678901234567890"
            
            response = await async_client.get(
                f"/api/risk/summary?wallet={different_wallet}",
                headers=valid_headers
            )
            
            assert response.status_code == 400
            assert "must match x-wallet-address header" in response.json()["detail"]

        @pytest.mark.asyncio
        async def test_risk_summary_rate_limit_exceeded(self, async_client, valid_wallet_address, 
                                                       valid_headers, mock_database_operations, 
                                                       mock_risk_calculator):
            """Test risk summary with rate limiting"""
            
            with patch('app.routes.check_rate_limit') as mock_rate_limit:
                mock_rate_limit.return_value = False
                
                response = await async_client.get(
                    f"/api/risk/summary?wallet={valid_wallet_address}",
                    headers=valid_headers
                )
                
                assert response.status_code == 429
                assert "Rate limit exceeded" in response.json()["detail"]

        @pytest.mark.asyncio
        async def test_risk_summary_calculation_error(self, async_client, valid_wallet_address, 
                                                     valid_headers, mock_database_operations, 
                                                     mock_rate_limiting):
            """Test risk summary when calculation fails"""
            
            with patch('app.routes.risk_calculator') as mock_calc:
                mock_calc.aggregate_risk_summary.side_effect = Exception("Calculation error")
                
                response = await async_client.get(
                    f"/api/risk/summary?wallet={valid_wallet_address}",
                    headers=valid_headers
                )
                
                assert response.status_code == 500

    class TestWalletAnalysisEndpoint:
        """Test /api/risk/analyze endpoint"""

        @pytest.mark.asyncio
        async def test_wallet_analysis_success(self, async_client, valid_wallet_address,
                                             valid_headers, mock_database_operations,
                                             mock_risk_calculator, mock_rate_limiting):
            """Test successful wallet analysis"""
            
            request_data = {
                "wallet_address": valid_wallet_address,
                "force_refresh": True,
                "protocols": ["aave_v3", "compound"]
            }
            
            response = await async_client.post(
                "/api/risk/analyze",
                headers=valid_headers,
                json=request_data
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "completed"
            assert data["wallet_address"] == valid_wallet_address
            assert "analysis_timestamp" in data
            assert "risk_summary" in data
            assert "alerts_generated" in data

        @pytest.mark.asyncio
        async def test_wallet_analysis_with_alerts(self, async_client, valid_wallet_address,
                                                  valid_headers, mock_database_operations,
                                                  mock_rate_limiting):
            """Test wallet analysis that generates alerts"""
            
            mock_alerts = [
                RiskAlert(
                    wallet_address=valid_wallet_address,
                    protocol="aave_v3",
                    alert_type="liquidation_risk",
                    severity="high",
                    title="High Liquidation Risk",
                    description="Health ratio approaching liquidation threshold",
                    recommendation="Consider adding collateral",
                    current_value=1.2,
                    threshold_value=1.1,
                    risk_score=75.0,
                    timestamp=datetime.utcnow()
                )
            ]
            
            with patch('app.routes.risk_calculator') as mock_calc:
                mock_summary = RiskSummary(
                    wallet_address=valid_wallet_address,
                    last_updated=datetime.utcnow(),
                    overall_risk_score=75.0,
                    risk_level="high",
                    health_ratio=1.2,
                    total_portfolio_value_usd=10000.0,
                    total_debt_usd=8000.0,
                    net_worth_usd=2000.0,
                    liquidation_risk=75.0,
                    volatility_risk=60.0,
                    concentration_risk=80.0,
                    liquidity_risk=40.0,
                    protocols=["aave_v3"],
                    protocol_exposure={"aave_v3": 1.0},
                    active_alerts_count=1,
                    critical_alerts_count=0
                )
                mock_calc.aggregate_risk_summary.return_value = mock_summary
                mock_calc.generate_alerts.return_value = mock_alerts
                
                request_data = {
                    "wallet_address": valid_wallet_address,
                    "force_refresh": True
                }
                
                response = await async_client.post(
                    "/api/risk/analyze",
                    headers=valid_headers,
                    json=request_data
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["alerts_generated"] == 1
                assert len(data["new_alerts"]) == 1
                assert data["new_alerts"][0]["severity"] == "high"

    class TestAlertsEndpoint:
        """Test /api/risk/alerts endpoint"""

        @pytest.mark.asyncio
        async def test_get_alerts_success(self, async_client, valid_wallet_address,
                                        valid_headers, mock_rate_limiting):
            """Test successful alerts retrieval"""
            
            mock_alert_docs = [
                {
                    "_id": "alert_id_1",
                    "wallet_address": valid_wallet_address,
                    "protocol": "aave_v3",
                    "alert_type": "liquidation_risk",
                    "severity": "high",
                    "title": "High Liquidation Risk",
                    "description": "Health ratio approaching liquidation threshold",
                    "recommendation": "Consider adding collateral",
                    "current_value": 1.2,
                    "threshold_value": 1.1,
                    "risk_score": 75.0,
                    "is_resolved": False,
                    "acknowledged": False,
                    "timestamp": datetime.utcnow(),
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                    "metadata": {}
                }
            ]
            
            with patch('app.routes.get_collection') as mock_get_collection:
                mock_collection = AsyncMock()
                mock_collection.count_documents.return_value = 1
                
                # Create a proper mock cursor
                mock_cursor = AsyncMock()
                mock_cursor.to_list.return_value = mock_alert_docs
                mock_collection.find.return_value.sort.return_value.skip.return_value.limit.return_value = mock_cursor
                
                mock_get_collection.return_value = mock_collection
                
                response = await async_client.get(
                    "/api/risk/alerts",
                    headers=valid_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["total_count"] == 1
                assert len(data["alerts"]) == 1
                assert data["alerts"][0]["severity"] == "high"
                assert "pagination" in data

        @pytest.mark.asyncio
        async def test_get_alerts_with_filters(self, async_client, valid_wallet_address,
                                             valid_headers, mock_rate_limiting):
            """Test alerts retrieval with filters"""
            
            with patch('app.routes.get_collection') as mock_get_collection:
                mock_collection = AsyncMock()
                mock_collection.count_documents.return_value = 0
                
                mock_cursor = AsyncMock()
                mock_cursor.to_list.return_value = []
                mock_collection.find.return_value.sort.return_value.skip.return_value.limit.return_value = mock_cursor
                
                mock_get_collection.return_value = mock_collection
                
                response = await async_client.get(
                    "/api/risk/alerts?severity=high&resolved=false",
                    headers=valid_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["total_count"] == 0
                assert len(data["alerts"]) == 0

        @pytest.mark.asyncio
        async def test_get_alerts_pagination(self, async_client, valid_wallet_address,
                                           valid_headers, mock_rate_limiting):
            """Test alerts pagination"""
            
            with patch('app.routes.get_collection') as mock_get_collection:
                mock_collection = AsyncMock()
                mock_collection.count_documents.return_value = 100
                
                mock_cursor = AsyncMock()
                mock_cursor.to_list.return_value = []
                mock_collection.find.return_value.sort.return_value.skip.return_value.limit.return_value = mock_cursor
                
                mock_get_collection.return_value = mock_collection
                
                response = await async_client.get(
                    "/api/risk/alerts?limit=10&skip=20",
                    headers=valid_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["pagination"]["limit"] == 10
                assert data["pagination"]["skip"] == 20
                assert data["pagination"]["has_more"] == True

    class TestResolveAlertEndpoint:
        """Test /api/risk/alerts/{alert_id}/resolve endpoint"""

        @pytest.mark.asyncio
        async def test_resolve_alert_success(self, async_client, valid_wallet_address,
                                           valid_headers, mock_rate_limiting):
            """Test successful alert resolution"""
            
            alert_id = "507f1f77bcf86cd799439011"  # Valid ObjectId format
            
            with patch('app.routes.get_collection') as mock_get_collection:
                mock_collection = AsyncMock()
                mock_result = MagicMock()
                mock_result.matched_count = 1
                mock_collection.update_one.return_value = mock_result
                mock_get_collection.return_value = mock_collection
                
                response = await async_client.post(
                    f"/api/risk/alerts/{alert_id}/resolve",
                    headers=valid_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["status"] == "resolved"
                assert data["alert_id"] == alert_id
                assert "resolved_at" in data

        @pytest.mark.asyncio
        async def test_resolve_alert_not_found(self, async_client, valid_wallet_address,
                                             valid_headers, mock_rate_limiting):
            """Test resolving non-existent alert"""
            
            alert_id = "507f1f77bcf86cd799439011"
            
            with patch('app.routes.get_collection') as mock_get_collection:
                mock_collection = AsyncMock()
                mock_result = MagicMock()
                mock_result.matched_count = 0
                mock_collection.update_one.return_value = mock_result
                mock_get_collection.return_value = mock_collection
                
                response = await async_client.post(
                    f"/api/risk/alerts/{alert_id}/resolve",
                    headers=valid_headers
                )
                
                assert response.status_code == 404
                assert "Alert not found" in response.json()["detail"]

        @pytest.mark.asyncio
        async def test_resolve_alert_invalid_id(self, async_client, valid_wallet_address,
                                               valid_headers, mock_rate_limiting):
            """Test resolving alert with invalid ID"""
            
            invalid_alert_id = "invalid_id"
            
            response = await async_client.post(
                f"/api/risk/alerts/{invalid_alert_id}/resolve",
                headers=valid_headers
            )
            
            assert response.status_code == 400
            assert "Invalid alert ID format" in response.json()["detail"]

    class TestHealthCheckEndpoint:
        """Test /api/risk/health endpoint"""

        @pytest.mark.asyncio
        async def test_health_check_success(self, async_client):
            """Test successful health check"""
            
            with patch('app.routes.db_manager') as mock_db_manager:
                mock_db_manager.health_check.return_value = {
                    "mongodb": {"status": "connected"},
                    "redis": {"status": "connected"}
                }
                
                response = await async_client.get("/api/risk/health")
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["status"] == "healthy"
                assert data["version"] == "1.0.0"
                assert "timestamp" in data

        @pytest.mark.asyncio
        async def test_health_check_failure(self, async_client):
            """Test health check when database is unhealthy"""
            
            with patch('app.routes.db_manager') as mock_db_manager:
                mock_db_manager.health_check.side_effect = Exception("Database connection failed")
                
                response = await async_client.get("/api/risk/health")
                
                assert response.status_code == 503
                data = response.json()
                
                assert data["status"] == "unhealthy"
                assert "error" in data

    class TestRootEndpoint:
        """Test root endpoint"""

        @pytest.mark.asyncio
        async def test_root_endpoint(self, async_client):
            """Test root endpoint information"""
            
            response = await async_client.get("/")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["service"] == "Risk Manager Agent"
            assert data["version"] == "1.0.0"
            assert data["status"] == "operational"
            assert "endpoints" in data
            assert "authentication" in data
            assert "rate_limit" in data

    class TestServiceInfoEndpoint:
        """Test /api/info endpoint"""

        @pytest.mark.asyncio
        async def test_service_info_success(self, async_client):
            """Test service info endpoint"""
            
            with patch('app.main.db_manager') as mock_db_manager:
                mock_db_manager.health_check.return_value = {
                    "mongodb": {"status": "connected", "latency_ms": 15.2},
                    "redis": {"status": "connected", "latency_ms": 5.1}
                }
                
                response = await async_client.get("/api/info")
                
                assert response.status_code == 200
                data = response.json()
                
                assert "service" in data
                assert "features" in data
                assert "database" in data
                assert "configuration" in data
                assert data["service"]["name"] == "Risk Manager Agent"
                assert data["service"]["version"] == "1.0.0"

        @pytest.mark.asyncio
        async def test_service_info_database_error(self, async_client):
            """Test service info when database check fails"""
            
            with patch('app.main.db_manager') as mock_db_manager:
                mock_db_manager.health_check.side_effect = Exception("Database error")
                
                response = await async_client.get("/api/info")
                
                assert response.status_code == 500

    class TestErrorHandling:
        """Test global error handling"""

        @pytest.mark.asyncio
        async def test_404_endpoint(self, async_client):
            """Test non-existent endpoint"""
            
            response = await async_client.get("/non-existent-endpoint")
            
            assert response.status_code == 404

        @pytest.mark.asyncio
        async def test_method_not_allowed(self, async_client, valid_headers):
            """Test method not allowed"""
            
            response = await async_client.delete(
                "/api/risk/status",
                headers=valid_headers
            )
            
            assert response.status_code == 405

    class TestConcurrency:
        """Test concurrent API requests"""

        @pytest.mark.asyncio
        async def test_concurrent_requests(self, async_client, valid_wallet_address,
                                         valid_headers, mock_database_operations,
                                         mock_risk_calculator, mock_rate_limiting):
            """Test handling of concurrent requests"""
            
            # Create multiple concurrent requests
            tasks = [
                async_client.get(
                    f"/api/risk/summary?wallet={valid_wallet_address}",
                    headers=valid_headers
                )
                for _ in range(10)
            ]
            
            responses = await asyncio.gather(*tasks)
            
            # All requests should succeed
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert data["wallet_address"] == valid_wallet_address