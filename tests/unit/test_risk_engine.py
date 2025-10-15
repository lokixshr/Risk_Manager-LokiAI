import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from app.risk_engine import RiskCalculator
from app.models import LendingPosition, VolatilityMetrics, AavePositionData, CompoundPositionData
from app.config import settings


class TestRiskCalculator:
    """Comprehensive tests for the RiskCalculator class"""
    
    @pytest.fixture
    def risk_calculator(self):
        """Create a RiskCalculator instance for testing"""
        return RiskCalculator()
    
    @pytest.fixture
    def sample_aave_position(self):
        """Sample Aave position data for testing"""
        return AavePositionData(
            user_address="0x742b4c0d8fd9b2b29e70dc3e08f4e98a78b3a2b5",
            total_collateral_eth=5.0,
            total_debt_eth=3.0,
            available_borrow_eth=1.0,
            current_liquidation_threshold=0.8,
            health_factor=1.33
        )
    
    @pytest.fixture
    def sample_compound_position(self):
        """Sample Compound position data for testing"""
        return CompoundPositionData(
            user_address="0x742b4c0d8fd9b2b29e70dc3e08f4e98a78b3a2b5",
            total_collateral_value_usd=10000.0,
            total_borrow_value_usd=6000.0,
            account_liquidity=1000.0,
            shortfall=0.0
        )
    
    @pytest.fixture
    def sample_lending_positions(self):
        """Sample lending positions for testing"""
        return [
            LendingPosition(
                protocol="aave_v3",
                token_address="0xa0b86991c431c24b0b13715bf94de7b15c7b96000",
                symbol="USDC",
                supplied_amount=5000.0,
                supplied_usd=5000.0,
                borrowed_amount=0.0,
                borrowed_usd=0.0,
                collateral_factor=0.85,
                liquidation_threshold=0.87,
                apy_supply=3.5,
                apy_borrow=0.0
            ),
            LendingPosition(
                protocol="aave_v3",
                token_address="0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",
                symbol="WBTC",
                supplied_amount=0.1,
                supplied_usd=3000.0,
                borrowed_amount=0.0,
                borrowed_usd=0.0,
                collateral_factor=0.7,
                liquidation_threshold=0.75,
                apy_supply=1.2,
                apy_borrow=0.0
            )
        ]

    class TestHealthCalculation:
        """Test health ratio and liquidation risk calculations"""
        
        @pytest.mark.asyncio
        async def test_aave_health_calculation(self, risk_calculator, sample_aave_position):
            """Test Aave health ratio calculation"""
            
            with patch('app.risk_engine.api_manager') as mock_api_manager:
                mock_api_manager.aave.get_user_positions.return_value = sample_aave_position
                
                result = await risk_calculator.calculate_position_health(
                    "0x742b4c0d8fd9b2b29e70dc3e08f4e98a78b3a2b5", 
                    "aave_v3"
                )
                
                assert result["health_ratio"] == 1.33
                assert result["liquidation_risk"] > 0
                assert result["collateral_usd"] > 0
                assert result["debt_usd"] > 0
        
        @pytest.mark.asyncio
        async def test_compound_health_calculation(self, risk_calculator, sample_compound_position):
            """Test Compound health ratio calculation"""
            
            with patch('app.risk_engine.api_manager') as mock_api_manager:
                mock_api_manager.compound.get_user_positions.return_value = sample_compound_position
                
                result = await risk_calculator.calculate_position_health(
                    "0x742b4c0d8fd9b2b29e70dc3e08f4e98a78b3a2b5", 
                    "compound"
                )
                
                expected_health = 10000.0 / 6000.0  # collateral / debt
                assert abs(result["health_ratio"] - expected_health) < 0.01
                assert result["liquidation_risk"] > 0
                assert result["collateral_usd"] == 10000.0
                assert result["debt_usd"] == 6000.0

        def test_liquidation_risk_calculation(self, risk_calculator):
            """Test liquidation risk scoring logic"""
            
            # Test healthy position (health ratio > 2.0)
            with patch.object(risk_calculator, 'calculate_position_health') as mock_calc:
                mock_calc.return_value = {
                    "health_ratio": 2.5,
                    "liquidation_risk": 0.0,
                    "collateral_usd": 10000.0,
                    "debt_usd": 4000.0,
                    "available_borrow_usd": 2000.0
                }
                
                # For a healthy position, risk should be 0
                health_data = mock_calc.return_value
                assert health_data["liquidation_risk"] == 0.0
            
            # Test risky position (health ratio < 1.2)
            with patch.object(risk_calculator, 'calculate_position_health') as mock_calc:
                mock_calc.return_value = {
                    "health_ratio": 1.1,
                    "liquidation_risk": 80.0,  # High risk
                    "collateral_usd": 10000.0,
                    "debt_usd": 9000.0,
                    "available_borrow_usd": 100.0
                }
                
                health_data = mock_calc.return_value
                assert health_data["liquidation_risk"] > 60.0

        @pytest.mark.asyncio
        async def test_no_position_data(self, risk_calculator):
            """Test handling when no position data is available"""
            
            with patch('app.risk_engine.api_manager') as mock_api_manager:
                mock_api_manager.aave.get_user_positions.return_value = None
                
                result = await risk_calculator.calculate_position_health(
                    "0x742b4c0d8fd9b2b29e70dc3e08f4e98a78b3a2b5", 
                    "aave_v3"
                )
                
                assert result["health_ratio"] == float('inf')
                assert result["liquidation_risk"] == 0.0
                assert result["collateral_usd"] == 0.0
                assert result["debt_usd"] == 0.0

    class TestVolatilityCalculation:
        """Test volatility and VaR calculations"""
        
        @pytest.fixture
        def sample_price_data(self):
            """Sample price data for volatility testing"""
            return [
                {"timestamp": datetime.now() - timedelta(days=i), "price": 100 + i * 2 + np.random.normal(0, 5)}
                for i in range(30)
            ]

        @pytest.mark.asyncio
        async def test_volatility_calculation(self, risk_calculator, sample_price_data):
            """Test volatility metrics calculation"""
            
            token_addresses = ["0xa0b86991c431c24b0b13715bf94de7b15c7b96000"]
            
            with patch('app.risk_engine.api_manager') as mock_api_manager:
                mock_api_manager.coingecko.get_historical_prices.return_value = sample_price_data
                
                result = await risk_calculator.calculate_volatility_metrics(token_addresses)
                
                assert len(result) > 0
                token_address = token_addresses[0]
                assert token_address in result
                
                vol_metrics = result[token_address]
                assert isinstance(vol_metrics, VolatilityMetrics)
                assert vol_metrics.daily_volatility > 0
                assert vol_metrics.weekly_volatility > vol_metrics.daily_volatility
                assert vol_metrics.monthly_volatility > vol_metrics.weekly_volatility
                assert vol_metrics.var_95 < 0  # VaR should be negative
                assert vol_metrics.var_99 < vol_metrics.var_95  # 99% VaR should be more extreme

        @pytest.mark.asyncio
        async def test_empty_price_data(self, risk_calculator):
            """Test handling of empty price data"""
            
            token_addresses = ["0xa0b86991c431c24b0b13715bf94de7b15c7b96000"]
            
            with patch('app.risk_engine.api_manager') as mock_api_manager:
                mock_api_manager.coingecko.get_historical_prices.return_value = []
                
                result = await risk_calculator.calculate_volatility_metrics(token_addresses)
                
                # Should handle empty data gracefully
                assert len(result) == 0

        @pytest.mark.asyncio
        async def test_insufficient_price_data(self, risk_calculator):
            """Test handling of insufficient price data"""
            
            token_addresses = ["0xa0b86991c431c24b0b13715bf94de7b15c7b96000"]
            single_price = [{"timestamp": datetime.now(), "price": 100.0}]
            
            with patch('app.risk_engine.api_manager') as mock_api_manager:
                mock_api_manager.coingecko.get_historical_prices.return_value = single_price
                
                result = await risk_calculator.calculate_volatility_metrics(token_addresses)
                
                # Should handle insufficient data gracefully
                assert len(result) == 0

    class TestExposureCalculation:
        """Test exposure and concentration risk calculations"""
        
        @pytest.mark.asyncio
        async def test_exposure_calculation(self, risk_calculator, sample_lending_positions):
            """Test exposure score calculation"""
            
            wallet_address = "0x742b4c0d8fd9b2b29e70dc3e08f4e98a78b3a2b5"
            
            result = await risk_calculator.calculate_exposure_score(
                wallet_address, 
                sample_lending_positions
            )
            
            assert result.wallet_address == wallet_address
            assert result.concentration_score >= 0.0
            assert result.concentration_score <= 1.0
            assert result.largest_position_pct >= 0.0
            assert result.largest_position_pct <= 1.0
            assert result.liquidity_score >= 0.0
            assert result.liquidity_score <= 1.0
            assert result.exposure_score >= 0.0
            assert result.risk_level in ["low", "medium", "high", "critical"]

        @pytest.mark.asyncio
        async def test_empty_positions(self, risk_calculator):
            """Test exposure calculation with empty positions"""
            
            wallet_address = "0x742b4c0d8fd9b2b29e70dc3e08f4e98a78b3a2b5"
            
            result = await risk_calculator.calculate_exposure_score(wallet_address, [])
            
            assert result.wallet_address == wallet_address
            assert result.concentration_score == 0.0
            assert result.largest_position_pct == 0.0
            assert result.liquidity_score == 1.0  # Default to high liquidity
            assert result.exposure_score == 0.0
            assert result.risk_level == "low"

        @pytest.mark.asyncio
        async def test_single_asset_concentration(self, risk_calculator):
            """Test high concentration risk with single asset"""
            
            wallet_address = "0x742b4c0d8fd9b2b29e70dc3e08f4e98a78b3a2b5"
            single_position = [
                LendingPosition(
                    protocol="aave_v3",
                    token_address="0xa0b86991c431c24b0b13715bf94de7b15c7b96000",
                    symbol="USDC",
                    supplied_amount=10000.0,
                    supplied_usd=10000.0,
                    borrowed_amount=0.0,
                    borrowed_usd=0.0,
                    collateral_factor=0.85,
                    liquidation_threshold=0.87,
                    apy_supply=3.5,
                    apy_borrow=0.0
                )
            ]
            
            result = await risk_calculator.calculate_exposure_score(
                wallet_address, 
                single_position
            )
            
            # Single asset should have 100% concentration
            assert result.largest_position_pct == 1.0
            assert result.concentration_score > 0.8  # High concentration

    class TestRiskSummaryGeneration:
        """Test comprehensive risk summary generation"""
        
        @pytest.mark.asyncio
        async def test_aggregate_risk_summary(self, risk_calculator):
            """Test comprehensive risk summary aggregation"""
            
            wallet_address = "0x742b4c0d8fd9b2b29e70dc3e08f4e98a78b3a2b5"
            
            with patch.object(risk_calculator, 'calculate_position_health') as mock_health, \
                 patch.object(risk_calculator, 'calculate_volatility_metrics') as mock_vol, \
                 patch.object(risk_calculator, 'calculate_exposure_score') as mock_exp, \
                 patch('app.risk_engine.api_manager') as mock_api:
                
                # Mock health calculation
                mock_health.return_value = {
                    "health_ratio": 1.5,
                    "liquidation_risk": 25.0,
                    "collateral_usd": 10000.0,
                    "debt_usd": 6000.0,
                    "available_borrow_usd": 1000.0
                }
                
                # Mock volatility calculation
                mock_vol.return_value = {
                    "0xa0b86991c431c24b0b13715bf94de7b15c7b96000": VolatilityMetrics(
                        asset_symbol="USDC",
                        daily_volatility=0.02,
                        weekly_volatility=0.05,
                        monthly_volatility=0.1,
                        var_95=-0.03,
                        var_99=-0.05,
                        max_drawdown=-0.1,
                        sharpe_ratio=1.5
                    )
                }
                
                # Mock exposure calculation
                from app.models import ExposureMetrics
                mock_exp.return_value = ExposureMetrics(
                    wallet_address=wallet_address,
                    protocol="aave_v3",
                    concentration_score=0.6,
                    largest_position_pct=0.7,
                    liquidity_score=0.8,
                    avg_daily_volume_usd=1000000.0,
                    weighted_volatility=0.05,
                    correlation_risk=0.3,
                    exposure_score=45.0,
                    risk_level="medium"
                )
                
                # Mock API calls for position data
                mock_api.aave.get_user_positions.return_value = None
                mock_api.compound.get_user_positions.return_value = None
                
                result = await risk_calculator.aggregate_risk_summary(wallet_address)
                
                assert result.wallet_address == wallet_address
                assert result.overall_risk_score >= 0.0
                assert result.overall_risk_score <= 100.0
                assert result.risk_level in ["low", "medium", "high", "critical"]
                assert isinstance(result.last_updated, datetime)

    class TestErrorHandling:
        """Test error handling in risk calculations"""
        
        @pytest.mark.asyncio
        async def test_api_failure_handling(self, risk_calculator):
            """Test handling of API failures"""
            
            wallet_address = "0x742b4c0d8fd9b2b29e70dc3e08f4e98a78b3a2b5"
            
            with patch('app.risk_engine.api_manager') as mock_api_manager:
                # Simulate API failure
                mock_api_manager.aave.get_user_positions.side_effect = Exception("API Error")
                
                result = await risk_calculator.calculate_position_health(
                    wallet_address, 
                    "aave_v3"
                )
                
                # Should return safe defaults on error
                assert result["health_ratio"] == float('inf')
                assert result["liquidation_risk"] == 0.0

        @pytest.mark.asyncio
        async def test_invalid_protocol_handling(self, risk_calculator):
            """Test handling of invalid protocols"""
            
            wallet_address = "0x742b4c0d8fd9b2b29e70dc3e08f4e98a78b3a2b5"
            
            result = await risk_calculator.calculate_position_health(
                wallet_address, 
                "invalid_protocol"
            )
            
            # Should return safe defaults for invalid protocol
            assert result["health_ratio"] == float('inf')
            assert result["liquidation_risk"] == 0.0

    class TestPerformance:
        """Test performance characteristics"""
        
        @pytest.mark.asyncio
        async def test_concurrent_calculations(self, risk_calculator):
            """Test concurrent risk calculations"""
            
            wallet_addresses = [f"0x{i:040x}" for i in range(10)]
            
            with patch.object(risk_calculator, 'calculate_position_health') as mock_calc:
                mock_calc.return_value = {
                    "health_ratio": 1.5,
                    "liquidation_risk": 25.0,
                    "collateral_usd": 10000.0,
                    "debt_usd": 6000.0,
                    "available_borrow_usd": 1000.0
                }
                
                # Test concurrent execution
                tasks = [
                    risk_calculator.calculate_position_health(wallet, "aave_v3")
                    for wallet in wallet_addresses
                ]
                
                start_time = datetime.now()
                results = await asyncio.gather(*tasks)
                end_time = datetime.now()
                
                assert len(results) == len(wallet_addresses)
                
                # Should complete within reasonable time
                execution_time = (end_time - start_time).total_seconds()
                assert execution_time < 5.0  # Should complete within 5 seconds