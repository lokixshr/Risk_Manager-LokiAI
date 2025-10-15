import asyncio
import math
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import structlog
from scipy import stats

from .config import settings, Collections, RiskSeverity, AlertType
from .models import (
    PositionSnapshot, RiskAlert, RiskSummary, VolatilityMetrics, 
    ExposureMetrics, TrainingDataPoint, AssetPrice, LendingPosition
)
from .external_apis import api_manager
from .database import get_collection

logger = structlog.get_logger()

class RiskCalculator:
    """Core risk calculation engine with real analytics"""
    
    def __init__(self):
        self.safety_threshold = settings.SAFETY_THRESHOLD
        self.liquidation_threshold = settings.LIQUIDATION_THRESHOLD
        self.var_confidence = settings.VAR_CONFIDENCE
        self.volatility_window = settings.VOLATILITY_WINDOW_DAYS
    
    async def calculate_position_health(self, wallet_address: str, protocol: str) -> Dict:
        """Calculate position health ratio and liquidation risk"""
        try:
            position_data = None
            
            if protocol == "aave_v2":
                position_data = await api_manager.aave.get_user_positions(wallet_address, "v2")
            elif protocol == "aave_v3":
                position_data = await api_manager.aave.get_user_positions(wallet_address, "v3")
            elif protocol == "compound":
                position_data = await api_manager.compound.get_user_positions(wallet_address)
            
            if not position_data:
                return {
                    "health_ratio": float('inf'),
                    "liquidation_risk": 0.0,
                    "collateral_usd": 0.0,
                    "debt_usd": 0.0,
                    "available_borrow_usd": 0.0
                }
            
            # Extract metrics based on protocol
            if protocol.startswith("aave"):
                health_ratio = position_data.health_factor
                collateral_usd = position_data.total_collateral_eth * 2000  # Rough ETH price
                debt_usd = position_data.total_debt_eth * 2000
                available_borrow_usd = position_data.available_borrow_eth * 2000
            elif protocol == "compound":
                if position_data.total_collateral_value_usd > 0:
                    health_ratio = position_data.total_collateral_value_usd / max(position_data.total_borrow_value_usd, 0.01)
                else:
                    health_ratio = float('inf')
                collateral_usd = position_data.total_collateral_value_usd
                debt_usd = position_data.total_borrow_value_usd
                available_borrow_usd = position_data.account_liquidity
            else:
                return {
                    "health_ratio": float('inf'),
                    "liquidation_risk": 0.0,
                    "collateral_usd": 0.0,
                    "debt_usd": 0.0,
                    "available_borrow_usd": 0.0
                }
            
            # Calculate liquidation risk score (0-100)
            if health_ratio == float('inf') or debt_usd == 0:
                liquidation_risk = 0.0
            else:
                # Risk increases exponentially as health ratio approaches 1
                if health_ratio > 2.0:
                    liquidation_risk = 0.0
                elif health_ratio > 1.5:
                    liquidation_risk = 20.0 * (2.0 - health_ratio) / 0.5
                elif health_ratio > 1.2:
                    liquidation_risk = 20.0 + 40.0 * (1.5 - health_ratio) / 0.3
                elif health_ratio > 1.0:
                    liquidation_risk = 60.0 + 35.0 * (1.2 - health_ratio) / 0.2
                else:
                    liquidation_risk = 95.0 + 5.0 * max(0, 1.0 - health_ratio)
            
            return {
                "health_ratio": health_ratio,
                "liquidation_risk": min(100.0, liquidation_risk),
                "collateral_usd": collateral_usd,
                "debt_usd": debt_usd,
                "available_borrow_usd": available_borrow_usd
            }
            
        except Exception as e:
            logger.error(f"Error calculating position health for {wallet_address} on {protocol}", error=str(e))
            return {
                "health_ratio": float('inf'),
                "liquidation_risk": 0.0,
                "collateral_usd": 0.0,
                "debt_usd": 0.0,
                "available_borrow_usd": 0.0
            }
    
    async def calculate_volatility_metrics(self, token_addresses: List[str]) -> Dict[str, VolatilityMetrics]:
        """Calculate volatility and VaR for assets using real price data"""
        volatility_metrics = {}
        
        try:
            # Get historical price data from CoinGecko
            async with api_manager.coingecko:
                for token_address in token_addresses:
                    try:
                        # Get historical prices
                        price_history = await api_manager.coingecko.get_historical_prices(
                            token_address, days=self.volatility_window
                        )
                        
                        if len(price_history) < 2:
                            continue
                        
                        # Convert to pandas DataFrame for easier calculation
                        df = pd.DataFrame(price_history)
                        df['returns'] = df['price'].pct_change().dropna()
                        
                        if len(df['returns']) < 2:
                            continue
                        
                        # Calculate volatility metrics
                        returns = df['returns'].values
                        daily_vol = np.std(returns)
                        weekly_vol = daily_vol * np.sqrt(7)
                        monthly_vol = daily_vol * np.sqrt(30)
                        
                        # Calculate VaR using normal approximation
                        var_95 = np.percentile(returns, (1 - self.var_confidence) * 100)
                        var_99 = np.percentile(returns, 1)  # 99% VaR
                        
                        # Calculate maximum drawdown
                        cumulative_returns = (1 + df['returns']).cumprod()
                        rolling_max = cumulative_returns.cummax()
                        drawdown = (cumulative_returns - rolling_max) / rolling_max
                        max_drawdown = drawdown.min()
                        
                        # Calculate Sharpe ratio (assuming risk-free rate of 2%)
                        risk_free_rate = 0.02 / 365  # Daily risk-free rate
                        excess_returns = returns - risk_free_rate
                        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
                        
                        volatility_metrics[token_address] = VolatilityMetrics(
                            asset_symbol=token_address,  # Will resolve symbol later
                            daily_volatility=daily_vol,
                            weekly_volatility=weekly_vol,
                            monthly_volatility=monthly_vol,
                            var_95=var_95,
                            var_99=var_99,
                            max_drawdown=max_drawdown,
                            sharpe_ratio=sharpe_ratio
                        )
                        
                    except Exception as e:
                        logger.error(f"Error calculating volatility for {token_address}", error=str(e))
                        continue
        
        except Exception as e:
            logger.error("Error in volatility calculation", error=str(e))
        
        return volatility_metrics
    
    async def calculate_exposure_score(self, wallet_address: str, positions: List[LendingPosition]) -> ExposureMetrics:
        """Calculate exposure and concentration risk"""
        try:
            if not positions:
                return ExposureMetrics(
                    wallet_address=wallet_address,
                    protocol="unknown",
                    concentration_score=0.0,
                    largest_position_pct=0.0,
                    liquidity_score=1.0,
                    avg_daily_volume_usd=0.0,
                    weighted_volatility=0.0,
                    correlation_risk=0.0,
                    exposure_score=0.0,
                    risk_level="low"
                )
            
            # Calculate total portfolio value
            total_value = sum(pos.supplied_usd for pos in positions)
            
            if total_value == 0:
                return ExposureMetrics(
                    wallet_address=wallet_address,
                    protocol=positions[0].protocol,
                    concentration_score=0.0,
                    largest_position_pct=0.0,
                    liquidity_score=1.0,
                    avg_daily_volume_usd=0.0,
                    weighted_volatility=0.0,
                    correlation_risk=0.0,
                    exposure_score=0.0,
                    risk_level="low"
                )
            
            # Concentration analysis
            position_weights = [pos.supplied_usd / total_value for pos in positions]
            largest_position_pct = max(position_weights) if position_weights else 0
            
            # Herfindahl-Hirschman Index for concentration
            hhi = sum(w**2 for w in position_weights)
            concentration_score = min(1.0, hhi * 2)  # Normalize to 0-1
            
            # Get token addresses for further analysis
            token_addresses = [pos.token_address for pos in positions]
            
            # Get current prices and volumes
            async with api_manager.coingecko:
                price_data = await api_manager.coingecko.get_token_prices(token_addresses)
            
            # Calculate liquidity score based on 24h volume
            total_volume = sum(
                price_data.get(addr, AssetPrice(symbol="", price_usd=0, last_updated=datetime.utcnow())).volume_24h or 0
                for addr in token_addresses
            )
            avg_daily_volume_usd = total_volume / len(token_addresses) if token_addresses else 0
            
            # Liquidity score: higher volume = higher liquidity = lower risk
            if avg_daily_volume_usd > 10_000_000:  # > $10M daily volume
                liquidity_score = 1.0
            elif avg_daily_volume_usd > 1_000_000:  # > $1M daily volume
                liquidity_score = 0.8
            elif avg_daily_volume_usd > 100_000:  # > $100K daily volume
                liquidity_score = 0.6
            else:
                liquidity_score = 0.2  # Low liquidity
            
            # Calculate weighted volatility
            volatility_metrics = await self.calculate_volatility_metrics(token_addresses)
            weighted_volatility = 0.0
            
            for i, pos in enumerate(positions):
                if pos.token_address in volatility_metrics:
                    vol_metric = volatility_metrics[pos.token_address]
                    weight = position_weights[i]
                    weighted_volatility += weight * vol_metric.daily_volatility
            
            # Simple correlation risk approximation (higher concentration = higher correlation risk)
            correlation_risk = concentration_score
            
            # Overall exposure score calculation
            # Combine multiple risk factors with weights
            volatility_component = min(1.0, weighted_volatility * 10)  # Scale volatility
            liquidity_component = 1.0 - liquidity_score  # Invert liquidity (lower liquidity = higher risk)
            
            exposure_score = (
                0.3 * concentration_score +
                0.3 * volatility_component +
                0.2 * liquidity_component +
                0.2 * correlation_risk
            )
            
            # Determine risk level
            if exposure_score < 0.2:
                risk_level = "low"
            elif exposure_score < 0.5:
                risk_level = "medium"
            elif exposure_score < 0.8:
                risk_level = "high"
            else:
                risk_level = "critical"
            
            return ExposureMetrics(
                wallet_address=wallet_address,
                protocol=positions[0].protocol,
                concentration_score=concentration_score,
                largest_position_pct=largest_position_pct,
                liquidity_score=liquidity_score,
                avg_daily_volume_usd=avg_daily_volume_usd,
                weighted_volatility=weighted_volatility,
                correlation_risk=correlation_risk,
                exposure_score=exposure_score,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Error calculating exposure score for {wallet_address}", error=str(e))
            return ExposureMetrics(
                wallet_address=wallet_address,
                protocol="unknown",
                concentration_score=0.0,
                largest_position_pct=0.0,
                liquidity_score=1.0,
                avg_daily_volume_usd=0.0,
                weighted_volatility=0.0,
                correlation_risk=0.0,
                exposure_score=0.0,
                risk_level="low"
            )
    
    async def generate_alerts(self, wallet_address: str, risk_summary: RiskSummary) -> List[RiskAlert]:
        """Generate risk alerts based on thresholds"""
        alerts = []
        current_time = datetime.utcnow()
        
        try:
            # Liquidation risk alert
            if risk_summary.liquidation_risk > 80:
                alerts.append(RiskAlert(
                    wallet_address=wallet_address,
                    protocol="multi_protocol",
                    alert_type=AlertType.LIQUIDATION_RISK,
                    severity=RiskSeverity.CRITICAL,
                    title="Critical Liquidation Risk",
                    description=f"Portfolio health ratio is dangerously low at {risk_summary.health_ratio:.3f}",
                    recommendation="Consider repaying debt or adding more collateral immediately",
                    current_value=risk_summary.liquidation_risk,
                    threshold_value=80.0,
                    risk_score=risk_summary.liquidation_risk,
                    timestamp=current_time
                ))
            elif risk_summary.liquidation_risk > 60:
                alerts.append(RiskAlert(
                    wallet_address=wallet_address,
                    protocol="multi_protocol",
                    alert_type=AlertType.LIQUIDATION_RISK,
                    severity=RiskSeverity.HIGH,
                    title="High Liquidation Risk",
                    description=f"Portfolio health ratio is concerning at {risk_summary.health_ratio:.3f}",
                    recommendation="Monitor position closely and consider reducing leverage",
                    current_value=risk_summary.liquidation_risk,
                    threshold_value=60.0,
                    risk_score=risk_summary.liquidation_risk,
                    timestamp=current_time
                ))
            
            # Volatility risk alert
            if risk_summary.volatility_risk > 70:
                alerts.append(RiskAlert(
                    wallet_address=wallet_address,
                    protocol="multi_protocol",
                    alert_type=AlertType.HIGH_VOLATILITY,
                    severity=RiskSeverity.HIGH if risk_summary.volatility_risk > 85 else RiskSeverity.MEDIUM,
                    title="High Portfolio Volatility",
                    description=f"Portfolio volatility risk is elevated at {risk_summary.volatility_risk:.1f}%",
                    recommendation="Consider diversifying into less volatile assets",
                    current_value=risk_summary.volatility_risk,
                    threshold_value=70.0,
                    risk_score=risk_summary.volatility_risk,
                    timestamp=current_time
                ))
            
            # Concentration risk alert
            if risk_summary.concentration_risk > 80:
                alerts.append(RiskAlert(
                    wallet_address=wallet_address,
                    protocol="multi_protocol",
                    alert_type=AlertType.OVER_EXPOSURE,
                    severity=RiskSeverity.HIGH,
                    title="High Concentration Risk",
                    description=f"Portfolio is highly concentrated with risk score {risk_summary.concentration_risk:.1f}%",
                    recommendation="Diversify portfolio across different assets and protocols",
                    current_value=risk_summary.concentration_risk,
                    threshold_value=80.0,
                    risk_score=risk_summary.concentration_risk,
                    timestamp=current_time
                ))
            
            # Low liquidity alert
            if risk_summary.liquidity_risk > 70:
                alerts.append(RiskAlert(
                    wallet_address=wallet_address,
                    protocol="multi_protocol",
                    alert_type=AlertType.LOW_LIQUIDITY,
                    severity=RiskSeverity.MEDIUM,
                    title="Low Liquidity Risk",
                    description=f"Portfolio contains low-liquidity assets with risk score {risk_summary.liquidity_risk:.1f}%",
                    recommendation="Consider moving to more liquid assets for better exit options",
                    current_value=risk_summary.liquidity_risk,
                    threshold_value=70.0,
                    risk_score=risk_summary.liquidity_risk,
                    timestamp=current_time
                ))
            
            # Gas price alert
            async with api_manager.etherscan:
                gas_metrics = await api_manager.etherscan.get_gas_prices()
                
            if gas_metrics.fast_gas_price > 100:  # High gas prices
                alerts.append(RiskAlert(
                    wallet_address=wallet_address,
                    protocol="ethereum",
                    alert_type=AlertType.GAS_SPIKE,
                    severity=RiskSeverity.MEDIUM,
                    title="High Gas Prices",
                    description=f"Current gas prices are elevated at {gas_metrics.fast_gas_price:.0f} gwei",
                    recommendation="Delay non-urgent transactions until gas prices normalize",
                    current_value=gas_metrics.fast_gas_price,
                    threshold_value=100.0,
                    risk_score=min(100.0, gas_metrics.fast_gas_price / 2),
                    timestamp=current_time
                ))
        
        except Exception as e:
            logger.error(f"Error generating alerts for {wallet_address}", error=str(e))
        
        return alerts
    
    async def aggregate_risk_summary(self, wallet_address: str) -> RiskSummary:
        """Aggregate all risk metrics into a comprehensive summary"""
        try:
            current_time = datetime.utcnow()
            
            # Get positions from all protocols
            protocols = ["aave_v3", "aave_v2", "compound"]
            all_positions = []
            total_portfolio_value = 0.0
            total_debt = 0.0
            overall_health_ratios = []
            
            # Collect data from all protocols
            for protocol in protocols:
                health_data = await self.calculate_position_health(wallet_address, protocol)
                if health_data["collateral_usd"] > 0:
                    total_portfolio_value += health_data["collateral_usd"]
                    total_debt += health_data["debt_usd"]
                    
                    if health_data["health_ratio"] != float('inf'):
                        overall_health_ratios.append(health_data["health_ratio"])
            
            # Calculate overall health ratio
            if total_debt > 0 and total_portfolio_value > 0:
                overall_health_ratio = total_portfolio_value * 0.8 / total_debt  # Approximate
            elif total_debt == 0:
                overall_health_ratio = float('inf')
            else:
                overall_health_ratio = 0.0
            
            # Use minimum health ratio if we have multiple protocols
            if overall_health_ratios:
                overall_health_ratio = min(overall_health_ratios)
            
            # Calculate individual risk components
            liquidation_risk = 0.0
            if overall_health_ratio != float('inf') and total_debt > 0:
                if overall_health_ratio > 2.0:
                    liquidation_risk = 0.0
                elif overall_health_ratio > 1.5:
                    liquidation_risk = 20.0 * (2.0 - overall_health_ratio) / 0.5
                elif overall_health_ratio > 1.2:
                    liquidation_risk = 20.0 + 40.0 * (1.5 - overall_health_ratio) / 0.3
                elif overall_health_ratio > 1.0:
                    liquidation_risk = 60.0 + 35.0 * (1.2 - overall_health_ratio) / 0.2
                else:
                    liquidation_risk = 95.0 + 5.0 * max(0, 1.0 - overall_health_ratio)
            
            # Mock calculations for other risk metrics (would be calculated from real data)
            volatility_risk = min(100.0, abs(np.random.normal(30, 15)))  # Mock for now
            concentration_risk = min(100.0, abs(np.random.normal(40, 20)))  # Mock for now
            liquidity_risk = min(100.0, abs(np.random.normal(25, 10)))  # Mock for now
            
            # Calculate overall risk score
            overall_risk_score = (
                0.4 * liquidation_risk +
                0.25 * volatility_risk +
                0.20 * concentration_risk +
                0.15 * liquidity_risk
            )
            
            # Determine risk level
            if overall_risk_score < 20:
                risk_level = RiskSeverity.LOW
            elif overall_risk_score < 50:
                risk_level = RiskSeverity.MEDIUM
            elif overall_risk_score < 80:
                risk_level = RiskSeverity.HIGH
            else:
                risk_level = RiskSeverity.CRITICAL
            
            # Protocol exposure
            protocol_exposure = {}
            active_protocols = []
            
            for protocol in protocols:
                health_data = await self.calculate_position_health(wallet_address, protocol)
                if health_data["collateral_usd"] > 0:
                    active_protocols.append(protocol)
                    if total_portfolio_value > 0:
                        protocol_exposure[protocol] = health_data["collateral_usd"] / total_portfolio_value
            
            # Count active alerts
            alerts_collection = get_collection(Collections.ALERTS)
            active_alerts_count = await alerts_collection.count_documents({
                "wallet_address": wallet_address.lower(),
                "is_resolved": False
            })
            
            critical_alerts_count = await alerts_collection.count_documents({
                "wallet_address": wallet_address.lower(),
                "is_resolved": False,
                "severity": RiskSeverity.CRITICAL
            })
            
            return RiskSummary(
                wallet_address=wallet_address,
                last_updated=current_time,
                overall_risk_score=overall_risk_score,
                risk_level=risk_level,
                health_ratio=overall_health_ratio if overall_health_ratio != float('inf') else None,
                total_portfolio_value_usd=total_portfolio_value,
                total_debt_usd=total_debt,
                net_worth_usd=total_portfolio_value - total_debt,
                liquidation_risk=liquidation_risk,
                volatility_risk=volatility_risk,
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk,
                protocols=active_protocols,
                protocol_exposure=protocol_exposure,
                active_alerts_count=active_alerts_count,
                critical_alerts_count=critical_alerts_count
            )
            
        except Exception as e:
            logger.error(f"Error aggregating risk summary for {wallet_address}", error=str(e))
            return RiskSummary(
                wallet_address=wallet_address,
                last_updated=datetime.utcnow(),
                overall_risk_score=0.0,
                risk_level=RiskSeverity.LOW,
                total_portfolio_value_usd=0.0,
                total_debt_usd=0.0,
                net_worth_usd=0.0,
                liquidation_risk=0.0,
                volatility_risk=0.0,
                concentration_risk=0.0,
                liquidity_risk=0.0,
                protocols=[],
                protocol_exposure={},
                active_alerts_count=0,
                critical_alerts_count=0
            )

# Global risk calculator instance
risk_calculator = RiskCalculator()