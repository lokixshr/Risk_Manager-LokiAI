import httpx
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Dict, List, Optional, Any, Union
import structlog
from datetime import datetime, timedelta
import json
from .config import settings, NETWORK_CONFIG
from .models import AssetPrice, AavePositionData, CompoundPositionData, CurvePoolData, GasMetrics

logger = structlog.get_logger()

class APIError(Exception):
    """Custom exception for API errors"""
    pass

class BaseAPIClient:
    def __init__(self, base_url: str, headers: Optional[Dict] = None):
        self.base_url = base_url
        self.default_headers = headers or {}
        self.client = None
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.default_headers,
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make HTTP request with retry logic"""
        if not self.client:
            raise RuntimeError("Client not initialized. Use 'async with' context.")
        
        try:
            response = await self.client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code} for {method} {endpoint}", 
                        error=str(e), status_code=e.response.status_code)
            raise APIError(f"API request failed: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Request error for {method} {endpoint}", error=str(e))
            raise APIError(f"Network error: {str(e)}")

class CoinGeckoClient(BaseAPIClient):
    def __init__(self):
        headers = {
            "x-cg-demo-api-key": settings.COINGECKO_API_KEY,
            "Content-Type": "application/json"
        }
        super().__init__(settings.COINGECKO_BASE_URL, headers)
    
    async def get_token_prices(self, token_addresses: List[str], vs_currency: str = "usd") -> Dict[str, AssetPrice]:
        """Get current token prices from CoinGecko"""
        addresses_str = ",".join(token_addresses)
        params = {
            "contract_addresses": addresses_str,
            "vs_currencies": vs_currency,
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true"
        }
        
        data = await self._make_request("GET", "/simple/token_price/ethereum", params=params)
        
        prices = {}
        for address, price_data in data.items():
            if price_data:
                prices[address.lower()] = AssetPrice(
                    symbol=address,  # We'll need to resolve symbol separately
                    price_usd=price_data.get(vs_currency, 0),
                    market_cap=price_data.get(f"{vs_currency}_market_cap"),
                    volume_24h=price_data.get(f"{vs_currency}_24h_vol"),
                    price_change_24h=price_data.get(f"{vs_currency}_24h_change"),
                    last_updated=datetime.utcnow()
                )
        
        return prices
    
    async def get_historical_prices(self, token_address: str, days: int = 7) -> List[Dict]:
        """Get historical price data for volatility calculation"""
        params = {
            "contract_address": token_address,
            "vs_currency": "usd",
            "days": str(days)
        }
        
        data = await self._make_request("GET", "/coins/ethereum/contract/{}/market_chart".format(token_address), params=params)
        
        # Convert price data to more usable format
        prices = []
        for timestamp, price in data.get("prices", []):
            prices.append({
                "timestamp": datetime.fromtimestamp(timestamp / 1000),
                "price": price
            })
        
        return prices

class SubgraphClient(BaseAPIClient):
    def __init__(self, subgraph_url: str):
        headers = {
            "Content-Type": "application/json"
        }
        if settings.THEGRAPH_API_KEY:
            headers["Authorization"] = f"Bearer {settings.THEGRAPH_API_KEY}"
        
        super().__init__(subgraph_url, headers)
    
    async def query(self, query: str, variables: Optional[Dict] = None) -> dict:
        """Execute GraphQL query"""
        payload = {
            "query": query,
            "variables": variables or {}
        }
        
        return await self._make_request("POST", "", json=payload)

class AaveClient:
    def __init__(self):
        self.v2_client = SubgraphClient(NETWORK_CONFIG["ethereum"]["subgraph_endpoints"]["aave_v2"])
        self.v3_client = SubgraphClient(NETWORK_CONFIG["ethereum"]["subgraph_endpoints"]["aave_v3"])
    
    async def get_user_positions(self, user_address: str, version: str = "v3") -> AavePositionData:
        """Get user positions from Aave"""
        client = self.v3_client if version == "v3" else self.v2_client
        
        query = """
        query GetUserReserves($userAddress: String!) {
          user(id: $userAddress) {
            reserves {
              currentATokenBalance
              currentStableDebt
              currentVariableDebt
              liquidityRate
              variableBorrowRate
              reserve {
                symbol
                decimals
                liquidationThreshold
                reserveLiquidationThreshold
                usageAsCollateralEnabled
                price {
                  priceInEth
                }
              }
            }
          }
          userReserves(where: {user: $userAddress}) {
            user {
              id
            }
            reserve {
              symbol
              name
              decimals
              liquidationThreshold
              reserveLiquidationThreshold
            }
            currentATokenBalance
            currentStableDebt  
            currentVariableDebt
          }
        }
        """
        
        async with client:
            result = await client.query(query, {"userAddress": user_address.lower()})
            
            if not result.get("data") or not result["data"].get("user"):
                return AavePositionData(
                    user_address=user_address,
                    total_collateral_eth=0.0,
                    total_debt_eth=0.0,
                    available_borrow_eth=0.0,
                    current_liquidation_threshold=0.0,
                    health_factor=float('inf')
                )
            
            # Calculate position metrics from subgraph data
            user_data = result["data"]["user"]
            reserves = user_data.get("reserves", [])
            
            total_collateral_eth = 0.0
            total_debt_eth = 0.0
            weighted_liquidation_threshold = 0.0
            
            for reserve in reserves:
                reserve_data = reserve["reserve"]
                atoken_balance = float(reserve.get("currentATokenBalance", 0))
                stable_debt = float(reserve.get("currentStableDebt", 0))
                variable_debt = float(reserve.get("currentVariableDebt", 0))
                
                if atoken_balance > 0:
                    collateral_eth = atoken_balance * float(reserve_data["price"]["priceInEth"])
                    total_collateral_eth += collateral_eth
                    weighted_liquidation_threshold += collateral_eth * float(reserve_data["liquidationThreshold"]) / 10000
                
                if stable_debt > 0 or variable_debt > 0:
                    debt_eth = (stable_debt + variable_debt) * float(reserve_data["price"]["priceInEth"])
                    total_debt_eth += debt_eth
            
            # Calculate health factor
            if total_debt_eth > 0 and total_collateral_eth > 0:
                avg_liquidation_threshold = weighted_liquidation_threshold / total_collateral_eth if total_collateral_eth > 0 else 0
                health_factor = (total_collateral_eth * avg_liquidation_threshold) / total_debt_eth
                available_borrow = max(0, total_collateral_eth * avg_liquidation_threshold - total_debt_eth)
            else:
                health_factor = float('inf')
                available_borrow = total_collateral_eth * (weighted_liquidation_threshold / total_collateral_eth if total_collateral_eth > 0 else 0)
            
            return AavePositionData(
                user_address=user_address,
                total_collateral_eth=total_collateral_eth,
                total_debt_eth=total_debt_eth,
                available_borrow_eth=available_borrow,
                current_liquidation_threshold=weighted_liquidation_threshold / total_collateral_eth if total_collateral_eth > 0 else 0,
                health_factor=health_factor
            )

class CompoundClient(BaseAPIClient):
    def __init__(self):
        super().__init__("https://api.compound.finance/api/v2")
    
    async def get_user_positions(self, user_address: str) -> CompoundPositionData:
        """Get user positions from Compound"""
        try:
            # Get account data
            account_data = await self._make_request("GET", f"/account?addresses[]={user_address}")
            
            if not account_data.get("accounts"):
                return CompoundPositionData(
                    user_address=user_address,
                    total_collateral_value_usd=0.0,
                    total_borrow_value_usd=0.0,
                    account_liquidity=0.0,
                    shortfall=0.0
                )
            
            account = account_data["accounts"][0]
            
            return CompoundPositionData(
                user_address=user_address,
                total_collateral_value_usd=float(account.get("total_collateral_value_in_eth", {}).get("value", 0)) * 2000,  # Rough ETH price
                total_borrow_value_usd=float(account.get("total_borrow_value_in_eth", {}).get("value", 0)) * 2000,
                account_liquidity=float(account.get("liquidity", {}).get("value", 0)),
                shortfall=float(account.get("shortfall", {}).get("value", 0))
            )
        except Exception as e:
            logger.error(f"Error fetching Compound data for {user_address}", error=str(e))
            return CompoundPositionData(
                user_address=user_address,
                total_collateral_value_usd=0.0,
                total_borrow_value_usd=0.0,
                account_liquidity=0.0,
                shortfall=0.0
            )

class CurveClient(BaseAPIClient):
    def __init__(self):
        super().__init__(settings.CURVE_API_BASE)
    
    async def get_pool_data(self, pool_address: str) -> Optional[CurvePoolData]:
        """Get Curve pool data"""
        try:
            # Get pool info
            pool_data = await self._make_request("GET", f"/getPools")
            
            for pool in pool_data.get("data", {}).get("poolData", []):
                if pool.get("address", "").lower() == pool_address.lower():
                    return CurvePoolData(
                        pool_address=pool_address,
                        pool_name=pool.get("name", "Unknown"),
                        virtual_price=float(pool.get("virtualPrice", 1.0)),
                        total_supply=float(pool.get("totalSupply", 0)),
                        liquidity_usd=float(pool.get("usdTotal", 0)),
                        daily_volume_usd=float(pool.get("dailyVolume", 0))
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching Curve pool data for {pool_address}", error=str(e))
            return None

class AlchemyClient(BaseAPIClient):
    def __init__(self):
        super().__init__(settings.ALCHEMY_URL)
    
    async def get_token_balances(self, wallet_address: str, token_addresses: Optional[List[str]] = None) -> List[Dict]:
        """Get token balances for a wallet"""
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getTokenBalances",
            "params": [wallet_address]
        }
        
        if token_addresses:
            payload["params"].append(token_addresses)
        
        try:
            response = await self._make_request("POST", "", json=payload)
            return response.get("result", {}).get("tokenBalances", [])
        except Exception as e:
            logger.error(f"Error fetching token balances for {wallet_address}", error=str(e))
            return []
    
    async def get_eth_balance(self, wallet_address: str) -> float:
        """Get ETH balance for a wallet"""
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "eth_getBalance",
            "params": [wallet_address, "latest"]
        }
        
        try:
            response = await self._make_request("POST", "", json=payload)
            balance_hex = response.get("result", "0x0")
            balance_wei = int(balance_hex, 16)
            return balance_wei / 10**18
        except Exception as e:
            logger.error(f"Error fetching ETH balance for {wallet_address}", error=str(e))
            return 0.0

class EtherscanClient(BaseAPIClient):
    def __init__(self):
        super().__init__("https://api.etherscan.io/api")
    
    async def get_gas_prices(self) -> GasMetrics:
        """Get current gas prices"""
        params = {
            "module": "gastracker",
            "action": "gasoracle",
            "apikey": settings.ETHERSCAN_API_KEY
        }
        
        try:
            data = await self._make_request("GET", "", params=params)
            result = data.get("result", {})
            
            return GasMetrics(
                chain_id=1,
                standard_gas_price=float(result.get("SafeGasPrice", 20)),
                fast_gas_price=float(result.get("ProposeGasPrice", 25)),
                instant_gas_price=float(result.get("FastGasPrice", 30)),
                liquidation_gas_cost_usd=0.0,  # Will be calculated based on gas price
                safe_gas_threshold=50.0  # 50 gwei threshold
            )
        except Exception as e:
            logger.error("Error fetching gas prices from Etherscan", error=str(e))
            return GasMetrics(
                chain_id=1,
                standard_gas_price=20.0,
                fast_gas_price=25.0,
                instant_gas_price=30.0,
                liquidation_gas_cost_usd=0.0,
                safe_gas_threshold=50.0
            )

class DefiLlamaClient(BaseAPIClient):
    def __init__(self):
        super().__init__(settings.DEFILLAMA_BASE_URL)
    
    async def get_protocol_tvl(self, protocol: str) -> float:
        """Get TVL for a specific protocol"""
        try:
            data = await self._make_request("GET", f"/protocol/{protocol}")
            
            # Get the most recent TVL value
            tvl_data = data.get("tvl", [])
            if tvl_data:
                return float(tvl_data[-1].get("totalLiquidityUSD", 0))
            return 0.0
            
        except Exception as e:
            logger.error(f"Error fetching TVL for protocol {protocol}", error=str(e))
            return 0.0
    
    async def get_chain_tvl(self, chain: str = "Ethereum") -> float:
        """Get total TVL for a chain"""
        try:
            data = await self._make_request("GET", "/chains")
            
            for chain_data in data:
                if chain_data.get("name") == chain:
                    return float(chain_data.get("tvl", 0))
            return 0.0
            
        except Exception as e:
            logger.error(f"Error fetching TVL for chain {chain}", error=str(e))
            return 0.0

class APIManager:
    """Central manager for all external API clients"""
    
    def __init__(self):
        self.coingecko = CoinGeckoClient()
        self.aave = AaveClient()
        self.compound = CompoundClient()
        self.curve = CurveClient()
        self.alchemy = AlchemyClient()
        self.etherscan = EtherscanClient()
        self.defillama = DefiLlamaClient()
    
    async def health_check(self) -> Dict[str, str]:
        """Check health of all external APIs"""
        health_status = {}
        
        # Test CoinGecko
        try:
            async with self.coingecko:
                await self.coingecko._make_request("GET", "/ping")
            health_status["coingecko"] = "healthy"
        except:
            health_status["coingecko"] = "unhealthy"
        
        # Test Etherscan
        try:
            async with self.etherscan:
                await self.etherscan.get_gas_prices()
            health_status["etherscan"] = "healthy"
        except:
            health_status["etherscan"] = "unhealthy"
        
        # Test DefiLlama
        try:
            async with self.defillama:
                await self.defillama.get_chain_tvl()
            health_status["defillama"] = "healthy"
        except:
            health_status["defillama"] = "unhealthy"
        
        # Test Alchemy
        try:
            async with self.alchemy:
                await self.alchemy.get_eth_balance("0x0000000000000000000000000000000000000000")
            health_status["alchemy"] = "healthy"
        except:
            health_status["alchemy"] = "unhealthy"
        
        return health_status

# Global API manager instance
api_manager = APIManager()