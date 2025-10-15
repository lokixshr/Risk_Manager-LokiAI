import asyncio
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
import redis.asyncio as aioredis
from pymongo import IndexModel, ASCENDING, DESCENDING
import structlog
from typing import Optional
from .config import settings, Collections

logger = structlog.get_logger()

class DatabaseManager:
    def __init__(self):
        self.mongo_client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self.redis_client: Optional[aioredis.Redis] = None
        
    async def connect(self):
        """Initialize database connections"""
        try:
            # MongoDB connection
            self.mongo_client = AsyncIOMotorClient(
                settings.MONGODB_URI,
                maxPoolSize=20,
                minPoolSize=5,
                maxIdleTimeMS=30000,
                serverSelectionTimeoutMS=5000,
                socketTimeoutMS=20000
            )
            self.database = self.mongo_client[settings.MONGO_DB_NAME]
            
            # Test MongoDB connection
            await self.mongo_client.admin.command('ping')
            logger.info("Connected to MongoDB Atlas", database=settings.MONGO_DB_NAME)
            
            # Create indexes
            await self._create_indexes()
            
            # Redis connection (if enabled)
            if settings.ENABLE_REDIS:
                self.redis_client = aioredis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=20,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # Test Redis connection
                await self.redis_client.ping()
                logger.info("Connected to Redis", url=settings.REDIS_URL)
            else:
                logger.warning("Redis is disabled - rate limiting will be limited")
                
        except Exception as e:
            logger.error("Failed to connect to databases", error=str(e))
            raise
    
    async def disconnect(self):
        """Close database connections"""
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("Disconnected from MongoDB")
            
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Disconnected from Redis")
    
    async def _create_indexes(self):
        """Create necessary MongoDB indexes for performance"""
        try:
            # Positions collection indexes
            positions_indexes = [
                IndexModel([("wallet_address", ASCENDING)]),
                IndexModel([("protocol", ASCENDING)]),
                IndexModel([("timestamp", DESCENDING)]),
                IndexModel([("wallet_address", ASCENDING), ("protocol", ASCENDING)]),
                IndexModel([("health_ratio", ASCENDING)]),  # For liquidation risk queries
            ]
            await self.database[Collections.POSITIONS].create_indexes(positions_indexes)
            
            # Alerts collection indexes
            alerts_indexes = [
                IndexModel([("wallet_address", ASCENDING)]),
                IndexModel([("alert_type", ASCENDING)]),
                IndexModel([("severity", ASCENDING)]),
                IndexModel([("timestamp", DESCENDING)]),
                IndexModel([("is_resolved", ASCENDING)]),
                IndexModel([("wallet_address", ASCENDING), ("is_resolved", ASCENDING)]),
            ]
            await self.database[Collections.ALERTS].create_indexes(alerts_indexes)
            
            # Metrics collection indexes
            metrics_indexes = [
                IndexModel([("wallet_address", ASCENDING)]),
                IndexModel([("metric_type", ASCENDING)]),
                IndexModel([("timestamp", DESCENDING)]),
                IndexModel([("wallet_address", ASCENDING), ("metric_type", ASCENDING)]),
            ]
            await self.database[Collections.METRICS].create_indexes(metrics_indexes)
            
            # Training data indexes (for ML)
            training_indexes = [
                IndexModel([("wallet_address", ASCENDING)]),
                IndexModel([("timestamp", DESCENDING)]),
                IndexModel([("liquidated", ASCENDING)]),  # Target variable
            ]
            await self.database[Collections.TRAINING_DATA].create_indexes(training_indexes)
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.error("Failed to create indexes", error=str(e))
            raise
    
    def get_collection(self, collection_name: str):
        """Get MongoDB collection"""
        if self.database is None:
            raise RuntimeError("Database not connected")
        return self.database[collection_name]
    
    async def health_check(self) -> dict:
        """Check database health status"""
        health = {
            "mongodb": {"status": "disconnected", "latency_ms": None},
            "redis": {"status": "disconnected", "latency_ms": None}
        }
        
        try:
            # MongoDB health check
            if self.mongo_client:
                import time
                start_time = time.time()
                await self.mongo_client.admin.command('ping')
                latency = (time.time() - start_time) * 1000
                health["mongodb"] = {"status": "connected", "latency_ms": round(latency, 2)}
        except Exception as e:
            health["mongodb"]["error"] = str(e)
        
        try:
            # Redis health check
            if self.redis_client:
                import time
                start_time = time.time()
                await self.redis_client.ping()
                latency = (time.time() - start_time) * 1000
                health["redis"] = {"status": "connected", "latency_ms": round(latency, 2)}
        except Exception as e:
            health["redis"]["error"] = str(e)
            
        return health

# Global database manager instance
db_manager = DatabaseManager()

# Convenience functions
def get_db() -> AsyncIOMotorDatabase:
    """Get database instance"""
    if db_manager.database is None:
        raise RuntimeError("Database not connected")
    return db_manager.database

def get_redis() -> Optional[aioredis.Redis]:
    """Get Redis instance"""
    return db_manager.redis_client

def get_collection(collection_name: str):
    """Get MongoDB collection"""
    return db_manager.get_collection(collection_name)