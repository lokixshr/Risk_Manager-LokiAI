"""
Database migration, backup, and management system
"""
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import structlog
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import IndexModel, ASCENDING, DESCENDING

logger = structlog.get_logger()

class DatabaseManager:
    """Database migration and management system"""
    
    def __init__(self, mongodb_uri: str, database_name: str):
        self.client = AsyncIOMotorClient(mongodb_uri)
        self.db = self.client[database_name]
        self.migrations_collection = self.db["_migrations"]
    
    async def run_migrations(self):
        """Run all pending database migrations"""
        migrations = [
            Migration001_InitialSchema(),
            Migration002_AddIndexes(),
            Migration003_UpdateAlertSchema(),
            Migration004_AddMetricsCollection(),
            Migration005_OptimizeIndexes()
        ]
        
        for migration in migrations:
            if not await self._is_migration_applied(migration.version):
                logger.info(f"Applying migration {migration.version}: {migration.description}")
                await migration.apply(self.db)
                await self._record_migration(migration)
                logger.info(f"Migration {migration.version} completed")
    
    async def _is_migration_applied(self, version: str) -> bool:
        """Check if migration has been applied"""
        result = await self.migrations_collection.find_one({"version": version})
        return result is not None
    
    async def _record_migration(self, migration):
        """Record that migration has been applied"""
        await self.migrations_collection.insert_one({
            "version": migration.version,
            "description": migration.description,
            "applied_at": datetime.utcnow()
        })
    
    async def backup_database(self, backup_path: str = None) -> str:
        """Create database backup"""
        if not backup_path:
            backup_path = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        logger.info(f"Creating database backup: {backup_path}")
        
        backup_data = {
            "created_at": datetime.utcnow().isoformat(),
            "collections": {}
        }
        
        # Backup all collections
        collections = await self.db.list_collection_names()
        for collection_name in collections:
            if not collection_name.startswith("_"):  # Skip system collections
                collection = self.db[collection_name]
                documents = await collection.find({}).to_list(length=None)
                backup_data["collections"][collection_name] = documents
        
        # Write backup file
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, default=str, indent=2)
        
        logger.info(f"Database backup created: {backup_path}")
        return backup_path


class BaseMigration:
    """Base class for database migrations"""
    
    version: str = ""
    description: str = ""
    
    async def apply(self, db):
        """Apply the migration"""
        raise NotImplementedError


class Migration001_InitialSchema(BaseMigration):
    version = "001"
    description = "Create initial schema and collections"
    
    async def apply(self, db):
        """Apply initial schema migration"""
        
        # Create collections with validation
        await db.create_collection("risk_manager_positions")
        await db.create_collection("risk_manager_alerts")
        await db.create_collection("risk_manager_metrics")
        await db.create_collection("risk_manager_training_data")
        
        logger.info("Initial collections created")


class Migration002_AddIndexes(BaseMigration):
    version = "002"
    description = "Add performance indexes"
    
    async def apply(self, db):
        """Add performance indexes"""
        
        # Positions indexes
        positions = db["risk_manager_positions"]
        await positions.create_indexes([
            IndexModel([("wallet_address", ASCENDING)]),
            IndexModel([("protocol", ASCENDING)]),
            IndexModel([("timestamp", DESCENDING)]),
            IndexModel([("wallet_address", ASCENDING), ("protocol", ASCENDING)]),
            IndexModel([("health_ratio", ASCENDING)])
        ])
        
        # Alerts indexes
        alerts = db["risk_manager_alerts"]
        await alerts.create_indexes([
            IndexModel([("wallet_address", ASCENDING)]),
            IndexModel([("alert_type", ASCENDING)]),
            IndexModel([("severity", ASCENDING)]),
            IndexModel([("timestamp", DESCENDING)]),
            IndexModel([("is_resolved", ASCENDING)]),
            IndexModel([("wallet_address", ASCENDING), ("is_resolved", ASCENDING)])
        ])
        
        # Metrics indexes
        metrics = db["risk_manager_metrics"]
        await metrics.create_indexes([
            IndexModel([("wallet_address", ASCENDING)]),
            IndexModel([("metric_type", ASCENDING)]),
            IndexModel([("timestamp", DESCENDING)]),
            IndexModel([("wallet_address", ASCENDING), ("metric_type", ASCENDING)])
        ])
        
        logger.info("Performance indexes created")


class Migration003_UpdateAlertSchema(BaseMigration):
    version = "003"
    description = "Update alert schema with new fields"
    
    async def apply(self, db):
        """Update alert schema"""
        
        alerts = db["risk_manager_alerts"]
        
        # Add new fields to existing alerts
        await alerts.update_many(
            {"acknowledged": {"$exists": False}},
            {"$set": {"acknowledged": False}}
        )
        
        await alerts.update_many(
            {"metadata": {"$exists": False}},
            {"$set": {"metadata": {}}}
        )
        
        logger.info("Alert schema updated")


class Migration004_AddMetricsCollection(BaseMigration):
    version = "004"
    description = "Add system metrics collection"
    
    async def apply(self, db):
        """Add system metrics collection"""
        
        await db.create_collection("system_metrics")
        
        system_metrics = db["system_metrics"]
        await system_metrics.create_indexes([
            IndexModel([("metric_name", ASCENDING)]),
            IndexModel([("timestamp", DESCENDING)]),
            IndexModel([("service", ASCENDING)]),
            IndexModel([("metric_name", ASCENDING), ("timestamp", DESCENDING)])
        ])
        
        logger.info("System metrics collection created")


class Migration005_OptimizeIndexes(BaseMigration):
    version = "005"
    description = "Optimize database indexes for better performance"
    
    async def apply(self, db):
        """Optimize indexes"""
        
        # Add compound indexes for common queries
        positions = db["risk_manager_positions"]
        await positions.create_index([
            ("wallet_address", ASCENDING),
            ("timestamp", DESCENDING),
            ("protocol", ASCENDING)
        ])
        
        # Add TTL index for old training data
        training_data = db["risk_manager_training_data"]
        await training_data.create_index(
            [("timestamp", ASCENDING)],
            expireAfterSeconds=90*24*3600  # 90 days
        )
        
        logger.info("Index optimization completed")


class BackupManager:
    """Database backup management"""
    
    def __init__(self, mongodb_uri: str, database_name: str):
        self.mongodb_uri = mongodb_uri
        self.database_name = database_name
        self.backup_dir = "backups"
        
        # Create backup directory
        os.makedirs(self.backup_dir, exist_ok=True)
    
    async def create_full_backup(self) -> str:
        """Create full database backup"""
        db_manager = DatabaseManager(self.mongodb_uri, self.database_name)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(self.backup_dir, f"full_backup_{timestamp}.json")
        
        return await db_manager.backup_database(backup_file)
    
    async def create_incremental_backup(self, since: datetime) -> str:
        """Create incremental backup since timestamp"""
        client = AsyncIOMotorClient(self.mongodb_uri)
        db = client[self.database_name]
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(self.backup_dir, f"incremental_backup_{timestamp}.json")
        
        backup_data = {
            "created_at": datetime.utcnow().isoformat(),
            "incremental_since": since.isoformat(),
            "collections": {}
        }
        
        # Backup only changed documents
        collections = ["risk_manager_positions", "risk_manager_alerts", "risk_manager_metrics"]
        
        for collection_name in collections:
            collection = db[collection_name]
            
            # Find documents modified since the timestamp
            query = {
                "$or": [
                    {"created_at": {"$gte": since}},
                    {"updated_at": {"$gte": since}}
                ]
            }
            
            documents = await collection.find(query).to_list(length=None)
            backup_data["collections"][collection_name] = documents
        
        with open(backup_file, 'w') as f:
            json.dump(backup_data, f, default=str, indent=2)
        
        await client.close()
        logger.info(f"Incremental backup created: {backup_file}")
        return backup_file
    
    async def restore_from_backup(self, backup_file: str):
        """Restore database from backup"""
        logger.info(f"Restoring database from backup: {backup_file}")
        
        with open(backup_file, 'r') as f:
            backup_data = json.load(f)
        
        client = AsyncIOMotorClient(self.mongodb_uri)
        db = client[self.database_name]
        
        for collection_name, documents in backup_data["collections"].items():
            if documents:
                collection = db[collection_name]
                
                # Drop existing collection
                await collection.drop()
                
                # Insert backed up documents
                await collection.insert_many(documents)
                
                logger.info(f"Restored {len(documents)} documents to {collection_name}")
        
        await client.close()
        logger.info("Database restore completed")
    
    def list_backups(self) -> List[Dict]:
        """List available backups"""
        backups = []
        
        for filename in os.listdir(self.backup_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.backup_dir, filename)
                stat = os.stat(filepath)
                
                backups.append({
                    "filename": filename,
                    "filepath": filepath,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
                })
        
        return sorted(backups, key=lambda x: x["created"], reverse=True)
    
    async def cleanup_old_backups(self, keep_days: int = 30):
        """Clean up old backup files"""
        cutoff_date = datetime.utcnow() - timedelta(days=keep_days)
        
        for backup in self.list_backups():
            created_date = datetime.fromisoformat(backup["created"])
            if created_date < cutoff_date:
                os.remove(backup["filepath"])
                logger.info(f"Removed old backup: {backup['filename']}")


async def main():
    """Main database management function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database Management Tool")
    parser.add_argument("--action", choices=["migrate", "backup", "restore", "list-backups"], required=True)
    parser.add_argument("--mongodb-uri", default=os.getenv("MONGODB_URI"))
    parser.add_argument("--database-name", default=os.getenv("MONGO_DB_NAME", "loki_agents"))
    parser.add_argument("--backup-file", help="Backup file for restore operation")
    parser.add_argument("--incremental", action="store_true", help="Create incremental backup")
    parser.add_argument("--since", help="Incremental backup since timestamp (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    if not args.mongodb_uri:
        print("Error: MongoDB URI is required (set MONGODB_URI env var or use --mongodb-uri)")
        return
    
    backup_manager = BackupManager(args.mongodb_uri, args.database_name)
    
    if args.action == "migrate":
        db_manager = DatabaseManager(args.mongodb_uri, args.database_name)
        await db_manager.run_migrations()
        print("Migrations completed successfully")
    
    elif args.action == "backup":
        if args.incremental:
            if args.since:
                since_date = datetime.fromisoformat(args.since)
            else:
                since_date = datetime.utcnow() - timedelta(hours=24)  # Last 24 hours
            
            backup_file = await backup_manager.create_incremental_backup(since_date)
        else:
            backup_file = await backup_manager.create_full_backup()
        
        print(f"Backup created: {backup_file}")
    
    elif args.action == "restore":
        if not args.backup_file:
            print("Error: --backup-file is required for restore operation")
            return
        
        await backup_manager.restore_from_backup(args.backup_file)
        print("Restore completed successfully")
    
    elif args.action == "list-backups":
        backups = backup_manager.list_backups()
        
        print(f"\nAvailable backups ({len(backups)} files):")
        print("-" * 80)
        for backup in backups:
            print(f"{backup['filename']:<40} {backup['size_mb']:>8.2f} MB  {backup['created']}")


if __name__ == "__main__":
    asyncio.run(main())