import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Set
import structlog
from concurrent.futures import ThreadPoolExecutor

from .config import settings, Collections
from .database import get_collection
from .risk_engine import risk_calculator
from .security import log_to_loki
from .models import TrainingDataPoint

logger = structlog.get_logger()

class BackgroundTaskManager:
    """Manages periodic background tasks for risk monitoring"""
    
    def __init__(self):
        self.is_running = False
        self.tasks: List[asyncio.Task] = []
        self.task_interval = settings.BACKGROUND_TASK_INTERVAL  # 10 minutes
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Track processed wallets to avoid duplicate processing
        self.processed_wallets: Set[str] = set()
        self.last_cleanup = datetime.utcnow()
    
    async def start(self):
        """Start all background tasks"""
        if self.is_running:
            logger.warning("Background tasks already running")
            return
        
        self.is_running = True
        logger.info("Starting background tasks")
        
        # Start main risk monitoring task
        self.tasks.append(
            asyncio.create_task(self._risk_monitoring_loop())
        )
        
        # Start metrics aggregation task
        self.tasks.append(
            asyncio.create_task(self._metrics_aggregation_loop())
        )
        
        # Start alert cleanup task
        self.tasks.append(
            asyncio.create_task(self._alert_cleanup_loop())
        )
        
        # Start training data collection task
        self.tasks.append(
            asyncio.create_task(self._training_data_collection_loop())
        )
        
        await log_to_loki("background_tasks_started", {
            "task_count": len(self.tasks),
            "interval_seconds": self.task_interval
        })
    
    async def stop(self):
        """Stop all background tasks"""
        if not self.is_running:
            return
        
        logger.info("Stopping background tasks")
        self.is_running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        self.tasks.clear()
        self.executor.shutdown(wait=True)
        
        await log_to_loki("background_tasks_stopped", {
            "stopped_at": datetime.utcnow().isoformat()
        })
    
    async def _risk_monitoring_loop(self):
        """Main risk monitoring loop - runs every 10 minutes"""
        while self.is_running:
            try:
                start_time = datetime.utcnow()
                logger.info("Starting risk monitoring cycle")
                
                # Get list of wallets to monitor
                wallets_to_monitor = await self._get_wallets_to_monitor()
                
                if not wallets_to_monitor:
                    logger.info("No wallets to monitor")
                else:
                    logger.info(f"Monitoring {len(wallets_to_monitor)} wallets")
                    
                    # Process wallets in batches to avoid overloading APIs
                    batch_size = 10
                    for i in range(0, len(wallets_to_monitor), batch_size):
                        batch = wallets_to_monitor[i:i + batch_size]
                        await self._process_wallet_batch(batch)
                        
                        # Small delay between batches
                        await asyncio.sleep(2)
                
                # Clean up processed wallets set periodically
                if datetime.utcnow() - self.last_cleanup > timedelta(hours=1):
                    self.processed_wallets.clear()
                    self.last_cleanup = datetime.utcnow()
                
                duration = (datetime.utcnow() - start_time).total_seconds()
                logger.info(f"Risk monitoring cycle completed in {duration:.1f}s")
                
                await log_to_loki("risk_monitoring_cycle_completed", {
                    "wallets_processed": len(wallets_to_monitor),
                    "duration_seconds": duration
                })
                
            except Exception as e:
                logger.error("Error in risk monitoring loop", error=str(e))
                await log_to_loki("risk_monitoring_error", {
                    "error": str(e)
                }, "error")
            
            # Wait for next cycle
            await asyncio.sleep(self.task_interval)
    
    async def _get_wallets_to_monitor(self) -> List[str]:
        """Get list of wallets that need monitoring"""
        try:
            # Get wallets from recent metrics (active wallets)
            metrics_collection = get_collection(Collections.METRICS)
            
            # Get wallets that have been active in the last 24 hours
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            active_wallets = await metrics_collection.distinct(
                "wallet_address",
                {"timestamp": {"$gte": cutoff_time}}
            )
            
            # Get wallets from recent positions
            positions_collection = get_collection(Collections.POSITIONS)
            position_wallets = await positions_collection.distinct(
                "wallet_address",
                {"timestamp": {"$gte": cutoff_time}}
            )
            
            # Combine and deduplicate
            all_wallets = list(set(active_wallets + position_wallets))
            
            # Filter out wallets processed recently
            wallets_to_process = [
                wallet for wallet in all_wallets 
                if wallet not in self.processed_wallets
            ]
            
            return wallets_to_process
            
        except Exception as e:
            logger.error("Error getting wallets to monitor", error=str(e))
            return []
    
    async def _process_wallet_batch(self, wallet_batch: List[str]):
        """Process a batch of wallets for risk monitoring"""
        tasks = []
        
        for wallet_address in wallet_batch:
            task = asyncio.create_task(self._monitor_wallet_risk(wallet_address))
            tasks.append(task)
        
        # Process batch concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log any errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing wallet {wallet_batch[i]}", error=str(result))
    
    async def _monitor_wallet_risk(self, wallet_address: str):
        """Monitor risk for a single wallet and generate alerts if needed"""
        try:
            # Calculate current risk summary
            risk_summary = await risk_calculator.aggregate_risk_summary(wallet_address)
            
            # Store risk metrics
            await self._store_risk_metrics(wallet_address, risk_summary)
            
            # Generate alerts if thresholds are breached
            alerts = await risk_calculator.generate_alerts(wallet_address, risk_summary)
            
            if alerts:
                await self._store_new_alerts(wallet_address, alerts)
                logger.info(f"Generated {len(alerts)} alerts for wallet {wallet_address}")
            
            # Mark wallet as processed
            self.processed_wallets.add(wallet_address)
            
            # Log monitoring activity
            await log_to_loki("wallet_risk_monitored", {
                "wallet_address": wallet_address,
                "risk_score": risk_summary.overall_risk_score,
                "risk_level": risk_summary.risk_level,
                "alerts_generated": len(alerts)
            })
            
        except Exception as e:
            logger.error(f"Error monitoring wallet {wallet_address}", error=str(e))
    
    async def _store_risk_metrics(self, wallet_address: str, risk_summary):
        """Store risk metrics in database"""
        try:
            metrics_collection = get_collection(Collections.METRICS)
            
            metric_doc = {
                "wallet_address": wallet_address,
                "metric_type": "risk_summary",
                "data": risk_summary.dict(),
                "timestamp": datetime.utcnow(),
                "created_at": datetime.utcnow()
            }
            
            await metrics_collection.insert_one(metric_doc)
            
        except Exception as e:
            logger.error(f"Error storing risk metrics for {wallet_address}", error=str(e))
    
    async def _store_new_alerts(self, wallet_address: str, alerts):
        """Store new alerts in database"""
        try:
            if not alerts:
                return
            
            alerts_collection = get_collection(Collections.ALERTS)
            alert_docs = []
            
            for alert in alerts:
                # Check if similar alert already exists and is unresolved
                existing_alert = await alerts_collection.find_one({
                    "wallet_address": wallet_address,
                    "alert_type": alert.alert_type,
                    "is_resolved": False
                })
                
                if not existing_alert:
                    alert_doc = alert.dict(by_alias=True)
                    alert_doc.pop("id", None)  # Remove id field for insertion
                    alert_docs.append(alert_doc)
            
            if alert_docs:
                await alerts_collection.insert_many(alert_docs)
                
                # Log critical alerts
                critical_alerts = [a for a in alert_docs if a["severity"] == "critical"]
                if critical_alerts:
                    await log_to_loki("critical_alerts_generated", {
                        "wallet_address": wallet_address,
                        "critical_alert_count": len(critical_alerts),
                        "alert_types": [a["alert_type"] for a in critical_alerts]
                    }, "error")
            
        except Exception as e:
            logger.error(f"Error storing alerts for {wallet_address}", error=str(e))
    
    async def _metrics_aggregation_loop(self):
        """Aggregate metrics for system overview - runs every hour"""
        while self.is_running:
            try:
                logger.info("Starting metrics aggregation")
                
                # Aggregate metrics by risk level
                await self._aggregate_risk_distribution()
                
                # Aggregate metrics by protocol
                await self._aggregate_protocol_metrics()
                
                # Calculate system-wide statistics
                await self._calculate_system_stats()
                
                logger.info("Metrics aggregation completed")
                
            except Exception as e:
                logger.error("Error in metrics aggregation", error=str(e))
            
            # Run every hour
            await asyncio.sleep(3600)
    
    async def _aggregate_risk_distribution(self):
        """Aggregate risk level distribution"""
        try:
            metrics_collection = get_collection(Collections.METRICS)
            
            # Get recent risk summaries
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            
            pipeline = [
                {
                    "$match": {
                        "metric_type": "risk_summary",
                        "timestamp": {"$gte": cutoff_time}
                    }
                },
                {
                    "$group": {
                        "_id": "$data.risk_level",
                        "count": {"$sum": 1},
                        "avg_risk_score": {"$avg": "$data.overall_risk_score"}
                    }
                }
            ]
            
            results = await metrics_collection.aggregate(pipeline).to_list(length=None)
            
            # Store aggregated results
            aggregation_doc = {
                "aggregation_type": "risk_distribution",
                "data": results,
                "timestamp": datetime.utcnow(),
                "period": "1hour"
            }
            
            await metrics_collection.insert_one(aggregation_doc)
            
        except Exception as e:
            logger.error("Error aggregating risk distribution", error=str(e))
    
    async def _aggregate_protocol_metrics(self):
        """Aggregate metrics by protocol"""
        try:
            positions_collection = get_collection(Collections.POSITIONS)
            
            # Get recent positions
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            
            pipeline = [
                {
                    "$match": {
                        "timestamp": {"$gte": cutoff_time}
                    }
                },
                {
                    "$group": {
                        "_id": "$protocol",
                        "total_value_usd": {"$sum": "$total_supplied_usd"},
                        "total_debt_usd": {"$sum": "$total_borrowed_usd"},
                        "unique_wallets": {"$addToSet": "$wallet_address"},
                        "avg_health_ratio": {"$avg": "$health_ratio"}
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "total_value_usd": 1,
                        "total_debt_usd": 1,
                        "unique_wallet_count": {"$size": "$unique_wallets"},
                        "avg_health_ratio": 1
                    }
                }
            ]
            
            results = await positions_collection.aggregate(pipeline).to_list(length=None)
            
            # Store results
            metrics_collection = get_collection(Collections.METRICS)
            aggregation_doc = {
                "aggregation_type": "protocol_metrics",
                "data": results,
                "timestamp": datetime.utcnow(),
                "period": "1hour"
            }
            
            await metrics_collection.insert_one(aggregation_doc)
            
        except Exception as e:
            logger.error("Error aggregating protocol metrics", error=str(e))
    
    async def _calculate_system_stats(self):
        """Calculate system-wide statistics"""
        try:
            # Count active alerts
            alerts_collection = get_collection(Collections.ALERTS)
            active_alerts = await alerts_collection.count_documents({"is_resolved": False})
            critical_alerts = await alerts_collection.count_documents({
                "is_resolved": False,
                "severity": "critical"
            })
            
            # Count tracked wallets
            metrics_collection = get_collection(Collections.METRICS)
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            tracked_wallets = len(await metrics_collection.distinct(
                "wallet_address",
                {"timestamp": {"$gte": cutoff_time}}
            ))
            
            # Store system stats
            stats_doc = {
                "aggregation_type": "system_stats",
                "data": {
                    "active_alerts": active_alerts,
                    "critical_alerts": critical_alerts,
                    "tracked_wallets": tracked_wallets,
                    "uptime_hours": (datetime.utcnow() - datetime.utcnow().replace(hour=0, minute=0, second=0)).total_seconds() / 3600
                },
                "timestamp": datetime.utcnow()
            }
            
            await metrics_collection.insert_one(stats_doc)
            
            await log_to_loki("system_stats_calculated", stats_doc["data"])
            
        except Exception as e:
            logger.error("Error calculating system stats", error=str(e))
    
    async def _alert_cleanup_loop(self):
        """Clean up old resolved alerts - runs daily"""
        while self.is_running:
            try:
                logger.info("Starting alert cleanup")
                
                # Remove resolved alerts older than 30 days
                cutoff_time = datetime.utcnow() - timedelta(days=30)
                
                alerts_collection = get_collection(Collections.ALERTS)
                result = await alerts_collection.delete_many({
                    "is_resolved": True,
                    "resolved_at": {"$lt": cutoff_time}
                })
                
                logger.info(f"Cleaned up {result.deleted_count} old alerts")
                
                await log_to_loki("alert_cleanup_completed", {
                    "alerts_deleted": result.deleted_count
                })
                
            except Exception as e:
                logger.error("Error in alert cleanup", error=str(e))
            
            # Run once daily
            await asyncio.sleep(86400)
    
    async def _training_data_collection_loop(self):
        """Collect training data for ML model - runs every hour"""
        while self.is_running:
            try:
                logger.info("Starting training data collection")
                
                # Get recent position snapshots
                positions_collection = get_collection(Collections.POSITIONS)
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                
                recent_positions = await positions_collection.find({
                    "timestamp": {"$gte": cutoff_time}
                }).to_list(length=None)
                
                training_data_collection = get_collection(Collections.TRAINING_DATA)
                training_docs = []
                
                for position in recent_positions:
                    # Create training data point
                    training_point = TrainingDataPoint(
                        wallet_address=position["wallet_address"],
                        protocol=position.get("protocol", "unknown"),
                        health_ratio=position.get("health_ratio", 0.0),
                        total_supplied_usd=position.get("total_supplied_usd", 0.0),
                        total_borrowed_usd=position.get("total_borrowed_usd", 0.0),
                        volatility_score=0.0,  # Would be calculated
                        liquidity_score=0.0,   # Would be calculated
                        concentration_score=0.0, # Would be calculated
                        gas_price_gwei=25.0,   # Would be fetched from gas oracle
                        liquidated_within_24h=False  # Would be determined by monitoring
                    )
                    
                    training_doc = training_point.dict()
                    training_docs.append(training_doc)
                
                if training_docs:
                    await training_data_collection.insert_many(training_docs)
                    logger.info(f"Collected {len(training_docs)} training data points")
                
            except Exception as e:
                logger.error("Error collecting training data", error=str(e))
            
            # Run every hour
            await asyncio.sleep(3600)

# Global background task manager
background_task_manager = BackgroundTaskManager()