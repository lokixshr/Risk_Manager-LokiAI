from fastapi import APIRouter, HTTPException, Depends, Header, Request
from fastapi.responses import JSONResponse
from typing import Optional, List
import structlog
from datetime import datetime, timedelta
import time

from .config import settings, Collections, RiskSeverity
from .models import (
    RiskSummary, SystemStatus, AlertsResponse, WalletAnalysisRequest,
    RiskAlert
)
from .database import get_collection, db_manager
from .external_apis import api_manager
from .risk_engine import risk_calculator
from .security import verify_wallet_address, check_rate_limit, log_to_loki

logger = structlog.get_logger()

# Track application startup time for uptime calculation
app_start_time = time.time()

# Global metrics for system status
request_count = 0
error_count = 0
response_times = []

router = APIRouter()

async def validate_wallet_header(x_wallet_address: Optional[str] = Header(None)) -> str:
    """Validate wallet address header"""
    if not x_wallet_address:
        raise HTTPException(
            status_code=401,
            detail="Missing x-wallet-address header"
        )
    
    if not verify_wallet_address(x_wallet_address):
        raise HTTPException(
            status_code=400,
            detail="Invalid wallet address format"
        )
    
    return x_wallet_address.lower()

async def apply_rate_limit(request: Request, wallet_address: str):
    """Apply rate limiting per wallet and IP"""
    client_ip = request.client.host if request.client else "unknown"
    
    # Check rate limit
    if not await check_rate_limit(wallet_address, client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Maximum 60 requests per minute per wallet."
        )

async def track_request_metrics(start_time: float, success: bool = True):
    """Track request metrics for system status"""
    global request_count, error_count, response_times
    
    request_count += 1
    if not success:
        error_count += 1
    
    response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    response_times.append(response_time)
    
    # Keep only last 1000 response times for rolling average
    if len(response_times) > 1000:
        response_times = response_times[-1000:]

@router.get("/api/risk/status", response_model=SystemStatus)
async def get_system_status():
    """Get overall system health and status"""
    start_time = time.time()
    
    try:
        # Calculate uptime
        uptime_seconds = int(time.time() - app_start_time)
        
        # Get tracked wallets count
        metrics_collection = get_collection(Collections.METRICS)
        tracked_wallets = await metrics_collection.distinct("wallet_address")
        
        # Get active alerts count
        alerts_collection = get_collection(Collections.ALERTS)
        active_alerts = await alerts_collection.count_documents({"is_resolved": False})
        
        # Database health check
        db_health = await db_manager.health_check()
        
        # External APIs health check
        api_health = await api_manager.health_check()
        
        # Calculate performance metrics
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        requests_per_minute = request_count  # Simplified - would need time window tracking
        error_rate = (error_count / max(request_count, 1)) * 100
        
        await track_request_metrics(start_time, True)
        
        status = SystemStatus(
            status="operational",
            version="1.0.0",
            uptime_seconds=uptime_seconds,
            tracked_wallets=len(tracked_wallets),
            active_alerts=active_alerts,
            database_status=db_health,
            external_apis_status=api_health,
            avg_response_time_ms=round(avg_response_time, 2),
            requests_per_minute=requests_per_minute,
            error_rate_percent=round(error_rate, 2)
        )
        
        # Log to LokiAI
        await log_to_loki("system_status", {
            "status": status.status,
            "tracked_wallets": status.tracked_wallets,
            "active_alerts": status.active_alerts,
            "uptime_seconds": status.uptime_seconds
        })
        
        return status
        
    except Exception as e:
        await track_request_metrics(start_time, False)
        logger.error("Error getting system status", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/api/risk/summary", response_model=RiskSummary)
async def get_risk_summary(
    request: Request,
    wallet: str,
    wallet_address: str = Depends(validate_wallet_header)
):
    """Get comprehensive risk summary for a wallet"""
    start_time = time.time()
    
    try:
        # Validate wallet parameter matches header
        if wallet.lower() != wallet_address:
            raise HTTPException(
                status_code=400,
                detail="Wallet parameter must match x-wallet-address header"
            )
        
        # Apply rate limiting
        await apply_rate_limit(request, wallet_address)
        
        # Generate risk summary
        risk_summary = await risk_calculator.aggregate_risk_summary(wallet_address)
        
        # Store metrics in database
        metrics_collection = get_collection(Collections.METRICS)
        await metrics_collection.insert_one({
            "wallet_address": wallet_address,
            "metric_type": "risk_summary",
            "data": risk_summary.dict(),
            "timestamp": datetime.utcnow(),
            "created_at": datetime.utcnow()
        })
        
        await track_request_metrics(start_time, True)
        
        # Log to LokiAI
        await log_to_loki("risk_summary_requested", {
            "wallet_address": wallet_address,
            "risk_score": risk_summary.overall_risk_score,
            "risk_level": risk_summary.risk_level,
            "total_value_usd": risk_summary.total_portfolio_value_usd
        })
        
        return risk_summary
        
    except HTTPException:
        await track_request_metrics(start_time, False)
        raise
    except Exception as e:
        await track_request_metrics(start_time, False)
        logger.error(f"Error getting risk summary for {wallet_address}", error=str(e))
        raise HTTPException(status_code=500, detail="Error calculating risk summary")

@router.post("/api/risk/analyze")
async def analyze_wallet(
    request: Request,
    analysis_request: WalletAnalysisRequest,
    wallet_address: str = Depends(validate_wallet_header)
):
    """Force manual risk analysis for a wallet"""
    start_time = time.time()
    
    try:
        # Validate wallet parameter matches header
        if analysis_request.wallet_address != wallet_address:
            raise HTTPException(
                status_code=400,
                detail="Request wallet address must match x-wallet-address header"
            )
        
        # Apply rate limiting
        await apply_rate_limit(request, wallet_address)
        
        # Perform comprehensive analysis
        logger.info(f"Starting manual analysis for wallet {wallet_address}")
        
        # Get risk summary
        risk_summary = await risk_calculator.aggregate_risk_summary(wallet_address)
        
        # Generate alerts
        alerts = await risk_calculator.generate_alerts(wallet_address, risk_summary)
        
        # Store alerts in database
        if alerts:
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
        
        # Store position snapshot
        positions_collection = get_collection(Collections.POSITIONS)
        position_snapshot = {
            "wallet_address": wallet_address,
            "protocol": "multi_protocol",
            "chain_id": 1,
            "total_supplied_usd": risk_summary.total_portfolio_value_usd,
            "total_borrowed_usd": risk_summary.total_debt_usd,
            "total_collateral_usd": risk_summary.total_portfolio_value_usd,
            "health_ratio": risk_summary.health_ratio,
            "timestamp": datetime.utcnow(),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        await positions_collection.insert_one(position_snapshot)
        
        await track_request_metrics(start_time, True)
        
        # Log to LokiAI
        await log_to_loki("wallet_analysis_completed", {
            "wallet_address": wallet_address,
            "risk_score": risk_summary.overall_risk_score,
            "alerts_generated": len(alerts),
            "force_refresh": analysis_request.force_refresh
        })
        
        return {
            "status": "completed",
            "wallet_address": wallet_address,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "risk_summary": risk_summary,
            "alerts_generated": len(alerts),
            "new_alerts": [alert.dict() for alert in alerts]
        }
        
    except HTTPException:
        await track_request_metrics(start_time, False)
        raise
    except Exception as e:
        await track_request_metrics(start_time, False)
        logger.error(f"Error analyzing wallet {wallet_address}", error=str(e))
        raise HTTPException(status_code=500, detail="Error performing wallet analysis")

@router.get("/api/risk/alerts", response_model=AlertsResponse)
async def get_alerts(
    request: Request,
    wallet_address: str = Depends(validate_wallet_header),
    limit: int = 50,
    skip: int = 0,
    severity: Optional[str] = None,
    alert_type: Optional[str] = None,
    resolved: Optional[bool] = None
):
    """Get risk alerts for a wallet with filtering options"""
    start_time = time.time()
    
    try:
        # Apply rate limiting
        await apply_rate_limit(request, wallet_address)
        
        # Build query filter
        query = {"wallet_address": wallet_address}
        
        if severity:
            query["severity"] = severity
        if alert_type:
            query["alert_type"] = alert_type
        if resolved is not None:
            query["is_resolved"] = resolved
        
        # Get alerts from database
        alerts_collection = get_collection(Collections.ALERTS)
        
        # Get total count
        total_count = await alerts_collection.count_documents(query)
        
        # Get alerts with pagination
        cursor = alerts_collection.find(query).sort("timestamp", -1).skip(skip).limit(limit)
        alert_docs = await cursor.to_list(length=limit)
        
        # Convert to RiskAlert models
        alerts = []
        for doc in alert_docs:
            doc["id"] = str(doc.get("_id"))
            alerts.append(RiskAlert(**doc))
        
        await track_request_metrics(start_time, True)
        
        # Log to LokiAI
        await log_to_loki("alerts_requested", {
            "wallet_address": wallet_address,
            "total_alerts": total_count,
            "filters": {
                "severity": severity,
                "alert_type": alert_type,
                "resolved": resolved
            }
        })
        
        return AlertsResponse(
            total_count=total_count,
            alerts=alerts,
            pagination={
                "limit": limit,
                "skip": skip,
                "has_more": skip + len(alerts) < total_count
            }
        )
        
    except HTTPException:
        await track_request_metrics(start_time, False)
        raise
    except Exception as e:
        await track_request_metrics(start_time, False)
        logger.error(f"Error getting alerts for {wallet_address}", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving alerts")

@router.post("/api/risk/alerts/{alert_id}/resolve")
async def resolve_alert(
    request: Request,
    alert_id: str,
    wallet_address: str = Depends(validate_wallet_header)
):
    """Mark an alert as resolved"""
    start_time = time.time()
    
    try:
        # Apply rate limiting
        await apply_rate_limit(request, wallet_address)
        
        # Update alert in database
        alerts_collection = get_collection(Collections.ALERTS)
        
        from bson import ObjectId
        try:
            object_id = ObjectId(alert_id)
        except:
            raise HTTPException(status_code=400, detail="Invalid alert ID format")
        
        result = await alerts_collection.update_one(
            {
                "_id": object_id,
                "wallet_address": wallet_address,
                "is_resolved": False
            },
            {
                "$set": {
                    "is_resolved": True,
                    "resolved_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        if result.matched_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Alert not found or already resolved"
            )
        
        await track_request_metrics(start_time, True)
        
        # Log to LokiAI
        await log_to_loki("alert_resolved", {
            "wallet_address": wallet_address,
            "alert_id": alert_id
        })
        
        return {
            "status": "resolved",
            "alert_id": alert_id,
            "resolved_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        await track_request_metrics(start_time, False)
        raise
    except Exception as e:
        await track_request_metrics(start_time, False)
        logger.error(f"Error resolving alert {alert_id}", error=str(e))
        raise HTTPException(status_code=500, detail="Error resolving alert")

@router.get("/api/risk/health")
async def health_check():
    """Simple health check endpoint"""
    try:
        # Check database connectivity
        await db_manager.health_check()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )