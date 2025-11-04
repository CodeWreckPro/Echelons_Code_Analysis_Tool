from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel
from app.services.insights_service import insights_service
from app.models.insights import (
    SubsystemHealth, RefactorAlert, CodebaseMetrics, ResourcePrediction,
    RiskArea, MaintenancePrediction, ComplexityPrediction, DashboardData
)

router = APIRouter()

# Response models matching our actual data structures
class DashboardResponse(BaseModel):
    subsystem_health: List[SubsystemHealth]
    refactor_alerts: List[RefactorAlert]
    codebase_metrics: CodebaseMetrics
    resource_predictions: List[ResourcePrediction]
    risk_areas: List[RiskArea]
    maintenance_predictions: List[MaintenancePrediction]
    complexity_predictions: List[ComplexityPrediction]
    generated_at: datetime

class HealthResponse(BaseModel):
    name: str
    status: str
    complexity_trend: str
    last_updated: datetime

class AlertResponse(BaseModel):
    file_path: str
    reason: str
    severity: str
    suggested_action: str
    estimated_effort: str

@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard_data(repo_path: str = Query(..., description="Path to the Git repository")):
    """
    Get comprehensive dashboard data including health scores, alerts, and AI-driven insights.
    Uses ML models for hotspot prediction and statistical analysis for metrics.
    """
    try:
        dashboard_data = insights_service.generate_dashboard_insights(repo_path)
        return dashboard_data
    except Exception as e:
        logger.error(f"Error generating dashboard insights: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard insights: {str(e)}")

@router.get("/health", response_model=List[HealthResponse])
async def get_subsystem_health(repo_path: str = Query(..., description="Path to the Git repository")):
    """
    Get health analysis for all subsystems using ML hotspot prediction and statistical analysis.
    """
    try:
        health_data = insights_service.analyze_subsystem_health(repo_path)
        return health_data
    except Exception as e:
        logger.error(f"Error analyzing subsystem health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze subsystem health: {str(e)}")

@router.get("/alerts", response_model=List[AlertResponse])
async def get_refactor_alerts(repo_path: str = Query(..., description="Path to the Git repository")):
    """
    Get AI-driven refactoring alerts based on ML hotspot prediction and code complexity analysis.
    """
    try:
        alerts = insights_service.identify_refactor_opportunities(repo_path)
        return alerts
    except Exception as e:
        logger.error(f"Error identifying refactor opportunities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to identify refactor opportunities: {str(e)}")

@router.get("/metrics", response_model=CodebaseMetrics)
async def get_codebase_metrics(repo_path: str = Query(..., description="Path to the Git repository")):
    """
    Get comprehensive codebase metrics including technical debt analysis and quality scores.
    """
    try:
        metrics = insights_service.calculate_codebase_metrics(repo_path)
        return metrics
    except Exception as e:
        logger.error(f"Error calculating codebase metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate codebase metrics: {str(e)}")

@router.get("/predictions/resource", response_model=List[ResourcePrediction])
async def get_resource_predictions(repo_path: str = Query(..., description="Path to the Git repository")):
    """
    Get AI-driven resource need predictions based on historical development patterns.
    """
    try:
        predictions = insights_service.predict_resource_needs(repo_path)
        return predictions
    except Exception as e:
        logger.error(f"Error predicting resource needs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to predict resource needs: {str(e)}")

@router.get("/predictions/maintenance", response_model=List[MaintenancePrediction])
async def get_maintenance_predictions(repo_path: str = Query(..., description="Path to the Git repository")):
    """
    Get AI-driven maintenance need predictions using ML models and trend analysis.
    """
    try:
        predictions = insights_service.predict_maintenance_needs(repo_path)
        return predictions
    except Exception as e:
        logger.error(f"Error predicting maintenance needs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to predict maintenance needs: {str(e)}")

@router.get("/predictions/complexity", response_model=List[ComplexityPrediction])
async def get_complexity_predictions(repo_path: str = Query(..., description="Path to the Git repository")):
    """
    Get AI-driven complexity trend predictions based on statistical analysis.
    """
    try:
        predictions = insights_service.predict_complexity_trends(repo_path)
        return predictions
    except Exception as e:
        logger.error(f"Error predicting complexity trends: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to predict complexity trends: {str(e)}")

@router.get("/risks", response_model=List[RiskArea])
async def get_risk_areas(repo_path: str = Query(..., description="Path to the Git repository")):
    """
    Get identified risk areas using ML risk assessment and heuristics.
    """
    try:
        risks = insights_service.identify_risk_areas(repo_path)
        return risks
    except Exception as e:
        logger.error(f"Error identifying risk areas: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to identify risk areas: {str(e)}")

@router.post("/notify/{platform}")
async def send_notification(platform: str, message: str, repo_path: str = Query(None, description="Optional repository path for context")):
    """
    Send AI-generated insights notifications to external platforms (Slack/Notion).
    """
    try:
        # For now, return a placeholder response
        # In a real implementation, this would integrate with external APIs
        return {
            "status": "success",
            "platform": platform,
            "message": message,
            "sent_at": datetime.now(),
            "note": "External integration not implemented yet"
        }
    except Exception as e:
        logger.error(f"Error sending notification: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send notification: {str(e)}")

# Add logging
import logging
logger = logging.getLogger(__name__)