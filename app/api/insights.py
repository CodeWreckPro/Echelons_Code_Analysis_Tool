from fastapi import APIRouter, HTTPException
from typing import List, Dict
from datetime import datetime
from pydantic import BaseModel
from app.services.insights_service import InsightsService

router = APIRouter()
insights_service = InsightsService()

class SubsystemHealth(BaseModel):
    name: str
    score: float
    trend: List[float]
    issues: List[str]
    recommendations: List[str]

class RefactorAlert(BaseModel):
    severity: str
    message: str
    affected_files: List[str]
    impact: str
    suggested_action: str

class CodebaseMetrics(BaseModel):
    total_lines: int
    test_coverage: float
    complexity_score: float
    dependency_score: float
    maintainability_index: float

class DashboardData(BaseModel):
    subsystem_health: List[SubsystemHealth]
    alerts: List[RefactorAlert]
    metrics: CodebaseMetrics
    predictions: Dict[str, any]

@router.get("/dashboard", response_model=DashboardData)
async def get_dashboard_data():
    """
    Get comprehensive dashboard data including health scores and alerts.
    """
    try:
        return insights_service.get_dashboard_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=List[SubsystemHealth])
async def get_subsystem_health():
    """
    Get health scores for all subsystems.
    """
    try:
        return insights_service.analyze_subsystem_health()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts", response_model=List[RefactorAlert])
async def get_refactor_alerts():
    """
    Get current refactoring alerts.
    """
    try:
        return insights_service.generate_refactor_alerts()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", response_model=CodebaseMetrics)
async def get_codebase_metrics():
    """
    Get current codebase metrics.
    """
    try:
        return insights_service.calculate_codebase_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictions")
async def get_codebase_predictions():
    """
    Get predictions about codebase health and potential issues.
    """
    try:
        return insights_service.generate_predictions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/notify/{platform}")
async def send_notification(platform: str, message: str):
    """
    Send notifications to external platforms (Slack/Notion).
    """
    try:
        return insights_service.send_notification(platform, message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))