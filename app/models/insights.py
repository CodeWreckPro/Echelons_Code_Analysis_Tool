from typing import List, Dict, Optional
from pydantic import BaseModel
from datetime import datetime

class SubsystemHealth(BaseModel):
    """Model for subsystem health data."""
    name: str
    score: float  # 0-100 scale
    trend: List[float]  # Historical trend data
    issues: List[str]
    recommendations: List[str]

class RefactorAlert(BaseModel):
    """Model for refactoring alerts."""
    severity: str  # high, medium, low
    message: str
    affected_files: List[str]
    impact: str
    suggested_action: str

class CodebaseMetrics(BaseModel):
    """Model for overall codebase metrics."""
    total_lines: int
    test_coverage: float  # 0-100 scale
    complexity_score: float
    dependency_score: float  # 0-1 scale
    maintainability_index: float  # 0-100 scale

class ResourcePrediction(BaseModel):
    """Model for resource requirement predictions."""
    developer_hours: Dict[str, int]  # weekly, monthly
    testing_hours: Dict[str, int]  # weekly, monthly
    review_hours: Dict[str, int]  # weekly, monthly
    priority_areas: List[Dict]

class RiskArea(BaseModel):
    """Model for identified risk areas."""
    file: str
    risk_score: float
    factors: Dict[str, float]
    mitigation: List[str]

class MaintenancePrediction(BaseModel):
    """Model for maintenance need predictions."""
    needs_maintenance: bool
    priority: str
    estimated_effort: Dict
    recommended_timeline: str

class ComplexityPrediction(BaseModel):
    """Model for complexity trend predictions."""
    trend: str  # increasing, decreasing, stable
    rate: float
    predictions: List[float]

class DashboardData(BaseModel):
    """Model for complete dashboard data."""
    subsystem_health: List[SubsystemHealth]
    alerts: List[RefactorAlert]
    metrics: CodebaseMetrics
    predictions: Dict  # Contains all prediction data

class NotificationConfig(BaseModel):
    """Model for notification configuration."""
    platform: str  # slack, notion
    message: str
    priority: Optional[str] = "normal"
    recipients: Optional[List[str]] = None