from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from datetime import datetime

class RefactorSuggestion(BaseModel):
    file_path: str
    suggestion_type: str  # "duplication", "complexity", "dependency"
    description: str
    impact: str
    effort_estimate: str
    code_snippet: Optional[str] = None
    suggested_changes: Optional[str] = None

class RefactorAnalysis(BaseModel):
    suggestions: List[RefactorSuggestion]
    overall_health: float
    priority_order: List[str]

class ModuleNode(BaseModel):
    id: str
    name: str
    type: str  # "module", "class", "function"
    size: int  # Lines of code or complexity
    color: str  # Hex color based on health score
    position: Dict[str, float]  # 3D coordinates

class ModuleConnection(BaseModel):
    source: str
    target: str
    strength: float  # Connection strength based on dependency frequency
    type: str  # "import", "inheritance", "function_call"

class CodeMap(BaseModel):
    nodes: List[ModuleNode]
    connections: List[ModuleConnection]
    metadata: Dict[str, Any]

class SubsystemHealth(BaseModel):
    """Model for subsystem health data."""
    name: str
    status: str  # healthy, stable, warning, critical
    complexity_trend: str
    last_updated: datetime

class RefactorAlert(BaseModel):
    """Model for refactoring alerts."""
    file_path: str
    reason: str
    severity: str  # high, medium, low
    suggested_action: str
    estimated_effort: str

class CodebaseMetrics(BaseModel):
    """Model for overall codebase metrics."""
    total_files: int
    total_commits: int
    average_complexity: float
    technical_debt_ratio: float
    code_quality_score: float
    last_analyzed: datetime

class ResourcePrediction(BaseModel):
    """Model for resource requirement predictions."""
    resource_type: str
    predicted_need: int
    confidence_score: float
    timeframe: str

class RiskArea(BaseModel):
    """Model for identified risk areas."""
    area_name: str
    risk_level: str
    potential_impact: str
    mitigation_strategy: str

class MaintenancePrediction(BaseModel):
    """Model for maintenance need predictions."""
    maintenance_type: str
    predicted_frequency: int
    timeframe: str
    confidence_score: float

class ComplexityPrediction(BaseModel):
    """Model for complexity trend predictions."""
    component: str
    predicted_complexity: float
    trend: str  # increasing, decreasing, stable
    confidence_score: float

class DashboardData(BaseModel):
    """Model for complete dashboard data."""
    subsystem_health: List[SubsystemHealth]
    refactor_alerts: List[RefactorAlert]
    codebase_metrics: CodebaseMetrics
    resource_predictions: List[ResourcePrediction]
    risk_areas: List[RiskArea]
    maintenance_predictions: List[MaintenancePrediction]
    complexity_predictions: List[ComplexityPrediction]
    generated_at: datetime

class NotificationConfig(BaseModel):
    """Model for notification configuration."""
    platform: str  # slack, notion
    message: str
    priority: Optional[str] = "normal"
    recipients: Optional[List[str]] = None