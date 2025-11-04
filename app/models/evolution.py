from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class ChangeMetrics(BaseModel):
    file_path: str
    lines_added: int
    lines_deleted: int
    complexity_before: float
    complexity_after: float
    change_type: str  # "addition", "modification", "deletion", "refactor"

class CommitAnalysis(BaseModel):
    commit_hash: str
    timestamp: datetime
    author: str
    message: str
    changes: List[ChangeMetrics]
    semantic_importance: float
    complexity_impact: float

class HotspotMetrics(BaseModel):
    file_path: str
    change_frequency: int
    last_modified: datetime
    complexity_trend: List[float]
    risk_score: float
    contributors: List[str]

class RefactoringPattern(BaseModel):
    pattern_type: str  # "extract_method", "rename", "move", "inline"
    frequency: int
    affected_files: List[str]
    impact_score: float

class CodeEvolutionInsights(BaseModel):
    start_date: datetime
    end_date: datetime
    total_commits: int
    hotspots: List[HotspotMetrics]
    refactoring_patterns: List[RefactoringPattern]
    complexity_trend: List[float]
    health_score: float