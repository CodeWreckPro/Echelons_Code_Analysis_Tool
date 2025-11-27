from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class SuggestionLocation(BaseModel):
    file_path: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None


class ActionableSuggestion(BaseModel):
    id: str
    title: str
    description: str
    priority: str = Field(..., description="high | medium | low")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    estimated_roi: float = Field(..., description="Estimated ROI impact score (0–100)")
    estimated_effort: str = Field(..., description="T-shirt size or hours")
    scope: str = Field(..., description="project | subsystem | file | pattern")
    categories: List[str] = Field(default_factory=list)
    locations: List[SuggestionLocation] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    suggested_diff: Optional[str] = None
    rationale: List[str] = Field(default_factory=list)
    horizon: str = Field(..., description="1w | 1m | 3m")
    metadata: Dict = Field(default_factory=dict)


class SuggestionsBundle(BaseModel):
    generated_at: str
    project_scope: List[ActionableSuggestion]
    subsystem_scope: List[ActionableSuggestion]
    file_scope: List[ActionableSuggestion]
    pattern_scope: List[ActionableSuggestion]
    version: str = "v1"