from fastapi import APIRouter, HTTPException
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
from app.services.git_analysis import GitAnalysisService
from app.services.embedding import EmbeddingService
from app.models.evolution import ChangeMetrics

router = APIRouter()
git_service = GitAnalysisService()
embedding_service = EmbeddingService()

class TimelineEntry(BaseModel):
    timestamp: datetime
    commit_hash: str
    author: str
    message: str
    changes: List[ChangeMetrics]
    complexity_delta: float
    semantic_importance: float

@router.get("/timeline", response_model=List[TimelineEntry])
async def get_code_evolution_timeline(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    path: Optional[str] = None
):
    """
    Get the code evolution timeline with semantic analysis of changes.
    """
    try:
        # Get commit history
        commits = git_service.get_commit_history(start_date, end_date, path)
        
        # Analyze each commit
        timeline = []
        for commit in commits:
            # Get change metrics
            changes = git_service.analyze_commit_changes(commit.hash)
            
            # Calculate complexity changes
            complexity_delta = git_service.calculate_complexity_delta(commit.hash)
            
            # Generate embeddings for semantic analysis
            semantic_importance = embedding_service.analyze_commit_importance(
                commit.message,
                changes
            )
            
            timeline.append(TimelineEntry(
                timestamp=commit.timestamp,
                commit_hash=commit.hash,
                author=commit.author,
                message=commit.message,
                changes=changes,
                complexity_delta=complexity_delta,
                semantic_importance=semantic_importance
            ))
        
        return timeline
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/hotspots")
async def get_evolution_hotspots(timeframe: Optional[str] = "1m"):
    """
    Get code evolution hotspots based on change frequency and impact.
    """
    try:
        return git_service.analyze_hotspots(timeframe)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/complexity-trend")
async def get_complexity_trend(
    path: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """
    Get the complexity trend for a specific file or directory over time.
    """
    try:
        return git_service.analyze_complexity_trend(path, start_date, end_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/refactoring-patterns")
async def get_refactoring_patterns(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """
    Identify and analyze refactoring patterns in the codebase.
    """
    try:
        return git_service.analyze_refactoring_patterns(start_date, end_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))