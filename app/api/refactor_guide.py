from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from app.services.refactor_service import RefactorService

router = APIRouter()
refactor_service = RefactorService()

class RefactorSuggestion(BaseModel):
    file_path: str
    suggestion_type: str  # "duplication", "complexity", "dependency"
    description: str
    impact: str
    effort_estimate: str
    code_snippet: Optional[str]
    suggested_changes: Optional[str]

class RefactorAnalysis(BaseModel):
    suggestions: List[RefactorSuggestion]
    overall_health: float
    priority_order: List[str]

@router.get("/analyze", response_model=RefactorAnalysis)
async def analyze_for_refactoring(path: str):
    """
    Analyze code for potential refactoring opportunities.
    """
    try:
        return refactor_service.analyze_code(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/duplicates", response_model=List[RefactorSuggestion])
async def find_code_duplicates(path: str, min_lines: int = 5):
    """
    Find code duplicates that could be refactored.
    """
    try:
        return refactor_service.find_duplicates(path, min_lines)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/complexity", response_model=List[RefactorSuggestion])
async def analyze_complexity(path: str, threshold: float = 10.0):
    """
    Find complex code that could benefit from refactoring.
    """
    try:
        return refactor_service.analyze_complexity(path, threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dependencies", response_model=List[RefactorSuggestion])
async def analyze_dependencies(path: str):
    """
    Analyze and suggest improvements for dependency relationships.
    """
    try:
        return refactor_service.analyze_dependencies(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))