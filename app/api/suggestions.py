from fastapi import APIRouter, HTTPException, Query
from typing import List
from app.services.suggestions_service import suggestions_service
from app.models.suggestions import SuggestionsBundle, ActionableSuggestion
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/bundle", response_model=SuggestionsBundle)
async def get_suggestions_bundle(repo_path: str = Query(..., description="Path to the Git repository")):
    """
    Return actionable suggestions across project, subsystem, file, and pattern scopes.
    Backward compatible: independent of /api/insights endpoints.
    """
    try:
        bundle = suggestions_service.generate_suggestions(repo_path)
        return bundle
    except Exception as e:
        logger.error(f"Error generating suggestions bundle: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate suggestions: {str(e)}")


@router.get("/file", response_model=List[ActionableSuggestion])
async def get_file_suggestions(repo_path: str = Query(...), file_path: str = Query(...)):
    """
    Return actionable suggestions focused on a single file.
    """
    try:
        bundle = suggestions_service.generate_suggestions(repo_path)
        suggestions = [s for s in bundle.file_scope if any(loc.file_path.endswith(file_path) for loc in s.locations)]
        return suggestions
    except Exception as e:
        logger.error(f"Error generating file suggestions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate file suggestions: {str(e)}")