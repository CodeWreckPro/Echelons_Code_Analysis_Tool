from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from app.services.storytelling_service import StorytellingService
from app.services.git_analysis import GitAnalysisService

router = APIRouter()
storytelling_service = StorytellingService()

class ChangeStory(BaseModel):
    commit_hash: str
    purpose: str
    system_impact: str
    affected_components: List[str]

@router.post("/analyze-commit", response_model=ChangeStory)
async def analyze_commit_story(commit_hash: str):
    """
    Analyze a commit and generate a human-readable "story" about its purpose and impact.
    """
    try:
        # Initialize Git service and get commit info
        git_service = GitAnalysisService()
        git_service.initialize_repo(".")  # Assuming current directory is the repo
        commit = git_service.repo.commit(commit_hash)
        
        # Get commit details
        commit_message = commit.message
        changes = git_service.analyze_commit_changes(commit_hash)
        
        # Generate the story
        story = storytelling_service.generate_commit_story(
            commit_message,
            changes
        )
        
        return ChangeStory(
            commit_hash=commit_hash,
            purpose=story['purpose'],
            system_impact=story['impact'],
            affected_components=story['components']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-pr")
async def analyze_pull_request_story(pr_url: str):
    """
    Analyze a pull request and generate a summary of its changes.
    """
    try:
        # This is a placeholder for PR analysis logic
        # In a real implementation, you would fetch PR data from GitHub/GitLab API
        story = storytelling_service.generate_pr_summary(pr_url)
        
        return {
            "pr_url": pr_url,
            "summary": story
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))