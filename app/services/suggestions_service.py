import os
from typing import List, Dict
from pathlib import Path
from datetime import datetime
from app.models.suggestions import ActionableSuggestion, SuggestionsBundle, SuggestionLocation
from ai.pipelines.suggestion_pipeline import suggestion_pipeline
from app.services.git_analysis import GitAnalysisService


class SuggestionsService:
    """
    Orchestrates metrics collection and hybrid pipeline to produce
    high-quality, actionable suggestions with confidence and ROI.

    - Lazy model loading via SuggestionPipeline
    - Fallback if model missing
    - Extensible metrics via GitAnalysisService and light static analysis
    """

    def __init__(self):
        self.git_service = GitAnalysisService()

    def _collect_repo_files(self, repo_path: str) -> Dict[str, str]:
        files: Dict[str, str] = {}
        for file_path in Path(repo_path).rglob("*.py"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    files[str(file_path)] = f.read()
            except Exception:
                continue
        return files

    def _file_volatility(self) -> Dict[str, int]:
        freq: Dict[str, int] = {}
        for commit in self.git_service.get_commit_statistics():
            fp = commit.get("file_path")
            if fp:
                freq[fp] = freq.get(fp, 0) + 1
        return freq

    def generate_suggestions(self, repo_path: str) -> SuggestionsBundle:
        self.git_service.initialize_repo(repo_path)

        files = self._collect_repo_files(repo_path)
        volatility = self._file_volatility()

        bundle = suggestion_pipeline.generate(repo_path, files, volatility)
        return bundle


# Singleton
suggestions_service = SuggestionsService()