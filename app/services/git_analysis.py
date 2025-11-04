import git
from datetime import datetime
from typing import List, Optional, Dict
from dataclasses import dataclass
import radon.complexity as radon
from app.models.evolution import ChangeMetrics
from app.services.embedding import EmbeddingService

@dataclass
class CommitInfo:
    hash: str
    timestamp: datetime
    author: str
    message: str

class GitAnalysisService:
    def __init__(self):
        self.repo = None
        self.embedding_service = EmbeddingService()

    def initialize_repo(self, path: str):
        """Initialize the Git repository."""
        try:
            self.repo = git.Repo(path)
        except git.InvalidGitRepositoryError:
            raise ValueError(f"Invalid Git repository path: {path}")

    def get_commit_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        path: Optional[str] = None
    ) -> List[CommitInfo]:
        """Get the commit history with optional date range and path filters."""
        if not self.repo:
            raise ValueError("Repository not initialized")

        commits = []
        for commit in self.repo.iter_commits(paths=path):
            commit_date = datetime.fromtimestamp(commit.committed_date)
            
            if start_date and commit_date < start_date:
                continue
            if end_date and commit_date > end_date:
                continue

            commits.append(CommitInfo(
                hash=commit.hexsha,
                timestamp=commit_date,
                author=f"{commit.author.name} <{commit.author.email}>",
                message=commit.message
            ))

        return commits

    def analyze_commit_changes(self, commit_hash: str) -> List[ChangeMetrics]:
        """Analyze changes in a specific commit."""
        if not self.repo:
            raise ValueError("Repository not initialized")

        commit = self.repo.commit(commit_hash)
        prev_commit = commit.parents[0] if commit.parents else None

        changes = []
        for diff in commit.diff(prev_commit):
            if diff.a_blob and diff.b_blob:
                # Analyze file changes
                old_content = diff.a_blob.data_stream.read().decode('utf-8', errors='ignore')
                new_content = diff.b_blob.data_stream.read().decode('utf-8', errors='ignore')

                # Calculate complexity metrics
                old_complexity = self._calculate_complexity(old_content)
                new_complexity = self._calculate_complexity(new_content)

                changes.append(ChangeMetrics(
                    file_path=diff.b_path,
                    lines_added=diff.stats['insertions'],
                    lines_deleted=diff.stats['deletions'],
                    complexity_before=old_complexity,
                    complexity_after=new_complexity,
                    change_type=self._determine_change_type(diff)
                ))

        return changes

    def _calculate_complexity(self, content: str) -> float:
        """Calculate code complexity using Radon."""
        try:
            return radon.cc_visit(content)
        except:
            return 0.0

    def _determine_change_type(self, diff) -> str:
        """Determine the type of change (addition, modification, deletion, refactor)."""
        if diff.new_file:
            return "addition"
        elif diff.deleted_file:
            return "deletion"
        elif self._is_refactor(diff):
            return "refactor"
        else:
            return "modification"

    def _is_refactor(self, diff) -> bool:
        """Determine if a change is likely a refactoring."""
        # Simple heuristic: if the number of lines changed is similar
        # and the content is similar, it's likely a refactor
        stats = diff.stats
        if not (stats['insertions'] and stats['deletions']):
            return False

        ratio = stats['insertions'] / stats['deletions']
        return 0.7 <= ratio <= 1.3

    def analyze_hotspots(self, timeframe: str) -> Dict:
        """Analyze code hotspots based on change frequency and impact."""
        if not self.repo:
            raise ValueError("Repository not initialized")

        hotspots = {}
        for commit in self.repo.iter_commits():
            for file in commit.stats.files:
                if file not in hotspots:
                    hotspots[file] = {
                        'change_count': 0,
                        'impact_score': 0,
                        'last_modified': commit.committed_datetime
                    }
                
                hotspots[file]['change_count'] += 1
                hotspots[file]['impact_score'] += self._calculate_change_impact(commit, file)

        return hotspots

    def _calculate_change_impact(self, commit, file_path: str) -> float:
        """Calculate the impact score of a change."""
        # Factors considered:
        # 1. Number of lines changed
        # 2. Complexity delta
        # 3. Semantic importance (based on commit message)
        
        stats = commit.stats.files[file_path]
        lines_changed = stats['insertions'] + stats['deletions']
        
        # Get complexity change
        complexity_delta = abs(self.calculate_complexity_delta(commit.hexsha))
        
        # Get semantic importance
        semantic_score = self.embedding_service.analyze_commit_importance(
            commit.message,
            []  # Pass empty changes list for now
        )
        
        # Weighted impact score
        return (
            0.4 * lines_changed +
            0.3 * complexity_delta +
            0.3 * semantic_score
        )

    def calculate_complexity_delta(self, commit_hash: str) -> float:
        """Calculate the complexity change introduced by a commit."""
        if not self.repo:
            raise ValueError("Repository not initialized")

        commit = self.repo.commit(commit_hash)
        prev_commit = commit.parents[0] if commit.parents else None
        
        if not prev_commit:
            return 0.0

        total_delta = 0.0
        for diff in commit.diff(prev_commit):
            if diff.a_blob and diff.b_blob:
                old_content = diff.a_blob.data_stream.read().decode('utf-8', errors='ignore')
                new_content = diff.b_blob.data_stream.read().decode('utf-8', errors='ignore')
                
                old_complexity = self._calculate_complexity(old_content)
                new_complexity = self._calculate_complexity(new_content)
                
                total_delta += new_complexity - old_complexity

        return total_delta

    def analyze_complexity_trend(
        self,
        path: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """Analyze complexity trend for a specific path over time."""
        if not self.repo:
            raise ValueError("Repository not initialized")

        trend = []
        for commit in self.repo.iter_commits(paths=path):
            commit_date = datetime.fromtimestamp(commit.committed_date)
            
            if start_date and commit_date < start_date:
                continue
            if end_date and commit_date > end_date:
                continue

            try:
                blob = commit.tree / path
                content = blob.data_stream.read().decode('utf-8', errors='ignore')
                complexity = self._calculate_complexity(content)
                
                trend.append({
                    'timestamp': commit_date,
                    'complexity': complexity,
                    'commit_hash': commit.hexsha
                })
            except:
                continue

        return trend

    def analyze_refactoring_patterns(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """Analyze refactoring patterns in the codebase."""
        if not self.repo:
            raise ValueError("Repository not initialized")

        patterns = {
            'extract_method': [],
            'rename': [],
            'move': [],
            'inline': []
        }

        for commit in self.repo.iter_commits():
            commit_date = datetime.fromtimestamp(commit.committed_date)
            
            if start_date and commit_date < start_date:
                continue
            if end_date and commit_date > end_date:
                continue

            # Analyze commit message for refactoring keywords
            msg = commit.message.lower()
            if 'refactor' in msg or 'refactoring' in msg:
                pattern_type = self._identify_refactoring_pattern(commit)
                if pattern_type:
                    patterns[pattern_type].append({
                        'commit_hash': commit.hexsha,
                        'timestamp': commit_date,
                        'message': commit.message,
                        'files': list(commit.stats.files.keys())
                    })

        return patterns

    def _identify_refactoring_pattern(self, commit) -> Optional[str]:
        """Identify the type of refactoring in a commit."""
        msg = commit.message.lower()
        
        if 'extract' in msg or 'split' in msg:
            return 'extract_method'
        elif 'rename' in msg:
            return 'rename'
        elif 'move' in msg:
            return 'move'
        elif 'inline' in msg:
            return 'inline'
        
        return None