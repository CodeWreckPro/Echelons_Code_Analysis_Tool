import git
from datetime import datetime
from typing import List, Optional, Dict
from dataclasses import dataclass
import radon.complexity as radon
from radon.metrics import h_visit
from radon.raw import analyze
from lizard import analyze_file, FileAnalyzer
from collections import defaultdict
import safety
import json

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

    def _analyze_code_metrics(self, content: str) -> Dict:
        """Analyze various code metrics for the given content."""
        try:
            # Cyclomatic complexity
            cc_results = radon.cc_visit(content)
            avg_complexity = sum(c.complexity for c in cc_results) / len(cc_results) if cc_results else 0

            # Halstead metrics
            h_results = h_visit(content)
            halstead_metrics = h_results.total.__dict__ if h_results else {}

            # Raw metrics (LOC, LLOC, etc.)
            raw_metrics = analyze(content)
            loc = raw_metrics.loc
            lloc = raw_metrics.lloc
            sloc = raw_metrics.sloc
            comments = raw_metrics.comments
            multi = raw_metrics.multi
            blank = raw_metrics.blank
            single_comments = raw_metrics.single_comments

            # Maintainability Index
            maintainability_index = (171 - 5.2 * avg_complexity - 0.23 * (lloc / 1) - 16.2 * (loc / 1))
            
            return {
                "cyclomatic_complexity": avg_complexity,
                "halstead": halstead_metrics,
                "loc": loc,
                "lloc": lloc,
                "sloc": sloc,
                "comments": comments,
                "multi": multi,
                "blank": blank,
                "single_comments": single_comments,
                "maintainability_index": maintainability_index,
            }
        except Exception:
            return {}

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
                old_metrics = self._analyze_code_metrics(old_content)
                new_metrics = self._analyze_code_metrics(new_content)

                changes.append(ChangeMetrics(
                    file_path=diff.b_path,
                    lines_added=diff.stats['insertions'],
                    lines_deleted=diff.stats['deletions'],
                    complexity_before=old_metrics.get("cyclomatic_complexity", 0),
                    complexity_after=new_metrics.get("cyclomatic_complexity", 0),
                    change_type=self._determine_change_type(diff)
                ))

        return changes

    def _calculate_complexity(self, content: str) -> float:
        """Calculate average cyclomatic complexity using Radon."""
        try:
            results = radon.cc_visit(content)
            if not results:
                return 0.0
            # Each result has a 'complexity' attribute; compute average
            total = sum(getattr(r, "complexity", 0.0) for r in results)
            return float(total) / float(len(results))
        except Exception:
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

    def analyze_file_metrics(self, file_path: str) -> Dict:
        """Analyze metrics for a single file using Lizard."""
        try:
            analysis = analyze_file(file_path)
            return {
                "nloc": analysis.nloc,
                "cyclomatic_complexity": analysis.cyclomatic_complexity,
                "token_count": analysis.token_count,
                "function_count": len(analysis.function_list),
                "functions": [
                    {
                        "name": func.name,
                        "cyclomatic_complexity": func.cyclomatic_complexity,
                        "nloc": func.nloc,
                        "token_count": func.token_count,
                        "start_line": func.start_line,
                        "end_line": func.end_line,
                    }
                    for func in analysis.function_list
                ],
            }
        except Exception:
            return {}

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
                metrics = self._analyze_code_metrics(content)
                
                trend.append({
                    'timestamp': commit_date,
                    'complexity': metrics.get("cyclomatic_complexity", 0),
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

    def get_authors_per_file(self, file_path: str) -> List[str]:
        """Get the list of authors who have modified a file."""
        if not self.repo:
            raise ValueError("Repository not initialized")

        authors = set()
        for commit in self.repo.iter_commits(paths=file_path):
            authors.add(commit.author.email)
        return list(authors)

    def calculate_bus_factor(self) -> Dict[str, int]:
        """Calculate the bus factor for each file in the repository."""
        if not self.repo:
            raise ValueError("Repository not initialized")

        file_authors = defaultdict(set)
        for commit in self.repo.iter_commits():
            for file_path in commit.stats.files:
                file_authors[file_path].add(commit.author.email)

        bus_factors = {file_path: len(authors) for file_path, authors in file_authors.items()}
        return bus_factors

    def analyze_dependencies(self) -> List[Dict]:
        """Analyze dependencies for vulnerabilities using the safety library."""
        if not self.repo:
            raise ValueError("Repository not initialized")

        vulnerabilities = []
        try:
            # Find requirements.txt in the root directory
            requirements_path = f"{self.repo.working_dir}/requirements.txt"
            with open(requirements_path, "r") as f:
                packages = [line.strip() for line in f if line.strip() and not line.startswith("#")]

            # Use safety to check for vulnerabilities
            vulns = safety.check(packages=packages)
            for vuln in vulns:
                vulnerabilities.append({
                    "package": vuln.name,
                    "version": vuln.version,
                    "vulnerability_id": vuln.vuln_id,
                    "spec": vuln.spec,
                    "description": vuln.description,
                })
        except FileNotFoundError:
            # Handle case where requirements.txt does not exist
            pass
        except Exception as e:
            # Handle other exceptions during analysis
            print(f"Error analyzing dependencies: {e}")

        return vulnerabilities

    def scan_for_todos(self) -> List[Dict]:
        """Scan the repository for TODO and FIXME comments."""
        if not self.repo:
            raise ValueError("Repository not initialized")

        todos = []
        for commit in self.repo.iter_commits():
            for blob in commit.tree.traverse():
                if blob.type == 'blob':
                    try:
                        content = blob.data_stream.read().decode('utf-8', errors='ignore')
                        for i, line in enumerate(content.splitlines()):
                            if 'TODO' in line or 'FIXME' in line:
                                todos.append({
                                    'file_path': blob.path,
                                    'line_number': i + 1,
                                    'line_content': line.strip(),
                                    'commit_hash': commit.hexsha,
                                })
                    except Exception:
                        continue
        return todos

    # --- Newly added methods to align with InsightsService expectations ---
    def get_commit_statistics(self) -> List[Dict]:
        """Return simplified commit statistics for downstream insights.

        Structure per commit:
        {
          'hash': str,
          'timestamp': int,  # POSIX seconds
          'author': str,
          'message': str,
          'file_path': str
        }
        """
        if not self.repo:
            raise ValueError("Repository not initialized")

        stats = []
        for commit in self.repo.iter_commits():
            try:
                files = list(commit.stats.files.keys())
                for file_path in files:
                    stats.append({
                        'hash': commit.hexsha,
                        'timestamp': int(getattr(commit, 'committed_date', 0)),
                        'author': f"{commit.author.name} <{commit.author.email}>",
                        'message': commit.message,
                        'file_path': file_path,
                    })
            except Exception:
                continue
        return stats

    def get_repository_statistics(self) -> Dict:
        """Return basic repository statistics expected by InsightsService.

        Currently includes:
          - total_files: count of blobs in HEAD tree
        """
        if not self.repo:
            raise ValueError("Repository not initialized")

        total_files = 0
        try:
            head_tree = self.repo.head.commit.tree
            for item in head_tree.traverse():
                # Blob objects represent files
                if item.type == 'blob':
                    total_files += 1
        except Exception:
            # Fallback via last commit stats if traversal fails
            try:
                last_commit = next(self.repo.iter_commits())
                total_files = len(last_commit.stats.files.keys())
            except Exception:
                total_files = 0

        return {
            'total_files': total_files,
        }

    def analyze_complexity_trends(self) -> List[Dict]:
        """Compute per-file complexity and change frequency across recent commits.

        Returns a list of dicts:
        {
          'file_path': str,
          'complexity': float,  # last observed average CC
          'change_frequency': int  # number of commits touching this file (recent window)
        }
        """
        if not self.repo:
            raise ValueError("Repository not initialized")

        # Aggregate per-file stats across a reasonable recent window
        file_changes: Dict[str, int] = {}
        file_complexity: Dict[str, float] = {}

        max_commits = 200
        count = 0
        for commit in self.repo.iter_commits():
            count += 1
            try:
                touched_files = list(commit.stats.files.keys())
            except Exception:
                touched_files = []

            for fp in touched_files:
                file_changes[fp] = file_changes.get(fp, 0) + 1
                # Try to read file content from this commit's tree
                try:
                    blob = commit.tree / fp
                    content = blob.data_stream.read().decode('utf-8', errors='ignore')
                    file_complexity[fp] = self._calculate_complexity(content)
                except Exception:
                    # If content not available (deleted/renamed), keep last known
                    pass

            if count >= max_commits:
                break

        result = []
        for fp, freq in file_changes.items():
            result.append({
                'file_path': fp,
                'complexity': float(file_complexity.get(fp, 0.0)),
                'change_frequency': int(freq),
            })
        return result