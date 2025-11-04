import ast
import networkx as nx
from typing import List, Dict
from pathlib import Path
import radon.complexity as radon
from radon.visitors import ComplexityVisitor
from app.models.insights import RefactorSuggestion, RefactorAnalysis

class RefactorService:
    def __init__(self):
        self.complexity_threshold = 10.0
        self.duplication_threshold = 5  # minimum lines for duplication
        
    def analyze_code(self, path: str) -> RefactorAnalysis:
        """Perform comprehensive code analysis for refactoring opportunities."""
        # Get all suggestions
        duplicates = self.find_duplicates(path)
        complexity_issues = self.analyze_complexity(path)
        dependency_issues = self.analyze_dependencies(path)
        
        # Combine all suggestions
        all_suggestions = duplicates + complexity_issues + dependency_issues
        
        # Calculate overall health score (0-100)
        health_score = self._calculate_health_score(
            all_suggestions,
            path
        )
        
        # Determine priority order
        priority_order = self._prioritize_suggestions(all_suggestions)
        
        return RefactorAnalysis(
            suggestions=all_suggestions,
            overall_health=health_score,
            priority_order=priority_order
        )
    
    def find_duplicates(self, path: str, min_lines: int = 5) -> List[RefactorSuggestion]:
        """Find duplicate code blocks."""
        suggestions = []
        file_contents = {}
        
        # Read all Python files
        for file_path in Path(path).rglob("*.py"):
            with open(file_path, "r") as f:
                content = f.read()
                file_contents[str(file_path)] = content.splitlines()
        
        # Find duplicates using rolling hash
        duplicates = self._find_duplicate_blocks(file_contents, min_lines)
        
        for locations in duplicates:
            suggestions.append(RefactorSuggestion(
                file_path=locations[0][0],
                suggestion_type="duplication",
                description=f"Found duplicate code block of {len(locations)} lines",
                impact="Code duplication increases maintenance burden and risk of inconsistent updates",
                effort_estimate="Medium",
                code_snippet=self._get_code_snippet(locations[0]),
                suggested_changes="Consider extracting this code into a shared function or utility"
            ))
        
        return suggestions
    
    def analyze_complexity(self, path: str, threshold: float = 10.0) -> List[RefactorSuggestion]:
        """Analyze code complexity and suggest refactoring."""
        suggestions = []
        
        for file_path in Path(path).rglob("*.py"):
            with open(file_path, "r") as f:
                content = f.read()
            
            # Parse the code
            try:
                tree = ast.parse(content)
            except:
                continue
            
            # Analyze complexity
            visitor = ComplexityVisitor.from_ast(tree)
            for complexity in visitor.functions:
                if complexity.complexity > threshold:
                    suggestions.append(RefactorSuggestion(
                        file_path=str(file_path),
                        suggestion_type="complexity",
                        description=f"Function '{complexity.name}' has high cyclomatic complexity ({complexity.complexity})",
                        impact="High complexity makes code harder to understand and maintain",
                        effort_estimate="High",
                        code_snippet=self._get_function_code(content, complexity),
                        suggested_changes=self._generate_complexity_suggestions(complexity)
                    ))
        
        return suggestions
    
    def analyze_dependencies(self, path: str) -> List[RefactorSuggestion]:
        """Analyze dependency relationships and suggest improvements."""
        suggestions = []
        dependency_graph = self._build_dependency_graph(path)
        
        # Find circular dependencies
        cycles = list(nx.simple_cycles(dependency_graph))
        for cycle in cycles:
            suggestions.append(RefactorSuggestion(
                file_path=cycle[0],  # Use first file in cycle
                suggestion_type="dependency",
                description=f"Found circular dependency between {', '.join(cycle)}",
                impact="Circular dependencies make code harder to maintain and test",
                effort_estimate="High",
                suggested_changes="Consider restructuring these modules to break the dependency cycle"
            ))
        
        # Find highly coupled modules
        for node in dependency_graph.nodes():
            in_degree = dependency_graph.in_degree(node)
            out_degree = dependency_graph.out_degree(node)
            if in_degree + out_degree > 10:  # Arbitrary threshold
                suggestions.append(RefactorSuggestion(
                    file_path=node,
                    suggestion_type="dependency",
                    description=f"Module has high coupling ({in_degree} incoming, {out_degree} outgoing dependencies)",
                    impact="High coupling makes the codebase more rigid and harder to change",
                    effort_estimate="Medium",
                    suggested_changes="Consider splitting this module into smaller, more focused modules"
                ))
        
        return suggestions
    
    def _find_duplicate_blocks(self, file_contents: Dict[str, List[str]], min_lines: int) -> List[List[tuple]]:
        """Find duplicate code blocks using rolling hash."""
        block_locations = {}
        duplicates = []
        
        for file_path, lines in file_contents.items():
            for i in range(len(lines) - min_lines + 1):
                block = "\n".join(lines[i:i+min_lines])
                block_hash = hash(block)
                
                if block_hash in block_locations:
                    # Found a duplicate
                    if len(block_locations[block_hash]) == 1:  # First duplicate
                        duplicates.append(block_locations[block_hash])
                    block_locations[block_hash].append((file_path, i, i+min_lines))
                else:
                    block_locations[block_hash] = [(file_path, i, i+min_lines)]
        
        return duplicates
    
    def _build_dependency_graph(self, path: str) -> nx.DiGraph:
        """Build a directed graph of module dependencies."""
        graph = nx.DiGraph()
        
        for file_path in Path(path).rglob("*.py"):
            with open(file_path, "r") as f:
                content = f.read()
            
            try:
                tree = ast.parse(content)
            except:
                continue
            
            # Add node for this file
            graph.add_node(str(file_path))
            
            # Find imports
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Add edges for dependencies
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            graph.add_edge(str(file_path), name.name)
                    else:  # ImportFrom
                        if node.module:
                            graph.add_edge(str(file_path), node.module)
        
        return graph
    
    def _calculate_health_score(self, suggestions: List[RefactorSuggestion], path: str) -> float:
        """Calculate overall code health score (0-100)."""
        total_files = len(list(Path(path).rglob("*.py")))
        if total_files == 0:
            return 100.0
        
        # Weight different types of issues
        weights = {
            "duplication": 1.0,
            "complexity": 1.5,
            "dependency": 2.0
        }
        
        # Calculate weighted issue score
        issue_score = sum(
            weights[s.suggestion_type]
            for s in suggestions
        )
        
        # Normalize score (0-100, where 100 is perfect health)
        health_score = max(0, 100 - (issue_score / total_files) * 20)
        return round(health_score, 2)
    
    def _prioritize_suggestions(self, suggestions: List[RefactorSuggestion]) -> List[str]:
        """Prioritize refactoring suggestions."""
        # Sort by type and impact
        priorities = {
            "dependency": 3,
            "complexity": 2,
            "duplication": 1
        }
        
        sorted_suggestions = sorted(
            suggestions,
            key=lambda s: priorities[s.suggestion_type],
            reverse=True
        )
        
        return [s.file_path for s in sorted_suggestions]
    
    def _get_code_snippet(self, location: tuple) -> str:
        """Get the code snippet for a given location."""
        file_path, start, end = location
        with open(file_path, "r") as f:
            lines = f.readlines()
        return "".join(lines[start:end])
    
    def _get_function_code(self, content: str, complexity_info) -> str:
        """Extract function code using line numbers from complexity info."""
        lines = content.splitlines()
        return "\n".join(lines[complexity_info.lineno-1:complexity_info.endline])
    
    def _generate_complexity_suggestions(self, complexity_info) -> str:
        """Generate specific suggestions for reducing complexity."""
        suggestions = []
        
        if complexity_info.complexity > 20:
            suggestions.append("Consider breaking this function into smaller, more focused functions")
        if complexity_info.complexity > 15:
            suggestions.append("Look for opportunities to simplify conditional logic")
        if complexity_info.complexity > 10:
            suggestions.append("Consider extracting complex conditions into well-named helper functions")
        
        return "\n".join(suggestions)