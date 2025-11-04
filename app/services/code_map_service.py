import ast
import networkx as nx
from typing import List, Dict
from pathlib import Path
import math
import colorsys
from app.models.insights import ModuleNode, ModuleConnection, CodeMap
from app.services.git_analysis import GitAnalysisService

class CodeMapService:
    def __init__(self):
        self.git_service = GitAnalysisService()
        self.graph = nx.DiGraph()
        
    def generate_map(self, path: str) -> CodeMap:
        """Generate a 3D visualization map of the codebase."""
        # Build dependency graph
        self._build_dependency_graph(path)
        
        # Calculate node positions using force-directed layout
        positions = self._calculate_3d_layout()
        
        # Generate nodes
        nodes = self._generate_nodes(positions)
        
        # Generate connections
        connections = self._generate_connections()
        
        # Calculate metadata
        metadata = self._calculate_metadata()
        
        return CodeMap(
            nodes=nodes,
            connections=connections,
            metadata=metadata
        )
    
    def identify_hotspots(self, path: str) -> List[ModuleNode]:
        """Identify and return hotspot nodes."""
        hotspots = []
        
        # Get commit history analysis
        commit_frequency = self._analyze_commit_frequency(path)
        
        # Get complexity metrics
        complexity_scores = self._analyze_complexity(path)
        
        # Combine metrics to identify hotspots
        for file_path, frequency in commit_frequency.items():
            complexity = complexity_scores.get(file_path, 0)
            if frequency > 10 or complexity > 20:  # Arbitrary thresholds
                hotspots.append(ModuleNode(
                    id=file_path,
                    name=Path(file_path).name,
                    type="module",
                    size=complexity,
                    color=self._calculate_health_color(frequency, complexity),
                    position=self._calculate_node_position(file_path)
                ))
        
        return hotspots
    
    def analyze_dependencies(self, path: str) -> List[ModuleConnection]:
        """Analyze and return module dependencies."""
        self._build_dependency_graph(path)
        return self._generate_connections()
    
    def get_module_details(self, module_id: str) -> Dict:
        """Get detailed information about a specific module."""
        if not Path(module_id).exists():
            raise ValueError(f"Module not found: {module_id}")
        
        with open(module_id, "r") as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
        except:
            raise ValueError(f"Unable to parse module: {module_id}")
        
        return {
            "name": Path(module_id).name,
            "size": len(content.splitlines()),
            "classes": self._extract_classes(tree),
            "functions": self._extract_functions(tree),
            "dependencies": self._get_module_dependencies(module_id),
            "complexity": self._calculate_module_complexity(content),
            "last_modified": self._get_last_modified(module_id)
        }
    
    def _build_dependency_graph(self, path: str):
        """Build a graph representation of module dependencies."""
        self.graph.clear()
        
        for file_path in Path(path).rglob("*.py"):
            with open(file_path, "r") as f:
                content = f.read()
            
            try:
                tree = ast.parse(content)
            except:
                continue
            
            # Add node
            str_path = str(file_path)
            self.graph.add_node(
                str_path,
                name=file_path.name,
                size=len(content.splitlines())
            )
            
            # Add edges for imports
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    self._add_import_edges(str_path, node)
    
    def _add_import_edges(self, source: str, node: ast.AST):
        """Add edges for import statements."""
        if isinstance(node, ast.Import):
            for name in node.names:
                self.graph.add_edge(source, name.name, type="import")
        elif isinstance(node, ast.ImportFrom) and node.module:
            self.graph.add_edge(source, node.module, type="import")
    
    def _calculate_3d_layout(self) -> Dict[str, tuple]:
        """Calculate 3D positions for nodes using force-directed layout."""
        # Use networkx's spring layout in 3D
        pos = nx.spring_layout(self.graph, dim=3)
        
        # Scale positions to reasonable range
        scale = 100
        return {
            node: {
                'x': pos[node][0] * scale,
                'y': pos[node][1] * scale,
                'z': pos[node][2] * scale
            }
            for node in self.graph.nodes()
        }
    
    def _generate_nodes(self, positions: Dict[str, Dict[str, float]]) -> List[ModuleNode]:
        """Generate node objects with positions."""
        nodes = []
        
        for node in self.graph.nodes():
            # Get node attributes
            attrs = self.graph.nodes[node]
            size = attrs.get('size', 1)
            
            # Calculate color based on various metrics
            color = self._calculate_node_color(node)
            
            nodes.append(ModuleNode(
                id=node,
                name=attrs.get('name', Path(node).name),
                type="module",
                size=size,
                color=color,
                position=positions[node]
            ))
        
        return nodes
    
    def _generate_connections(self) -> List[ModuleConnection]:
        """Generate connection objects from graph edges."""
        connections = []
        
        for source, target, data in self.graph.edges(data=True):
            connections.append(ModuleConnection(
                source=source,
                target=target,
                strength=data.get('weight', 1.0),
                type=data.get('type', 'import')
            ))
        
        return connections
    
    def _calculate_metadata(self) -> Dict:
        """Calculate metadata about the codebase structure."""
        return {
            "total_modules": self.graph.number_of_nodes(),
            "total_dependencies": self.graph.number_of_edges(),
            "avg_dependencies": self.graph.number_of_edges() / max(1, self.graph.number_of_nodes()),
            "strongly_connected_components": len(list(nx.strongly_connected_components(self.graph))),
            "modularity": self._calculate_modularity()
        }
    
    def _calculate_modularity(self) -> float:
        """Calculate modularity score for the dependency graph."""
        try:
            communities = nx.community.greedy_modularity_communities(self.graph.to_undirected())
            return nx.community.modularity(self.graph.to_undirected(), communities)
        except:
            return 0.0
    
    def _calculate_node_color(self, node: str) -> str:
        """Calculate node color based on health metrics."""
        # Get various metrics
        in_degree = self.graph.in_degree(node)
        out_degree = self.graph.out_degree(node)
        size = self.graph.nodes[node].get('size', 1)
        
        # Calculate health score (0-1)
        coupling = (in_degree + out_degree) / max(1, self.graph.number_of_nodes())
        size_factor = min(1.0, size / 1000)  # Normalize size
        health = 1 - (coupling * 0.5 + size_factor * 0.5)
        
        # Convert to HSL color (red->yellow->green)
        hue = health * 0.3  # 0 (red) to 0.3 (green)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        
        # Convert to hex
        return "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
    
    def _calculate_health_color(self, frequency: int, complexity: float) -> str:
        """Calculate color based on health metrics."""
        # Normalize metrics
        norm_freq = min(1.0, frequency / 50)  # Cap at 50 commits
        norm_complexity = min(1.0, complexity / 30)  # Cap at complexity of 30
        
        # Calculate health (0-1, where 0 is worst)
        health = 1 - (norm_freq * 0.4 + norm_complexity * 0.6)
        
        # Convert to color
        hue = health * 0.3  # 0 (red) to 0.3 (green)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        
        return "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
    
    def _calculate_node_position(self, file_path: str) -> Dict[str, float]:
        """Calculate 3D position for a node."""
        # Use path structure to influence position
        parts = Path(file_path).parts
        depth = len(parts)
        
        # Calculate position based on path structure
        angle = hash(parts[-2]) % 360 if len(parts) > 1 else 0
        radius = depth * 20
        
        return {
            'x': radius * math.cos(math.radians(angle)),
            'y': depth * 10,
            'z': radius * math.sin(math.radians(angle))
        }
    
    def _analyze_commit_frequency(self, path: str) -> Dict[str, int]:
        """Analyze commit frequency for files."""
        frequency = {}
        for commit in self.git_service.repo.iter_commits():
            for file_path in commit.stats.files:
                if file_path not in frequency:
                    frequency[file_path] = 0
                frequency[file_path] += 1
        return frequency
    
    def _analyze_complexity(self, path: str) -> Dict[str, float]:
        """Analyze code complexity for files."""
        complexity = {}
        for file_path in Path(path).rglob("*.py"):
            with open(file_path, "r") as f:
                content = f.read()
            complexity[str(file_path)] = self._calculate_module_complexity(content)
        return complexity
    
    def _calculate_module_complexity(self, content: str) -> float:
        """Calculate complexity score for a module."""
        try:
            tree = ast.parse(content)
            visitor = ComplexityVisitor.from_ast(tree)
            return sum(item.complexity for item in visitor.functions)
        except:
            return 0.0
    
    def _extract_classes(self, tree: ast.AST) -> List[Dict]:
        """Extract class information from AST."""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append({
                    "name": node.name,
                    "methods": len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                    "line_number": node.lineno
                })
        return classes
    
    def _extract_functions(self, tree: ast.AST) -> List[Dict]:
        """Extract function information from AST."""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    "name": node.name,
                    "args": len(node.args.args),
                    "line_number": node.lineno
                })
        return functions
    
    def _get_module_dependencies(self, module_id: str) -> List[str]:
        """Get dependencies for a specific module."""
        return list(self.graph.successors(module_id))
    
    def _get_last_modified(self, module_id: str) -> str:
        """Get last modification time for a module."""
        for commit in self.git_service.repo.iter_commits(paths=[module_id]):
            return commit.committed_datetime.isoformat()
        return None