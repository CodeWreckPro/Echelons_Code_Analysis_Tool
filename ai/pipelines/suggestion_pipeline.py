import os
from typing import Dict, List, Optional
from pathlib import Path
import joblib
import numpy as np
from dataclasses import dataclass

# Lightweight metrics
from radon.complexity import cc_visit
from radon.metrics import mi_visit

# Local models
try:
    from app.models.suggestions import ActionableSuggestion, SuggestionsBundle, SuggestionLocation
except Exception:
    # Allow CI/standalone training to run without app imports
    ActionableSuggestion = None  # type: ignore
    SuggestionsBundle = None  # type: ignore
    SuggestionLocation = None  # type: ignore


@dataclass
class LoadedSuggestionModel:
    model: Optional[object]
    features: Optional[List[str]]
    version: str


class SuggestionPipeline:
    """
    Hybrid pipeline combining:
    - Supervised ML scoring (priority, ROI)
    - Rule-based code smell detection
    - Heuristic refactor patterns
    - Optional LLM-assisted fine-tuning (stubbed for future)
    """

    def __init__(self, model_path: str = "ai/models/suggestion_model.joblib"):
        self.model_path = model_path
        self._loaded: Optional[LoadedSuggestionModel] = None

    def _lazy_load(self) -> LoadedSuggestionModel:
        if self._loaded:
            return self._loaded
        if os.path.exists(self.model_path):
            try:
                bundle = joblib.load(self.model_path)
                # Support both legacy 'model' (wrapper) and new separate regressors
                model = bundle.get("model")
                if not model and bundle.get("model_priority") and bundle.get("model_roi"):
                    model = {
                        "model_priority": bundle.get("model_priority"),
                        "model_roi": bundle.get("model_roi"),
                    }
                self._loaded = LoadedSuggestionModel(
                    model=model,
                    features=bundle.get("features", []),
                    version=bundle.get("version", "v1"),
                )
                return self._loaded
            except Exception:
                pass
        # Fallback: no model
        self._loaded = LoadedSuggestionModel(model=None, features=None, version="v0-fallback")
        return self._loaded

    def _score_with_ml(self, feature_row: Dict[str, float]) -> Dict[str, float]:
        loaded = self._lazy_load()
        if not loaded.model or not loaded.features:
            # Heuristic fallback
            complexity = feature_row.get("avg_function_complexity", 0.0)
            duplication = feature_row.get("duplication_ratio", 0.0)
            volatility = feature_row.get("file_change_frequency", 0.0)
            priority_score = np.clip(0.5 * (complexity / 10.0) + 0.3 * duplication + 0.2 * volatility, 0.0, 1.0)
            roi_score = float(np.clip(60 + 30 * duplication + 20 * (complexity / 10.0), 0.0, 100.0))
            return {"priority": priority_score, "roi": roi_score, "confidence": 0.6}

        # Predict via ML model
        try:
            X = np.array([[feature_row.get(f, 0.0) for f in loaded.features]])
            # If model is a dict of two regressors
            if isinstance(loaded.model, dict) and "model_priority" in loaded.model and "model_roi" in loaded.model:
                p1 = loaded.model["model_priority"].predict(X)
                p2 = loaded.model["model_roi"].predict(X)
                priority_score = float(np.clip(p1[0], 0.0, 1.0))
                roi_score = float(np.clip(p2[0], 0.0, 100.0))
                return {"priority": priority_score, "roi": roi_score, "confidence": 0.8}
            else:
                # Wrapper or sklearn pipeline with 2-d outputs
                y_pred = loaded.model.predict(X)
                priority_score = float(np.clip(y_pred[0][0], 0.0, 1.0)) if np.ndim(y_pred) > 1 else float(np.clip(y_pred[0], 0.0, 1.0))
                roi_score = float(np.clip(y_pred[0][1], 0.0, 100.0)) if np.ndim(y_pred) > 1 and len(y_pred[0]) > 1 else float(50.0)
                return {"priority": priority_score, "roi": roi_score, "confidence": 0.75}
        except Exception:
            # Fallback
            return {"priority": 0.5, "roi": 50.0, "confidence": 0.5}

    def _analyze_file_metrics(self, file_path: str, content: str) -> Dict[str, float]:
        try:
            blocks = cc_visit(content)
            mi = mi_visit(content, False)
            avg_complexity = float(np.mean([b.complexity for b in blocks])) if blocks else 0.0
            max_complexity = float(np.max([b.complexity for b in blocks])) if blocks else 0.0
            long_funcs = sum(1 for b in blocks if (getattr(b, 'endline', getattr(b, 'lineno', 0)) - getattr(b, 'lineno', 0)) >= 60)
            deep_nesting = sum(1 for b in blocks if b.complexity >= 12)
            return {
                "avg_function_complexity": avg_complexity,
                "max_function_complexity": max_complexity,
                "maintainability_index": float(mi),
                "long_function_count": float(long_funcs),
                "deep_nesting_count": float(deep_nesting),
            }
        except Exception:
            return {"avg_function_complexity": 0.0, "maintainability_index": 80.0}

    def _detect_duplication(self, file_bodies: List[str]) -> float:
        if not file_bodies:
            return 0.0
        # Very light-weight duplication ratio: fraction of identical normalized bodies
        norm = ["".join("".join(s.split()) for s in body.splitlines()) for body in file_bodies]
        unique = len(set(norm))
        return float(np.clip(1.0 - (unique / max(1, len(norm))), 0.0, 1.0))

    def _priority_label(self, score: float) -> str:
        if score >= 0.75: return "high"
        if score >= 0.5: return "medium"
        return "low"

    def _effort_label(self, complexity: float, dup_ratio: float) -> str:
        est = complexity * 0.5 + dup_ratio * 2.0
        if est >= 8: return "high"
        if est >= 4: return "medium"
        return "low"

    def generate(self, repo_path: str, files: Dict[str, str], volatility: Dict[str, int]) -> SuggestionsBundle:
        """
        Generate suggestions across project/subsystem/file/pattern scopes.

        Args:
            repo_path: path to repo
            files: map of file_path -> content
            volatility: map of file_path -> change frequency
        """
        # Compute duplication ratio per coarse buckets
        file_bodies = list(files.values())
        duplication_ratio = self._detect_duplication(file_bodies)

        project_scope: List[ActionableSuggestion] = []
        subsystem_scope: List[ActionableSuggestion] = []
        file_scope: List[ActionableSuggestion] = []
        pattern_scope: List[ActionableSuggestion] = []

        # Project-level recommendation: test strategy and CI improvements
        feat_row = {"avg_function_complexity": np.mean([self._analyze_file_metrics(fp, c).get("avg_function_complexity", 0) for fp, c in files.items()]) if files else 0.0,
                    "duplication_ratio": duplication_ratio,
                    "file_change_frequency": float(np.mean(list(volatility.values())) if volatility else 0.0)}
        ml = self._score_with_ml(feat_row)
        project_scope.append(ActionableSuggestion(
            id="proj-test-ci",
            title="Establish test pyramid and stabilize CI",
            description="Adopt unit→integration→e2e mix; gate PRs; add flaky test quarantine",
            priority=self._priority_label(ml["priority"]),
            confidence_score=ml["confidence"],
            estimated_roi=ml["roi"],
            estimated_effort="medium",
            scope="project",
            categories=["testability", "ci", "quality"],
            actions=[
                "Add coverage thresholds (80% unit, 50% integration)",
                "Enable PR checks: lint, tests, security scan",
                "Introduce flaky test quarantine with retries and tagging"
            ],
            rationale=["Volatility and duplication increase change risk; solid tests reduce regressions"],
            horizon="1m",
            metadata={"duplication_ratio": duplication_ratio}
        ))

        # Per-file suggestions
        for fp, content in files.items():
            m = self._analyze_file_metrics(fp, content)
            freq = float(volatility.get(fp, 0))
            row = {"avg_function_complexity": m.get("avg_function_complexity", 0.0),
                   "duplication_ratio": duplication_ratio,
                   "file_change_frequency": float(np.clip(freq / 10.0, 0.0, 1.0))}
            s = self._score_with_ml(row)

            # Complexity reduction
            if m.get("max_function_complexity", 0.0) >= 15 or m.get("deep_nesting_count", 0.0) >= 1:
                file_scope.append(ActionableSuggestion(
                    id=f"complexity-{Path(fp).name}",
                    title=f"Decompose complex functions in {Path(fp).name}",
                    description=f"Split functions with CC≥{int(m.get('max_function_complexity', 0))} and flatten nesting",
                    priority=self._priority_label(s["priority"]),
                    confidence_score=s["confidence"],
                    estimated_roi=float(np.clip(70 + 1.5 * m.get("deep_nesting_count", 0.0), 0, 100)),
                    estimated_effort=self._effort_label(m.get("max_function_complexity", 0.0), duplication_ratio),
                    scope="file",
                    categories=["refactor", "complexity"],
                    locations=[SuggestionLocation(file_path=fp)],
                    actions=["Extract helper functions", "Unnest conditionals", "Introduce guard clauses"],
                    rationale=["High CC harms readability and testability"],
                    horizon="1w",
                    metadata={"maintainability_index": m.get("maintainability_index", 80.0)}
                ))

            # Duplication consolidation
            if duplication_ratio >= 0.2:
                pattern_scope.append(ActionableSuggestion(
                    id=f"dup-{Path(fp).name}",
                    title="Consolidate duplicate logic into shared helpers",
                    description="Extract repeated loops/validation into shared utils to reduce copy-paste",
                    priority=self._priority_label(0.55),
                    confidence_score=0.7,
                    estimated_roi=float(np.clip(60 + duplication_ratio * 40, 0, 100)),
                    estimated_effort="low",
                    scope="pattern",
                    categories=["refactor", "duplication"],
                    locations=[SuggestionLocation(file_path=fp)],
                    actions=["Create sanitize_input()", "Centralize validation"],
                    rationale=["Duplicate code increases defect probability and maintenance overhead"],
                    horizon="1w",
                    metadata={"duplication_ratio": duplication_ratio}
                ))

        # Simple subsystem bucketing by top-level directories
        subsys_map: Dict[str, List[str]] = {}
        for fp in files.keys():
            parts = Path(fp).parts
            if len(parts) > 1:
                subsys = parts[0]
                subsys_map.setdefault(subsys, []).append(fp)

        for subsys, fps in subsys_map.items():
            if len(fps) >= 3:
                subsystem_scope.append(ActionableSuggestion(
                    id=f"subsys-mod-{subsys}",
                    title=f"Modularize {subsys} boundaries",
                    description="Extract clear interfaces, reduce coupling, define anti-corruption layer",
                    priority="medium",
                    confidence_score=0.65,
                    estimated_roi=75.0,
                    estimated_effort="medium",
                    scope="subsystem",
                    categories=["architecture", "modularity"],
                    locations=[SuggestionLocation(file_path=f) for f in fps[:3]],
                    actions=["Define service interfaces", "Encapsulate shared state", "Introduce dependency inversion"],
                    rationale=["Subsystem sprawl complicates ownership and review"],
                    horizon="1m",
                    metadata={"files": len(fps)}
                ))

        # Bundle
        from datetime import datetime
        return SuggestionsBundle(
            generated_at=datetime.now().isoformat(),
            project_scope=project_scope,
            subsystem_scope=subsystem_scope,
            file_scope=file_scope,
            pattern_scope=pattern_scope,
            version=self._lazy_load().version,
        )


# Singleton for app services
suggestion_pipeline = SuggestionPipeline()