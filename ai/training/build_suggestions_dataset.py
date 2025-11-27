"""
Build a training dataset for the Actionable Suggestions scoring model.

Features per file:
- avg_function_complexity (radon cc)
- duplication_ratio (lightweight normalization across files)
- file_change_frequency (commits touching file)
- maintainability_index (radon mi)
- ownership_concentration (top author share of touches)

Usage:
  python ai/training/build_suggestions_dataset.py \
    --repo-path <path-to-repo> \
    --output-csv ai/training/suggestions_features.csv
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import sys

from radon.complexity import cc_visit
from radon.metrics import mi_visit

try:
    # Ensure project root on sys.path when invoked directly
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
except Exception:
    pass

from app.services.git_analysis import GitAnalysisService


def _read_repo_files(repo_path: str) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for file_path in Path(repo_path).rglob("*.py"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                files[str(file_path)] = f.read()
        except Exception:
            continue
    return files


def _duplication_ratio(file_bodies: List[str]) -> float:
    if not file_bodies:
        return 0.0
    # Normalize by removing whitespace
    norm = ["".join("".join(s.split()) for s in body.splitlines()) for body in file_bodies]
    unique = len(set(norm))
    return float(np.clip(1.0 - (unique / max(1, len(norm))), 0.0, 1.0))


def _file_metrics(content: str) -> Tuple[float, float]:
    """Return (avg_function_complexity, maintainability_index)."""
    try:
        blocks = cc_visit(content)
        mi = mi_visit(content, False)
        avg_cc = float(np.mean([b.complexity for b in blocks])) if blocks else 0.0
        return avg_cc, float(mi)
    except Exception:
        return 0.0, 80.0


def _commit_stats(repo_path: str) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    """
    Return:
    - touches: file_path -> change count
    - authors: file_path -> {author -> count}
    """
    git = GitAnalysisService()
    git.initialize_repo(repo_path)
    touches: Dict[str, int] = {}
    authors: Dict[str, Dict[str, int]] = {}
    for s in git.get_commit_statistics():
        fp = s.get("file_path")
        au = s.get("author", "unknown")
        if not fp:
            continue
        touches[fp] = touches.get(fp, 0) + 1
        if fp not in authors:
            authors[fp] = {}
        authors[fp][au] = authors[fp].get(au, 0) + 1
    return touches, authors


def _ownership_concentration(authors: Dict[str, int]) -> float:
    if not authors:
        return 0.0
    total = sum(authors.values())
    top = max(authors.values())
    return float(np.clip(top / max(1, total), 0.0, 1.0))


def build_dataset(repo_path: str, output_csv: str) -> str:
    files = _read_repo_files(repo_path)
    touches, authors_map = _commit_stats(repo_path)
    dup_ratio = _duplication_ratio(list(files.values()))

    rows: List[Dict] = []
    for fp, content in files.items():
        avg_cc, mi = _file_metrics(content)
        freq = touches.get(fp, 0)
        owners = authors_map.get(fp, {})
        own_conc = _ownership_concentration(owners)

        rows.append({
            "file_path": fp,
            "avg_function_complexity": float(avg_cc),
            "duplication_ratio": float(dup_ratio),
            "file_change_frequency": float(freq),
            "maintainability_index": float(mi),
            "ownership_concentration": float(own_conc),
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


def main():
    parser = argparse.ArgumentParser(description="Build Suggestions training dataset")
    parser.add_argument("--repo-path", required=True, help="Path to Git repository")
    parser.add_argument("--output-csv", default="ai/training/suggestions_features.csv", help="Output CSV path")
    args = parser.parse_args()
    out = build_dataset(args.repo_path, args.output_csv)
    print(f"Saved features to {out}")


if __name__ == "__main__":
    main()