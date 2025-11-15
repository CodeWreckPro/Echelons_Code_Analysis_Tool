import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import os
import sys

# Ensure repo root is in sys.path if PYTHONPATH isn't set externally
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from app.services.insights_service import InsightsService
except Exception as e:
    raise RuntimeError(
        f"Failed to import InsightsService. Ensure PYTHONPATH includes repo root. Original error: {e}"
    )


def serialize(value):
    """Recursively convert complex types to JSON-serializable structures."""
    # Pydantic v2
    if hasattr(value, "model_dump"):
        return value.model_dump()
    # Pydantic v1
    if hasattr(value, "dict") and callable(getattr(value, "dict")):
        return value.dict()
    # datetime
    if isinstance(value, datetime):
        return value.isoformat()
    # pathlib.Path
    if isinstance(value, Path):
        return str(value)
    # numpy types
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    # sets/tuples/lists
    # List/tuple
    if isinstance(value, (list, tuple)):
        return [serialize(v) for v in value]
    if isinstance(value, set):
        return [serialize(v) for v in value]
    # Dict
    if isinstance(value, dict):
        return {k: serialize(v) for k, v in value.items()}
    # Fallback
    return value


def main():
    parser = argparse.ArgumentParser(description="Generate EchoLens insights JSON for a target repository")
    parser.add_argument("--repo-path", required=True, help="Path to the target repo clone")
    parser.add_argument("--output-path", required=True, help="Where to write insights JSON")
    args = parser.parse_args()

    # InsightsService constructor takes no arguments; methods take repo_path
    service = InsightsService()

    # Generate dashboard (composite) and individual sections to mirror API outputs
    dashboard = service.generate_dashboard_insights(args.repo_path)
    subsystem_health = service.analyze_subsystem_health(args.repo_path)
    refactor_alerts = service.identify_refactor_opportunities(args.repo_path)
    metrics = service.calculate_codebase_metrics(args.repo_path)
    resource_predictions = service.predict_resource_needs(args.repo_path)
    maintenance_predictions = service.predict_maintenance_needs(args.repo_path)
    complexity_predictions = service.predict_complexity_trends(args.repo_path)
    risk_areas = service.identify_risk_areas(args.repo_path)

    result = {
        "dashboard": serialize(dashboard),
        "subsystem_health": serialize(subsystem_health),
        "refactor_alerts": serialize(refactor_alerts),
        "metrics": serialize(metrics),
        "predictions": {
            "resource": serialize(resource_predictions),
            "maintenance": serialize(maintenance_predictions),
            "complexity": serialize(complexity_predictions),
        },
        "risk_areas": serialize(risk_areas),
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()