import argparse
import json
import os
import sys

# Ensure repo root is in sys.path if PYTHONPATH isn't set externally
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from app.services.insights_service import InsightsService
except Exception as e:
    raise RuntimeError(f"Failed to import InsightsService. Ensure PYTHONPATH includes repo root. Original error: {e}")


def to_serializable(value):
    # Convert Pydantic models or other complex types to dicts
    try:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
    except Exception:
        pass
    return value


def main():
    parser = argparse.ArgumentParser(description="Generate EchoLens insights JSON for a target repository")
    parser.add_argument("--repo-path", required=True, help="Path to the target repo clone")
    parser.add_argument("--output-path", required=True, help="Where to write insights JSON")
    args = parser.parse_args()

    service = InsightsService(repo_path=args.repo_path)

    result = {
        "dashboard": to_serializable(service.get_dashboard_data()),
        "subsystem_health": to_serializable(service.get_subsystem_health()),
        "refactor_alerts": to_serializable(service.get_refactor_alerts()),
        "metrics": to_serializable(service.get_codebase_metrics()),
        "predictions": {
            "resource_risk": to_serializable(service.predict_resource_risk()),
            "maintenance_risk": to_serializable(service.predict_maintenance_risk()),
            "complexity_risk": to_serializable(service.predict_complexity_risk()),
        },
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()