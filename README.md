# EchoLens - AI-Driven Codebase Intelligence Platform

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🔍 Overview
EchoLens is a cutting-edge codebase intelligence platform that leverages AI to provide deep insights into software projects. By combining advanced visualization, machine learning, and predictive analytics, EchoLens helps development teams understand, maintain, and improve their codebase with unprecedented clarity.

## ✨ Key Features

### 🕒 Code Evolution Timeline
- **AI-Powered Git Analysis**: Intelligent parsing of commit history with semantic understanding
- **Visual Change Tracking**: Interactive timeline of code evolution
- **Semantic Analysis**: Advanced embedding technologies for understanding code changes

### 🎯 AI-Powered Change Storytelling
- **Smart Commit Analysis**: Natural language summaries of code changes
- **PR Impact Assessment**: Understand the scope and impact of changes
- **Component Relationship Mapping**: Visualize how changes affect different parts of the system

### 🔥 Hotspot & Risk Prediction
- **ML-Driven Risk Analysis**: Identify high-risk files before they cause problems
- **Complexity Tracking**: Monitor and alert on increasing code complexity
- **Change Pattern Detection**: Identify potentially problematic code patterns

### 🛠️ AI Refactor Guide
- **Smart Refactoring Suggestions**: AI-powered code improvement recommendations
- **Code Smell Detection**: Automated identification of potential issues
- **Best Practice Alignment**: Suggestions for improving code quality

### 🗺️ Interactive 3D Code Map
- **3D Visualization**: Immersive codebase exploration using Three.js
- **Dependency Analysis**: Interactive visualization of module relationships
- **Real-time Updates**: Live updates as your code evolves

### 📊 Predictive Insights Dashboard
- **Health Scoring**: Real-time subsystem health monitoring
- **Smart Alerts**: Proactive "Refactor Now" notifications
- **Trend Analysis**: Track and predict code quality trends

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/echolens.git
cd echolens
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Run the development server:
```bash
uvicorn app.main:app --reload
```

## Project Structure

```
echolens/
├── app/
│   ├── api/            # API endpoints
│   ├── core/           # Core business logic
│   ├── models/         # Database models
│   └── services/       # Service layer
├── ai/
│   ├── models/         # ML models
│   └── training/       # Training scripts
├── frontend/
│   ├── components/     # React components
│   └── pages/         # Page layouts
├── docs/              # Documentation
├── notebooks/         # Jupyter notebooks
└── tests/            # Test suite
```

## Documentation

- [Architecture Overview](docs/architecture.md)
- [API Documentation](docs/api.md)
- [ML Model Training](docs/ml_training.md)

## Model Retraining and Artifacts (v2)

- Retrain the hotspot prediction model using `ai/training/train_hotspot_model.py`.
- Example command: `python ai/training/train_hotspot_model.py`
- The training script saves v2 model artifacts under `ai/models`:
  - `ai/models/hotspot_prediction_model_v2.joblib`
  - `ai/models/hotspot_prediction_scaler_v2.joblib`
  - `ai/models/hotspot_prediction_features_v2.joblib`
  - `ai/models/hotspot_prediction_explainer_v2.joblib`
- The API prefers these v2 artifacts and falls back to v1 if missing.
- No notebook is required to serve the API.

## Notebook Note

- The notebook `notebooks/hotspot_prediction_training.ipynb` is optional and not part of the API runtime.
- Use it for experimentation, visualization, or alternative training workflows.
- Ensure any trained artifacts are saved to `ai/models` with the expected filenames so the API can load them.

## How Model Artifacts Are Loaded

- Loader location: `app/services/insights_service.py` in `InsightsService._load_models()`.
- Artifacts directory: `ai/models` with expected filenames (v2 preferred):
  - `hotspot_prediction_model_v2.joblib`
  - `hotspot_prediction_scaler_v2.joblib`
  - `hotspot_prediction_features_v2.joblib`
  - `hotspot_prediction_explainer_v2.joblib`
- Behavior: If v2 artifacts are missing, the service falls back to v1 names and logs a warning; the API still runs.
- Usage: Loaded artifacts power hotspot/risk predictions in `/api/insights/dashboard` and `/api/insights/predictions/*`.
- Deployment tip: Ensure the `ai/models` v2 artifacts exist before starting `uvicorn`.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
## Actionable Suggestions Engine — Overview

EchoLens includes an Actionable Suggestions Engine that produces high-quality, context-aware recommendations across project, subsystem, file, and pattern scopes. Suggestions include confidence scores, estimated ROI/effort, explicit rationales, and prioritized actions (high/medium/low).

Outputs are published to GitHub Pages as two JSON files per repo:
- `insights/<owner>/<repo>.json` — dashboard metrics, predictions, risks
- `insights/<owner>/<repo>-suggestions.json` — actionable suggestions bundle

The Pages UI displays both JSONs in two rows for quick review.

## Architecture, Models, and Data Flow

- Services: `InsightsService` (existing), `SuggestionsService` (new)
- Pipeline: `ai/pipelines/suggestion_pipeline.py` (hybrid ML+rules)
- Training: `ai/training/suggestion_model.py`
- Models: `ai/models/suggestion_model.joblib` (lazy-loaded; optional)
- API: `/api/suggestions` endpoints for bundle and per-file suggestions
- CI: `.github/workflows/generate-insights.yml` publishes both JSON outputs

Data Flow:
- CI clones target repo → runs `ci/generate_insights.py` → writes insights and suggestions JSONs → publishes to Pages.
- FastAPI app exposes live endpoints for programmatic access.

## How the Suggestion Model is Trained

The training script expects a CSV of tabular features derived from:
- Hotspot, churn, complexity, maintainability index
- Duplication ratio, ownership concentration, volatility
- Static-analysis findings and code smell rules

Targets: `priority_score (0–1)`, `roi_score (0–100)`. The model only scores; textual suggestions are generated by rule/heuristics for explainability.

Run:
```
python ai/training/suggestion_model.py --features-csv features.csv --output ai/models/suggestion_model.joblib
```

## How to Extend the Suggestion Ruleset

Edit `ai/pipelines/suggestion_pipeline.py` and add new detectors or actions. The pipeline merges ML scores with rules and heuristics; contributions should include:
- Detection logic
- Action list
- Rationale and horizon
- Confidence and ROI mapping

## Limitations & Future Work

- LLM-assisted fine-tuning is stubbed for future integration.
- Suggestions depend on available metrics; missing data falls back to heuristics.
- Security dependency scans require `requirements.txt` and may be skipped if not present.