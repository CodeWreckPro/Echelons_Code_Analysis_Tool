# EchoLens Architecture Documentation

## System Overview

EchoLens is an AI-driven codebase intelligence platform that provides advanced visualization, analysis, and predictive capabilities for large software projects. The system is designed with a modular architecture that separates concerns and allows for easy extension and maintenance.

## Core Components

### 1. Code Evolution Timeline
- **Purpose**: Track and visualize the evolution of code over time
- **Key Components**:
  - Git Analysis Service
  - Embedding Service for semantic analysis
  - Timeline visualization using D3.js

### 2. AI-Powered Change Storytelling
- **Purpose**: Generate human-readable explanations of code changes
- **Key Components**:
  - Storytelling Service
  - Language Models for text generation
  - Template-based narrative generation

### 3. Hotspot & Risk Prediction
- **Purpose**: Identify high-risk areas in the codebase
- **Key Components**:
  - ML models for risk assessment
  - Complexity analysis
  - Activity pattern analysis

### 4. AI Refactor Guide
- **Purpose**: Provide automated refactoring suggestions
- **Key Components**:
  - Code analysis engine
  - Pattern detection
  - Recommendation system

### 5. Interactive 3D Code Map
- **Purpose**: Visualize codebase structure and relationships
- **Key Components**:
  - Three.js visualization
  - Dependency analysis
  - Interactive navigation

### 6. Predictive Insights Dashboard
- **Purpose**: Monitor codebase health and trends
- **Key Components**:
  - Health scoring system
  - Alert generation
  - Integration with external tools

## Data Flow

```mermaid
graph TD
    A[Git Repository] --> B[Git Analysis Service]
    B --> C[Embedding Service]
    B --> D[Storytelling Service]
    C --> E[Code Evolution Timeline]
    D --> E
    B --> F[Hotspot Detection]
    F --> G[Risk Prediction Model]
    G --> H[Predictive Dashboard]
    B --> I[Code Map Generator]
    I --> J[3D Visualization]
    B --> K[Refactor Analysis]
    K --> L[Suggestion Engine]
```

## AI/ML Components

### 1. Embedding Models
- **Technology**: CodeBERT
- **Purpose**: Generate semantic embeddings for code and text
- **Usage**: Change analysis, similarity detection

### 2. Language Models
- **Technology**: BART
- **Purpose**: Generate human-readable explanations
- **Usage**: Change storytelling, documentation

### 3. Risk Prediction Models
- **Technology**: Custom ML models
- **Purpose**: Identify potential issues
- **Usage**: Hotspot detection, risk assessment

### ML Pipeline v2
- **Model**: `LightGBM` classifier with supervised hotspot labels
- **Explainability**: `SHAP TreeExplainer` persisted for consistent explanations
- **Feature Set (Epic 1)**:
  - Per-file metrics: `nloc`, `cyclomatic_complexity`, `token_count`, `function_count`
  - Activity metrics: `change_frequency`, `bus_factor`, `todo_count`
  - Derived features: `complexity_per_nloc`, `tokens_per_function`
  - Repo posture: `vulnerability_count` (from dependency scan)
- **Artifacts**: Saved under `ai/models`
  - `hotspot_prediction_model_v2.joblib`
  - `hotspot_prediction_scaler_v2.joblib`
  - `hotspot_prediction_features_v2.joblib`
  - `hotspot_prediction_explainer_v2.joblib`
- **Service Loading**: `InsightsService` prefers v2 artifacts and falls back to v1 names if missing
- **Training Script**: `ai/training/train_hotspot_model.py`
  - Collects repo metrics via `GitAnalysisService`
  - Engineers features and derives `is_hotspot` labels using normalized risk signal
  - Trains LightGBM with early stopping and persists artifacts

## External Integrations

1. **Version Control**
   - Git repositories
   - GitHub/GitLab APIs

2. **Communication**
   - Slack integration
   - Notion integration

3. **CI/CD**
   - Jenkins/GitHub Actions hooks
   - Automated reporting

## Security Considerations

1. **Code Access**
   - Secure repository access
   - Access control integration

2. **Data Privacy**
   - Local processing of sensitive code
   - Configurable data retention

3. **API Security**
   - Authentication/Authorization
   - Rate limiting

## Performance Optimization

1. **Caching Strategy**
   - Git operation results
   - Embedding computations
   - Visualization data

2. **Computation Distribution**
   - Background processing
   - Task queuing
   - Resource management

## Deployment Architecture

```mermaid
graph LR
    A[Frontend] --> B[API Gateway]
    B --> C[Core Services]
    C --> D[ML Services]
    C --> E[Database]
    C --> F[Cache]
    D --> G[Model Storage]
```

## Configuration Management

1. **Environment Variables**
   - API keys
   - Service endpoints
   - Feature flags

2. **Model Configuration**
   - Model parameters
   - Training settings
   - Inference settings

## Monitoring and Logging

1. **System Health**
   - Service metrics
   - Performance monitoring
   - Error tracking

2. **Usage Analytics**
   - Feature usage
   - User interactions
   - Performance metrics

## Future Extensions

1. **Additional Analysis Types**
   - Architecture compliance
   - Security vulnerability detection
   - Technical debt assessment

2. **Enhanced Visualizations**
   - VR/AR integration
   - Real-time collaboration
   - Custom visualization plugins

3. **Advanced AI Features**
   - Code generation
   - Automated testing
   - Performance optimization

---

## Actionable Suggestions Engine — Overview

The Suggestions Engine augments Insights with a second, structured output focused on actionable refactoring and optimization guidance. It uses a hybrid approach:
- Supervised ML scoring for priority/ROI
- Rule-based detection of code smells and anti-patterns
- Heuristic refactor patterns
- Optional LLM-assisted fine-tuning (future)

## Architecture, Models, and Data Flow

Components:
- `app/services/suggestions_service.py` — Orchestrates repo metrics and pipeline
- `ai/pipelines/suggestion_pipeline.py` — Hybrid reasoning and scoring
- `ai/training/suggestion_model.py` — Independent retraining entrypoint
- `ai/models/suggestion_model.joblib` — Lazy-loaded (optional)
- `app/api/suggestions.py` — API endpoints

Data Flow:
1. Repo metrics collected from Git history and static analysis
2. Feature vector constructed per scope
3. ML scores computed (priority, ROI) or heuristics used if missing
4. Rules generate explicit actions, rationale, locations, horizons
5. Bundle assembled and returned/published

## How the Suggestion Model is Trained

Training consumes a CSV of features built from:
- Cyclomatic complexity (per-function)
- Maintainability index
- Churn/volatility and ownership concentration
- Duplication ratio via AST/syntax normalization
- Static-analysis findings and smell rules

Targets are synthesized if unavailable. Two regressors estimate priority and ROI, bundled and saved via joblib with feature versioning.

## How to Extend the Suggestion Ruleset

Add new detectors in `suggestion_pipeline.py` and map them to actions with clear rationale. Keep outputs:
- Specific (files/lines)
- Prioritized (high/medium/low)
- Explainable (reasons list)
- Actionable (clear steps)

## Limitations & Future Work

- LLM reasoning integration is stubbed for future expansion
- Security checks depend on `requirements.txt`
- Duplicate detection is lightweight; can be replaced with AST graph similarity

## Development Guidelines

1. **Code Organization**
   - Feature-based structure
   - Clear separation of concerns
   - Consistent naming conventions

2. **Testing Strategy**
   - Unit tests
   - Integration tests
   - Performance tests

3. **Documentation**
   - API documentation
   - User guides
   - Development guides