#!/usr/bin/env python3
"""
Hotspot Prediction Model Training Script

This script trains the machine learning model for predicting code hotspots
based on historical repository data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import shap
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.git_analysis import GitAnalysisService


def collect_training_data(repo_path: str) -> pd.DataFrame:
    """
    Collect historical data from the Git repository for training.

    Builds a per-file dataset with metrics from Lizard, commit statistics,
    TODO counts, and bus factor. Also includes repository-level vulnerability
    count to reflect security posture.

    Args:
        repo_path (str): Path to the Git repository

    Returns:
        pd.DataFrame: Training data with per-file metrics
    """
    print(f"Collecting training data from repository: {repo_path}")

    git_service = GitAnalysisService()
    git_service.initialize_repo(repo_path)

    # Enumerate files from HEAD tree
    print("Analyzing file metrics...")
    head_tree = git_service.repo.head.commit.tree
    file_rows = []
    code_exts = {".py", ".js", ".ts", ".tsx", ".java", ".cs", ".go"}

    for item in head_tree.traverse():
        if item.type != 'blob':
            continue
        rel_path = item.path
        # Skip non-code files
        if Path(rel_path).suffix.lower() not in code_exts:
            continue

        abs_path = str(Path(git_service.repo.working_tree_dir) / rel_path)
        metrics = git_service.analyze_file_metrics(abs_path) or {}

        # Flatten core metrics
        row = {
            'file_path': rel_path,
            'nloc': metrics.get('nloc', 0),
            'cyclomatic_complexity': metrics.get('cyclomatic_complexity', 0.0),
            'token_count': metrics.get('token_count', 0),
            'function_count': metrics.get('function_count', 0),
        }
        file_rows.append(row)

    df = pd.DataFrame(file_rows)

    # Commit statistics -> change frequency
    print("Analyzing commit history...")
    commit_stats = git_service.get_commit_statistics()
    commit_df = pd.DataFrame(commit_stats)
    if not commit_df.empty:
        freq_df = commit_df.groupby('file_path').size().reset_index(name='change_frequency')
        df = pd.merge(df, freq_df, on='file_path', how='left')
    else:
        df['change_frequency'] = 0

    # Bus factor
    print("Calculating bus factor...")
    bus_map = git_service.calculate_bus_factor()
    df['bus_factor'] = df['file_path'].map(bus_map).fillna(0).astype(int)

    # TODO counts
    print("Scanning for TODOs...")
    todos = git_service.scan_for_todos()
    todo_df = pd.DataFrame(todos)
    if not todo_df.empty:
        todo_counts = todo_df.groupby('file_path').size().reset_index(name='todo_count')
        df = pd.merge(df, todo_counts, on='file_path', how='left')
    else:
        df['todo_count'] = 0
    df['todo_count'] = df['todo_count'].fillna(0).astype(int)

    # Dependency vulnerability count (repo-level)
    print("Analyzing dependencies...")
    dependencies = git_service.analyze_dependencies()
    vulnerability_count = len(dependencies) if isinstance(dependencies, list) else 0
    df['vulnerability_count'] = vulnerability_count

    # Ensure numeric types and fill NaNs
    for col in ['nloc', 'cyclomatic_complexity', 'token_count', 'function_count', 'change_frequency', 'bus_factor']:
        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)

    print(f"Collected data for {len(df)} files.")
    return df


def create_synthetic_data():
    """Create synthetic training data for demonstration purposes."""
    print("Creating synthetic training data...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic but realistic data
    data = {
        'file_path': [f'src/module_{i%50}/file_{i}.py' for i in range(n_samples)],
        'loc': np.random.randint(10, 500, size=n_samples),
        'cyclomatic_complexity': np.random.randint(1, 20, size=n_samples),
        'token_count': np.random.randint(50, 2000, size=n_samples),
        'maintainability_index': np.random.uniform(0, 100, size=n_samples),
        'commit_frequency': np.random.poisson(lam=5, size=n_samples),
        'lines_added': np.random.poisson(lam=10, size=n_samples),
        'lines_removed': np.random.poisson(lam=5, size=n_samples),
        'author_count': np.random.randint(1, 5, size=n_samples),
        'todo_count': np.random.choice([0, 1, 2], size=n_samples, p=[0.8, 0.15, 0.05]),
        'vulnerable_dependencies': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'has_bug': np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    print(f"Synthetic data created with {len(df)} samples.")
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer relevant features for hotspot prediction.
    
    Args:
        df (pd.DataFrame): Raw training data
        
    Returns:
        pd.DataFrame: Feature-engineered dataset
    """
    print("Engineering features...")
    
    # Fill NaN values
    df = df.fillna(0)

    # Core engineered features (Epic 1)
    df['complexity_per_nloc'] = df['cyclomatic_complexity'] / (df['nloc'] + 1)
    df['tokens_per_function'] = df['token_count'] / (df['function_count'] + 1)
    df['hotspot_signal'] = (
        0.4 * (df['change_frequency']) +
        0.3 * (df['cyclomatic_complexity']) +
        0.2 * (df['token_count']) +
        0.1 * (df['todo_count'])
    )

    # Min-max normalize select signals for label derivation
    def min_max(series: pd.Series) -> pd.Series:
        s = series.astype(float)
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0.0

    norm_signal = min_max(df['hotspot_signal'])
    threshold = np.quantile(norm_signal, 0.8) if len(norm_signal) > 0 else 0.0
    df['is_hotspot'] = (norm_signal >= threshold).astype(int)

    print(f"Engineered {len(df.columns)} features for {len(df)} files")
    return df


def train_model(features: pd.DataFrame):
    """
    Train the LightGBM model for hotspot prediction.
    
    Args:
        features (pd.DataFrame): Feature-engineered dataset
        
    Returns:
        tuple: (trained_model, scaler, feature_names, explainer)
    """
    print("Training hotspot prediction model...")
    
    # Select feature columns (exclude target and file_path)
    feature_cols = [col for col in features.columns if col not in ['file_path', 'is_hotspot']]
    
    X = features[feature_cols]
    y = features['is_hotspot']
    
    print(f"Training with {len(feature_cols)} features:")
    for col in feature_cols:
        print(f"  - {col}")
    
    # Check class balance
    class_balance = y.value_counts(normalize=True)
    print(f"Class distribution: {dict(class_balance)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with balanced class weights
    model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        is_unbalance=True
    )
    
    model.fit(X_train_scaled, y_train,
              eval_set=[(X_test_scaled, y_test)],
              eval_metric='logloss',
              callbacks=[lgb.early_stopping(100, verbose=True)])
    
    # Evaluate model
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"\nModel Performance:")
    print(f"Train accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    # Detailed evaluation
    y_pred = model.predict(X_test_scaled)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

    # Create SHAP explainer
    print("\nCreating SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    
    return model, scaler, feature_cols, explainer


def save_model(model, scaler, feature_names, explainer, model_path, scaler_path, features_path, explainer_path):
    """
    Save the trained model and associated data.
    
    Args:
        model: Trained LightGBM model
        scaler: Fitted StandardScaler
        feature_names: List of feature names
        explainer: SHAP explainer
        model_path: Path to save the model
        scaler_path: Path to save the scaler
        features_path: Path to save feature names
        explainer_path: Path to save the explainer
    """
    print(f"\nSaving model artifacts...")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save scaler
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Save feature names
    joblib.dump(feature_names, features_path)
    print(f"Feature names saved to {features_path}")

    # Save SHAP explainer
    joblib.dump(explainer, explainer_path)
    print(f"SHAP explainer saved to {explainer_path}")


def main():
    """Main training function."""
    print("=== Hotspot Prediction Model Training ===\n")
    
    # Configuration
    repo_path = str(project_root)  # Use project root repository
    model_dir = Path("ai/models")
    model_path = model_dir / "hotspot_prediction_model_v2.joblib"
    scaler_path = model_dir / "hotspot_prediction_scaler_v2.joblib"
    features_path = model_dir / "hotspot_prediction_features_v2.joblib"
    explainer_path = model_dir / "hotspot_prediction_explainer_v2.joblib"
    
    try:
        # Collect training data
        print("Step 1: Data Collection")
        raw_data = collect_training_data(repo_path)
        if raw_data.empty:
            print("No data collected from repo, using synthetic data.")
            raw_data = create_synthetic_data()

        print(f"Collected {len(raw_data)} data points\n")
        
        # Engineer features
        print("Step 2: Feature Engineering")
        features = engineer_features(raw_data)
        print(f"Created dataset with {len(features)} files\n")
        
        # Train model
        print("Step 3: Model Training")
        model, scaler, feature_names, explainer = train_model(features)
        print()
        
        # Save model
        print("Step 4: Model Persistence")
        save_model(model, scaler, feature_names, explainer, model_path, scaler_path, features_path, explainer_path)
        print()
        
        print("✅ Training completed successfully!")
        print(f"Model ready for use in: {model_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)