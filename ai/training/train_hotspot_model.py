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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.git_analysis import GitAnalysisService


def collect_training_data(repo_path):
    """
    Collect historical data from the Git repository for training.
    
    Args:
        repo_path (str): Path to the Git repository
        
    Returns:
        pd.DataFrame: Training data with commit and file metrics
    """
    print(f"Collecting training data from repository: {repo_path}")
    
    git_service = GitAnalysisService()
    git_service.initialize_repo(repo_path)
    
    data = []
    commit_count = 0
    
    try:
        for commit in git_service.repo.iter_commits():
            commit_count += 1
            if commit_count % 100 == 0:
                print(f"Processed {commit_count} commits...")
            
            for file_path, stats in commit.stats.files.items():
                # Skip binary files and very large files
                if not file_path.endswith(('.py', '.js', '.java', '.cpp', '.c', '.ts', '.rb', '.go')):
                    continue
                
                data.append({
                    'file_path': file_path,
                    'lines_changed': stats.get('lines', 0),
                    'insertions': stats.get('insertions', 0),
                    'deletions': stats.get('deletions', 0),
                    'commit_frequency': 1,
                    'author_count': len(set(commit.author.email for commit in git_service.repo.iter_commits(paths=file_path, max_count=10))),
                    'is_hotspot': False
                })
    
    except Exception as e:
        print(f"Error during data collection: {e}")
        print(f"Collected data from {commit_count} commits")
    
    if not data:
        print("No data collected. Using synthetic data for demonstration.")
        return create_synthetic_data()
    
    return pd.DataFrame(data)


def create_synthetic_data():
    """Create synthetic training data for demonstration purposes."""
    print("Creating synthetic training data...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic but realistic data
    data = {
        'file_path': [f'src/module_{i%50}/file_{i}.py' for i in range(n_samples)],
        'lines_changed': np.random.poisson(lam=5, size=n_samples),
        'insertions': np.random.poisson(lam=3, size=n_samples),
        'deletions': np.random.poisson(lam=2, size=n_samples),
        'commit_frequency': np.random.poisson(lam=1, size=n_samples) + 1,
        'author_count': np.random.randint(1, 6, size=n_samples),
        'is_hotspot': False
    }
    
    df = pd.DataFrame(data)
    
    # Mark some files as hotspots based on realistic criteria
    hotspot_criteria = (
        (df['commit_frequency'] > df['commit_frequency'].quantile(0.8)) &
        (df['lines_changed'] > df['lines_changed'].quantile(0.7)) &
        (df['author_count'] > 2)
    )
    
    df.loc[hotspot_criteria, 'is_hotspot'] = True
    
    # Ensure we have a reasonable balance
    hotspot_ratio = df['is_hotspot'].mean()
    print(f"Hotspot ratio in synthetic data: {hotspot_ratio:.2%}")
    
    return df


def engineer_features(df):
    """
    Engineer relevant features for hotspot prediction.
    
    Args:
        df (pd.DataFrame): Raw training data
        
    Returns:
        pd.DataFrame: Feature-engineered dataset
    """
    print("Engineering features...")
    
    # Aggregate by file path
    features = df.groupby('file_path').agg({
        'lines_changed': ['sum', 'mean', 'std'],
        'insertions': ['sum', 'mean'],
        'deletions': ['sum', 'mean'],
        'commit_frequency': 'sum',
        'author_count': 'max',
        'is_hotspot': 'any'  # If file was ever a hotspot
    }).reset_index()
    
    # Flatten column names
    features.columns = ['_'.join(col).strip() if col[1] else col[0] for col in features.columns]
    features = features.rename(columns={'file_path_': 'file_path', 'is_hotspot_any': 'is_hotspot'})
    
    # Fill NaN values in std columns
    std_cols = [col for col in features.columns if col.endswith('_std')]
    features[std_cols] = features[std_cols].fillna(0)
    
    # Calculate additional features
    features['change_rate'] = features['lines_changed_sum'] / (features['commit_frequency_sum'] + 1)
    features['insertion_deletion_ratio'] = features['insertions_sum'] / (features['deletions_sum'] + 1)
    features['change_volatility'] = features['lines_changed_std'] / (features['lines_changed_mean'] + 1)
    features['author_diversity'] = features['author_count_max'] / (features['commit_frequency_sum'] + 1)
    
    # Handle infinite values
    features = features.replace([np.inf, -np.inf], 0)
    
    print(f"Engineered {len(features.columns)} features for {len(features)} files")
    return features


def train_model(features):
    """
    Train the Random Forest model for hotspot prediction.
    
    Args:
        features (pd.DataFrame): Feature-engineered dataset
        
    Returns:
        tuple: (trained_model, scaler, feature_names)
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
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
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
    
    return model, scaler, feature_cols


def save_model(model, scaler, feature_names, model_path, scaler_path, features_path):
    """
    Save the trained model and associated data.
    
    Args:
        model: Trained RandomForest model
        scaler: Fitted StandardScaler
        feature_names: List of feature names
        model_path: Path to save the model
        scaler_path: Path to save the scaler
        features_path: Path to save feature names
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


def main():
    """Main training function."""
    print("=== Hotspot Prediction Model Training ===\n")
    
    # Configuration
    repo_path = "../.."  # Use project root repository
    model_path = "../models/hotspot_prediction_model.joblib"
    scaler_path = "../models/hotspot_prediction_scaler.joblib"
    features_path = "../models/hotspot_prediction_features.joblib"
    
    try:
        # Collect training data
        print("Step 1: Data Collection")
        raw_data = collect_training_data(repo_path)
        print(f"Collected {len(raw_data)} data points\n")
        
        # Engineer features
        print("Step 2: Feature Engineering")
        features = engineer_features(raw_data)
        print(f"Created dataset with {len(features)} files\n")
        
        # Train model
        print("Step 3: Model Training")
        model, scaler, feature_names = train_model(features)
        print()
        
        # Save model
        print("Step 4: Model Persistence")
        save_model(model, scaler, feature_names, model_path, scaler_path, features_path)
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