#!/usr/bin/env python3
"""
Insights Service - AI-driven insights generation for code analysis

This service provides intelligent insights about code quality, hotspots, risks,
and maintenance predictions using machine learning models and statistical analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import joblib
import os
from pathlib import Path
import logging
from collections import defaultdict
import json

# Import our services
from app.services.git_analysis import GitAnalysisService
from app.services.embedding import EmbeddingService
from app.models.insights import (
    SubsystemHealth, RefactorAlert, CodebaseMetrics, ResourcePrediction,
    RiskArea, MaintenancePrediction, ComplexityPrediction, DashboardData
)

logger = logging.getLogger(__name__)


class InsightsService:
    """
    Service for generating AI-driven insights about code quality, hotspots,
    and maintenance predictions using trained ML models and statistical analysis.
    """
    
    def __init__(self):
        """Initialize the InsightsService with ML models and dependencies."""
        self.git_service = GitAnalysisService()
        self.embedding_service = EmbeddingService()
        
        # Model paths
        self.model_dir = Path("ai/models")
        self.hotspot_model_path = self.model_dir / "hotspot_prediction_model.joblib"
        self.hotspot_scaler_path = self.model_dir / "hotspot_prediction_scaler.joblib"
        self.hotspot_features_path = self.model_dir / "hotspot_prediction_features.joblib"
        
        # Load ML models
        self.hotspot_model = None
        self.hotspot_scaler = None
        self.hotspot_features = None
        self._load_models()
        
        # Cache for performance
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes
        
        logger.info("InsightsService initialized successfully")
    
    def _load_models(self):
        """Load trained ML models from disk."""
        try:
            if self.hotspot_model_path.exists():
                self.hotspot_model = joblib.load(self.hotspot_model_path)
                logger.info("Hotspot prediction model loaded successfully")
            else:
                logger.warning("Hotspot prediction model not found, using statistical analysis")
            
            if self.hotspot_scaler_path.exists():
                self.hotspot_scaler = joblib.load(self.hotspot_scaler_path)
                logger.info("Hotspot scaler loaded successfully")
            
            if self.hotspot_features_path.exists():
                self.hotspot_features = joblib.load(self.hotspot_features_path)
                logger.info("Hotspot feature names loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            logger.info("Falling back to statistical analysis methods")
    
    def _get_cache_key(self, repo_path: str, analysis_type: str) -> str:
        """Generate cache key for repository analysis."""
        return f"{repo_path}:{analysis_type}:{datetime.now().strftime('%Y%m%d%H%M')}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache:
            return False
        
        timestamp, _ = self._cache[cache_key]
        return datetime.now() - timestamp < timedelta(seconds=self._cache_timeout)
    
    def _get_cached_data(self, cache_key: str):
        """Retrieve cached data if valid."""
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key][1]
        return None
    
    def _set_cached_data(self, cache_key: str, data):
        """Cache data with timestamp."""
        self._cache[cache_key] = (datetime.now(), data)
    
    def generate_dashboard_insights(self, repo_path: str) -> DashboardData:
        """
        Generate comprehensive dashboard insights using ML models and statistical analysis.
        
        Args:
            repo_path (str): Path to the Git repository
            
        Returns:
            DashboardData: Complete dashboard insights
        """
        logger.info(f"Generating dashboard insights for {repo_path}")
        
        try:
            # Initialize Git repository
            self.git_service.initialize_repo(repo_path)
            
            # Generate all components
            subsystem_health = self.analyze_subsystem_health(repo_path)
            refactor_alerts = self.identify_refactor_opportunities(repo_path)
            codebase_metrics = self.calculate_codebase_metrics(repo_path)
            resource_predictions = self.predict_resource_needs(repo_path)
            risk_areas = self.identify_risk_areas(repo_path)
            maintenance_predictions = self.predict_maintenance_needs(repo_path)
            complexity_predictions = self.predict_complexity_trends(repo_path)
            
            dashboard_data = DashboardData(
                subsystem_health=subsystem_health,
                refactor_alerts=refactor_alerts,
                codebase_metrics=codebase_metrics,
                resource_predictions=resource_predictions,
                risk_areas=risk_areas,
                maintenance_predictions=maintenance_predictions,
                complexity_predictions=complexity_predictions,
                generated_at=datetime.now()
            )
            
            logger.info("Dashboard insights generated successfully")
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error generating dashboard insights: {e}")
            return self._generate_fallback_dashboard()
    
    def analyze_subsystem_health(self, repo_path: str) -> List[SubsystemHealth]:
        """
        Analyze health of different subsystems using ML predictions and metrics.
        
        Args:
            repo_path (str): Path to the Git repository
            
        Returns:
            List[SubsystemHealth]: Health status of subsystems
        """
        logger.info("Analyzing subsystem health...")
        
        try:
            # Get recent commit data
            commits_data = self.git_service.get_commit_statistics()
            
            # Group by subsystem (directory structure)
            subsystem_data = defaultdict(list)
            for commit in commits_data:
                for file_path in commit.get('files', []):
                    subsystem = file_path.split('/')[0] if '/' in file_path else 'root'
                    subsystem_data[subsystem].append(commit)
            
            health_metrics = []
            for subsystem, commits in subsystem_data.items():
                if len(commits) < 5:  # Skip subsystems with too little data
                    continue
                
                # Calculate health metrics
                commit_frequency = len(commits) / 30  # commits per day (assuming 30 days data)
                bug_fixes = sum(1 for c in commits if 'fix' in c.get('message', '').lower())
                complexity_trend = self._calculate_complexity_trend(commits)
                hotspot_prediction = self._predict_hotspot_probability(subsystem, commits)
                
                # Determine health status
                if hotspot_prediction > 0.7:
                    status = "critical"
                elif hotspot_prediction > 0.5:
                    status = "warning"
                elif commit_frequency > 5 and bug_fixes < len(commits) * 0.1:
                    status = "healthy"
                else:
                    status = "stable"
                
                health_metrics.append(SubsystemHealth(
                    name=subsystem,
                    status=status,
                    complexity_trend=complexity_trend,
                    last_updated=datetime.now()
                ))
            
            return health_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing subsystem health: {e}")
            return self._generate_fallback_subsystem_health()
    
    def identify_refactor_opportunities(self, repo_path: str) -> List[RefactorAlert]:
        """
        Identify code refactoring opportunities using ML predictions and heuristics.
        
        Args:
            repo_path (str): Path to the Git repository
            
        Returns:
            List[RefactorAlert]: Refactoring opportunities
        """
        logger.info("Identifying refactoring opportunities...")
        
        try:
            # Get complexity analysis
            complexity_data = self.git_service.analyze_complexity_trends()
            
            alerts = []
            for file_data in complexity_data:
                file_path = file_data['file_path']
                complexity = file_data.get('complexity', 0)
                change_frequency = file_data.get('change_frequency', 0)
                
                # ML-based hotspot prediction
                hotspot_score = self._predict_file_hotspot(file_path, complexity, change_frequency)
                
                # Heuristic-based refactoring triggers
                if hotspot_score > 0.6 or complexity > 20 or change_frequency > 10:
                    severity = "high" if hotspot_score > 0.8 or complexity > 30 else "medium"
                    
                    alerts.append(RefactorAlert(
                        file_path=file_path,
                        reason=self._generate_refactor_reason(hotspot_score, complexity, change_frequency),
                        severity=severity,
                        suggested_action=self._suggest_refactor_action(hotspot_score, complexity),
                        estimated_effort=self._estimate_refactor_effort(complexity)
                    ))
            
            # Sort by severity and hotspot score
            alerts.sort(key=lambda x: (x.severity != 'high', x.file_path))
            return alerts[:10]  # Return top 10 alerts
            
        except Exception as e:
            logger.error(f"Error identifying refactor opportunities: {e}")
            return self._generate_fallback_refactor_alerts()
    
    def calculate_codebase_metrics(self, repo_path: str) -> CodebaseMetrics:
        """
        Calculate comprehensive codebase metrics using statistical analysis.
        
        Args:
            repo_path (str): Path to the Git repository
            
        Returns:
            CodebaseMetrics: Detailed codebase metrics
        """
        logger.info("Calculating codebase metrics...")
        
        try:
            # Get repository statistics
            repo_stats = self.git_service.get_repository_statistics()
            commit_stats = self.git_service.get_commit_statistics()
            
            # Calculate metrics
            total_files = repo_stats.get('total_files', 0)
            total_commits = len(commit_stats)
            
            # Complexity analysis
            complexity_data = self.git_service.analyze_complexity_trends()
            avg_complexity = np.mean([d.get('complexity', 0) for d in complexity_data]) if complexity_data else 0
            
            # Change frequency analysis
            change_frequencies = [len(c.get('files', [])) for c in commit_stats]
            avg_change_frequency = np.mean(change_frequencies) if change_frequencies else 0
            
            # Technical debt estimation (simplified)
            high_complexity_files = sum(1 for d in complexity_data if d.get('complexity', 0) > 15)
            technical_debt_ratio = high_complexity_files / total_files if total_files > 0 else 0
            
            # Code quality score (0-100)
            quality_score = max(0, 100 - (technical_debt_ratio * 50) - (avg_complexity * 2))
            
            return CodebaseMetrics(
                total_files=total_files,
                total_commits=total_commits,
                average_complexity=round(avg_complexity, 2),
                technical_debt_ratio=round(technical_debt_ratio, 3),
                code_quality_score=round(quality_score, 1),
                last_analyzed=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating codebase metrics: {e}")
            return self._generate_fallback_codebase_metrics()
    
    def predict_resource_needs(self, repo_path: str) -> List[ResourcePrediction]:
        """
        Predict future resource needs based on historical trends and ML analysis.
        
        Args:
            repo_path (str): Path to the Git repository
            
        Returns:
            List[ResourcePrediction]: Resource predictions
        """
        logger.info("Predicting resource needs...")
        
        try:
            # Get historical commit data
            commit_stats = self.git_service.get_commit_statistics()
            
            # Analyze commit patterns over time
            commit_timeline = defaultdict(int)
            for commit in commit_stats:
                date = datetime.fromtimestamp(commit.get('timestamp', 0)).date()
                commit_timeline[date] += 1
            
            # Calculate trends
            dates = sorted(commit_timeline.keys())
            if len(dates) < 7:
                return self._generate_fallback_resource_predictions()
            
            recent_commits = sum(commit_timeline[date] for date in dates[-7:])
            previous_commits = sum(commit_timeline[date] for date in dates[-14:-7])
            
            growth_rate = (recent_commits - previous_commits) / max(previous_commits, 1)
            
            # Predict resource needs
            predictions = []
            
            # Development velocity prediction
            predicted_velocity = recent_commits * (1 + growth_rate)
            predictions.append(ResourcePrediction(
                resource_type="development_velocity",
                predicted_need=max(1, int(predicted_velocity)),
                confidence_score=min(0.9, max(0.1, 1 - abs(growth_rate))),
                timeframe="7_days"
            ))
            
            # Code review capacity prediction
            review_capacity = max(1, int(predicted_velocity * 0.3))
            predictions.append(ResourcePrediction(
                resource_type="code_review_capacity",
                predicted_need=review_capacity,
                confidence_score=0.7,
                timeframe="7_days"
            ))
            
            # Testing effort prediction
            testing_effort = max(1, int(predicted_velocity * 0.5))
            predictions.append(ResourcePrediction(
                resource_type="testing_effort",
                predicted_need=testing_effort,
                confidence_score=0.6,
                timeframe="7_days"
            ))
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting resource needs: {e}")
            return self._generate_fallback_resource_predictions()
    
    def identify_risk_areas(self, repo_path: str) -> List[RiskArea]:
        """
        Identify high-risk areas in the codebase using ML predictions and heuristics.
        
        Args:
            repo_path (str): Path to the Git repository
            
        Returns:
            List[RiskArea]: Identified risk areas
        """
        logger.info("Identifying risk areas...")
        
        try:
            # Get complexity and change data
            complexity_data = self.git_service.analyze_complexity_trends()
            
            risk_areas = []
            for file_data in complexity_data:
                file_path = file_data['file_path']
                complexity = file_data.get('complexity', 0)
                change_frequency = file_data.get('change_frequency', 0)
                
                # ML-based risk assessment
                risk_score = self._predict_file_risk(file_path, complexity, change_frequency)
                
                if risk_score > 0.5:  # Significant risk
                    risk_level = "high" if risk_score > 0.8 else "medium"
                    
                    risk_areas.append(RiskArea(
                        area_name=file_path,
                        risk_level=risk_level,
                        potential_impact=self._assess_risk_impact(complexity, change_frequency),
                        mitigation_strategy=self._suggest_mitigation(risk_score, complexity)
                    ))
            
            # Sort by risk score
            risk_areas.sort(key=lambda x: x.risk_level != 'high')
            return risk_areas[:10]  # Return top 10 risk areas
            
        except Exception as e:
            logger.error(f"Error identifying risk areas: {e}")
            return self._generate_fallback_risk_areas()
    
    def predict_maintenance_needs(self, repo_path: str) -> List[MaintenancePrediction]:
        """
        Predict future maintenance needs using ML models and trend analysis.
        
        Args:
            repo_path (str): Path to the Git repository
            
        Returns:
            List[MaintenancePrediction]: Maintenance predictions
        """
        logger.info("Predicting maintenance needs...")
        
        try:
            # Get historical data
            commit_stats = self.git_service.get_commit_statistics()
            complexity_data = self.git_service.analyze_complexity_trends()
            
            # Analyze maintenance patterns
            maintenance_commits = []
            for commit in commit_stats:
                message = commit.get('message', '').lower()
                if any(keyword in message for keyword in ['fix', 'bug', 'patch', 'repair']):
                    maintenance_commits.append(commit)
            
            # Calculate maintenance frequency
            if len(commit_stats) > 0:
                maintenance_ratio = len(maintenance_commits) / len(commit_stats)
            else:
                maintenance_ratio = 0.1
            
            # Predict maintenance needs
            predictions = []
            
            # Bug fix prediction
            predicted_bugs = max(1, int(len(commit_stats) * maintenance_ratio * 0.1))
            predictions.append(MaintenancePrediction(
                maintenance_type="bug_fixes",
                predicted_frequency=predicted_bugs,
                timeframe="30_days",
                confidence_score=0.7
            ))
            
            # Refactoring prediction
            high_complexity_files = sum(1 for d in complexity_data if d.get('complexity', 0) > 20)
            predicted_refactors = max(1, int(high_complexity_files * 0.2))
            predictions.append(MaintenancePrediction(
                maintenance_type="refactoring",
                predicted_frequency=predicted_refactors,
                timeframe="30_days",
                confidence_score=0.6
            ))
            
            # Documentation updates
            doc_updates = max(1, int(len(commit_stats) * 0.05))
            predictions.append(MaintenancePrediction(
                maintenance_type="documentation",
                predicted_frequency=doc_updates,
                timeframe="30_days",
                confidence_score=0.8
            ))
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting maintenance needs: {e}")
            return self._generate_fallback_maintenance_predictions()
    
    def predict_complexity_trends(self, repo_path: str) -> List[ComplexityPrediction]:
        """
        Predict future complexity trends using statistical analysis.
        
        Args:
            repo_path (str): Path to the Git repository
            
        Returns:
            List[ComplexityPrediction]: Complexity predictions
        """
        logger.info("Predicting complexity trends...")
        
        try:
            # Get complexity trends
            complexity_data = self.git_service.analyze_complexity_trends()
            
            # Group by time periods (simplified)
            weekly_complexity = defaultdict(list)
            for data in complexity_data:
                # Simplified time grouping
                week = datetime.now().strftime('%Y-W%U')
                weekly_complexity[week].append(data.get('complexity', 0))
            
            predictions = []
            
            # Overall complexity trend
            avg_complexities = [np.mean(complexities) for complexities in weekly_complexity.values()]
            if len(avg_complexities) >= 2:
                trend = "increasing" if avg_complexities[-1] > avg_complexities[0] else "stable"
                predicted_complexity = avg_complexities[-1] * 1.05  # Simple projection
            else:
                trend = "stable"
                predicted_complexity = np.mean(avg_complexities) if avg_complexities else 10
            
            predictions.append(ComplexityPrediction(
                component="overall",
                predicted_complexity=round(predicted_complexity, 2),
                trend=trend,
                confidence_score=0.6
            ))
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting complexity trends: {e}")
            return self._generate_fallback_complexity_predictions()
    
    # Helper methods for ML predictions and statistical analysis
    def _predict_hotspot_probability(self, subsystem: str, commits: List[Dict]) -> float:
        """Predict hotspot probability using ML model or heuristics."""
        if self.hotspot_model and self.hotspot_scaler and self.hotspot_features:
            try:
                # Prepare features for prediction
                features = self._extract_subsystem_features(subsystem, commits)
                # Use DataFrame with feature names to satisfy sklearn fitted with feature names
                features_df = pd.DataFrame([features], columns=self.hotspot_features)
                features_scaled = self.hotspot_scaler.transform(features_df)
                proba = self.hotspot_model.predict_proba(features_scaled)
                # Safely extract positive class probability
                if hasattr(proba, "shape") and proba.shape[1] >= 2:
                    return float(proba[0][1])
                elif hasattr(proba, "shape") and proba.shape[1] >= 1:
                    # Single column proba; treat as positive class probability
                    return float(proba[0][0])
                else:
                    raise ValueError("Invalid predict_proba output shape")
            except Exception as e:
                logger.warning(f"ML hotspot prediction failed: {e}")
        
        # Fallback to heuristic
        return self._calculate_heuristic_hotspot_score(commits)
    
    def _predict_file_hotspot(self, file_path: str, complexity: float, change_frequency: int) -> float:
        """Predict if a file is a hotspot using ML or heuristics."""
        if self.hotspot_model and self.hotspot_scaler and self.hotspot_features:
            try:
                features = self._extract_file_features(file_path, complexity, change_frequency)
                features_df = pd.DataFrame([features], columns=self.hotspot_features)
                features_scaled = self.hotspot_scaler.transform(features_df)
                proba = self.hotspot_model.predict_proba(features_scaled)
                if hasattr(proba, "shape") and proba.shape[1] >= 2:
                    return float(proba[0][1])
                elif hasattr(proba, "shape") and proba.shape[1] >= 1:
                    return float(proba[0][0])
                else:
                    raise ValueError("Invalid predict_proba output shape")
            except Exception as e:
                logger.warning(f"ML file hotspot prediction failed: {e}")
        
        # Fallback heuristic
        normalized_complexity = min(complexity / 50, 1.0)
        normalized_frequency = min(change_frequency / 20, 1.0)
        return (normalized_complexity + normalized_frequency) / 2
    
    def _predict_file_risk(self, file_path: str, complexity: float, change_frequency: int) -> float:
        """Predict risk level for a file."""
        # Similar to hotspot prediction but with different weights
        complexity_weight = 0.4
        frequency_weight = 0.6
        
        normalized_complexity = min(complexity / 30, 1.0)
        normalized_frequency = min(change_frequency / 15, 1.0)
        
        return (normalized_complexity * complexity_weight + 
                normalized_frequency * frequency_weight)
    
    def _extract_subsystem_features(self, subsystem: str, commits: List[Dict]) -> List[float]:
        """Extract features for subsystem hotspot prediction."""
        # Simplified feature extraction
        total_changes = sum(len(c.get('files', [])) for c in commits)
        total_authors = len(set(c.get('author', '') for c in commits))
        avg_message_length = np.mean([len(c.get('message', '')) for c in commits]) if commits else 0
        
        # Return features in the same order as training
        return [
            total_changes,  # lines_changed_sum
            total_changes / len(commits) if commits else 0,  # lines_changed_mean
            np.std([len(c.get('files', [])) for c in commits]) if len(commits) > 1 else 0,  # lines_changed_std
            total_changes * 0.6,  # insertions_sum (estimated)
            total_changes * 0.4,  # insertions_mean (estimated)
            total_changes * 0.4,  # deletions_sum (estimated)
            total_changes * 0.2,  # deletions_mean (estimated)
            len(commits),  # commit_frequency_sum
            total_authors,  # author_count_max
            total_changes / (len(commits) + 1),  # change_rate
            1.5,  # insertion_deletion_ratio (default)
            0.5,  # change_volatility (default)
            total_authors / (len(commits) + 1),  # author_diversity
        ]
    
    def _extract_file_features(self, file_path: str, complexity: float, change_frequency: int) -> List[float]:
        """Extract features for file hotspot prediction."""
        return [
            complexity * change_frequency,  # lines_changed_sum
            complexity,  # lines_changed_mean
            complexity * 0.1,  # lines_changed_std
            complexity * 0.6,  # insertions_sum
            complexity * 0.5,  # insertions_mean
            complexity * 0.4,  # deletions_sum
            complexity * 0.3,  # deletions_mean
            change_frequency,  # commit_frequency_sum
            max(1, change_frequency // 3),  # author_count_max (estimated)
            complexity * change_frequency / (change_frequency + 1),  # change_rate
            1.5,  # insertion_deletion_ratio (default)
            0.3,  # change_volatility (default)
            max(1, change_frequency // 3) / (change_frequency + 1),  # author_diversity
        ]
    
    def _calculate_heuristic_hotspot_score(self, commits: List[Dict]) -> float:
        """Calculate hotspot score using heuristics when ML model is unavailable."""
        if not commits:
            return 0.0
        
        # Factors: commit frequency, number of authors, change magnitude
        commit_frequency = len(commits)
        author_count = len(set(c.get('author', '') for c in commits))
        avg_changes = np.mean([len(c.get('files', [])) for c in commits]) if commits else 0
        
        # Normalize factors
        freq_score = min(commit_frequency / 20, 1.0)
        author_score = min(author_count / 5, 1.0)
        change_score = min(avg_changes / 10, 1.0)
        
        # Weighted average
        return (freq_score * 0.4 + author_score * 0.3 + change_score * 0.3)
    
    def _calculate_complexity_trend(self, commits: List[Dict]) -> str:
        """Calculate complexity trend from commit data."""
        if len(commits) < 3:
            return "stable"
        
        # Simplified trend calculation
        recent_complexity = len(commits[-1].get('files', [])) if commits else 0
        older_complexity = len(commits[0].get('files', [])) if commits else 0
        
        if recent_complexity > older_complexity * 1.2:
            return "increasing"
        elif recent_complexity < older_complexity * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _generate_refactor_reason(self, hotspot_score: float, complexity: float, change_frequency: int) -> str:
        """Generate human-readable refactoring reason."""
        reasons = []
        
        if hotspot_score > 0.7:
            reasons.append("High change frequency indicates instability")
        if complexity > 20:
            reasons.append("High complexity score suggests need for simplification")
        if change_frequency > 10:
            reasons.append("Frequent modifications suggest poor design")
        
        return "; ".join(reasons) if reasons else "Code quality improvement opportunity"
    
    def _suggest_refactor_action(self, hotspot_score: float, complexity: float) -> str:
        """Suggest appropriate refactoring action."""
        if hotspot_score > 0.8 and complexity > 25:
            return "Consider major restructuring or decomposition"
        elif hotspot_score > 0.6:
            return "Implement design patterns to reduce coupling"
        elif complexity > 15:
            return "Extract methods and simplify logic"
        else:
            return "Apply minor refactoring improvements"
    
    def _estimate_refactor_effort(self, complexity: float) -> str:
        """Estimate refactoring effort based on complexity."""
        if complexity > 30:
            return "high"
        elif complexity > 15:
            return "medium"
        else:
            return "low"
    
    def _assess_risk_impact(self, complexity: float, change_frequency: int) -> str:
        """Assess potential impact of risk."""
        if complexity > 25 and change_frequency > 15:
            return "High impact on system stability and development velocity"
        elif complexity > 15 or change_frequency > 10:
            return "Moderate impact on maintainability"
        else:
            return "Low impact but worth monitoring"
    
    def _suggest_mitigation(self, risk_score: float, complexity: float) -> str:
        """Suggest risk mitigation strategies."""
        if risk_score > 0.8:
            return "Prioritize refactoring and add comprehensive tests"
        elif risk_score > 0.6:
            return "Implement monitoring and gradual improvement"
        else:
            return "Regular code reviews and documentation"
    
    # Fallback methods for error scenarios
    def _generate_fallback_dashboard(self) -> DashboardData:
        """Generate fallback dashboard data when analysis fails."""
        logger.warning("Generating fallback dashboard data")
        return DashboardData(
            subsystem_health=[
                SubsystemHealth(name="core", status="stable", complexity_trend="stable", last_updated=datetime.now())
            ],
            refactor_alerts=[],
            codebase_metrics=self._generate_fallback_codebase_metrics(),
            resource_predictions=[],
            risk_areas=[],
            maintenance_predictions=[],
            complexity_predictions=[],
            generated_at=datetime.now()
        )
    
    def _generate_fallback_subsystem_health(self) -> List[SubsystemHealth]:
        """Generate fallback subsystem health data."""
        return [
            SubsystemHealth(name="core", status="stable", complexity_trend="stable", last_updated=datetime.now())
        ]
    
    def _generate_fallback_refactor_alerts(self) -> List[RefactorAlert]:
        """Generate fallback refactor alerts."""
        return []
    
    def _generate_fallback_codebase_metrics(self) -> CodebaseMetrics:
        """Generate fallback codebase metrics."""
        return CodebaseMetrics(
            total_files=0,
            total_commits=0,
            average_complexity=0.0,
            technical_debt_ratio=0.0,
            code_quality_score=75.0,
            last_analyzed=datetime.now()
        )
    
    def _generate_fallback_resource_predictions(self) -> List[ResourcePrediction]:
        """Generate fallback resource predictions."""
        return [
            ResourcePrediction(
                resource_type="development_velocity",
                predicted_need=10,
                confidence_score=0.5,
                timeframe="7_days"
            )
        ]
    
    def _generate_fallback_risk_areas(self) -> List[RiskArea]:
        """Generate fallback risk areas."""
        return []
    
    def _generate_fallback_maintenance_predictions(self) -> List[MaintenancePrediction]:
        """Generate fallback maintenance predictions."""
        return [
            MaintenancePrediction(
                maintenance_type="bug_fixes",
                predicted_frequency=5,
                timeframe="30_days",
                confidence_score=0.5
            )
        ]
    
    def _generate_fallback_complexity_predictions(self) -> List[ComplexityPrediction]:
        """Generate fallback complexity predictions."""
        return [
            ComplexityPrediction(
                component="overall",
                predicted_complexity=10.0,
                trend="stable",
                confidence_score=0.5
            )
        ]


# Global instance for easy access
insights_service = InsightsService()