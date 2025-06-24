"""
Bioactivity Prediction Model for BioPath SHAP Demo

This module implements enterprise-grade ensemble machine learning models for
natural compound bioactivity prediction, with hyperparameter optimization,
model validation, and deployment-ready features.
"""

import logging
import warnings
import pickle
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    BaggingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    cross_val_score, 
    StratifiedKFold,
    GridSearchCV,
    train_test_split
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import optuna
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class ModelPerformance:
    """Data class for storing model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    cross_val_mean: float
    cross_val_std: float
    confusion_matrix: List[List[int]]
    training_time: float
    n_samples: int
    n_features: int


@dataclass
class ModelConfig:
    """Configuration for bioactivity prediction models."""
    model_type: str = 'ensemble'
    use_hyperparameter_optimization: bool = True
    optimization_trials: int = 100
    cross_validation_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    feature_selection: bool = True
    feature_importance_threshold: float = 0.001
    ensemble_voting: str = 'soft'  # 'hard' or 'soft'
    

class BioactivityPredictor:
    """
    Enterprise-grade ensemble model for natural compound bioactivity prediction.
    
    This class implements state-of-the-art machine learning techniques optimized
    for molecular bioactivity prediction with full model lifecycle management.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the bioactivity predictor.
        
        Args:
            config: Model configuration object
        """
        self.config = config or ModelConfig()
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.performance_metrics = None
        self.training_history = []
        self.is_fitted = False
        
        # Model selection and hyperparameters
        self._init_base_models()
        self._init_hyperparameter_spaces()
        
        logging.info(f"BioactivityPredictor initialized with {self.config.model_type} approach")
    
    def _init_base_models(self):
        """Initialize base models for ensemble learning."""
        self.base_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.config.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.config.random_state
            ),
            'logistic_regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.config.random_state
            ),
            'svm': SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=self.config.random_state
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=self.config.random_state
            )
        }
    
    def _init_hyperparameter_spaces(self):
        """Initialize hyperparameter optimization spaces."""
        self.param_spaces = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'lbfgs']
            }
        }
    
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray, 
            feature_names: Optional[List[str]] = None,
            validation_split: bool = True) -> 'BioactivityPredictor':
        """
        Train the bioactivity prediction model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            feature_names: Optional feature names for interpretability
            validation_split: Whether to use train/validation split
            
        Returns:
            Self for method chaining
        """
        start_time = datetime.now()
        logging.info(f"Training bioactivity predictor on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Store feature names
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # Split data if validation is requested
        if validation_split:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, 
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        # Feature scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        # Feature selection if requested
        if self.config.feature_selection:
            X_train_scaled, selected_features = self._select_features(X_train_scaled, y_train)
            self.selected_feature_indices = selected_features
            self.feature_names = [self.feature_names[i] for i in selected_features]
            if X_val is not None:
                X_val_scaled = X_val_scaled[:, selected_features]
        
        # Model training based on configuration
        if self.config.model_type == 'ensemble':
            self.model = self._train_ensemble_model(X_train_scaled, y_train)
        elif self.config.model_type == 'optimized':
            self.model = self._train_optimized_model(X_train_scaled, y_train)
        else:
            # Single model training
            self.model = self._train_single_model(X_train_scaled, y_train, self.config.model_type)
        
        # Calculate performance metrics
        if X_val is not None:
            self.performance_metrics = self._calculate_performance(
                X_train_scaled, y_train, X_val_scaled, y_val
            )
        else:
            # Use cross-validation for performance estimation
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train,
                cv=StratifiedKFold(n_splits=self.config.cross_validation_folds, shuffle=True, random_state=self.config.random_state),
                scoring='accuracy'
            )
            self.performance_metrics = ModelPerformance(
                accuracy=cv_scores.mean(),
                precision=0.0,  # Will be calculated during prediction
                recall=0.0,
                f1_score=0.0,
                roc_auc=0.0,
                cross_val_mean=cv_scores.mean(),
                cross_val_std=cv_scores.std(),
                confusion_matrix=[[0, 0], [0, 0]],
                training_time=(datetime.now() - start_time).total_seconds(),
                n_samples=X.shape[0],
                n_features=len(self.feature_names)
            )
        
        # Store training history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'performance': asdict(self.performance_metrics),
            'config': asdict(self.config)
        })
        
        self.is_fitted = True
        training_time = (datetime.now() - start_time).total_seconds()
        
        logging.info(f"Model training completed in {training_time:.2f} seconds")
        logging.info(f"Cross-validation accuracy: {self.performance_metrics.cross_val_mean:.3f} (+/- {self.performance_metrics.cross_val_std * 2:.3f})")
        
        return self
    
    def _select_features(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Select important features using ensemble-based feature importance.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Tuple of (selected_features, selected_indices)
        """
        # Use Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=50, random_state=self.config.random_state)
        rf.fit(X, y)
        
        # Select features above threshold
        feature_importance = rf.feature_importances_
        selected_indices = np.where(feature_importance >= self.config.feature_importance_threshold)[0]
        
        # Ensure we have at least 10 features
        if len(selected_indices) < 10:
            selected_indices = np.argsort(feature_importance)[-10:]
        
        logging.info(f"Feature selection: {len(selected_indices)}/{X.shape[1]} features selected")
        
        return X[:, selected_indices], selected_indices.tolist()
    
    def _train_ensemble_model(self, X: np.ndarray, y: np.ndarray) -> VotingClassifier:
        """
        Train an ensemble model using voting classifier.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Trained voting classifier
        """
        # Select subset of models for ensemble
        ensemble_models = [
            ('rf', self.base_models['random_forest']),
            ('gb', self.base_models['gradient_boosting']),
            ('lr', self.base_models['logistic_regression'])
        ]
        
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=ensemble_models,
            voting=self.config.ensemble_voting
        )
        
        # Train ensemble
        voting_clf.fit(X, y)
        
        logging.info(f"Ensemble model trained with {len(ensemble_models)} base models")
        
        return voting_clf
    
    def _train_optimized_model(self, X: np.ndarray, y: np.ndarray):
        """
        Train a hyperparameter-optimized model using Optuna.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Optimized model
        """
        def objective(trial):
            # Select model type
            model_name = trial.suggest_categorical('model', ['random_forest', 'gradient_boosting', 'logistic_regression'])
            
            # Get model with suggested hyperparameters
            if model_name == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                    max_depth=trial.suggest_int('max_depth', 5, 20),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 4),
                    random_state=self.config.random_state,
                    n_jobs=-1
                )
            elif model_name == 'gradient_boosting':
                model = GradientBoostingClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                    max_depth=trial.suggest_int('max_depth', 3, 10),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    subsample=trial.suggest_float('subsample', 0.6, 1.0),
                    random_state=self.config.random_state
                )
            else:  # logistic_regression
                model = LogisticRegression(
                    C=trial.suggest_float('C', 0.01, 100, log=True),
                    max_iter=1000,
                    random_state=self.config.random_state
                )
            
            # Cross-validation score
            cv_scores = cross_val_score(
                model, X, y,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.random_state),
                scoring='f1'
            )
            
            return cv_scores.mean()
        
        # Optimize hyperparameters
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.optimization_trials, show_progress_bar=True)
        
        # Train best model
        best_params = study.best_params
        model_name = best_params.pop('model')
        
        if model_name == 'random_forest':
            best_model = RandomForestClassifier(**best_params, random_state=self.config.random_state, n_jobs=-1)
        elif model_name == 'gradient_boosting':
            best_model = GradientBoostingClassifier(**best_params, random_state=self.config.random_state)
        else:
            best_model = LogisticRegression(**best_params, max_iter=1000, random_state=self.config.random_state)
        
        best_model.fit(X, y)
        
        logging.info(f"Hyperparameter optimization completed. Best model: {model_name}")
        logging.info(f"Best F1 score: {study.best_value:.3f}")
        
        return best_model
    
    def _train_single_model(self, X: np.ndarray, y: np.ndarray, model_type: str):
        """
        Train a single model type.
        
        Args:
            X: Training features
            y: Training labels
            model_type: Type of model to train
            
        Returns:
            Trained model
        """
        if model_type not in self.base_models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = self.base_models[model_type]
        model.fit(X, y)
        
        logging.info(f"Single model training completed: {model_type}")
        
        return model
    
    def _calculate_performance(self, 
                              X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray) -> ModelPerformance:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            ModelPerformance object with all metrics
        """
        # Predictions
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # Cross-validation on training set
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=StratifiedKFold(n_splits=self.config.cross_validation_folds, shuffle=True, random_state=self.config.random_state),
            scoring='accuracy'
        )
        
        # Calculate metrics
        performance = ModelPerformance(
            accuracy=accuracy_score(y_val, y_pred),
            precision=precision_score(y_val, y_pred, average='binary'),
            recall=recall_score(y_val, y_pred, average='binary'),
            f1_score=f1_score(y_val, y_pred, average='binary'),
            roc_auc=roc_auc_score(y_val, y_pred_proba) if y_pred_proba is not None else 0.0,
            cross_val_mean=cv_scores.mean(),
            cross_val_std=cv_scores.std(),
            confusion_matrix=confusion_matrix(y_val, y_pred).tolist(),
            training_time=0.0,  # Will be set by caller
            n_samples=len(X_train) + len(X_val),
            n_features=X_train.shape[1]
        )
        
        return performance
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Apply preprocessing
        X_scaled = self.scaler.transform(X)
        if hasattr(self, 'selected_feature_indices'):
            X_scaled = X_scaled[:, self.selected_feature_indices]
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability prediction")
        
        # Apply preprocessing
        X_scaled = self.scaler.transform(X)
        if hasattr(self, 'selected_feature_indices'):
            X_scaled = X_scaled[:, self.selected_feature_indices]
        
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Extract feature importance based on model type
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'estimators_'):
            # For ensemble models, try to get from first estimator
            if hasattr(self.model.estimators_[0], 'feature_importances_'):
                importances = self.model.estimators_[0].feature_importances_
            else:
                # Fallback to coefficient magnitude for linear models
                importances = np.abs(self.model.estimators_[0].coef_[0])
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            # No feature importance available
            importances = np.ones(len(self.feature_names)) / len(self.feature_names)
        
        return dict(zip(self.feature_names, importances))
    
    def save_model(self, filepath: Union[str, Path]):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model components
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'performance_metrics': asdict(self.performance_metrics),
            'config': asdict(self.config),
            'training_history': self.training_history,
            'selected_feature_indices': getattr(self, 'selected_feature_indices', None)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logging.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Union[str, Path]):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.performance_metrics = ModelPerformance(**model_data['performance_metrics'])
        self.config = ModelConfig(**model_data['config'])
        self.training_history = model_data['training_history']
        if 'selected_feature_indices' in model_data:
            self.selected_feature_indices = model_data['selected_feature_indices']
        
        self.is_fitted = True
        
        logging.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the model.
        
        Returns:
            Dictionary containing model summary information
        """
        if not self.is_fitted:
            return {"status": "Model not fitted"}
        
        summary = {
            "model_type": self.config.model_type,
            "is_fitted": self.is_fitted,
            "n_features": len(self.feature_names),
            "feature_selection_used": hasattr(self, 'selected_feature_indices'),
            "performance": asdict(self.performance_metrics),
            "config": asdict(self.config),
            "training_history_length": len(self.training_history)
        }
        
        # Add model-specific information
        if hasattr(self.model, 'n_estimators'):
            summary["n_estimators"] = self.model.n_estimators
        if hasattr(self.model, 'estimators_'):
            summary["ensemble_size"] = len(self.model.estimators_)
        
        return summary


# Utility functions for external use
def create_bioactivity_model(model_type: str = 'ensemble', 
                           optimize_hyperparameters: bool = True) -> BioactivityPredictor:
    """
    Create a bioactivity prediction model with optimal configuration.
    
    Args:
        model_type: Type of model ('ensemble', 'optimized', or specific model name)
        optimize_hyperparameters: Whether to use hyperparameter optimization
        
    Returns:
        Configured BioactivityPredictor instance
    """
    config = ModelConfig(
        model_type=model_type,
        use_hyperparameter_optimization=optimize_hyperparameters,
        optimization_trials=50 if optimize_hyperparameters else 0
    )
    
    return BioactivityPredictor(config)


def evaluate_model_performance(model: BioactivityPredictor, 
                             X_test: np.ndarray, 
                             y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained BioactivityPredictor
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of performance metrics
    """
    if not model.is_fitted:
        raise ValueError("Model must be fitted before evaluation")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model.model, 'predict_proba') else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.0
    }
    
    return metrics


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BioPath Bioactivity Predictor")
    parser.add_argument('--data', type=str, required=True, help='Path to training data CSV')
    parser.add_argument('--target', type=str, required=True, help='Target column name')
    parser.add_argument('--model-type', type=str, default='ensemble', help='Model type')
    parser.add_argument('--output', type=str, default='model.pkl', help='Output model path')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.data)
    X = df.drop(columns=[args.target]).values
    y = df[args.target].values
    
    # Create and train model
    model = create_bioactivity_model(args.model_type)
    model.fit(X, y)
    
    # Save model
    model.save_model(args.output)
    
    print(f"Model trained and saved to {args.output}")
    print(f"Performance: {model.performance_metrics.cross_val_mean:.3f} (+/- {model.performance_metrics.cross_val_std * 2:.3f})")


if __name__ == "__main__":
    main()
