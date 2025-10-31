"""
Hyperparameter optimization framework for model tuning.
Implements Bayesian optimization, grid search, and random search.
"""

import numpy as np
from typing import Dict, Any, List, Callable, Optional, Tuple
from sklearn.model_selection import cross_val_score, StratifiedKFold
import logging
import time

from src.model.base_model import BaseModel, ModelConfig
from src.model.model_factory import ModelFactory

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Base class for hyperparameter optimization.
    Provides common functionality for different optimization strategies.
    """
    
    def __init__(self, model_type: str, param_space: Dict[str, Any],
                 scoring: str = 'accuracy', cv: int = 5, n_jobs: int = -1):
        """
        Initialize the optimizer.
        
        Args:
            model_type: Type of model to optimize
            param_space: Dictionary defining parameter search space
            scoring: Scoring metric for evaluation
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
        """
        self.model_type = model_type
        self.param_space = param_space
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.best_params = None
        self.best_score = None
        self.optimization_history = []
        
    def optimize(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run optimization. To be implemented by subclasses.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Dictionary with optimization results
        """
        raise NotImplementedError("Subclasses must implement optimize()")
    
    def _evaluate_params(self, params: Dict[str, Any], X: np.ndarray, 
                        y: np.ndarray) -> float:
        """
        Evaluate a set of hyperparameters using cross-validation.
        
        Args:
            params: Hyperparameters to evaluate
            X: Training features
            y: Training labels
            
        Returns:
            Mean cross-validation score
        """
        try:
            # Create model with these parameters
            model = ModelFactory.create_model(self.model_type, **params)
            
            # Use sklearn's cross_val_score for traditional ML models
            if hasattr(model, 'model') and model.model is not None:
                scores = cross_val_score(
                    model.model, X, y,
                    cv=self.cv,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs
                )
                mean_score = np.mean(scores)
            else:
                # For custom models, do manual cross-validation
                mean_score = self._manual_cross_validation(model, X, y)
            
            # Store in history
            self.optimization_history.append({
                'params': params.copy(),
                'score': mean_score
            })
            
            return mean_score
            
        except Exception as e:
            logger.warning(f"Error evaluating params {params}: {e}")
            return -np.inf
    
    def _manual_cross_validation(self, model: BaseModel, X: np.ndarray, 
                                 y: np.ndarray) -> float:
        """Manual cross-validation for custom models."""
        kfold = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kfold.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model.train(X_train, y_train)
            
            # Evaluate
            predictions = model.predict(X_val)
            accuracy = np.mean(predictions == y_val)
            scores.append(accuracy)
        
        return np.mean(scores)
    
    def get_best_model(self) -> BaseModel:
        """Create and return a model with the best parameters."""
        if self.best_params is None:
            raise ValueError("No optimization has been run yet.")
        
        return ModelFactory.create_model(self.model_type, **self.best_params)


class GridSearchOptimizer(HyperparameterOptimizer):
    """Grid search hyperparameter optimization."""
    
    def optimize(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Perform grid search over parameter space.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting grid search for {self.model_type}")
        start_time = time.time()
        
        # Generate all parameter combinations
        param_combinations = self._generate_param_grid()
        total_combinations = len(param_combinations)
        
        logger.info(f"Evaluating {total_combinations} parameter combinations")
        
        best_score = -np.inf
        best_params = None
        
        for i, params in enumerate(param_combinations):
            logger.info(f"Evaluating combination {i+1}/{total_combinations}: {params}")
            
            score = self._evaluate_params(params, X, y)
            
            if score > best_score:
                best_score = score
                best_params = params
                logger.info(f"New best score: {best_score:.4f}")
        
        self.best_params = best_params
        self.best_score = best_score
        
        elapsed_time = time.time() - start_time
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'n_iterations': total_combinations,
            'elapsed_time': elapsed_time,
            'optimization_history': self.optimization_history
        }
    
    def _generate_param_grid(self) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters."""
        import itertools
        
        # Get parameter names and values
        param_names = list(self.param_space.keys())
        param_values = [self.param_space[name] for name in param_names]
        
        # Generate all combinations
        combinations = []
        for values in itertools.product(*param_values):
            param_dict = dict(zip(param_names, values))
            combinations.append(param_dict)
        
        return combinations


class RandomSearchOptimizer(HyperparameterOptimizer):
    """Random search hyperparameter optimization."""
    
    def __init__(self, model_type: str, param_space: Dict[str, Any],
                 n_iter: int = 50, scoring: str = 'accuracy', 
                 cv: int = 5, n_jobs: int = -1):
        """
        Initialize random search optimizer.
        
        Args:
            model_type: Type of model to optimize
            param_space: Dictionary defining parameter distributions
            n_iter: Number of random samples to try
            scoring: Scoring metric
            cv: Number of CV folds
            n_jobs: Number of parallel jobs
        """
        super().__init__(model_type, param_space, scoring, cv, n_jobs)
        self.n_iter = n_iter
    
    def optimize(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Perform random search over parameter space.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting random search for {self.model_type}")
        start_time = time.time()
        
        best_score = -np.inf
        best_params = None
        
        for i in range(self.n_iter):
            # Sample random parameters
            params = self._sample_params()
            
            logger.info(f"Iteration {i+1}/{self.n_iter}: {params}")
            
            score = self._evaluate_params(params, X, y)
            
            if score > best_score:
                best_score = score
                best_params = params
                logger.info(f"New best score: {best_score:.4f}")
        
        self.best_params = best_params
        self.best_score = best_score
        
        elapsed_time = time.time() - start_time
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'n_iterations': self.n_iter,
            'elapsed_time': elapsed_time,
            'optimization_history': self.optimization_history
        }
    
    def _sample_params(self) -> Dict[str, Any]:
        """Sample random parameters from distributions."""
        params = {}
        
        for param_name, param_dist in self.param_space.items():
            if isinstance(param_dist, list):
                # Discrete choice
                params[param_name] = np.random.choice(param_dist)
            elif isinstance(param_dist, tuple) and len(param_dist) == 2:
                # Continuous range (min, max)
                if isinstance(param_dist[0], int):
                    params[param_name] = np.random.randint(param_dist[0], param_dist[1] + 1)
                else:
                    params[param_name] = np.random.uniform(param_dist[0], param_dist[1])
            elif callable(param_dist):
                # Custom distribution function
                params[param_name] = param_dist()
            else:
                params[param_name] = param_dist
        
        return params


class BayesianOptimizer(HyperparameterOptimizer):
    """Bayesian optimization using Optuna."""
    
    def __init__(self, model_type: str, param_space: Dict[str, Any],
                 n_trials: int = 50, scoring: str = 'accuracy',
                 cv: int = 5, n_jobs: int = -1):
        """
        Initialize Bayesian optimizer.
        
        Args:
            model_type: Type of model to optimize
            param_space: Dictionary defining parameter search space
            n_trials: Number of optimization trials
            scoring: Scoring metric
            cv: Number of CV folds
            n_jobs: Number of parallel jobs
        """
        super().__init__(model_type, param_space, scoring, cv, n_jobs)
        self.n_trials = n_trials
        self.study = None
    
    def optimize(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Perform Bayesian optimization using Optuna.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Dictionary with optimization results
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError(
                "Optuna not installed. Install with: pip install optuna"
            )
        
        logger.info(f"Starting Bayesian optimization for {self.model_type}")
        start_time = time.time()
        
        # Create objective function
        def objective(trial):
            # Sample parameters
            params = self._sample_params_optuna(trial)
            
            # Evaluate
            score = self._evaluate_params(params, X, y)
            
            return score
        
        # Create study and optimize
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        elapsed_time = time.time() - start_time
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': self.n_trials,
            'elapsed_time': elapsed_time,
            'optimization_history': self.optimization_history,
            'study': self.study
        }
    
    def _sample_params_optuna(self, trial) -> Dict[str, Any]:
        """Sample parameters using Optuna trial."""
        params = {}
        
        for param_name, param_config in self.param_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get('type', 'categorical')
                
                if param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_type == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
            elif isinstance(param_config, list):
                # Simple categorical
                params[param_name] = trial.suggest_categorical(param_name, param_config)
            else:
                params[param_name] = param_config
        
        return params
    
    def plot_optimization_history(self):
        """Plot optimization history using Optuna."""
        if self.study is None:
            raise ValueError("No optimization has been run yet.")
        
        try:
            import optuna.visualization as vis
            
            # Plot optimization history
            fig1 = vis.plot_optimization_history(self.study)
            fig1.show()
            
            # Plot parameter importances
            fig2 = vis.plot_param_importances(self.study)
            fig2.show()
            
        except ImportError:
            logger.warning("Plotly not installed. Cannot create visualizations.")


class AutoTuner:
    """
    Automated hyperparameter tuning pipeline.
    Combines multiple optimization strategies.
    """
    
    def __init__(self, model_type: str, optimization_budget: str = 'medium'):
        """
        Initialize auto-tuner.
        
        Args:
            model_type: Type of model to tune
            optimization_budget: 'low', 'medium', or 'high'
        """
        self.model_type = model_type
        self.optimization_budget = optimization_budget
        self.best_optimizer = None
        
    def tune(self, X: np.ndarray, y: np.ndarray, 
             param_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Automatically tune hyperparameters.
        
        Args:
            X: Training features
            y: Training labels
            param_space: Optional custom parameter space
            
        Returns:
            Dictionary with tuning results
        """
        if param_space is None:
            param_space = self._get_default_param_space()
        
        # Choose optimization strategy based on budget
        if self.optimization_budget == 'low':
            # Quick random search
            optimizer = RandomSearchOptimizer(
                self.model_type, param_space, n_iter=20, cv=3
            )
        elif self.optimization_budget == 'high':
            # Thorough Bayesian optimization
            optimizer = BayesianOptimizer(
                self.model_type, param_space, n_trials=100, cv=5
            )
        else:
            # Medium: Bayesian with moderate trials
            optimizer = BayesianOptimizer(
                self.model_type, param_space, n_trials=50, cv=5
            )
        
        logger.info(f"Auto-tuning {self.model_type} with {self.optimization_budget} budget")
        
        result = optimizer.optimize(X, y)
        self.best_optimizer = optimizer
        
        return result
    
    def _get_default_param_space(self) -> Dict[str, Any]:
        """Get default parameter space for the model type."""
        default_spaces = {
            'random_forest': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10}
            },
            'xgboost': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0}
            },
            'lightgbm': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'max_depth': {'type': 'int', 'low': 3, 'high': 15},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'num_leaves': {'type': 'int', 'low': 20, 'high': 150}
            },
            'logistic_regression': {
                'C': {'type': 'float', 'low': 0.001, 'high': 100, 'log': True},
                'max_iter': {'type': 'int', 'low': 100, 'high': 1000}
            }
        }
        
        return default_spaces.get(self.model_type, {})
    
    def get_best_model(self) -> BaseModel:
        """Get the best model from tuning."""
        if self.best_optimizer is None:
            raise ValueError("No tuning has been performed yet.")
        
        return self.best_optimizer.get_best_model()
