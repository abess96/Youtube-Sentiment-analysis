"""
Example usage of hyperparameter optimization framework.
Demonstrates grid search, random search, Bayesian optimization, and auto-tuning.
"""

import numpy as np
from src.model.hyperparameter_optimizer import (
    GridSearchOptimizer, RandomSearchOptimizer, 
    BayesianOptimizer, AutoTuner
)


def create_sample_data(num_samples=1000, num_features=20, num_classes=3):
    """Create sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(num_samples, num_features)
    y = np.random.randint(0, num_classes, num_samples)
    return X, y


def example_grid_search():
    """Example of grid search optimization."""
    print("\n" + "="*60)
    print("Grid Search Optimization Example")
    print("="*60)
    
    # Create sample data
    X, y = create_sample_data(num_samples=500)
    
    # Define parameter grid
    param_space = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5]
    }
    
    # Create optimizer
    optimizer = GridSearchOptimizer(
        model_type='random_forest',
        param_space=param_space,
        scoring='accuracy',
        cv=3
    )
    
    print(f"Parameter space: {param_space}")
    print(f"Total combinations: {3 * 3 * 2} = 18")
    
    # Run optimization
    print("\nRunning grid search...")
    result = optimizer.optimize(X, y)
    
    print(f"\nOptimization completed!")
    print(f"Best parameters: {result['best_params']}")
    print(f"Best score: {result['best_score']:.4f}")
    print(f"Time elapsed: {result['elapsed_time']:.2f} seconds")
    print(f"Iterations: {result['n_iterations']}")
    
    # Get best model
    best_model = optimizer.get_best_model()
    print(f"\nBest model created: {best_model.config.model_name}")


def example_random_search():
    """Example of random search optimization."""
    print("\n" + "="*60)
    print("Random Search Optimization Example")
    print("="*60)
    
    X, y = create_sample_data(num_samples=500)
    
    # Define parameter distributions
    param_space = {
        'n_estimators': (50, 200),  # Range for integers
        'max_depth': (3, 20),
        'learning_rate': (0.01, 0.3),  # Range for floats
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]  # Discrete choices
    }
    
    # Create optimizer
    optimizer = RandomSearchOptimizer(
        model_type='xgboost',
        param_space=param_space,
        n_iter=30,  # Try 30 random combinations
        scoring='accuracy',
        cv=3
    )
    
    print(f"Parameter space: {param_space}")
    print(f"Number of iterations: 30")
    
    # Run optimization
    print("\nRunning random search...")
    result = optimizer.optimize(X, y)
    
    print(f"\nOptimization completed!")
    print(f"Best parameters: {result['best_params']}")
    print(f"Best score: {result['best_score']:.4f}")
    print(f"Time elapsed: {result['elapsed_time']:.2f} seconds")
    
    # Show top 5 parameter combinations
    history = sorted(result['optimization_history'], 
                    key=lambda x: x['score'], reverse=True)
    print("\nTop 5 parameter combinations:")
    for i, entry in enumerate(history[:5], 1):
        print(f"{i}. Score: {entry['score']:.4f}, Params: {entry['params']}")


def example_bayesian_optimization():
    """Example of Bayesian optimization using Optuna."""
    print("\n" + "="*60)
    print("Bayesian Optimization Example (Optuna)")
    print("="*60)
    
    try:
        import optuna
    except ImportError:
        print("Optuna not installed. Skipping Bayesian optimization example.")
        print("Install with: pip install optuna")
        return
    
    X, y = create_sample_data(num_samples=500)
    
    # Define parameter space for Bayesian optimization
    param_space = {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
        'max_depth': {'type': 'int', 'low': 3, 'high': 15},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
        'num_leaves': {'type': 'int', 'low': 20, 'high': 150}
    }
    
    # Create optimizer
    optimizer = BayesianOptimizer(
        model_type='lightgbm',
        param_space=param_space,
        n_trials=40,
        scoring='accuracy',
        cv=3
    )
    
    print(f"Parameter space: {param_space}")
    print(f"Number of trials: 40")
    
    # Run optimization
    print("\nRunning Bayesian optimization...")
    result = optimizer.optimize(X, y)
    
    print(f"\nOptimization completed!")
    print(f"Best parameters: {result['best_params']}")
    print(f"Best score: {result['best_score']:.4f}")
    print(f"Time elapsed: {result['elapsed_time']:.2f} seconds")
    
    # Show optimization progress
    print("\nOptimization progress (first 10 trials):")
    for i, entry in enumerate(result['optimization_history'][:10], 1):
        print(f"Trial {i}: Score = {entry['score']:.4f}")
    
    # Optionally plot optimization history
    # optimizer.plot_optimization_history()


def example_auto_tuner():
    """Example of automated hyperparameter tuning."""
    print("\n" + "="*60)
    print("Auto-Tuner Example")
    print("="*60)
    
    X, y = create_sample_data(num_samples=500)
    
    # Test different budget levels
    budgets = ['low', 'medium']
    
    for budget in budgets:
        print(f"\n{'='*60}")
        print(f"Auto-tuning with '{budget}' budget")
        print(f"{'='*60}")
        
        # Create auto-tuner
        tuner = AutoTuner(
            model_type='random_forest',
            optimization_budget=budget
        )
        
        # Run tuning
        print(f"Starting auto-tuning...")
        result = tuner.tune(X, y)
        
        print(f"\nTuning completed!")
        print(f"Best parameters: {result['best_params']}")
        print(f"Best score: {result['best_score']:.4f}")
        print(f"Time elapsed: {result['elapsed_time']:.2f} seconds")
        
        # Get best model
        best_model = tuner.get_best_model()
        print(f"Best model ready for use: {best_model.config.model_name}")


def example_comparison():
    """Compare different optimization strategies."""
    print("\n" + "="*60)
    print("Optimization Strategy Comparison")
    print("="*60)
    
    X, y = create_sample_data(num_samples=500)
    
    results = {}
    
    # 1. Grid Search (small grid)
    print("\n1. Running Grid Search...")
    grid_params = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }
    grid_opt = GridSearchOptimizer('random_forest', grid_params, cv=3)
    grid_result = grid_opt.optimize(X, y)
    results['Grid Search'] = {
        'score': grid_result['best_score'],
        'time': grid_result['elapsed_time'],
        'iterations': grid_result['n_iterations']
    }
    
    # 2. Random Search
    print("\n2. Running Random Search...")
    random_params = {
        'n_estimators': (50, 200),
        'max_depth': (3, 20),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 5)
    }
    random_opt = RandomSearchOptimizer('random_forest', random_params, n_iter=20, cv=3)
    random_result = random_opt.optimize(X, y)
    results['Random Search'] = {
        'score': random_result['best_score'],
        'time': random_result['elapsed_time'],
        'iterations': random_result['n_iterations']
    }
    
    # 3. Bayesian Optimization (if available)
    try:
        import optuna
        print("\n3. Running Bayesian Optimization...")
        bayes_params = {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 200},
            'max_depth': {'type': 'int', 'low': 3, 'high': 20},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 10},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 5}
        }
        bayes_opt = BayesianOptimizer('random_forest', bayes_params, n_trials=20, cv=3)
        bayes_result = bayes_opt.optimize(X, y)
        results['Bayesian'] = {
            'score': bayes_result['best_score'],
            'time': bayes_result['elapsed_time'],
            'iterations': bayes_result['n_trials']
        }
    except ImportError:
        print("\n3. Bayesian Optimization skipped (Optuna not installed)")
    
    # Print comparison
    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)
    print(f"{'Method':<20} {'Best Score':<12} {'Time (s)':<12} {'Iterations':<12}")
    print("-" * 60)
    for method, metrics in results.items():
        print(f"{method:<20} {metrics['score']:<12.4f} "
              f"{metrics['time']:<12.2f} {metrics['iterations']:<12}")


def example_custom_param_space():
    """Example with custom parameter space."""
    print("\n" + "="*60)
    print("Custom Parameter Space Example")
    print("="*60)
    
    X, y = create_sample_data(num_samples=500)
    
    # Define custom parameter space for LightGBM
    param_space = {
        'n_estimators': {'type': 'int', 'low': 100, 'high': 500},
        'max_depth': {'type': 'int', 'low': 5, 'high': 20},
        'learning_rate': {'type': 'float', 'low': 0.001, 'high': 0.1, 'log': True},
        'num_leaves': {'type': 'int', 'low': 31, 'high': 255},
        'min_child_samples': {'type': 'int', 'low': 10, 'high': 50},
        'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0},
        'colsample_bytree': {'type': 'float', 'low': 0.5, 'high': 1.0}
    }
    
    print("Custom parameter space for LightGBM:")
    for param, config in param_space.items():
        print(f"  {param}: {config}")
    
    # Use auto-tuner with custom space
    tuner = AutoTuner('lightgbm', optimization_budget='medium')
    
    print("\nRunning optimization with custom parameter space...")
    result = tuner.tune(X, y, param_space=param_space)
    
    print(f"\nOptimization completed!")
    print(f"Best parameters: {result['best_params']}")
    print(f"Best score: {result['best_score']:.4f}")


if __name__ == "__main__":
    print("Hyperparameter Optimization Examples")
    print("="*60)
    
    try:
        # Run examples
        example_grid_search()
        example_random_search()
        example_bayesian_optimization()
        example_auto_tuner()
        example_comparison()
        example_custom_param_space()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
