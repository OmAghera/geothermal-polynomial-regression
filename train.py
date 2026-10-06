import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# Ignore convergence warnings for a cleaner console output
warnings.filterwarnings('ignore')

def build_pipeline(degree, model_type):
    """Builds a scikit-learn pipeline with standard scaling, polynomial features, and regularized regression."""
    if model_type == 'ridge':
        reg = Ridge(max_iter=5000)
    elif model_type == 'lasso':
        reg = Lasso(max_iter=5000)
    elif model_type == 'elasticnet':
        reg = ElasticNet(max_iter=5000)
    else:
        raise ValueError("Invalid model type")
        
    return Pipeline([
        ('scaler_in', StandardScaler()),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler_poly', StandardScaler()),
        ('reg', reg)
    ])

def train_and_evaluate(var_id, train_file, test_file, max_degree):
    print(f"\n{'='*40}")
    print(f"--- Starting Phase {var_id} (max_degree={max_degree}) ---")
    print(f"{'='*40}")
    
    # Load datasets
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    X_train = train_df.drop('y', axis=1)
    y_train = train_df['y']
    X_test = test_df # Test set does not have 'y'
    
    # Define cross-validation strategy (5-fold)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    best_score = float('inf')
    best_params = None
    best_model = None
    
    # We will search over these regularization strengths
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
    l1_ratios = [0.2, 0.5, 0.8]
    
    for degree in range(1, max_degree + 1):
        print(f"Evaluating Polynomial Degree {degree}...")
        for model_type in ['ridge', 'lasso', 'elasticnet']:
            pipeline = build_pipeline(degree, model_type)
            
            # Setup parameter grid for this model type
            param_grid = {'reg__alpha': alphas}
            if model_type == 'elasticnet':
                param_grid['reg__l1_ratio'] = l1_ratios
                
            # Perform Grid Search with Cross Validation
            search = GridSearchCV(
                pipeline, 
                param_grid, 
                cv=cv, 
                scoring='neg_mean_squared_error', 
                n_jobs=-1 # Use all available CPU cores for speed
            )
            search.fit(X_train, y_train)
            
            # Scoring is negative MSE, so we negate it back
            cv_mse = -search.best_score_
            
            if cv_mse < best_score:
                best_score = cv_mse
                best_params = {
                    'degree': degree,
                    'model': model_type,
                    'params': search.best_params_,
                    'cv_mse': cv_mse
                }
                best_model = search.best_estimator_

    print("\n[SUCCESS] Best Configuration Found:")
    print(f"  - Model Type : {best_params['model'].capitalize()}")
    print(f"  - Degree     : {best_params['degree']}")
    print(f"  - Parameters : {best_params['params']}")
    print(f"  - CV MSE     : {best_params['cv_mse']:.6f}")
    
    # Generate predictions on the unseen test set
    predictions = best_model.predict(X_test)
    pred_file = f"BT2024088_pred_var{var_id}.csv"
    
    # Save to CSV in the requested format
    pd.DataFrame({'y': predictions}).to_csv(pred_file, index=False)
    print(f"\nSaved predictions to -> {pred_file}")
    
    return best_model, best_params

if __name__ == '__main__':
    # Phase 1: Turbine Optimization
    train_file_1 = "OneDrive_1_10-6-2026/BT2024088_train_var1.csv"
    test_file_1 = "OneDrive_1_10-6-2026/BT2024088_test_var1.csv"
    train_and_evaluate(var_id=1, train_file=train_file_1, test_file=test_file_1, max_degree=10)
    
    # Phase 2: Reservoir Mapping
    train_file_2 = "OneDrive_1_10-6-2026/BT2024088_train_var2.csv"
    test_file_2 = "OneDrive_1_10-6-2026/BT2024088_test_var2.csv"
    train_and_evaluate(var_id=2, train_file=train_file_2, test_file=test_file_2, max_degree=20)
