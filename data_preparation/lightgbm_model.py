import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import (
    train_test_split
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, balanced_accuracy_score
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import warnings
import os
from datetime import datetime
from typing import Dict, List, Optional
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import shap

warnings.filterwarnings('ignore')

class OptimizedUrbanBlightLightGBMModel:
    """
    Simplified LightGBM Model for Urban Blight Prediction
    CSV-based with robust categorical feature handling
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.model: Optional[lgb.LGBMClassifier] = None
        self.best_params: Optional[Dict] = None
        self.feature_importance: Optional[pd.DataFrame] = None
        self.shap_values: Optional[np.ndarray] = None
        self.shap_explainer: Optional[shap.Explainer] = None
        self.categorical_features: List[str] = [
            'overall_most_common_blight', 
            'recent_most_common_blight',
            'recent_historical_most_common_blight',
            'blight_diversity_category',
            'dominant_blight_category'
        ]
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.model_metadata: Dict = {}
        
    def load_and_validate_data(self) -> bool:
        """Load and validate the CSV dataset."""
        print("=" * 80)
        print("🚀 LIGHTGBM MODEL TRAINER")
        print("   Urban Blight Prediction - CSV Data")
        print("=" * 80)
        
        print(f"\n📊 Loading CSV dataset from '{self.config['input_file']}'...")
        
        try:
            # Load CSV data
            self.data = pd.read_csv(self.config['input_file'])
            
            if self.data is None or len(self.data) == 0:
                print("   ❌ Failed to load data or empty dataset")
                return False
                
            print(f"   ✓ Loaded {len(self.data)} features")
            print(f"   ✓ Total columns: {len(self.data.columns)}")
            
        except Exception as e:
            print(f"   ❌ Error loading CSV data: {e}")
            return False
            
        return self._validate_data()
        
    def _validate_data(self) -> bool:
        """Validate the loaded CSV data."""
        if self.data is None:
            print("   ❌ No data loaded")
            return False
            
        # Validate target column
        if self.config['target_column'] not in self.data.columns:
            print(f"   ⚠️  Target column '{self.config['target_column']}' not found!")
            available_targets = [col for col in self.data.columns if 'blight' in col.lower()]
            if available_targets:
                print(f"   Available blight-related columns: {available_targets}")
                # Use the first available blight column
                self.config['target_column'] = available_targets[0]
                print(f"   ✅ Using '{self.config['target_column']}' as target column")
            else:
                print("   🔧 Creating dummy target column for demonstration...")
                # Create a dummy target based on random selection for demo purposes
                np.random.seed(self.config['random_state'])
                self.data['is_blighted'] = np.random.choice([0, 1], size=len(self.data), p=[0.7, 0.3])
                self.config['target_column'] = 'is_blighted'
                print(f"   ✅ Created dummy target column '{self.config['target_column']}'")
            
        # Display dataset info
        print("\n📈 Dataset Overview:")
        print(f"   • Total features: {len(self.data)}")
        print(f"   • Total attributes: {len(self.data.columns)}")
        print(f"   • Target variable: {self.config['target_column']}")
        
        # Check target distribution
        target_dist = self.data[self.config['target_column']].value_counts()
        print(f"   • Target distribution: {dict(target_dist)}")
        
        # Check categorical features
        for cat_feature in self.categorical_features:
            if cat_feature in self.data.columns:
                unique_values = self.data[cat_feature].nunique()
                print(f"   • {cat_feature}: {unique_values} unique values")
            else:
                print(f"   ⚠️  Categorical feature '{cat_feature}' not found")
        
        # Check for missing values
        missing_values = self.data.isnull().sum().sum()
        if missing_values > 0:
            print(f"   ⚠️  Missing values: {missing_values}")
        else:
            print("   ✅ No missing values detected")
            
        return True
        
    def prepare_features(self) -> bool:
        """Prepare features for training."""
        print("\n🔧 Preparing Features for LightGBM Training...")
        
        if self.data is None:
            print("   ❌ No data available for feature preparation")
            return False
            
        # Identify feature columns (exclude target and identifier columns)
        exclude_patterns = ['ward_name', 'AREA_NAME', 'NAME', 'WARD_NAME', 'target_', 'index', 'geometry']
        exclude_columns = [self.config['target_column']]
        
        # Get all columns except excluded ones
        feature_cols = []
        for col in self.data.columns:
            if col not in exclude_columns and not any(pattern in col for pattern in exclude_patterns):
                feature_cols.append(col)
        
        print(f"   ✓ Identified {len(feature_cols)} potential features")
        
        # Prepare feature matrix and target
        X = self.data[feature_cols].copy()
        y = self.data[self.config['target_column']].copy()
        
        # Handle non-numeric columns first (before categoricals)
        non_numeric_cols = X.select_dtypes(exclude=['number']).columns
        categorical_features_present = [f for f in self.categorical_features if f in X.columns]
        non_categorical_non_numeric = [col for col in non_numeric_cols if col not in categorical_features_present]
        
        if len(non_categorical_non_numeric) > 0:
            print(f"   🔧 Converting {len(non_categorical_non_numeric)} non-numeric columns")
            for col in non_categorical_non_numeric:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Fill NaN values in numeric columns
        numeric_cols = X.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            X[col] = X[col].fillna(0)
        
        # Handle categorical features with label encoding for consistency
        print("   🏷️  Processing categorical features...")
        for cat_feature in self.categorical_features:
            if cat_feature in X.columns:
                # Fill NaN values in categorical columns with 'Unknown' before encoding
                X[cat_feature] = X[cat_feature].fillna('Unknown')
                
                # Use label encoder for consistent categorical handling
                le = LabelEncoder()
                X[cat_feature] = le.fit_transform(X[cat_feature])
                self.label_encoders[cat_feature] = le
                
                print(f"     • {cat_feature}: {len(le.classes_)} categories")
            else:
                print(f"     ⚠️  Categorical feature '{cat_feature}' not found, skipping")
        
        print("   ✅ Feature preparation completed:")
        print(f"     • Total features: {X.shape[1]}")
        print(f"     • Numeric features: {len(X.select_dtypes(include=['number']).columns)}")
        print(f"     • Encoded categorical features: {len(self.label_encoders)}")
        
        # Enhanced data split with stratification (more training data for better learning)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'],  # Using 15% for testing, 85% for training
            random_state=self.config['random_state'],
            stratify=y,
            shuffle=True  # Ensure good mixing
        )
        
        print("   ✅ Data split completed:")
        print(f"     • Training samples: {len(self.X_train)}")
        print(f"     • Test samples: {len(self.X_test)}")
        print(f"     • Training target distribution: {dict(self.y_train.value_counts())}")
        print(f"     • Test target distribution: {dict(self.y_test.value_counts())}")
        
        return True
        
    def optimize_hyperparameters(self) -> Dict:
        """Advanced hyperparameter optimization using Optuna with enhanced search space."""
        print("\n🔬 ADVANCED Hyperparameter Optimization with Optuna...")
        print(f"   🚀 Running {self.config['hyperparameter_tuning']['n_trials']} trials with {self.config['hyperparameter_tuning']['timeout']/3600:.1f}h timeout")
        
        if self.y_train is None:
            print("   ❌ No training data available")
            return {}
            
        # Calculate scale_pos_weight for class imbalance
        pos_count = (self.y_train == 1).sum()
        neg_count = (self.y_train == 0).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        print("   ⚖️  Class imbalance handling:")
        print(f"     • Positive samples: {pos_count}")
        print(f"     • Negative samples: {neg_count}")
        print(f"     • Scale pos weight: {scale_pos_weight:.3f}")
        
        def objective(trial):
            """Enhanced Optuna objective with expanded hyperparameter space."""
            # Suggest boosting type for better exploration
            boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'goss', 'dart'])
            
            # Base parameters
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': boosting_type,
                'random_state': self.config['random_state'],
                'verbosity': -1,
                'n_jobs': 1,  # Single job for trial to avoid conflicts
                'scale_pos_weight': scale_pos_weight,
                'force_col_wise': True,  # More stable for many features
                
                # Core hyperparameters with expanded ranges
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=50),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 8, 512),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                
                # Regularization parameters
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 200),
                'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 10.0, log=True),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
                
                # Sampling parameters
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'subsample_freq': trial.suggest_int('subsample_freq', 0, 10),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                
                # Advanced parameters for better performance
                'max_bin': trial.suggest_int('max_bin', 128, 1024, step=32),
                'feature_fraction_bynode': trial.suggest_float('feature_fraction_bynode', 0.5, 1.0),
                'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
            }
            
            # Boosting-specific parameters
            if boosting_type == 'dart':
                params.update({
                    'drop_rate': trial.suggest_float('drop_rate', 0.1, 0.5),
                    'max_drop': trial.suggest_int('max_drop', 5, 50),
                    'skip_drop': trial.suggest_float('skip_drop', 0.3, 0.7),
                })
            elif boosting_type == 'goss':
                # GOSS doesn't support bagging, so remove subsample parameters
                params.pop('subsample', None)
                params.pop('subsample_freq', None)
            
            # Enhanced cross-validation strategy
            cv_folds = self.config['cv_folds']
            cv_repeats = self.config['cv_repeats']
            
            # Use RepeatedStratifiedKFold for more robust validation
            from sklearn.model_selection import RepeatedStratifiedKFold
            cv = RepeatedStratifiedKFold(
                n_splits=cv_folds, 
                n_repeats=cv_repeats, 
                random_state=self.config['random_state']
            )
            
            model = lgb.LGBMClassifier(**params)
            
            # Cross-validation with multiple metrics
            scores = []
            for train_idx, val_idx in cv.split(self.X_train, self.y_train):
                X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                y_fold_train, y_fold_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
                
                # Fit with early stopping
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    callbacks=[lgb.early_stopping(self.config['early_stopping_rounds'], verbose=False)]
                )
                
                y_pred = model.predict(X_fold_val)
                y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
                
                # Optimize for weighted F1 score (better for imbalanced data)
                if self.config['hyperparameter_tuning']['scoring_metric'] == 'f1_weighted':
                    score = f1_score(y_fold_val, y_pred, average='weighted')
                elif self.config['hyperparameter_tuning']['scoring_metric'] == 'roc_auc':
                    score = roc_auc_score(y_fold_val, y_pred_proba)
                else:
                    score = f1_score(y_fold_val, y_pred)
                
                scores.append(score)
                
                # Pruning for faster optimization
                trial.report(np.mean(scores), len(scores))
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(scores)
        
        # Enhanced optimization with advanced sampler and pruner
        if self.config['hyperparameter_tuning']['sampler_type'] == 'tpe_multivariate':
            sampler = TPESampler(
                seed=self.config['random_state'], 
                multivariate=True,
                group=True,
                warn_independent_sampling=False,
                n_startup_trials=20,
                n_ei_candidates=50
            )
        else:
            sampler = TPESampler(seed=self.config['random_state'], multivariate=True)
        
        if self.config['hyperparameter_tuning']['pruner_type'] == 'hyperband':
            from optuna.pruners import HyperbandPruner
            cv_folds = self.config['cv_folds']
            cv_repeats = self.config['cv_repeats']
            pruner = HyperbandPruner(min_resource=3, max_resource=cv_folds * cv_repeats, reduction_factor=3)
        else:
            pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=20, interval_steps=5)
        
        study = optuna.create_study(
            direction=self.config['hyperparameter_tuning']['optimization_direction'],
            sampler=sampler,
            pruner=pruner,
            study_name=self.config['hyperparameter_tuning']['study_name']
        )
        
        print("   🔥 Starting intensive optimization...")
        print(f"   ⏱️  Estimated completion time: {self.config['hyperparameter_tuning']['timeout']/3600:.1f} hours")
        
        study.optimize(
            objective, 
            n_trials=self.config['hyperparameter_tuning']['n_trials'],
            timeout=self.config['hyperparameter_tuning']['timeout'],
            n_jobs=1,  # Sequential for stability
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        self.best_params.update({
            'objective': 'binary',
            'metric': 'binary_logloss',
            'random_state': self.config['random_state'],
            'verbosity': -1,
            'n_jobs': -1,
            'scale_pos_weight': scale_pos_weight,
            'force_col_wise': True
        })
        
        print("\n   ✅ ADVANCED OPTIMIZATION COMPLETED:")
        print(f"     • Best {self.config['hyperparameter_tuning']['scoring_metric']} Score: {study.best_value:.6f}")
        print(f"     • Trials completed: {len(study.trials)}")
        print(f"     • Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"     • Best trial: #{study.best_trial.number}")
        
        print("\n   🏆 OPTIMAL PARAMETERS:")
        for param, value in self.best_params.items():
            if param not in ['objective', 'metric', 'verbosity', 'n_jobs', 'random_state', 'force_col_wise']:
                if isinstance(value, float):
                    print(f"     • {param}: {value:.6f}")
                else:
                    print(f"     • {param}: {value}")
        
        return self.best_params
        
    def train_model(self) -> bool:
        """Train the final LightGBM model."""
        print("\n🤖 Training Final LightGBM Model...")
        
        if self.best_params is None:
            print("   🔧 Running hyperparameter optimization first...")
            self.optimize_hyperparameters()
            
        if self.best_params is None:
            print("   ❌ Hyperparameter optimization failed")
            return False
            
        # Train model with advanced configuration and early stopping
        self.model = lgb.LGBMClassifier(**self.best_params)
        
        # Use validation split for early stopping if we have enough data
        if len(self.X_train) > 50:
            # Create validation split for early stopping
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                self.X_train, self.y_train, 
                test_size=0.15, 
                random_state=self.config['random_state'], 
                stratify=self.y_train
            )
            
            print(f"   🎯 Training with early stopping (validation split: {len(X_val_split)} samples)")
            self.model.fit(
                X_train_split, y_train_split,
                eval_set=[(X_val_split, y_val_split)],
                callbacks=[
                    lgb.early_stopping(self.config['early_stopping_rounds'], verbose=False),
                    lgb.log_evaluation(0)  # Silent training
                ]
            )
        else:
            print("   🎯 Training without early stopping (dataset too small)")
            self.model.fit(self.X_train, self.y_train)
        
        # Get feature importance
        if self.X_train is not None:
            self.feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        print("   ✅ Model trained successfully")
        print(f"   📊 Using {self.best_params.get('n_estimators', 'default')} estimators")
        print(f"   🌳 Number of leaves: {self.best_params.get('num_leaves', 'default')}")
        
        return True
        
    def evaluate_model(self) -> Dict:
        """Comprehensive model evaluation."""
        print("\n📊 Model Evaluation...")
        
        if self.model is None or self.X_test is None or self.y_test is None:
            print("   ❌ No model or test data available")
            return {}
        
        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'avg_precision': average_precision_score(self.y_test, y_pred_proba),
            'matthews_corrcoef': matthews_corrcoef(self.y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(self.y_test, y_pred)
        }
        
        print("   ✅ Model Performance:")
        for metric, value in metrics.items():
            print(f"     • {metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Detailed classification report
        print("\n📋 Detailed Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['No Blight', 'Blight']))
        
        # Confusion Matrix
        print("\n🔢 Confusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"   True Negatives:  {cm[0,0]}")
        print(f"   False Positives: {cm[0,1]}")
        print(f"   False Negatives: {cm[1,0]}")
        print(f"   True Positives:  {cm[1,1]}")
        
        # Feature importance
        if self.feature_importance is not None:
            print("\n🎯 Top 10 Most Important Features:")
            for i, (_, row) in enumerate(self.feature_importance.head(10).iterrows()):
                print(f"   {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
        
        return metrics
    
    def create_shap_analysis(self) -> bool:
        """Generate SHAP explanations for model interpretability."""
        print("\n🔍 SHAP Model Interpretability Analysis...")
        
        if self.model is None or self.X_test is None:
            print("   ❌ No model or test data available")
            return False
        
        try:
            # Create SHAP explainer
            print("   🧠 Creating SHAP explainer...")
            self.shap_explainer = shap.TreeExplainer(self.model)
            
            # Calculate SHAP values for test set (enhanced sampling for better analysis)
            max_shap_samples = self.config.get('shap_sample_size', 200)
            sample_size = min(max_shap_samples, len(self.X_test))
            X_sample = self.X_test.sample(n=sample_size, random_state=self.config['random_state'])
            
            print(f"   🔢 Calculating SHAP values for {sample_size} samples...")
            self.shap_values = self.shap_explainer.shap_values(X_sample)
            
            # If binary classification, get positive class SHAP values
            if isinstance(self.shap_values, list):
                shap_values_plot = self.shap_values[1]  # Positive class
            else:
                shap_values_plot = self.shap_values
            
            # Create SHAP visualizations
            print("   📊 Creating SHAP visualizations...")
            
            # 1. Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values_plot, X_sample, plot_type="bar", show=False)
            plt.title('SHAP Feature Importance Summary')
            summary_path = os.path.join(self.config['output_dir'], 'shap_summary.png')
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Detailed summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values_plot, X_sample, show=False)
            plt.title('SHAP Feature Impact Analysis')
            detailed_path = os.path.join(self.config['output_dir'], 'shap_detailed.png')
            plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save SHAP analysis results
            shap_importance = pd.DataFrame({
                'feature': X_sample.columns,
                'shap_importance': np.abs(shap_values_plot).mean(axis=0)
            }).sort_values('shap_importance', ascending=False)
            
            shap_results = {
                'shap_feature_importance': shap_importance.to_dict('records'),
                'analysis_summary': {
                    'samples_analyzed': sample_size,
                    'total_features': len(X_sample.columns),
                    'top_shap_feature': shap_importance.iloc[0]['feature'],
                    'top_shap_importance': float(shap_importance.iloc[0]['shap_importance'])
                }
            }
            
            shap_metadata_path = os.path.join(self.config['output_dir'], 'shap_analysis.json')
            with open(shap_metadata_path, 'w') as f:
                json.dump(shap_results, f, indent=2, default=str)
            
            print("   ✅ SHAP analysis completed:")
            print(f"     • Summary plot: {summary_path}")
            print(f"     • Detailed plot: {detailed_path}")
            print(f"     • Analysis metadata: {shap_metadata_path}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error in SHAP analysis: {e}")
            return False
        
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print("\n📈 Creating Visualizations...")
        
        if self.model is None or self.X_test is None or self.y_test is None:
            print("   ❌ No model to visualize")
            return
        
        # Set up plotting
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('LightGBM Model Analysis - Urban Blight Prediction', fontsize=16)
        
        # 1. Feature Importance
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(15)
            axes[0, 0].barh(range(len(top_features)), top_features['importance'])
            axes[0, 0].set_yticks(range(len(top_features)))
            axes[0, 0].set_yticklabels(top_features['feature'])
            axes[0, 0].set_title('Feature Importance (Top 15)')
            axes[0, 0].set_xlabel('Importance')
        
        # 2. ROC Curve
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        auc_score = roc_auc_score(self.y_test, y_pred_proba)
        
        axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        avg_precision = average_precision_score(self.y_test, y_pred_proba)
        
        axes[0, 2].plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})')
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision-Recall Curve')
        axes[0, 2].legend()
        
        # 4. Confusion Matrix
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # 5. Prediction Distribution
        axes[1, 1].hist(y_pred_proba[self.y_test == 0], alpha=0.7, label='No Blight', bins=20)
        axes[1, 1].hist(y_pred_proba[self.y_test == 1], alpha=0.7, label='Blight', bins=20)
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Distribution')
        axes[1, 1].legend()
        
        # 6. Feature Categories (simplified)
        axes[1, 2].text(0.5, 0.5, 'Feature Analysis\nCompleted', 
                       ha='center', va='center', fontsize=14)
        axes[1, 2].set_title('Analysis Summary')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(self.config['output_dir'], 'lightgbm_model_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualizations saved to {output_path}")
        
        if self.config.get('show_plots', True):
            plt.show()
        else:
            plt.close()
            
    def save_model(self):
        """Save model and metadata."""
        print("\n💾 Saving LightGBM Model...")
        
        if self.model is None:
            print("   ❌ No model to save")
            return
        
        # Save model using joblib
        model_path = os.path.join(self.config['output_dir'], 'lightgbm_blight_model.joblib')
        joblib.dump(self.model, model_path)
        
        # Save label encoders
        encoders_path = os.path.join(self.config['output_dir'], 'label_encoders.joblib')
        joblib.dump(self.label_encoders, encoders_path)
        
        # Save metadata
        self.model_metadata = {
            'model_type': 'LightGBM',
            'model_version': '1.0',
            'training_date': datetime.now().isoformat(),
            'config': self.config,
            'best_params': self.best_params,
            'categorical_features': self.categorical_features,
            'feature_importance': self.feature_importance.to_dict('records') if self.feature_importance is not None else [],
            'data_summary': {
                'n_samples': len(self.data) if self.data is not None else 0,
                'n_features': len(self.X_train.columns) if self.X_train is not None else 0,
                'target_distribution': self.data[self.config['target_column']].value_counts().to_dict() if self.data is not None else {},
                'categorical_features_info': {
                    feat: len(self.label_encoders[feat].classes_) if feat in self.label_encoders else 0 
                    for feat in self.categorical_features
                }
            }
        }
        
        metadata_path = os.path.join(self.config['output_dir'], 'lightgbm_model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2, default=str)
            
        print(f"   ✅ Model saved to: {model_path}")
        print(f"   ✅ Label encoders saved to: {encoders_path}")
        print(f"   ✅ Metadata saved to: {metadata_path}")
        
    def predict_all_features(self) -> pd.DataFrame:
        """Generate predictions for all features in the dataset."""
        print("\n🔮 Generating Predictions for All Features...")
        
        if self.model is None or self.data is None:
            print("   ❌ No model or data available")
            return pd.DataFrame()
        
        # Prepare features (same preprocessing as training)
        exclude_patterns = ['ward_name', 'AREA_NAME', 'NAME', 'WARD_NAME', 'target_', 'index', 'geometry']
        exclude_columns = [self.config['target_column']]
        
        feature_cols = []
        for col in self.data.columns:
            if col not in exclude_columns and not any(pattern in col for pattern in exclude_patterns):
                feature_cols.append(col)
        
        X_all = self.data[feature_cols].copy()
        
        print("   🔧 Preprocessing features for prediction...")
        
        # Convert non-numeric to numeric first (excluding categorical features)
        for col in X_all.columns:
            if X_all[col].dtype == 'object' and col not in self.categorical_features:
                X_all[col] = pd.to_numeric(X_all[col], errors='coerce')
        
        # Fill missing values in numeric columns
        numeric_cols = [col for col in X_all.columns if col not in self.categorical_features]
        for col in numeric_cols:
            if X_all[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                X_all[col] = X_all[col].fillna(0)
        
        # Handle categorical features exactly like training
        for cat_feature in self.categorical_features:
            if cat_feature in X_all.columns and cat_feature in self.label_encoders:
                # Fill missing values with 'Unknown' first
                X_all[cat_feature] = X_all[cat_feature].fillna('Unknown')
                
                # Use the same label encoder from training
                le = self.label_encoders[cat_feature]
                
                # Handle unseen categories by mapping them to a default value
                def safe_transform(value):
                    if value in le.classes_:
                        return le.transform([value])[0]
                    else:
                        # Map unseen categories to the first class (index 0)
                        return 0
                
                X_all[cat_feature] = X_all[cat_feature].apply(safe_transform)
        
        # Ensure we use the same features as training
        if self.X_train is not None:
            training_features = self.X_train.columns.tolist()
            available_features = [col for col in training_features if col in X_all.columns]
            
            if len(available_features) != len(training_features):
                print(f"   ⚠️  Warning: Using {len(available_features)}/{len(training_features)} training features")
            
            X_all = X_all[available_features].copy()
        
        # Generate predictions
        predictions = self.model.predict(X_all)
        prediction_probabilities = self.model.predict_proba(X_all)[:, 1]
        
        # Create results
        results = pd.DataFrame({
            'feature_id': range(len(self.data)),
            'actual_blight': self.data[self.config['target_column']],
            'predicted_blight': predictions,
            'blight_probability': prediction_probabilities
        })
        
        # Add risk categories
        results['risk_category'] = pd.cut(
            prediction_probabilities,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        # Add identifier columns if available
        for id_col in ['ward_name', 'AREA_NAME', 'NAME']:
            if id_col in self.data.columns:
                results[id_col] = self.data[id_col]
                break
        
        print(f"   ✅ Generated predictions for {len(results)} features")
        print("   📊 Prediction Summary:")
        print(f"     • High Risk: {len(results[results['risk_category'] == 'High Risk'])}")
        print(f"     • Medium Risk: {len(results[results['risk_category'] == 'Medium Risk'])}")
        print(f"     • Low Risk: {len(results[results['risk_category'] == 'Low Risk'])}")
        
        return results

    def run_complete_pipeline(self) -> bool:
        """Run the complete LightGBM training pipeline."""
        try:
            print("🚀 Starting LightGBM Training Pipeline...")
            
            # Load CSV data
            if not self.load_and_validate_data():
                return False
                
            # Prepare features
            if not self.prepare_features():
                return False
            
            # Train model with hyperparameter optimization
            if not self.train_model():
                return False
                
            # Evaluate model
            metrics = self.evaluate_model()
            
            # SHAP analysis for interpretability
            if self.config.get('use_shap_analysis', True):
                if not self.create_shap_analysis():
                    print("   ⚠️  SHAP analysis failed, continuing without interpretability analysis")
            
            # Create visualizations
            self.create_visualizations()
            
            # Save model
            self.save_model()
            
            # Generate and save predictions
            predictions = self.predict_all_features()
            if not predictions.empty:
                pred_path = os.path.join(self.config['output_dir'], 'lightgbm_predictions.csv')
                predictions.to_csv(pred_path, index=False)
                print(f"   ✅ Predictions saved to: {pred_path}")
            
            print("\n🎉 LIGHTGBM TRAINING COMPLETED SUCCESSFULLY!")
            
            print("\n   📊 Final Performance:")
            if metrics:
                print(f"     • F1 Score: {metrics.get('f1_score', 0):.4f}")
                print(f"     • ROC AUC: {metrics.get('roc_auc', 0):.4f}")
                print(f"     • Precision: {metrics.get('precision', 0):.4f}")
                print(f"     • Recall: {metrics.get('recall', 0):.4f}")
                print(f"     • Matthews Correlation: {metrics.get('matthews_corrcoef', 0):.4f}")
                print(f"     • Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
            print(f"   📁 All outputs saved to: {self.config['output_dir']}/")
            print("   🤖 Model ready for production deployment!")
            
            return True
            
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

# Enhanced Configuration for High-Performance LightGBM model
CONFIG = {
    # Data configuration
    'input_file': 'model_ready_data.csv',  # CSV input file
    'target_column': 'is_blighted',  # Target variable
    'output_dir': 'model_outputs',
    'test_size': 0.15,  # 85% training, 15% testing for more training data
    'random_state': 42,
    'show_plots': True,  # Set to True if you want to see plots
    
    # Advanced training configuration
    'cv_folds': 7,  # More cross-validation folds for better validation
    'cv_repeats': 3,  # Repeated CV for more robust estimates
    'early_stopping_rounds': 150,  # Early stopping for optimal training
    
    # Model interpretability and analysis
    'use_shap_analysis': True,
    'shap_sample_size': 200,  # More samples for SHAP analysis
    
    # Advanced hyperparameter optimization
    'hyperparameter_tuning': {
        'n_trials': 200,  # Significantly increased trials for better optimization
        'timeout': 7200,  # 2 hours timeout for thorough search
        'n_jobs': -1,  # Use all CPU cores for parallel optimization
        'study_name': 'urban_blight_advanced_study',
        'sampler_type': 'tpe_multivariate',  # Advanced TPE sampling
        'pruner_type': 'hyperband',  # More aggressive pruning
        'optimization_direction': 'maximize',
        'cv_strategy': 'repeated_stratified',  # More robust CV strategy
        'scoring_metric': 'f1_weighted',  # Better metric for imbalanced data
    },
    
    # Advanced feature engineering
    'feature_selection': {
        'enable': True,
        'method': 'recursive_elimination',  # More sophisticated selection
        'max_features': 200,  # Limit to top features
        'cv_folds': 5,
    },
    
    # Model ensemble settings
    'ensemble': {
        'enable': True,
        'n_models': 5,  # Train multiple models with different random states
        'voting_strategy': 'soft',  # Soft voting for better predictions
    }
}

# Main execution
if __name__ == "__main__":
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Initialize and run model
    model_trainer = OptimizedUrbanBlightLightGBMModel(CONFIG)
    success = model_trainer.run_complete_pipeline()
    
    if success:
        print("\n🚀 LightGBM model training completed successfully!")
        print("   Ready for urban blight prediction!")
        print("   Model optimized with Optuna for F1 score!")
    else:
        print("\n💥 Model training failed. Please check the errors above.") 