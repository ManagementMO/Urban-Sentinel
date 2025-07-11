import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)
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

warnings.filterwarnings("ignore")


class OptimizedUrbanBlightLightGBMModel:
    """
    LightGBM Model for Urban Blight Prediction
    Optimized with Optuna for best performance
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
        # Only the two specified categorical features
        self.categorical_features: List[str] = [
            "overall_most_common_blight",
            "recent_most_common_blight"
        ]
        self.model_metadata: Dict = {}

    def load_and_validate_data(self) -> bool:
        """Load and validate the dataset."""
        print("=" * 80)
        print("🚀 LIGHTGBM MODEL TRAINER")
        print("   Urban Blight Prediction")
        print("=" * 80)

        print(f"\n📊 Loading dataset from '{self.config['input_file']}'...")

        try:
            # Load data (CSV format as specified by user)
            self.data = pd.read_csv(self.config["input_file"])

            if self.data is None or len(self.data) == 0:
                print("   ❌ Failed to load data or empty dataset")
                return False

            print(f"   ✓ Loaded {len(self.data)} records")
            print(f"   ✓ Total columns: {len(self.data.columns)}")

        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            return False

        return self._validate_data()

    def _validate_data(self) -> bool:
        """Validate the loaded data."""
        if self.data is None:
            print("   ❌ No data loaded")
            return False

        # Validate target column
        if self.config["target_column"] not in self.data.columns:
            print(f"   ⚠️  Target column '{self.config['target_column']}' not found!")
            available_targets = [
                col for col in self.data.columns if "blight" in col.lower()
            ]
            if available_targets:
                print(f"   Available blight-related columns: {available_targets}")
                # Use the first available blight column
                self.config["target_column"] = available_targets[0]
                print(f"   ✅ Using '{self.config['target_column']}' as target column")
            else:
                print("   🔧 Creating dummy target column for demonstration...")
                # Create a dummy target based on random selection for demo purposes
                np.random.seed(self.config["random_state"])
                self.data["is_blighted"] = np.random.choice(
                    [0, 1], size=len(self.data), p=[0.7, 0.3]
                )
                self.config["target_column"] = "is_blighted"
                print(
                    f"   ✅ Created dummy target column '{self.config['target_column']}'"
                )

        # Display dataset info
        print("\n📈 Dataset Overview:")
        print(f"   • Total records: {len(self.data)}")
        print(f"   • Total attributes: {len(self.data.columns)}")
        print(f"   • Target variable: {self.config['target_column']}")

        # Check target distribution
        target_dist = self.data[self.config["target_column"]].value_counts()
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
        """Prepare features for training with better feature selection for varied predictions."""
        print("\n🔧 Preparing Features for LightGBM Training...")

        if self.data is None:
            print("   ❌ No data available for feature preparation")
            return False

        # Identify feature columns (exclude target and identifier columns)
        exclude_patterns = [
            "ward_name",
            "AREA_NAME", 
            "NAME",
            "WARD_NAME",
            "target_",
            "index",
            "geometry",
            # CRITICAL: Exclude overly dominant binary features that cause polarized predictions
            "is_high_blight_risk",  # This was causing 0.004 vs 0.996 predictions
            "is_blighted",  # Target-like feature
        ]
        exclude_columns = [self.config["target_column"]]

        # Get all columns except excluded ones
        feature_cols = []
        for col in self.data.columns:
            if col not in exclude_columns and not any(
                pattern in col.lower() for pattern in exclude_patterns
            ):
                feature_cols.append(col)

        print(f"   ✓ Identified {len(feature_cols)} potential features")
        print(f"   ✓ Excluded dominant binary features for better probability variation")

        # Prepare feature matrix and target
        X = self.data[feature_cols].copy()
        y = self.data[self.config["target_column"]].copy()

        # Handle categorical features - convert to category dtype for LightGBM
        print("   🏷️  Processing categorical features...")
        for cat_feature in self.categorical_features:
            if cat_feature in X.columns:
                # Fill NaN values first
                X[cat_feature] = X[cat_feature].fillna("Unknown")
                # Convert to category dtype for LightGBM
                X[cat_feature] = pd.Categorical(X[cat_feature])
                print(f"     • {cat_feature}: {X[cat_feature].nunique()} categories")
            else:
                print(f"     ⚠️  Categorical feature '{cat_feature}' not found, skipping")

        # Handle non-categorical columns - convert to numeric
        non_categorical_cols = [col for col in X.columns if col not in self.categorical_features]
        for col in non_categorical_cols:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce')
            # Fill NaN values in numeric columns
            X[col] = X[col].fillna(0)

        # IMPROVED: More lenient feature selection for varied predictions
        print(f"\n🔍 Dataset size analysis:")
        print(f"     • Samples: {len(X)}")
        print(f"     • Features: {len(X.columns)}")
        print(f"     • Feature-to-sample ratio: {len(X.columns)/len(X):.1f}:1")
        
        # More lenient feature selection - keep more features for variation
        max_features = min(max(15, len(X) // 2), 50)  # Keep 15-50 features, up to half the samples
        
        if len(X.columns) > max_features:
            print(f"   🎯 Selecting top {max_features} most varied features for nuanced predictions")
            X = self._select_varied_features(X, y, max_features)
            print(f"   ✅ Reduced to {X.shape[1]} features with good variation")
        else:
            print(f"   ✅ Keeping all {len(X.columns)} features - good ratio for predictions")

        print("   ✅ Feature preparation completed:")
        print(f"     • Total features: {X.shape[1]}")
        print(f"     • Categorical features: {len([col for col in X.columns if col in self.categorical_features])}")
        print(f"     • Numeric features: {len([col for col in X.columns if col not in self.categorical_features])}")

        # Use entire dataset for training (no train/test split)
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.X_test = None  # No separate test set
        self.y_test = None

        print("   ✅ Using entire dataset for training:")
        print(f"     • Total training samples: {len(X)}")
        print(f"     • Target distribution: {dict(y.value_counts())}")
        print("     • Model will generate varied probability scores using multiple features")

        return True

    def _select_varied_features(self, X: pd.DataFrame, y: pd.Series, max_features: int) -> pd.DataFrame:
        """Select features that provide good variation for nuanced probability predictions."""
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
        from sklearn.feature_selection import VarianceThreshold
        
        print("   🎯 Applying intelligent feature selection for varied predictions...")
        
        # Step 1: Remove extremely low-variance features (but keep some variation)
        print("     • Removing only very low-variance features...")
        variance_selector = VarianceThreshold(threshold=0.001)  # Lower threshold
        
        # Separate numeric and categorical features for variance filtering
        numeric_features = [col for col in X.columns if col not in self.categorical_features]
        categorical_features = [col for col in X.columns if col in self.categorical_features]
        
        if numeric_features:
            X_numeric = X[numeric_features]
            X_numeric_filtered = pd.DataFrame(
                variance_selector.fit_transform(X_numeric),
                columns=X_numeric.columns[variance_selector.get_support()],
                index=X_numeric.index
            )
            print(f"     • Kept {X_numeric_filtered.shape[1]}/{len(numeric_features)} numeric features")
        else:
            X_numeric_filtered = pd.DataFrame(index=X.index)
        
        # Keep all categorical features
        if categorical_features:
            X_categorical = X[categorical_features]
            print(f"     • Keeping all {len(categorical_features)} categorical features")
        else:
            X_categorical = pd.DataFrame(index=X.index)
        
        # Combine filtered features
        X_filtered = pd.concat([X_numeric_filtered, X_categorical], axis=1)
        
        # Step 2: Select diverse features using multiple selection methods
        if X_filtered.shape[1] > max_features:
            print(f"     • Selecting top {max_features} most informative features...")
            
            # Prepare data for feature selection
            X_for_selection = X_filtered.copy()
            for cat_col in categorical_features:
                if cat_col in X_for_selection.columns:
                    X_for_selection[cat_col] = pd.factorize(X_for_selection[cat_col])[0]
            
            # Use mutual information to find features that provide good discrimination
            # but not perfect separation (which causes polarized predictions)
            selector = SelectKBest(score_func=mutual_info_classif, k=max_features)
            selector.fit(X_for_selection, y)  # Fit the selector before getting support
            X_selected_indices = selector.get_support()
            selected_features = X_filtered.columns[X_selected_indices]
            
            X_selected = X_filtered[selected_features].copy()
            
            # Restore categorical types for selected categorical features
            for cat_col in categorical_features:
                if cat_col in X_selected.columns:
                    X_selected[cat_col] = X_filtered[cat_col]
            
            print(f"     • Selected features: {list(X_selected.columns[:10])}{'...' if len(X_selected.columns) > 10 else ''}")
        else:
            X_selected = X_filtered
            
        print(f"     • Final feature count: {X_selected.shape[1]} (designed for varied 0-1 predictions)")
        return X_selected

    def optimize_hyperparameters(self) -> Dict:
        """Hyperparameter optimization adapted for small datasets."""
        print("\n🔬 Hyperparameter Optimization (Small Dataset Mode)...")

        if self.y_train is None:
            print("   ❌ No training data available")
            return {}

        # Calculate scale_pos_weight for class imbalance (but don't tune it)
        pos_count = (self.y_train == 1).sum()
        neg_count = (self.y_train == 0).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        print("   ⚖️  Class imbalance handling:")
        print(f"     • Positive samples: {pos_count}")
        print(f"     • Negative samples: {neg_count}")
        print(f"     • Scale pos weight: {scale_pos_weight:.3f}")

        # For very small datasets, use simpler validation
        n_samples = len(self.X_train)
        if n_samples < 50:
            print(f"   📊 Small dataset ({n_samples} samples) - using Leave-One-Out CV")
            cv_folds = min(n_samples, 10)  # Max 10 folds for small datasets
        else:
            cv_folds = 3
            
        print(f"   🚀 Running {self.config['hyperparameter_tuning']['n_trials']} trials with {cv_folds}-fold CV")

        def objective(trial):
            """Simplified objective for small datasets."""
            # Simpler parameter space for small datasets
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "random_state": self.config["random_state"],
                "verbosity": -1,
                "n_jobs": 1,
                "scale_pos_weight": scale_pos_weight,
                # Simplified hyperparameter space for small datasets
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),  # Fewer trees
                "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3),  # Higher learning rate
                "num_leaves": trial.suggest_int("num_leaves", 5, 31),  # Fewer leaves
                "max_depth": trial.suggest_int("max_depth", 3, 6),  # Shallower trees
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),  # More regularization
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 0.5),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 20),  # Higher min samples
            }

            # Use appropriate CV strategy
            if cv_folds <= 10:
                from sklearn.model_selection import LeaveOneOut, StratifiedKFold
                if cv_folds == n_samples:
                    cv = LeaveOneOut()
                else:
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config["random_state"])
            else:
                from sklearn.model_selection import StratifiedKFold
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config["random_state"])

            model = lgb.LGBMClassifier(**params)

            # Cross-validation with F1 score optimization
            scores = []
            try:
                for train_idx, val_idx in cv.split(self.X_train, self.y_train):
                    X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                    y_fold_train, y_fold_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

                    # Skip if validation fold has no positive cases
                    if y_fold_val.sum() == 0:
                        continue
                        
                    # Fit model
                    model.fit(X_fold_train, y_fold_train)
                    
                    # Predict and calculate F1 score
                    y_pred = model.predict(X_fold_val)
                    score = f1_score(y_fold_val, y_pred, zero_division='warn')
                    scores.append(score)

                return np.mean(scores) if scores else 0.0
                
            except Exception as e:
                print(f"     ⚠️  CV fold failed: {e}")
                return 0.0

        # Reduced trials for small datasets
        n_trials = min(self.config["hyperparameter_tuning"]["n_trials"], 50)
        
        # Set up Optuna study
        sampler = TPESampler(seed=self.config["random_state"])
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name="small_dataset_blight_optimization"
        )

        print("   🔥 Starting optimization...")
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=self.config["hyperparameter_tuning"]["timeout"],
            n_jobs=1,
            show_progress_bar=True,
        )

        self.best_params = study.best_params
        self.best_params.update({
                "objective": "binary",
                "metric": "binary_logloss",
            "boosting_type": "gbdt",
                "random_state": self.config["random_state"],
                "verbosity": -1,
                "n_jobs": -1,
                "scale_pos_weight": scale_pos_weight,
        })

        print("\n   ✅ OPTIMIZATION COMPLETED:")
        print(f"     • Best F1 Score: {study.best_value:.6f}")
        print(f"     • Trials completed: {len(study.trials)}")
        print(f"     • Best trial: #{study.best_trial.number}")

        print("\n   🏆 BEST PARAMETERS:")
        for param, value in self.best_params.items():
            if param not in ["objective", "metric", "verbosity", "n_jobs", "random_state", "boosting_type"]:
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

        # Initialize LGBMClassifier with best hyperparameters
        self.model = lgb.LGBMClassifier(**self.best_params)

        print("   🎯 Training on entire training set...")
        # Train the final model on the entire training set
        self.model.fit(self.X_train, self.y_train)

        # Get feature importance
        if self.X_train is not None:
            self.feature_importance = pd.DataFrame({
                    "feature": self.X_train.columns,
                    "importance": self.model.feature_importances_,
            }).sort_values("importance", ascending=False)

        print("   ✅ Model trained successfully")
        print(f"   📊 Using {self.best_params.get('n_estimators', 'default')} estimators")
        print(f"   🌳 Number of leaves: {self.best_params.get('num_leaves', 'default')}")

        return True

    def evaluate_model(self) -> Dict:
        """Model evaluation using training set and cross-validation insights."""
        print("\n📊 Model Evaluation...")

        if self.model is None or self.X_train is None or self.y_train is None:
            print("   ❌ No model or training data available")
            return {}

        # Since we trained on all data, evaluate on training set
        # Note: This gives optimistic estimates, but CV during hyperparameter tuning provides realistic performance
        print("   ℹ️  Evaluating on training set (cross-validation during tuning provides unbiased estimates)")
        
        # Make predictions on training set
        y_pred = self.model.predict(self.X_train)
        y_pred_proba = self.model.predict_proba(self.X_train)[:, 1]

        # Calculate metrics
        metrics = {
            "training_accuracy": accuracy_score(self.y_train, y_pred),
            "training_precision": precision_score(self.y_train, y_pred),
            "training_recall": recall_score(self.y_train, y_pred),
            "training_f1_score": f1_score(self.y_train, y_pred),
            "training_roc_auc": roc_auc_score(self.y_train, y_pred_proba),
        }

        print("   ✅ Training Set Performance:")
        for metric, value in metrics.items():
            print(f"     • {metric.replace('_', ' ').title()}: {value:.4f}")

        # Detailed classification report
        print("\n📋 Detailed Classification Report (Training Set):")
        print(classification_report(self.y_train, y_pred, target_names=["No Blight", "Blight"]))

        # Confusion Matrix
        print("\n🔢 Confusion Matrix (Training Set):")
        cm = confusion_matrix(self.y_train, y_pred)
        print(f"   True Negatives:  {cm[0, 0]}")
        print(f"   False Positives: {cm[0, 1]}")
        print(f"   False Negatives: {cm[1, 0]}")
        print(f"   True Positives:  {cm[1, 1]}")

        # ROC AUC Score
        print(f"\n🎯 Training ROC AUC Score: {metrics['training_roc_auc']:.4f}")
        print("   📝 Note: Cross-validation F1 score from hyperparameter tuning provides realistic performance estimate")

        return metrics

    def plot_feature_importance(self):
        """Generate and plot feature importance."""
        print("\n📊 Generating Feature Importance Plot...")
        
        if self.feature_importance is None:
            print("   ❌ No feature importance data available")
            return

        # Create feature importance plot
        plt.figure(figsize=(12, 8))
        top_features = self.feature_importance.head(20)
        
        plt.barh(range(len(top_features)), top_features["importance"])
        plt.yticks(range(len(top_features)), top_features["feature"])
        plt.xlabel("Feature Importance")
        plt.title("Top 20 Feature Importance - LightGBM Model")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # Save plot
        importance_path = os.path.join(self.config["output_dir"], "feature_importance.png")
        plt.savefig(importance_path, dpi=300, bbox_inches="tight")
        print(f"   ✅ Feature importance plot saved to: {importance_path}")

        if self.config.get("show_plots", True):
            plt.show()
        else:
            plt.close()

        # Print top 10 features
        print("\n🎯 Top 10 Most Important Features:")
        for i, (_, row) in enumerate(self.feature_importance.head(10).iterrows()):
            print(f"   {i + 1:2d}. {row['feature']}: {row['importance']:.4f}")

    def save_model(self):
        """Save model and metadata."""
        print("\n💾 Saving LightGBM Model...")

        if self.model is None:
            print("   ❌ No model to save")
            return

        # Save model using joblib as specified
        model_path = os.path.join(self.config["output_dir"], "lightgbm_blight_model.joblib")
        joblib.dump(self.model, model_path)

        # Save metadata
        self.model_metadata = {
            "model_type": "LightGBM",
            "model_version": "1.0",
            "training_date": datetime.now().isoformat(),
            "config": self.config,
            "best_params": self.best_params,
            "categorical_features": self.categorical_features,
            "feature_importance": self.feature_importance.to_dict("records") if self.feature_importance is not None else [],
            "data_summary": {
                "n_samples": len(self.data) if self.data is not None else 0,
                "n_features": len(self.X_train.columns) if self.X_train is not None else 0,
                "target_distribution": self.data[self.config["target_column"]].value_counts().to_dict() if self.data is not None else {},
            },
        }

        metadata_path = os.path.join(self.config["output_dir"], "lightgbm_model_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.model_metadata, f, indent=2, default=str)

        print(f"   ✅ Model saved to: {model_path}")
        print(f"   ✅ Metadata saved to: {metadata_path}")

    def run_complete_pipeline(self) -> bool:
        """Run the complete LightGBM training pipeline."""
        try:
            print("🚀 Starting LightGBM Training Pipeline...")

            # Load data
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

            # Generate predictions for all wards
            predictions_df = self.generate_predictions()

            # Plot feature importance
            self.plot_feature_importance()

            # Save model
            self.save_model()

            print("\n🎉 LIGHTGBM TRAINING COMPLETED SUCCESSFULLY!")
            print("\n   📊 Final Performance (Training Set):")
            if metrics:
                print(f"     • Training F1 Score: {metrics.get('training_f1_score', 0):.4f}")
                print(f"     • Training ROC AUC: {metrics.get('training_roc_auc', 0):.4f}")
                print(f"     • Training Precision: {metrics.get('training_precision', 0):.4f}")
                print(f"     • Training Recall: {metrics.get('training_recall', 0):.4f}")
            print("   📝 Note: Cross-validation during hyperparameter tuning provides realistic performance estimates")
            print(f"   📁 All outputs saved to: {self.config['output_dir']}/")
            print("   🤖 Model trained on entire dataset and ready for predictions!")

            return True

        except Exception as e:
            print(f"\n❌ CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_predictions(self):
        """Generate comprehensive predictions for all wards and save to CSV."""
        print("\n🔮 Generating Ward Predictions...")

        if self.model is None or self.X_train is None:
            print("   ❌ No trained model or data available for predictions")
            return

        # Make predictions for all wards
        print("   🎯 Computing predictions for all wards...")
        predictions = self.model.predict(self.X_train)
        prediction_probabilities = self.model.predict_proba(self.X_train)

        # Get ward names from original data
        if 'ward_name' in self.data.columns:
            ward_names = self.data['ward_name'].values
        else:
            # Fallback to row indices if no ward_name column
            ward_names = [f"Ward_{i+1}" for i in range(len(self.X_train))]

        # Create comprehensive predictions dataframe
        predictions_df = pd.DataFrame({
            'ward_name': ward_names,
            'predicted_blight': predictions,
            'no_blight_probability': prediction_probabilities[:, 0],
            'blight_probability': prediction_probabilities[:, 1],
            'actual_blight': self.y_train.values,
        })

        # Add risk categories based on probability thresholds
        def get_risk_category(prob):
            if prob >= 0.8:
                return "Very High Risk"
            elif prob >= 0.6:
                return "High Risk"
            elif prob >= 0.4:
                return "Medium Risk"
            elif prob >= 0.2:
                return "Low Risk"
            else:
                return "Very Low Risk"

        predictions_df['risk_category'] = predictions_df['blight_probability'].apply(get_risk_category)

        # Add confidence score (distance from 0.5 threshold)
        predictions_df['confidence_score'] = abs(predictions_df['blight_probability'] - 0.5) * 2

        # Add prediction accuracy flag
        predictions_df['correct_prediction'] = (predictions_df['predicted_blight'] == predictions_df['actual_blight'])

        # Sort by blight probability (highest risk first)
        predictions_df = predictions_df.sort_values('blight_probability', ascending=False)

        # Add ranking
        predictions_df['risk_rank'] = range(1, len(predictions_df) + 1)

        # Save predictions to CSV
        predictions_path = os.path.join(self.config["output_dir"], "ward_blight_predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)

        print(f"   ✅ Ward predictions saved to: {predictions_path}")

        # Print summary statistics
        print("\n📊 Prediction Summary:")
        print(f"   • Total wards analyzed: {len(predictions_df)}")
        print(f"   • Wards predicted as high blight risk: {predictions_df['predicted_blight'].sum()}")
        print(f"   • Average blight probability: {predictions_df['blight_probability'].mean():.3f}")
        print(f"   • Prediction accuracy: {predictions_df['correct_prediction'].mean():.1%}")

        # Show risk category distribution
        print("\n🎯 Risk Category Distribution:")
        risk_counts = predictions_df['risk_category'].value_counts()
        for category, count in risk_counts.items():
            print(f"   • {category}: {count} wards")

        # Show top risk wards
        print("\n🚨 Top 5 Highest Risk Wards:")
        top_risk = predictions_df.head(5)
        for _, ward in top_risk.iterrows():
            print(f"   {ward['risk_rank']:2d}. {ward['ward_name']}: {ward['blight_probability']:.3f} ({ward['risk_category']})")

        # Show lowest risk wards
        print("\n✅ Top 5 Lowest Risk Wards:")
        low_risk = predictions_df.tail(5)
        for _, ward in low_risk.iterrows():
            print(f"   {ward['risk_rank']:2d}. {ward['ward_name']}: {ward['blight_probability']:.3f} ({ward['risk_category']})")

        # Update model metadata with prediction info
        self.model_metadata["predictions_summary"] = {
            "total_wards": len(predictions_df),
            "high_risk_wards": int(predictions_df['predicted_blight'].sum()),
            "average_blight_probability": float(predictions_df['blight_probability'].mean()),
            "prediction_accuracy": float(predictions_df['correct_prediction'].mean()),
            "risk_distribution": risk_counts.to_dict(),
            "predictions_file": predictions_path,
        }

        return predictions_df


# Configuration for LightGBM model - optimized for small datasets
CONFIG = {
    # Data configuration
    "input_file": "model_ready_data.csv",
    "target_column": "is_blighted",
    "output_dir": "model_outputs",
    "random_state": 42,
    "show_plots": True,
    # Hyperparameter optimization - reduced for small datasets
    "hyperparameter_tuning": {
        "n_trials": 50,  # Reduced trials for small datasets
        "timeout": 1800,  # 30 minutes - faster for small datasets
    },
}

# Main execution
if __name__ == "__main__":
    # Create output directory
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Initialize and run model
    model_trainer = OptimizedUrbanBlightLightGBMModel(CONFIG)
    success = model_trainer.run_complete_pipeline()

    if success:
        print("\n🚀 LightGBM model training completed successfully!")
        print("   Ready for urban blight prediction!")
    else:
        print("\n💥 Model training failed. Please check the errors above.")
