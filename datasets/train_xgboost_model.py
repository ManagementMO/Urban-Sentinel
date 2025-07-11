import pandas as pd
import geopandas as gpd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, 
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import optuna
from optuna.samplers import TPESampler
import shap

warnings.filterwarnings('ignore')

class UrbanBlightXGBoostModel:
    """
    Advanced XGBoost Model for Urban Blight Prediction
    Optimized for Toronto Ward-Based Data with Comprehensive Hyperparameter Tuning
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.model: Optional[xgb.XGBClassifier] = None
        self.best_params: Optional[Dict] = None
        self.feature_importance: Optional[pd.DataFrame] = None
        self.label_encoders: Dict = {}
        self.scaler: Optional[StandardScaler] = None
        self.selected_features: Optional[List[str]] = None
        self.model_metadata: Dict = {}
        
    def load_and_prepare_data(self) -> bool:
        """Load and prepare the ward-based dataset for training."""
        print("=" * 80)
        print("🚀 ADVANCED XGBOOST MODEL TRAINER")
        print("   Urban Blight Prediction - Toronto Ward Analysis")
        print("=" * 80)
        
        print(f"\n📊 Loading dataset from '{self.config['input_file']}'...")
        
        try:
            # Load data based on file extension
            if self.config['input_file'].endswith('.csv'):
                self.data = pd.read_csv(self.config['input_file'])
                print(f"   ✓ Loaded CSV file with {len(self.data)} wards")
            elif self.config['input_file'].endswith('.geojson'):
                self.data = gpd.read_file(self.config['input_file'])
                print(f"   ✓ Loaded GeoJSON file with {len(self.data)} wards")
            else:
                # Try CSV first, then GeoJSON
                try:
                    self.data = pd.read_csv(self.config['input_file'])
                    print(f"   ✓ Loaded as CSV file with {len(self.data)} wards")
                except:
                    self.data = gpd.read_file(self.config['input_file'])
                    print(f"   ✓ Loaded as GeoJSON file with {len(self.data)} wards")
            
            # Validate data was loaded
            if self.data is None:
                print(f"   ❌ Failed to load data from {self.config['input_file']}")
                return False
                
            print(f"   ✓ Features available: {len(self.data.columns)}")
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            return False
            
        # Display basic info
        print(f"\n📈 Dataset Overview:")
        print(f"   • Total wards: {len(self.data)}")
        print(f"   • Target variable: {self.config['target_column']}")
        
        # Check target distribution
        if self.config['target_column'] in self.data.columns:
            target_dist = self.data[self.config['target_column']].value_counts()
            print(f"   • Target distribution: {dict(target_dist)}")
        else:
            print(f"   ❌ Target column '{self.config['target_column']}' not found!")
            return False
            
        return True
        
    def engineer_features(self) -> bool:
        """Advanced feature engineering and selection."""
        print(f"\n🔧 Advanced Feature Engineering...")
        
        # Validate data is loaded
        if self.data is None:
            print("   ❌ No data loaded. Cannot engineer features.")
            return False
        
        # Exclude non-feature columns
        exclude_cols = [
            'geometry', 'ward_name', 'AREA_NAME', 'NAME', 'WARD_NAME',
            self.config['target_column'], 'target_total_requests',
            'target_blight_requests', 'target_blight_rate'
        ]
        
        # Get feature columns
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        print(f"   ✓ Base features: {len(feature_cols)}")
        
        # Create additional engineered features
        self._create_interaction_features()
        self._create_ratio_features()
        self._create_polynomial_features()
        
        # Update feature columns after engineering
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        print(f"   ✓ Total features after engineering: {len(feature_cols)}")
        
        # Prepare features and target
        X = self.data[feature_cols].copy()
        y = self.data[self.config['target_column']].copy()
        
        # Handle datetime columns - exclude them completely
        datetime_cols = X.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            print(f"   ⚠️  Excluding {len(datetime_cols)} datetime columns: {list(datetime_cols)}")
            X = X.drop(columns=datetime_cols)
        
        # Handle categorical features
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_features) > 0:
            print(f"   🔤 Encoding {len(categorical_features)} categorical features...")
            for col in categorical_features:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Ensure all remaining columns are numeric
        non_numeric_cols = X.select_dtypes(exclude=['number']).columns
        if len(non_numeric_cols) > 0:
            print(f"   ⚠️  Converting {len(non_numeric_cols)} non-numeric columns to numeric")
            for col in non_numeric_cols:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                
        # Handle missing values
        X = X.fillna(0)
        
        # Final check: ensure all columns are numeric
        final_dtypes = X.dtypes
        non_numeric_final = final_dtypes[~final_dtypes.apply(lambda x: np.issubdtype(x, np.number))]
        if len(non_numeric_final) > 0:
            print(f"   ❌ ERROR: Still have non-numeric columns: {list(non_numeric_final.index)}")
            return False
        
        print(f"   ✓ All features are numeric, ready for ML")
        
        # Feature selection
        if self.config['feature_selection']['enabled']:
            X = self._select_features(X, y)
            
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        print(f"   ✓ Training set: {len(self.X_train)} samples")
        print(f"   ✓ Test set: {len(self.X_test)} samples")
        print(f"   ✓ Final features: {self.X_train.shape[1]}")
        
        return True
        
    def _create_interaction_features(self):
        """Create interaction features between important variables."""
        print(f"   🔗 Creating interaction features...")
        
        # Key feature groups for interactions
        blight_features = [col for col in self.data.columns if 'blight' in col.lower()]
        trend_features = [col for col in self.data.columns if 'trend' in col.lower()]
        ratio_features = [col for col in self.data.columns if 'ratio' in col.lower()]
        
        interactions_created = 0
        
        # Blight × Trend interactions
        if len(blight_features) >= 2:
            for i, feat1 in enumerate(blight_features[:3]):  # Limit to avoid explosion
                for feat2 in blight_features[i+1:4]:
                    if feat1 != feat2:
                        interaction_name = f"interact_{feat1}_{feat2}"
                        self.data[interaction_name] = self.data[feat1] * self.data[feat2]
                        interactions_created += 1
                        
        # Trend × Ratio interactions
        if len(trend_features) > 0 and len(ratio_features) > 0:
            for trend_feat in trend_features[:2]:
                for ratio_feat in ratio_features[:2]:
                    interaction_name = f"interact_{trend_feat}_{ratio_feat}"
                    self.data[interaction_name] = self.data[trend_feat] * self.data[ratio_feat]
                    interactions_created += 1
                    
        print(f"     ✓ Created {interactions_created} interaction features")
        
    def _create_ratio_features(self):
        """Create additional ratio features."""
        print(f"   📊 Creating ratio features...")
        
        ratios_created = 0
        
        # Blight intensity ratios
        if 'blight_weighted_score' in self.data.columns and 'total_requests' in self.data.columns:
            self.data['blight_intensity_ratio'] = (
                self.data['blight_weighted_score'] / (self.data['total_requests'] + 1)
            )
            ratios_created += 1
            
        # Trend stability ratio
        if 'trend_overall_slope' in self.data.columns and 'trend_overall_r2' in self.data.columns:
            self.data['trend_stability_ratio'] = (
                self.data['trend_overall_r2'] / (abs(self.data['trend_overall_slope']) + 1)
            )
            ratios_created += 1
            
        # Seasonal consistency ratio
        seasonal_cols = [col for col in self.data.columns if 'seasonal' in col.lower()]
        if len(seasonal_cols) >= 2:
            seasonal_sum = self.data[seasonal_cols].sum(axis=1)
            seasonal_max = self.data[seasonal_cols].max(axis=1)
            self.data['seasonal_consistency_ratio'] = seasonal_max / (seasonal_sum + 1e-10)
            ratios_created += 1
            
        print(f"     ✓ Created {ratios_created} ratio features")
        
    def _create_polynomial_features(self):
        """Create polynomial features for key variables."""
        print(f"   🔢 Creating polynomial features...")
        
        # Key features for polynomial transformation
        poly_candidates = [
            'blight_weighted_score', 'total_requests', 'trend_overall_slope',
            'blight_rate', 'service_type_diversity'
        ]
        
        poly_created = 0
        for col in poly_candidates:
            if col in self.data.columns:
                # Square terms
                self.data[f"{col}_squared"] = self.data[col] ** 2
                # Log terms (for positive values)
                if (self.data[col] > 0).all():
                    self.data[f"{col}_log"] = np.log1p(self.data[col])
                poly_created += 2
                
        print(f"     ✓ Created {poly_created} polynomial features")
        
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Advanced feature selection using multiple methods."""
        print(f"   🎯 Feature Selection...")
        
        method = self.config['feature_selection']['method']
        k = self.config['feature_selection']['k_features']
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        else:
            print(f"   ⚠️  Unknown feature selection method: {method}")
            return X
            
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        print(f"   ✓ Selected {len(selected_features)} features using {method}")
        self.selected_features = selected_features
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
    def optimize_hyperparameters(self) -> Dict:
        """Advanced hyperparameter optimization using Optuna."""
        print(f"\n🔬 Hyperparameter Optimization with Optuna...")
        
        def objective(trial):
            """Optuna objective function."""
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': self.config['random_state'],
                'verbosity': 0,
                
                # Tree parameters
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                
                # Boosting parameters
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                
                # Regularization
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                
                # Advanced parameters
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
            }
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config['random_state'])
            model = xgb.XGBClassifier(**params)
            
            scores = cross_val_score(
                model, self.X_train, self.y_train, 
                cv=cv, scoring='roc_auc', n_jobs=-1
            )
            
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.config['random_state'])
        )
        
        study.optimize(
            objective, 
            n_trials=self.config['hyperparameter_tuning']['n_trials'],
            timeout=self.config['hyperparameter_tuning']['timeout']
        )
        
        self.best_params = study.best_params
        self.best_params['objective'] = 'binary:logistic'
        self.best_params['eval_metric'] = 'auc'
        self.best_params['random_state'] = self.config['random_state']
        self.best_params['verbosity'] = 0
        
        print(f"   ✓ Best CV Score: {study.best_value:.4f}")
        print(f"   ✓ Best parameters found after {len(study.trials)} trials")
        
        return self.best_params
        
    def train_model(self) -> bool:
        """Train the final XGBoost model with optimized parameters."""
        print(f"\n🤖 Training Final XGBoost Model...")
        
        # Get optimized parameters
        if self.best_params is None:
            self.best_params = self.optimize_hyperparameters()
            
        # Train final model
        self.model = xgb.XGBClassifier(**self.best_params)
        
        # Fit with evaluation set (early stopping is handled by n_estimators from optimization)
        eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        
        self.model.fit(
            self.X_train, self.y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"   ✓ Model trained successfully")
        
        # Check if best_iteration is available (depends on XGBoost version)
        try:
            if hasattr(self.model, 'best_iteration'):
                print(f"   ✓ Best iteration: {self.model.best_iteration}")
            else:
                print(f"   ✓ Used {self.best_params.get('n_estimators', 'default')} estimators")
        except:
            print(f"   ✓ Training completed")
        
        return True
        
    def evaluate_model(self) -> Dict:
        """Comprehensive model evaluation."""
        print(f"\n📊 Comprehensive Model Evaluation...")
        
        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Basic metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_pred_proba)
        avg_precision = average_precision_score(self.y_test, y_pred_proba)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': auc,
            'avg_precision': avg_precision
        }
        
        print(f"   ✓ Test Results:")
        print(f"     • Accuracy: {accuracy:.4f}")
        print(f"     • Precision: {precision:.4f}")
        print(f"     • Recall: {recall:.4f}")
        print(f"     • F1 Score: {f1:.4f}")
        print(f"     • ROC AUC: {auc:.4f}")
        print(f"     • Avg Precision: {avg_precision:.4f}")
        
        # Detailed classification report
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\n🔍 Confusion Matrix:")
        print(cm)
        
        # Feature importance
        print(f"\n🎯 Top 10 Most Important Features:")
        for i, (_, row) in enumerate(self.feature_importance.head(10).iterrows()):
            print(f"   {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
            
        return metrics
        
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print(f"\n📈 Creating Visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('XGBoost Model Analysis - Urban Blight Prediction', fontsize=16)
        
        # 1. Feature Importance
        top_features = self.feature_importance.head(15)
        axes[0, 0].barh(top_features['feature'], top_features['importance'])
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
        axes[1, 1].hist(y_pred_proba[self.y_test == 0], alpha=0.7, label='Low Risk', bins=20)
        axes[1, 1].hist(y_pred_proba[self.y_test == 1], alpha=0.7, label='High Risk', bins=20)
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Distribution')
        axes[1, 1].legend()
        
        # 6. Learning Curves
        results = self.model.evals_result()
        if results:
            train_auc = results['validation_0']['auc']
            test_auc = results['validation_1']['auc']
            epochs = range(len(train_auc))
            
            axes[1, 2].plot(epochs, train_auc, label='Training AUC')
            axes[1, 2].plot(epochs, test_auc, label='Validation AUC')
            axes[1, 2].set_xlabel('Epochs')
            axes[1, 2].set_ylabel('AUC')
            axes[1, 2].set_title('Learning Curves')
            axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(self.config['output_dir'] + '/model_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✓ Visualizations saved to {self.config['output_dir']}/model_analysis.png")
        
    def explain_model(self):
        """Generate SHAP explanations for model interpretability."""
        print(f"\n🔍 Generating SHAP Explanations...")
        
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.X_test)
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, self.X_test, show=False)
            plt.savefig(self.config['output_dir'] + '/shap_summary.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Feature importance plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, self.X_test, plot_type="bar", show=False)
            plt.savefig(self.config['output_dir'] + '/shap_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"   ✓ SHAP explanations saved to {self.config['output_dir']}/")
            
        except Exception as e:
            print(f"   ⚠️  SHAP explanation failed: {e}")
            
    def save_model(self):
        """Save the trained model and metadata."""
        print(f"\n💾 Saving Model and Metadata...")
        
        # Save model
        model_path = f"{self.config['output_dir']}/xgboost_urban_blight_model.pkl"
        joblib.dump(self.model, model_path)
        
        # Save metadata
        self.model_metadata = {
            'model_type': 'XGBoost',
            'model_version': '1.0',
            'training_date': datetime.now().isoformat(),
            'config': self.config,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance.to_dict('records'),
            'selected_features': self.selected_features,
            'label_encoders': {k: v.classes_.tolist() for k, v in self.label_encoders.items()},
            'data_summary': {
                'n_samples': len(self.data),
                'n_features': self.X_train.shape[1],
                'target_distribution': self.data[self.config['target_column']].value_counts().to_dict()
            }
        }
        
        metadata_path = f"{self.config['output_dir']}/model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2, default=str)
            
        print(f"   ✓ Model saved to: {model_path}")
        print(f"   ✓ Metadata saved to: {metadata_path}")
        
    def predict_all_wards(self) -> pd.DataFrame:
        """Generate predictions for all wards in the dataset."""
        print(f"\n🔮 Generating Predictions for All Wards...")
        
        # Prepare features for all wards
        exclude_cols = [
            'geometry', 'ward_name', 'AREA_NAME', 'NAME', 'WARD_NAME',
            self.config['target_column'], 'target_total_requests',
            'target_blight_requests', 'target_blight_rate'
        ]
        
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        X_all = self.data[feature_cols].copy()
        
        # Handle datetime columns - exclude them completely
        datetime_cols = X_all.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            X_all = X_all.drop(columns=datetime_cols)
        
        # Apply same preprocessing
        for col in self.label_encoders:
            if col in X_all.columns:
                X_all[col] = self.label_encoders[col].transform(X_all[col].astype(str))
        
        # Ensure all remaining columns are numeric
        non_numeric_cols = X_all.select_dtypes(exclude=['number']).columns
        if len(non_numeric_cols) > 0:
            for col in non_numeric_cols:
                X_all[col] = pd.to_numeric(X_all[col], errors='coerce')
                
        X_all = X_all.fillna(0)
        
        if self.selected_features:
            X_all = X_all[self.selected_features]
            
        # Generate predictions
        predictions = self.model.predict(X_all)
        prediction_probabilities = self.model.predict_proba(X_all)[:, 1]
        
        # Create results dataframe
        ward_name_col = 'ward_name' if 'ward_name' in self.data.columns else 'AREA_NAME'
        results_cols = [ward_name_col, self.config['target_column']]
        if 'AREA_NAME' in self.data.columns and ward_name_col != 'AREA_NAME':
            results_cols.append('AREA_NAME')
            
        results = self.data[results_cols].copy()
        results['predicted_risk'] = predictions
        results['risk_probability'] = prediction_probabilities
        results['risk_category'] = pd.cut(
            prediction_probabilities,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        print(f"   ✓ Generated predictions for {len(results)} wards")
        print(f"   ✓ Risk distribution: {results['risk_category'].value_counts().to_dict()}")
        
        return results
        
    def run_complete_pipeline(self) -> bool:
        """Run the complete model training pipeline."""
        try:
            # Load and prepare data
            if not self.load_and_prepare_data():
                return False
                
            # Feature engineering
            if not self.engineer_features():
                return False
                
            # Train model
            if not self.train_model():
                return False
                
            # Evaluate model
            metrics = self.evaluate_model()
            
            # Create visualizations
            self.create_visualizations()
            
            # Model explanations
            self.explain_model()
            
            # Save everything
            self.save_model()
            
            # Generate predictions
            predictions = self.predict_all_wards()
            predictions.to_csv(f"{self.config['output_dir']}/ward_predictions.csv", index=False)
            
            print(f"\n🎉 TRAINING COMPLETE!")
            print(f"   📊 Final Model Performance:")
            print(f"     • ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"     • F1 Score: {metrics['f1_score']:.4f}")
            print(f"     • Precision: {metrics['precision']:.4f}")
            print(f"     • Recall: {metrics['recall']:.4f}")
            print(f"   📁 All outputs saved to: {self.config['output_dir']}/")
            
            return True
            
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

# Configuration
CONFIG = {
    'input_file': 'model_ready_data.csv',  # Changed to CSV by default
    'target_column': 'is_high_blight_risk',
    'output_dir': 'model_outputs',
    'test_size': 0.2,
    'random_state': 42,
    
    'feature_selection': {
        'enabled': True,
        'method': 'mutual_info',  # 'mutual_info' or 'f_classif'
        'k_features': 30  # Number of top features to select
    },
    
    'hyperparameter_tuning': {
        'n_trials': 100,  # Number of Optuna trials
        'timeout': 1800   # Timeout in seconds (30 minutes)
    }
}

# Run the training pipeline
if __name__ == "__main__":
    import os
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Initialize and run the model
    model_trainer = UrbanBlightXGBoostModel(CONFIG)
    success = model_trainer.run_complete_pipeline()
    
    if success:
        print("\n🚀 XGBoost model training completed successfully!")
        print("   Ready for deployment and urban blight prediction!")
    else:
        print("\n💥 Model training failed. Please check the errors above.") 