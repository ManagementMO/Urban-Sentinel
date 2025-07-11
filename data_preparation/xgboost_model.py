import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import warnings
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import optuna
from optuna.samplers import TPESampler
import shap
import geopandas as gpd

warnings.filterwarnings('ignore')

class OptimizedUrbanBlightXGBoostModel:
    """
    Optimized XGBoost Model for Urban Blight Prediction
    Specifically designed for the comprehensive CSV output from geocode_wards_csv.py
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
        self.selected_features: Optional[List[str]] = None
        self.model_metadata: Dict = {}
        self.geojson_data: Optional[gpd.GeoDataFrame] = None
        
    def load_and_validate_data(self) -> bool:
        """Load and validate the comprehensive CSV dataset."""
        print("=" * 80)
        print("🚀 OPTIMIZED XGBOOST MODEL TRAINER")
        print("   Urban Blight Prediction - Comprehensive Feature Set")
        print("=" * 80)
        
        print(f"\n📊 Loading dataset from '{self.config['input_file']}'...")
        
        try:
            # Load CSV data
            self.data = pd.read_csv(self.config['input_file'])
            
            if self.data is None or len(self.data) == 0:
                print("   ❌ Failed to load data or empty dataset")
                return False
                
            print(f"   ✓ Loaded {len(self.data)} wards")
            print(f"   ✓ Total columns: {len(self.data.columns)}")
            
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            return False
            
        # Load GeoJSON data if available
        self._load_geojson_data()
        
        return self._validate_data()
        
    def _load_geojson_data(self):
        """Load GeoJSON data for spatial visualization."""
        geojson_file = self.config.get('geojson_file')
        if not geojson_file:
            # Try to infer GeoJSON filename from CSV filename
            if self.config['input_file'].endswith('.csv'):
                geojson_file = self.config['input_file'].replace('.csv', '.geojson')
            else:
                print("   ⚠️  No GeoJSON file specified, spatial visualization will be unavailable")
                return
                
        if os.path.exists(geojson_file):
            try:
                print(f"   🗺️  Loading GeoJSON from '{geojson_file}'...")
                self.geojson_data = gpd.read_file(geojson_file)
                print(f"   ✓ Loaded {len(self.geojson_data)} geometries")
                
                # Check if ward names match
                csv_wards = set(self.data.get('ward_name', []))
                geojson_wards = set()
                
                # Try different possible ward name columns
                for col in ['ward_name', 'AREA_NAME', 'NAME', 'WARD_NAME']:
                    if col in self.geojson_data.columns:
                        geojson_wards = set(self.geojson_data[col])
                        break
                        
                matched_wards = csv_wards.intersection(geojson_wards)
                if len(matched_wards) > 0:
                    print(f"   ✓ Matched {len(matched_wards)} wards between CSV and GeoJSON")
                else:
                    print("   ⚠️  No matching wards found between CSV and GeoJSON")
                    
            except Exception as e:
                print(f"   ⚠️  Error loading GeoJSON: {e}")
                self.geojson_data = None
        else:
            print(f"   ⚠️  GeoJSON file not found: {geojson_file}")
            
    def _validate_data(self) -> bool:
        """Validate the loaded data."""
            
        # Validate target column
        if self.config['target_column'] not in self.data.columns:
            print(f"   ❌ Target column '{self.config['target_column']}' not found!")
            available_targets = [col for col in self.data.columns if col.startswith('is_') or col.startswith('risk_')]
            if available_targets:
                print(f"   Available target columns: {available_targets}")
            return False
            
        # Display dataset info
        print(f"\n📈 Dataset Overview:")
        print(f"   • Total wards: {len(self.data)}")
        print(f"   • Total features: {len(self.data.columns)}")
        print(f"   • Target variable: {self.config['target_column']}")
        
        # Check target distribution
        target_dist = self.data[self.config['target_column']].value_counts()
        print(f"   • Target distribution: {dict(target_dist)}")
        
        # Check for missing values
        missing_values = self.data.isnull().sum().sum()
        if missing_values > 0:
            print(f"   ⚠️  Missing values: {missing_values}")
        else:
            print(f"   ✅ No missing values detected")
            
        return True
        
    def prepare_features(self) -> bool:
        """Prepare features for training with intelligent selection."""
        print(f"\n🔧 Preparing Features for Training...")
        
        if self.data is None:
            print("   ❌ No data loaded")
            return False
        
        # Identify feature columns (exclude identifiers and targets)
        exclude_patterns = [
            'ward_name', 'AREA_NAME', 'NAME', 'WARD_NAME',
            'target_', 'is_', 'risk_', 'geometry'
        ]
        
        # Get all columns
        all_columns = list(self.data.columns)
        
        # Filter out excluded columns
        feature_cols = []
        for col in all_columns:
            if not any(pattern in col for pattern in exclude_patterns):
                feature_cols.append(col)
        
        print(f"   ✓ Identified {len(feature_cols)} potential features")
        
        # Prepare feature matrix and target
        X = self.data[feature_cols].copy()
        y = self.data[self.config['target_column']].copy()
        
        # Handle any remaining non-numeric columns
        non_numeric_cols = X.select_dtypes(exclude=['number']).columns
        if len(non_numeric_cols) > 0:
            print(f"   🔧 Converting {len(non_numeric_cols)} non-numeric columns")
            for col in non_numeric_cols:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Fill any remaining NaN values
        X = X.fillna(0)
        
        # Verify all columns are numeric
        final_non_numeric = X.select_dtypes(exclude=['number']).columns
        if len(final_non_numeric) > 0:
            print(f"   ❌ Still have non-numeric columns: {list(final_non_numeric)}")
            return False
            
        print(f"   ✅ All {X.shape[1]} features are numeric")
        
        # Remove features with zero variance
        zero_var_cols = X.columns[X.var() == 0]
        if len(zero_var_cols) > 0:
            print(f"   🧹 Removing {len(zero_var_cols)} zero-variance features")
            X = X.drop(columns=zero_var_cols)
        
        # Remove highly correlated features
        if self.config.get('remove_correlated_features', True):
            X = self._remove_correlated_features(X)
        
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
        
        print(f"   ✅ Data prepared successfully:")
        print(f"     • Training samples: {len(self.X_train)}")
        print(f"     • Test samples: {len(self.X_test)}")
        print(f"     • Final features: {self.X_train.shape[1]}")
        
        return True
        
    def _remove_correlated_features(self, X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features."""
        print(f"   🔍 Removing features with correlation > {threshold}")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find highly correlated pairs
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        if len(to_drop) > 0:
            print(f"     • Removing {len(to_drop)} highly correlated features")
            X = X.drop(columns=to_drop)
        else:
            print(f"     • No highly correlated features found")
            
        return X
        
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Advanced feature selection."""
        print(f"   🎯 Feature Selection using {self.config['feature_selection']['method']}")
        
        method = self.config['feature_selection']['method']
        k = min(self.config['feature_selection']['k_features'], X.shape[1])
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        else:
            print(f"   ⚠️  Unknown method {method}, using mutual_info")
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        print(f"     • Selected {len(selected_features)} features from {X.shape[1]}")
        self.selected_features = selected_features
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
    def optimize_hyperparameters(self) -> Dict:
        """Optimize hyperparameters using Optuna."""
        print(f"\n🔬 Hyperparameter Optimization...")
        
        def objective(trial):
            """Optuna objective function."""
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': self.config['random_state'],
                'verbosity': 0,
                'n_jobs': -1,
                
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
        self.best_params['n_jobs'] = -1
        
        print(f"   ✅ Optimization completed:")
        print(f"     • Best CV Score: {study.best_value:.4f}")
        print(f"     • Trials completed: {len(study.trials)}")
        
        return self.best_params
        
    def train_model(self) -> bool:
        """Train the final XGBoost model."""
        print(f"\n🤖 Training Final XGBoost Model...")
        
        if self.best_params is None:
            print("   🔧 Running hyperparameter optimization first...")
            self.optimize_hyperparameters()
            
        # Train model
        self.model = xgb.XGBClassifier(**self.best_params)
        self.model.fit(self.X_train, self.y_train)
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"   ✅ Model trained successfully")
        print(f"   📊 Using {self.best_params.get('n_estimators', 'default')} estimators")
        
        return True
        
    def evaluate_model(self) -> Dict:
        """Comprehensive model evaluation."""
        print(f"\n📊 Model Evaluation...")
        
        if self.model is None:
            print("   ❌ No model trained")
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
            'avg_precision': average_precision_score(self.y_test, y_pred_proba)
        }
        
        print(f"   ✅ Model Performance:")
        for metric, value in metrics.items():
            print(f"     • {metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Feature importance
        print(f"\n🎯 Top 10 Most Important Features:")
        for i, (_, row) in enumerate(self.feature_importance.head(10).iterrows()):
            print(f"   {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
        
        return metrics
        
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print(f"\n📈 Creating Visualizations...")
        
        if self.model is None:
            print("   ❌ No model to visualize")
            return
        
        # Set up plotting
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('XGBoost Model Analysis - Urban Blight Prediction', fontsize=16)
        
        # 1. Feature Importance
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
        axes[1, 1].hist(y_pred_proba[self.y_test == 0], alpha=0.7, label='Low Risk', bins=20)
        axes[1, 1].hist(y_pred_proba[self.y_test == 1], alpha=0.7, label='High Risk', bins=20)
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Distribution')
        axes[1, 1].legend()
        
        # 6. Feature Categories
        feature_categories = self._categorize_features()
        if feature_categories:
            category_counts = {k: len(v) for k, v in feature_categories.items()}
            axes[1, 2].pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
            axes[1, 2].set_title('Feature Categories')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(self.config['output_dir'], 'model_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualizations saved to {output_path}")
        
        if self.config.get('show_plots', True):
            plt.show()
        else:
            plt.close()
            
    def _categorize_features(self) -> Dict[str, List[str]]:
        """Categorize features by type."""
        if self.selected_features is None:
            return {}
            
        categories = {
            'Volume': [],
            'Blight': [],
            'Temporal': [],
            'Trend': [],
            'Seasonal': [],
            'Ratio': [],
            'Statistical': [],
            'Diversity': [],
            'Other': []
        }
        
        for feature in self.selected_features:
            if any(word in feature.lower() for word in ['requests_', 'total_']):
                categories['Volume'].append(feature)
            elif 'blight' in feature.lower():
                categories['Blight'].append(feature)
            elif any(word in feature.lower() for word in ['temporal_', 'weekday_', 'weekend_', 'hour_', 'morning_', 'afternoon_', 'evening_']):
                categories['Temporal'].append(feature)
            elif 'trend' in feature.lower():
                categories['Trend'].append(feature)
            elif 'seasonal' in feature.lower():
                categories['Seasonal'].append(feature)
            elif 'ratio' in feature.lower():
                categories['Ratio'].append(feature)
            elif any(word in feature.lower() for word in ['stats_', 'monthly_', 'quarterly_', 'weekly_']):
                categories['Statistical'].append(feature)
            elif 'diversity' in feature.lower():
                categories['Diversity'].append(feature)
            else:
                categories['Other'].append(feature)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
        
    def explain_model(self):
        """Generate SHAP explanations."""
        print(f"\n🔍 Generating SHAP Explanations...")
        
        if self.model is None:
            print("   ❌ No model to explain")
            return
        
        try:
            # Create explainer
            explainer = shap.TreeExplainer(self.model)
            
            # Use a subset of test data for efficiency
            n_samples = min(100, len(self.X_test))
            sample_indices = np.random.choice(len(self.X_test), n_samples, replace=False)
            X_sample = self.X_test.iloc[sample_indices]
            
            shap_values = explainer.shap_values(X_sample)
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            
            output_path = os.path.join(self.config['output_dir'], 'shap_summary.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ SHAP summary saved to {output_path}")
            
            if self.config.get('show_plots', True):
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"   ⚠️  SHAP explanation failed: {e}")
            
    def save_model(self):
        """Save model and metadata."""
        print(f"\n💾 Saving Model...")
        
        if self.model is None:
            print("   ❌ No model to save")
            return
        
        # Save model
        model_path = os.path.join(self.config['output_dir'], 'xgboost_urban_blight_model.pkl')
        joblib.dump(self.model, model_path)
        
        # Save metadata
        self.model_metadata = {
            'model_type': 'XGBoost',
            'model_version': '2.0',
            'training_date': datetime.now().isoformat(),
            'config': self.config,
            'best_params': self.best_params,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance.to_dict('records') if self.feature_importance is not None else [],
            'data_summary': {
                'n_samples': len(self.data) if self.data is not None else 0,
                'n_features': len(self.selected_features) if self.selected_features else 0,
                'target_distribution': self.data[self.config['target_column']].value_counts().to_dict() if self.data is not None else {}
            }
        }
        
        metadata_path = os.path.join(self.config['output_dir'], 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2, default=str)
            
        print(f"   ✅ Model saved to: {model_path}")
        print(f"   ✅ Metadata saved to: {metadata_path}")
        
    def predict_all_wards(self) -> pd.DataFrame:
        """Generate predictions for all wards."""
        print(f"\n🔮 Generating Predictions for All Wards...")
        
        if self.model is None or self.data is None:
            print("   ❌ No model or data available")
            return pd.DataFrame()
        
        # Prepare features
        exclude_patterns = [
            'ward_name', 'AREA_NAME', 'NAME', 'WARD_NAME',
            'target_', 'is_', 'risk_', 'geometry'
        ]
        
        feature_cols = []
        for col in self.data.columns:
            if not any(pattern in col for pattern in exclude_patterns):
                feature_cols.append(col)
        
        X_all = self.data[feature_cols].copy()
        
        # Apply same preprocessing
        X_all = X_all.fillna(0)
        
        # Select same features
        if self.selected_features:
            X_all = X_all[self.selected_features]
            
        # Generate predictions
        predictions = self.model.predict(X_all)
        prediction_probabilities = self.model.predict_proba(X_all)[:, 1]
        
        # Create results
        ward_col = 'ward_name' if 'ward_name' in self.data.columns else 'AREA_NAME'
        results = pd.DataFrame({
            'ward_name': self.data[ward_col] if ward_col in self.data.columns else range(len(self.data)),
            'actual_risk': self.data[self.config['target_column']],
            'predicted_risk': predictions,
            'risk_probability': prediction_probabilities
        })
        
        # Add risk categories
        results['risk_category'] = pd.cut(
            prediction_probabilities,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        print(f"   ✅ Generated predictions for {len(results)} wards")
        
        return results
        
    def create_prediction_geojson(self, predictions: pd.DataFrame) -> bool:
        """Create GeoJSON file with predictions for Mapbox visualization."""
        print(f"\n🗺️  Creating Prediction GeoJSON for Mapbox...")
        
        if self.geojson_data is None:
            print("   ❌ No GeoJSON data available - skipping spatial output")
            return False
            
        if predictions.empty:
            print("   ❌ No predictions available")
            return False
            
        try:
            # Find the ward name column in GeoJSON
            ward_col_geojson = None
            for col in ['ward_name', 'AREA_NAME', 'NAME', 'WARD_NAME']:
                if col in self.geojson_data.columns:
                    ward_col_geojson = col
                    break
                    
            if ward_col_geojson is None:
                print("   ❌ Could not find ward name column in GeoJSON")
                return False
                
            # Debug: Print column info before merge
            print(f"   🔍 Debug - Before merge:")
            print(f"     • GeoJSON columns: {list(self.geojson_data.columns)}")
            print(f"     • Predictions columns: {list(predictions.columns)}")
            
            # Merge predictions with geometry
            prediction_gdf = self.geojson_data.merge(
                predictions,
                left_on=ward_col_geojson,
                right_on='ward_name',
                how='left',
                suffixes=('_geojson', '_prediction')
            )
            
            # Debug: Print column info after merge
            print(f"   🔍 Debug - After merge:")
            print(f"     • Merged columns: {list(prediction_gdf.columns)}")
            print(f"     • Duplicate columns: {prediction_gdf.columns.duplicated().sum()}")
            
            # Fill missing predictions with default values
            prediction_gdf['predicted_risk'] = prediction_gdf['predicted_risk'].fillna(0)
            prediction_gdf['risk_probability'] = prediction_gdf['risk_probability'].fillna(0.0)
            
            # Handle categorical risk_category column properly
            if 'risk_category' in prediction_gdf.columns:
                if prediction_gdf['risk_category'].dtype.name == 'category':
                    # Add 'Unknown' to categories if it doesn't exist
                    if 'Unknown' not in prediction_gdf['risk_category'].cat.categories:
                        prediction_gdf['risk_category'] = prediction_gdf['risk_category'].cat.add_categories(['Unknown'])
                    prediction_gdf['risk_category'] = prediction_gdf['risk_category'].fillna('Unknown')
                else:
                    prediction_gdf['risk_category'] = prediction_gdf['risk_category'].fillna('Unknown')
            
            prediction_gdf['actual_risk'] = prediction_gdf['actual_risk'].fillna(0)
            
            # Add additional visualization-friendly columns
            prediction_gdf['risk_score_normalized'] = (
                prediction_gdf['risk_probability'] * 100
            ).round(1)
            
            # Create color coding for Mapbox
            def get_risk_color(prob):
                if prob >= 0.7:
                    return '#d73027'  # High risk - Red
                elif prob >= 0.4:
                    return '#fc8d59'  # Medium risk - Orange
                elif prob >= 0.2:
                    return '#fee08b'  # Low-medium risk - Yellow
                else:
                    return '#91cf60'  # Low risk - Green
                    
            prediction_gdf['risk_color'] = prediction_gdf['risk_probability'].apply(get_risk_color)
            
            # Create risk intensity for fill opacity
            prediction_gdf['risk_intensity'] = np.clip(
                prediction_gdf['risk_probability'] * 0.8 + 0.2, 0.2, 1.0
            )
            
            # Add descriptive text for popups
            prediction_gdf['risk_description'] = prediction_gdf.apply(
                lambda row: f"Risk Level: {row['risk_category']}<br>" +
                           f"Probability: {row['risk_score_normalized']:.1f}%<br>" +
                           f"Prediction: {'High Risk' if row['predicted_risk'] == 1 else 'Low Risk'}<br>" +
                           f"Actual: {'High Risk' if row['actual_risk'] == 1 else 'Low Risk'}",
                axis=1
            )
            
            # Select columns for final output, handling potential suffixes
            base_columns = [
                ward_col_geojson, 'ward_name', 'actual_risk', 'predicted_risk',
                'risk_probability', 'risk_score_normalized', 'risk_category',
                'risk_color', 'risk_intensity', 'risk_description', 'geometry'
            ]
            
            # Find available columns, checking for suffixes
            available_columns = []
            for col in base_columns:
                if col in prediction_gdf.columns:
                    available_columns.append(col)
                elif f'{col}_prediction' in prediction_gdf.columns:
                    available_columns.append(f'{col}_prediction')
                elif f'{col}_geojson' in prediction_gdf.columns:
                    available_columns.append(f'{col}_geojson')
            
            # Always include geometry if it exists
            if 'geometry' in prediction_gdf.columns:
                if 'geometry' not in available_columns:
                    available_columns.append('geometry')
            
            final_gdf = prediction_gdf[available_columns].copy()
            
            # Rename columns to remove suffixes for clean output
            rename_dict = {}
            for col in final_gdf.columns:
                if col.endswith('_prediction'):
                    base_name = col.replace('_prediction', '')
                    if base_name not in final_gdf.columns:
                        rename_dict[col] = base_name
                elif col.endswith('_geojson'):
                    base_name = col.replace('_geojson', '')
                    if base_name not in final_gdf.columns:
                        rename_dict[col] = base_name
            
            if rename_dict:
                final_gdf = final_gdf.rename(columns=rename_dict)
            
            # Check for and handle any remaining duplicate columns
            duplicate_cols = final_gdf.columns[final_gdf.columns.duplicated()].tolist()
            if duplicate_cols:
                print(f"   ⚠️  Found duplicate columns: {duplicate_cols}")
                # Remove duplicate columns by keeping only the first occurrence
                final_gdf = final_gdf.loc[:, ~final_gdf.columns.duplicated()]
                print(f"   ✓ Removed duplicate columns")
            
            # Debug: Print column info
            print(f"   📊 Final GeoDataFrame info:")
            print(f"     • Columns: {list(final_gdf.columns)}")
            print(f"     • Shape: {final_gdf.shape}")
            print(f"     • Duplicate columns: {final_gdf.columns.duplicated().sum()}")
            
            # Save GeoJSON
            output_path = os.path.join(self.config['output_dir'], 'ward_predictions.geojson')
            final_gdf.to_file(output_path, driver='GeoJSON')
            
            print(f"   ✅ Prediction GeoJSON saved to: {output_path}")
            print(f"   📊 Statistics:")
            print(f"     • Total wards: {len(final_gdf)}")
            print(f"     • High risk wards: {len(final_gdf[final_gdf['predicted_risk'] == 1])}")
            print(f"     • Medium risk wards: {len(final_gdf[final_gdf['risk_category'] == 'Medium Risk'])}")
            print(f"     • Low risk wards: {len(final_gdf[final_gdf['risk_category'] == 'Low Risk'])}")
            print(f"   🎨 Ready for Mapbox visualization!")
            
            # Also create a simplified version for web use
            simplified_gdf = final_gdf.copy()
            # Simplify geometries for web performance
            simplified_gdf['geometry'] = simplified_gdf['geometry'].simplify(tolerance=0.001)
            
            simplified_path = os.path.join(self.config['output_dir'], 'ward_predictions_simplified.geojson')
            simplified_gdf.to_file(simplified_path, driver='GeoJSON')
            print(f"   ✅ Simplified GeoJSON saved to: {simplified_path}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error creating prediction GeoJSON: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def save_predictions_with_geometry(self, predictions: pd.DataFrame) -> bool:
        """Save predictions with geometry in CSV format for easy inspection."""
        print(f"\n📄 Saving Predictions with Geometry (CSV)...")
        
        try:
            # Find the ward name column in GeoJSON
            ward_col_geojson = None
            for col in ['ward_name', 'AREA_NAME', 'NAME', 'WARD_NAME']:
                if col in self.geojson_data.columns:
                    ward_col_geojson = col
                    break
                    
            if ward_col_geojson is None:
                print("   ❌ Could not find ward name column in GeoJSON")
                return False
                
            # Merge predictions with geometry data
            geometry_df = self.geojson_data[[ward_col_geojson, 'geometry']].copy()
            
            # Convert geometry to Well-Known Text (WKT) for CSV readability
            geometry_df['geometry_wkt'] = geometry_df['geometry'].apply(lambda x: x.wkt if x is not None else '')
            
            # Merge with predictions
            predictions_with_geom = predictions.merge(
                geometry_df,
                left_on='ward_name',
                right_on=ward_col_geojson,
                how='left'
            )
            
            # Reorganize columns for better readability
            column_order = [
                'ward_name', 'actual_risk', 'predicted_risk', 'risk_probability', 
                'risk_category', 'geometry_wkt'
            ]
            
            # Add any additional columns that exist
            for col in predictions_with_geom.columns:
                if col not in column_order and col != ward_col_geojson and col != 'geometry':
                    column_order.append(col)
            
            # Filter to only include columns that exist
            available_columns = [col for col in column_order if col in predictions_with_geom.columns]
            final_df = predictions_with_geom[available_columns]
            
            # Save to CSV
            output_path = os.path.join(self.config['output_dir'], 'ward_predictions_with_geometry.csv')
            final_df.to_csv(output_path, index=False)
            
            print(f"   ✅ Predictions with geometry saved to: {output_path}")
            print(f"   📊 Includes {len(final_df)} wards with WKT geometry for inspection")
            print(f"   💡 Geometry column 'geometry_wkt' contains Well-Known Text format")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error saving predictions with geometry: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def run_complete_pipeline(self) -> bool:
        """Run the complete training pipeline."""
        try:
            print("🚀 Starting Complete XGBoost Training Pipeline...")
            
            # Load data
            if not self.load_and_validate_data():
                return False
                
            # Prepare features
            if not self.prepare_features():
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
            
            # Save model
            self.save_model()
            
            # Generate predictions
            predictions = self.predict_all_wards()
            if not predictions.empty:
                # Save basic CSV predictions
                pred_path = os.path.join(self.config['output_dir'], 'ward_predictions.csv')
                predictions.to_csv(pred_path, index=False)
                print(f"   ✅ Predictions saved to: {pred_path}")
                
                # Create GeoJSON for Mapbox visualization
                self.create_prediction_geojson(predictions)
                
                # Save CSV with geometry for inspection
                if self.geojson_data is not None:
                    self.save_predictions_with_geometry(predictions)
            
            print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print(f"   📊 Final Performance:")
            if metrics:
                print(f"     • ROC AUC: {metrics.get('roc_auc', 0):.4f}")
                print(f"     • F1 Score: {metrics.get('f1_score', 0):.4f}")
                print(f"     • Precision: {metrics.get('precision', 0):.4f}")
                print(f"     • Recall: {metrics.get('recall', 0):.4f}")
            print(f"   📁 All outputs saved to: {self.config['output_dir']}/")
            
            if self.geojson_data is not None:
                print(f"   🗺️  GeoJSON files ready for Mapbox visualization!")
                print(f"     • Full detail: ward_predictions.geojson")
                print(f"     • Web optimized: ward_predictions_simplified.geojson")
            
            return True
            
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

# Optimized Configuration
CONFIG = {
    'input_file': 'model_ready_data.csv',
    'geojson_file': 'model_ready_data.geojson',  # Optional: for spatial visualization
    'target_column': 'is_high_blight_risk',
    'output_dir': 'model_outputs',
    'test_size': 0.2,
    'random_state': 42,
    'show_plots': False,  # Set to True if you want to see plots
    'remove_correlated_features': True,
    
    'feature_selection': {
        'enabled': True,
        'method': 'mutual_info',  # 'mutual_info' or 'f_classif'
        'k_features': 50  # Increased for comprehensive feature set
    },
    
    'hyperparameter_tuning': {
        'n_trials': 100,
        'timeout': 1800  # 30 minutes
    }
}

# Main execution
if __name__ == "__main__":
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Initialize and run model
    model_trainer = OptimizedUrbanBlightXGBoostModel(CONFIG)
    success = model_trainer.run_complete_pipeline()
    
    if success:
        print("\n🚀 XGBoost model training completed successfully!")
        print("   Ready for deployment and urban blight prediction!")
        print("   Model optimized for comprehensive CSV feature set!")
    else:
        print("\n💥 Model training failed. Please check the errors above.") 