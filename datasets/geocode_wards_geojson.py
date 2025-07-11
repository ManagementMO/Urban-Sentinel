import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class WardBasedModelDataGenerator:
    """
    Comprehensive Toronto Ward-Based Model Data Generator
    Optimized for Gradient Boosting Algorithms (XGBoost, LightGBM, CatBoost)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.df = None
        self.wards_gdf = None
        self.features_df = None
        self.final_gdf = None
        self.feature_metadata = {}
        
        # Advanced blight indicators with weights
        self.blight_indicators = {
            # Critical Infrastructure Failures (High Weight)
            'Road - Pot hole': 0.9,
            'Road - Damaged': 0.9,
            'Road - Sinking': 1.0,
            'Sidewalk - Damaged / Concrete': 0.8,
            'Sidewalk - Damaged /Brick/Interlock': 0.8,
            'Catch Basin - Blocked / Flooding': 0.9,
            
            # Visual Blight & Neglect (Medium-High Weight)
            'Graffiti': 0.7,
            'Graffiti - Private Property': 0.8,
            'Graffiti - Public Property': 0.6,
            'Road - Graffiti Complaint': 0.7,
            'Sidewalk - Graffiti Complaint': 0.7,
            
            # Waste & Sanitation (Medium Weight)
            'Litter / Bin / Overflow or Not Picked Up': 0.6,
            'Litter / Illegal Dumping Cleanup': 0.8,
            'Illegal Dumping': 0.9,
            'Illegal Dumping / Discharge': 0.9,
            'Garbage Collection - Missed Pick-Up': 0.5,
            
            # Property Neglect (High Weight)
            'Long Grass and Weeds': 0.8,
            'Property Standards': 0.9,
            'Construction-Unsafe/Untidy Condition': 0.9,
        }
        
        # Seasonal patterns for time-based features
        self.seasonal_complaints = {
            'winter': ['Snow / Ice Removal', 'Heating', 'Storm Clean Up'],
            'spring': ['Long Grass and Weeds', 'Property Standards'],
            'summer': ['Graffiti', 'Litter', 'Noise'],
            'fall': ['Storm Clean Up', 'Leaf Collection']
        }
        
    def load_and_validate_data(self) -> bool:
        """Load and validate all input data with comprehensive error handling."""
        print("=" * 80)
        print("🏗️  COMPREHENSIVE WARD-BASED MODEL DATA GENERATOR")
        print("   Optimized for Gradient Boosting Algorithms")
        print("=" * 80)
        
        # Check ward boundaries
        if not os.path.exists(self.config['ward_boundaries_file']):
            print(f"\n❌ ERROR: Ward boundaries file not found!")
            print(f"📋 DOWNLOAD INSTRUCTIONS:")
            print("1. Go to: https://open.toronto.ca/dataset/city-wards/")
            print("2. Download 'City Wards Data' as GeoJSON")
            print(f"3. Save as '{self.config['ward_boundaries_file']}'")
            return False
            
        # Check service requests
        if not os.path.exists(self.config['service_requests_csv']):
            print(f"\n❌ ERROR: Service requests file not found!")
            print(f"   Expected: {self.config['service_requests_csv']}")
            return False
            
        # Load service requests
        print(f"\n📊 Loading service requests...")
        try:
            self.df = pd.read_csv(self.config['service_requests_csv'])
            print(f"   ✓ Loaded {len(self.df):,} service requests")
        except Exception as e:
            print(f"   ❌ Error loading service requests: {e}")
            return False
            
        # Load ward boundaries
        print(f"\n🗺️  Loading ward boundaries...")
        try:
            self.wards_gdf = gpd.read_file(self.config['ward_boundaries_file'])
            print(f"   ✓ Loaded {len(self.wards_gdf)} ward boundaries")
        except Exception as e:
            print(f"   ❌ Error loading ward boundaries: {e}")
            return False
            
        return True
        
    def clean_and_prepare_data(self) -> bool:
        """Advanced data cleaning and preparation."""
        print(f"\n🧹 Advanced data cleaning and preparation...")
        
        # Convert dates with multiple format handling
        date_formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d']
        self.df['Creation Date'] = pd.to_datetime(self.df['Creation Date'], errors='coerce')
        
        # Extract temporal features
        self.df['year'] = self.df['Creation Date'].dt.year
        self.df['month'] = self.df['Creation Date'].dt.month
        self.df['day_of_week'] = self.df['Creation Date'].dt.dayofweek
        self.df['quarter'] = self.df['Creation Date'].dt.quarter
        self.df['season'] = self.df['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })
        
        # Clean ward names with advanced normalization
        self.df['Ward_Clean'] = (self.df['Ward']
                                .str.replace(r'\s*\(\d+\)', '', regex=True)
                                .str.strip()
                                .str.title()
                                .str.replace(r'\s+', ' ', regex=True))
        
        # Remove invalid records
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['Creation Date', 'Ward', 'Service Request Type'])
        
        # Filter to valid years
        valid_years = range(self.config['history_start_year'], 
                          self.config['target_year'] + 1)
        self.df = self.df[self.df['year'].isin(valid_years)]
        
        print(f"   ✓ Cleaned data: {len(self.df):,} records (removed {initial_count - len(self.df):,})")
        print(f"   ✓ Date range: {self.df['year'].min()} to {self.df['year'].max()}")
        print(f"   ✓ Unique wards: {self.df['Ward_Clean'].nunique()}")
        print(f"   ✓ Unique service types: {self.df['Service Request Type'].nunique()}")
        
        # Identify ward name column in boundaries
        ward_cols = ['AREA_NAME', 'NAME', 'Ward_Name', 'WARD_NAME', 'ward_name']
        self.ward_name_col = None
        for col in ward_cols:
            if col in self.wards_gdf.columns:
                self.ward_name_col = col
                break
                
        if self.ward_name_col is None:
            print(f"   ❌ Could not identify ward name column in boundaries")
            print(f"   Available columns: {list(self.wards_gdf.columns)}")
            return False
            
        # Normalize ward names in boundaries
        self.wards_gdf[self.ward_name_col] = (self.wards_gdf[self.ward_name_col]
                                             .str.strip()
                                             .str.title()
                                             .str.replace(r'\s+', ' ', regex=True))
        
        # Ensure CRS
        if self.wards_gdf.crs is None:
            self.wards_gdf = self.wards_gdf.set_crs('EPSG:4326')
            
        print(f"   ✓ Using '{self.ward_name_col}' as ward name column")
        
        return True
        
    def create_advanced_features(self) -> bool:
        """Create comprehensive features optimized for gradient boosting."""
        print(f"\n🔧 Creating advanced features for gradient boosting...")
        
        # Get ward matching info
        service_wards = set(self.df['Ward_Clean'].unique())
        boundary_wards = set(self.wards_gdf[self.ward_name_col].unique())
        matched_wards = service_wards.intersection(boundary_wards)
        
        print(f"   ✓ Service requests: {len(service_wards)} wards")
        print(f"   ✓ Boundary data: {len(boundary_wards)} wards")
        print(f"   ✓ Matched wards: {len(matched_wards)}")
        
        if len(matched_wards) < len(service_wards) * 0.8:
            print(f"   ⚠️  Low match rate: {len(matched_wards)}/{len(service_wards)}")
            
        # Create comprehensive features for each ward
        ward_features = []
        
        for ward in matched_wards:
            ward_data = self.df[self.df['Ward_Clean'] == ward].copy()
            features = self._create_ward_features(ward, ward_data)
            ward_features.append(features)
            
        self.features_df = pd.DataFrame(ward_features)
        print(f"   ✓ Created {len(self.features_df.columns)} features for {len(self.features_df)} wards")
        
        return True
        
    def _create_ward_features(self, ward: str, ward_data: pd.DataFrame) -> Dict:
        """Create comprehensive features for a single ward."""
        features = {'ward_name': ward}
        
        # Historical data (training period)
        historical_data = ward_data[
            (ward_data['year'] >= self.config['history_start_year']) & 
            (ward_data['year'] <= self.config['history_end_year'])
        ]
        
        # Target data (prediction period)
        target_data = ward_data[ward_data['year'] == self.config['target_year']]
        
        # === BASIC VOLUME FEATURES ===
        features.update(self._create_volume_features(historical_data, target_data))
        
        # === BLIGHT-SPECIFIC FEATURES ===
        features.update(self._create_blight_features(historical_data, target_data))
        
        # === TEMPORAL FEATURES ===
        features.update(self._create_temporal_features(historical_data))
        
        # === DIVERSITY FEATURES ===
        features.update(self._create_diversity_features(historical_data))
        
        # === TREND FEATURES ===
        features.update(self._create_trend_features(historical_data))
        
        # === SEASONAL FEATURES ===
        features.update(self._create_seasonal_features(historical_data))
        
        # === RATIO FEATURES ===
        features.update(self._create_ratio_features(historical_data))
        
        # === STATISTICAL FEATURES ===
        features.update(self._create_statistical_features(historical_data))
        
        return features
        
    def _create_volume_features(self, historical_data: pd.DataFrame, target_data: pd.DataFrame) -> Dict:
        """Create volume-based features."""
        years_span = self.config['history_end_year'] - self.config['history_start_year'] + 1
        
        return {
            'total_requests': len(historical_data),
            'total_requests_per_year': len(historical_data) / years_span,
            'target_total_requests': len(target_data),
            'requests_density': len(historical_data) / years_span if years_span > 0 else 0
        }
        
    def _create_blight_features(self, historical_data: pd.DataFrame, target_data: pd.DataFrame) -> Dict:
        """Create blight-specific features with weighted scores."""
        # Weighted blight score
        blight_data = historical_data[
            historical_data['Service Request Type'].isin(self.blight_indicators.keys())
        ]
        
        # Calculate weighted blight score
        weighted_blight_score = 0
        for _, row in blight_data.iterrows():
            complaint_type = row['Service Request Type']
            weight = self.blight_indicators.get(complaint_type, 0.5)
            weighted_blight_score += weight
            
        # Target blight data
        target_blight = target_data[
            target_data['Service Request Type'].isin(self.blight_indicators.keys())
        ]
        
        years_span = self.config['history_end_year'] - self.config['history_start_year'] + 1
        
        return {
            'blight_requests_total': len(blight_data),
            'blight_requests_per_year': len(blight_data) / years_span,
            'blight_weighted_score': weighted_blight_score,
            'blight_weighted_score_per_year': weighted_blight_score / years_span,
            'blight_rate': len(blight_data) / len(historical_data) if len(historical_data) > 0 else 0,
            'target_blight_requests': len(target_blight),
            'target_blight_rate': len(target_blight) / len(target_data) if len(target_data) > 0 else 0
        }
        
    def _create_temporal_features(self, historical_data: pd.DataFrame) -> Dict:
        """Create temporal pattern features."""
        if len(historical_data) == 0:
            return {f'temporal_{key}': 0 for key in [
                'weekday_rate', 'weekend_rate', 'peak_hour_rate',
                'morning_rate', 'afternoon_rate', 'evening_rate', 'night_rate'
            ]}
            
        # Day of week patterns
        weekday_count = len(historical_data[historical_data['day_of_week'] < 5])
        weekend_count = len(historical_data[historical_data['day_of_week'] >= 5])
        
        # Hour patterns (if available)
        if 'Creation Date' in historical_data.columns:
            historical_data['hour'] = historical_data['Creation Date'].dt.hour
            morning = len(historical_data[historical_data['hour'].between(6, 11)])
            afternoon = len(historical_data[historical_data['hour'].between(12, 17)])
            evening = len(historical_data[historical_data['hour'].between(18, 23)])
            night = len(historical_data[~historical_data['hour'].between(6, 23)])
            peak_hours = len(historical_data[historical_data['hour'].between(9, 17)])
        else:
            morning = afternoon = evening = night = peak_hours = 0
            
        total = len(historical_data)
        
        return {
            'temporal_weekday_rate': weekday_count / total if total > 0 else 0,
            'temporal_weekend_rate': weekend_count / total if total > 0 else 0,
            'temporal_peak_hour_rate': peak_hours / total if total > 0 else 0,
            'temporal_morning_rate': morning / total if total > 0 else 0,
            'temporal_afternoon_rate': afternoon / total if total > 0 else 0,
            'temporal_evening_rate': evening / total if total > 0 else 0,
            'temporal_night_rate': night / total if total > 0 else 0
        }
        
    def _create_diversity_features(self, historical_data: pd.DataFrame) -> Dict:
        """Create service type diversity features."""
        if len(historical_data) == 0:
            return {
                'service_type_diversity': 0,
                'division_diversity': 0,
                'top_service_dominance': 0
            }
            
        # Service type diversity (entropy)
        service_counts = historical_data['Service Request Type'].value_counts()
        service_probs = service_counts / service_counts.sum()
        service_entropy = -np.sum(service_probs * np.log2(service_probs + 1e-10))
        
        # Division diversity
        if 'Division' in historical_data.columns:
            division_counts = historical_data['Division'].value_counts()
            division_probs = division_counts / division_counts.sum()
            division_entropy = -np.sum(division_probs * np.log2(division_probs + 1e-10))
        else:
            division_entropy = 0
            
        # Top service dominance
        top_service_rate = service_counts.iloc[0] / len(historical_data) if len(service_counts) > 0 else 0
        
        return {
            'service_type_diversity': service_entropy,
            'division_diversity': division_entropy,
            'top_service_dominance': top_service_rate
        }
        
    def _create_trend_features(self, historical_data: pd.DataFrame) -> Dict:
        """Create trend analysis features."""
        if len(historical_data) == 0:
            return {f'trend_{key}': 0 for key in [
                'overall_slope', 'overall_r2', 'blight_slope', 'blight_r2',
                'recent_vs_early', 'acceleration'
            ]}
            
        # Yearly trends
        yearly_counts = historical_data.groupby('year').size()
        if len(yearly_counts) > 1:
            years = yearly_counts.index.values
            counts = yearly_counts.values
            
            # Overall trend
            overall_slope = np.polyfit(years, counts, 1)[0]
            overall_r2 = np.corrcoef(years, counts)[0, 1] ** 2 if len(years) > 1 else 0
            
            # Blight trend
            blight_data = historical_data[
                historical_data['Service Request Type'].isin(self.blight_indicators.keys())
            ]
            yearly_blight = blight_data.groupby('year').size()
            
            if len(yearly_blight) > 1:
                blight_slope = np.polyfit(yearly_blight.index.values, yearly_blight.values, 1)[0]
                blight_r2 = np.corrcoef(yearly_blight.index.values, yearly_blight.values)[0, 1] ** 2
            else:
                blight_slope = blight_r2 = 0
                
            # Recent vs early comparison
            mid_year = years[len(years)//2]
            early_avg = yearly_counts[yearly_counts.index < mid_year].mean()
            recent_avg = yearly_counts[yearly_counts.index >= mid_year].mean()
            recent_vs_early = (recent_avg - early_avg) / early_avg if early_avg > 0 else 0
            
            # Acceleration (second derivative)
            if len(years) > 2:
                acceleration = np.polyfit(years, counts, 2)[0]
            else:
                acceleration = 0
                
        else:
            overall_slope = overall_r2 = blight_slope = blight_r2 = 0
            recent_vs_early = acceleration = 0
            
        return {
            'trend_overall_slope': overall_slope,
            'trend_overall_r2': overall_r2,
            'trend_blight_slope': blight_slope,
            'trend_blight_r2': blight_r2,
            'trend_recent_vs_early': recent_vs_early,
            'trend_acceleration': acceleration
        }
        
    def _create_seasonal_features(self, historical_data: pd.DataFrame) -> Dict:
        """Create seasonal pattern features."""
        if len(historical_data) == 0:
            return {f'seasonal_{season}_rate': 0 for season in ['winter', 'spring', 'summer', 'fall']}
            
        total = len(historical_data)
        seasonal_features = {}
        
        for season in ['winter', 'spring', 'summer', 'fall']:
            seasonal_data = historical_data[historical_data['season'] == season]
            seasonal_features[f'seasonal_{season}_rate'] = len(seasonal_data) / total if total > 0 else 0
            
        return seasonal_features
        
    def _create_ratio_features(self, historical_data: pd.DataFrame) -> Dict:
        """Create ratio-based features."""
        if len(historical_data) == 0:
            return {f'ratio_{key}': 0 for key in [
                'blight_to_total', 'infrastructure_to_total', 'quality_to_total'
            ]}
            
        total = len(historical_data)
        
        # Blight ratio
        blight_count = len(historical_data[
            historical_data['Service Request Type'].isin(self.blight_indicators.keys())
        ])
        
        # Infrastructure issues
        infrastructure_types = ['Road', 'Sidewalk', 'Traffic', 'Bridge']
        infrastructure_count = len(historical_data[
            historical_data['Service Request Type'].str.contains('|'.join(infrastructure_types), na=False)
        ])
        
        # Quality of life issues
        quality_types = ['Noise', 'Bylaw', 'Parking', 'Animal']
        quality_count = len(historical_data[
            historical_data['Service Request Type'].str.contains('|'.join(quality_types), na=False)
        ])
        
        return {
            'ratio_blight_to_total': blight_count / total if total > 0 else 0,
            'ratio_infrastructure_to_total': infrastructure_count / total if total > 0 else 0,
            'ratio_quality_to_total': quality_count / total if total > 0 else 0
        }
        
    def _create_statistical_features(self, historical_data: pd.DataFrame) -> Dict:
        """Create statistical features."""
        if len(historical_data) == 0:
            return {f'stats_{key}': 0 for key in [
                'yearly_std', 'yearly_cv', 'monthly_std', 'monthly_cv',
                'peak_month_ratio', 'consistency_score'
            ]}
            
        # Yearly statistics
        yearly_counts = historical_data.groupby('year').size()
        yearly_std = yearly_counts.std() if len(yearly_counts) > 1 else 0
        yearly_cv = yearly_std / yearly_counts.mean() if yearly_counts.mean() > 0 else 0
        
        # Monthly statistics
        monthly_counts = historical_data.groupby('month').size()
        monthly_std = monthly_counts.std() if len(monthly_counts) > 1 else 0
        monthly_cv = monthly_std / monthly_counts.mean() if monthly_counts.mean() > 0 else 0
        
        # Peak month ratio
        peak_month_ratio = monthly_counts.max() / monthly_counts.mean() if monthly_counts.mean() > 0 else 0
        
        # Consistency score (inverse of coefficient of variation)
        consistency_score = 1 / (yearly_cv + 1) if yearly_cv >= 0 else 0
        
        return {
            'stats_yearly_std': yearly_std,
            'stats_yearly_cv': yearly_cv,
            'stats_monthly_std': monthly_std,
            'stats_monthly_cv': monthly_cv,
            'stats_peak_month_ratio': peak_month_ratio,
            'stats_consistency_score': consistency_score
        }
        
    def merge_and_finalize(self) -> bool:
        """Merge features with ward boundaries and create final dataset."""
        print(f"\n🔗 Merging features with ward boundaries...")
        
        # Merge with ward boundaries
        self.final_gdf = self.wards_gdf.merge(
            self.features_df,
            left_on=self.ward_name_col,
            right_on='ward_name',
            how='inner'
        )
        
        print(f"   ✓ Merged dataset: {len(self.final_gdf)} wards")
        
        # Create target variables
        self._create_target_variables()
        
        # Add feature metadata
        self._add_feature_metadata()
        
        # Validate features
        self._validate_features()
        
        return True
        
    def _create_target_variables(self):
        """Create multiple target variables for different modeling approaches."""
        print(f"   🎯 Creating target variables...")
        
        # Binary high-risk classification (75th percentile)
        blight_threshold_75 = self.final_gdf['target_blight_requests'].quantile(0.75)
        self.final_gdf['is_high_blight_risk'] = (
            self.final_gdf['target_blight_requests'] > blight_threshold_75
        ).astype(int)
        
        # Binary extreme-risk classification (90th percentile)
        blight_threshold_90 = self.final_gdf['target_blight_requests'].quantile(0.90)
        self.final_gdf['is_extreme_blight_risk'] = (
            self.final_gdf['target_blight_requests'] > blight_threshold_90
        ).astype(int)
        
        # Multi-class risk levels
        self.final_gdf['risk_level'] = pd.cut(
            self.final_gdf['target_blight_requests'],
            bins=[0, 
                  self.final_gdf['target_blight_requests'].quantile(0.33),
                  self.final_gdf['target_blight_requests'].quantile(0.67),
                  self.final_gdf['target_blight_requests'].max()],
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        # Continuous risk score (normalized)
        max_blight = self.final_gdf['target_blight_requests'].max()
        self.final_gdf['risk_score'] = (
            self.final_gdf['target_blight_requests'] / max_blight if max_blight > 0 else 0
        )
        
        print(f"   ✓ High-risk wards (75th percentile): {self.final_gdf['is_high_blight_risk'].sum()}")
        print(f"   ✓ Extreme-risk wards (90th percentile): {self.final_gdf['is_extreme_blight_risk'].sum()}")
        
    def _add_feature_metadata(self):
        """Add comprehensive feature metadata."""
        feature_cols = [col for col in self.final_gdf.columns 
                       if col not in ['geometry', self.ward_name_col, 'ward_name']]
        
        # Categorize features
        self.feature_metadata = {
            'volume_features': [col for col in feature_cols if col.startswith('total_') or col.startswith('requests_')],
            'blight_features': [col for col in feature_cols if 'blight' in col],
            'temporal_features': [col for col in feature_cols if col.startswith('temporal_')],
            'trend_features': [col for col in feature_cols if col.startswith('trend_')],
            'seasonal_features': [col for col in feature_cols if col.startswith('seasonal_')],
            'ratio_features': [col for col in feature_cols if col.startswith('ratio_')],
            'statistical_features': [col for col in feature_cols if col.startswith('stats_')],
            'diversity_features': [col for col in feature_cols if 'diversity' in col or 'dominance' in col],
            'target_features': [col for col in feature_cols if col.startswith('target_') or col.startswith('is_') or col.startswith('risk_')]
        }
        
    def _validate_features(self):
        """Validate feature quality and completeness."""
        print(f"   🔍 Validating feature quality...")
        
        feature_cols = [col for col in self.final_gdf.columns 
                       if col not in ['geometry', self.ward_name_col, 'ward_name']]
        
        # Check for missing values
        missing_counts = self.final_gdf[feature_cols].isnull().sum()
        if missing_counts.sum() > 0:
            print(f"   ⚠️  Missing values found in {(missing_counts > 0).sum()} features")
            
        # Separate numeric and categorical columns
        numeric_cols = self.final_gdf[feature_cols].select_dtypes(include=[np.number]).columns
        categorical_cols = self.final_gdf[feature_cols].select_dtypes(include=['category']).columns
        other_cols = [col for col in feature_cols if col not in numeric_cols and col not in categorical_cols]
        
        # Handle infinite values in numeric columns
        if len(numeric_cols) > 0:
            inf_counts = np.isinf(self.final_gdf[numeric_cols]).sum()
            if inf_counts.sum() > 0:
                print(f"   ⚠️  Infinite values found in {(inf_counts > 0).sum()} numeric features")
                # Replace inf with large finite values
                self.final_gdf[numeric_cols] = self.final_gdf[numeric_cols].replace([np.inf, -np.inf], [1e10, -1e10])
            
            # Fill missing values in numeric columns
            self.final_gdf[numeric_cols] = self.final_gdf[numeric_cols].fillna(0)
            
        # Handle categorical columns
        if len(categorical_cols) > 0:
            print(f"   🔧 Processing {len(categorical_cols)} categorical features...")
            for col in categorical_cols:
                if self.final_gdf[col].isnull().any():
                    # For categorical columns, use the first category or 'Unknown'
                    if col == 'risk_level':
                        # For risk_level, use 'Low' as default
                        self.final_gdf[col] = self.final_gdf[col].cat.add_categories(['Low'])
                        self.final_gdf[col] = self.final_gdf[col].fillna('Low')
                    else:
                        # For other categorical columns, add 'Unknown' category
                        if 'Unknown' not in self.final_gdf[col].cat.categories:
                            self.final_gdf[col] = self.final_gdf[col].cat.add_categories(['Unknown'])
                        self.final_gdf[col] = self.final_gdf[col].fillna('Unknown')
                        
        # Handle other columns (object, etc.)
        if len(other_cols) > 0:
            for col in other_cols:
                if self.final_gdf[col].dtype == 'object':
                    self.final_gdf[col] = self.final_gdf[col].fillna('Unknown')
                else:
                    self.final_gdf[col] = self.final_gdf[col].fillna(0)
        
        print(f"   ✓ Feature validation complete")
        
    def save_results(self) -> bool:
        """Save the final dataset with comprehensive metadata."""
        print(f"\n💾 Saving comprehensive model-ready dataset...")
        
        # Save main GeoJSON
        self.final_gdf.to_file(self.config['output_geojson'], driver='GeoJSON')
        
        # Save feature metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'config': self.config,
            'data_summary': {
                'total_wards': len(self.final_gdf),
                'total_features': len([col for col in self.final_gdf.columns 
                                     if col not in ['geometry', self.ward_name_col, 'ward_name']]),
                'date_range': f"{self.config['history_start_year']}-{self.config['history_end_year']}",
                'target_year': self.config['target_year'],
                'blight_indicators': len(self.blight_indicators),
                'high_risk_wards': int(self.final_gdf['is_high_blight_risk'].sum()),
                'extreme_risk_wards': int(self.final_gdf['is_extreme_blight_risk'].sum())
            },
            'feature_metadata': self.feature_metadata,
            'blight_indicators': self.blight_indicators,
            'target_variables': {
                'is_high_blight_risk': 'Binary classification (75th percentile)',
                'is_extreme_blight_risk': 'Binary classification (90th percentile)',
                'risk_level': 'Multi-class classification (Low/Medium/High)',
                'risk_score': 'Continuous risk score (0-1)'
            }
        }
        
        metadata_file = self.config['output_geojson'].replace('.geojson', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        print(f"   ✓ Main dataset: {self.config['output_geojson']}")
        print(f"   ✓ Metadata: {metadata_file}")
        
        return True
        
    def print_summary(self):
        """Print comprehensive summary of the generated dataset."""
        print(f"\n📈 COMPREHENSIVE DATASET SUMMARY")
        print("=" * 80)
        
        # Basic stats
        print(f"🏢 Dataset Overview:")
        print(f"   • Total wards: {len(self.final_gdf)}")
        print(f"   • Training period: {self.config['history_start_year']}-{self.config['history_end_year']}")
        print(f"   • Target year: {self.config['target_year']}")
        print(f"   • Total features: {len([col for col in self.final_gdf.columns if col not in ['geometry', self.ward_name_col, 'ward_name']])}")
        
        # Target variable distribution
        print(f"\n🎯 Target Variable Distribution:")
        print(f"   • High-risk wards: {self.final_gdf['is_high_blight_risk'].sum()}/{len(self.final_gdf)} ({self.final_gdf['is_high_blight_risk'].mean():.1%})")
        print(f"   • Extreme-risk wards: {self.final_gdf['is_extreme_blight_risk'].sum()}/{len(self.final_gdf)} ({self.final_gdf['is_extreme_blight_risk'].mean():.1%})")
        
        # Feature categories
        print(f"\n🔧 Feature Categories:")
        for category, features in self.feature_metadata.items():
            if features and not category.endswith('_features'):
                continue
            print(f"   • {category.replace('_', ' ').title()}: {len(features)}")
            
        # Top features by variance (proxy for importance)
        numeric_cols = self.final_gdf.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols 
                       if col not in ['geometry', self.ward_name_col, 'ward_name'] 
                       and not col.startswith('target_')]
        
        if len(feature_cols) > 0:
            variances = self.final_gdf[feature_cols].var().sort_values(ascending=False)
            print(f"\n📊 Top Features by Variance:")
            for i, (feature, variance) in enumerate(variances.head(10).items()):
                print(f"   {i+1:2d}. {feature}: {variance:.4f}")
                
        print(f"\n✅ SUCCESS! Comprehensive dataset ready for gradient boosting!")
        print(f"   📁 File: {self.config['output_geojson']}")
        print(f"   🎯 Recommended target: 'is_high_blight_risk' for binary classification")
        print(f"   🎯 Alternative targets: 'risk_level' (multi-class) or 'risk_score' (regression)")
        
    def run(self) -> bool:
        """Run the complete data generation pipeline."""
        try:
            if not self.load_and_validate_data():
                return False
                
            if not self.clean_and_prepare_data():
                return False
                
            if not self.create_advanced_features():
                return False
                
            if not self.merge_and_finalize():
                return False
                
            if not self.save_results():
                return False
                
            self.print_summary()
            return True
            
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

# Configuration
CONFIG = {
    'service_requests_csv': 'SRALL.csv',
    'ward_boundaries_file': 'ward_data.geojson',
    'output_geojson': 'model_ready_data.geojson',
    'history_start_year': 2014,
    'history_end_year': 2023,
    'target_year': 2024
}

# Run the generator
if __name__ == "__main__":
    generator = WardBasedModelDataGenerator(CONFIG)
    success = generator.run()
    
    if success:
        print("\n🎉 Model-ready dataset generation completed successfully!")
        print("   Ready for XGBoost, LightGBM, CatBoost, or any gradient boosting algorithm!")
    else:
        print("\n💥 Dataset generation failed. Please check the errors above.") 