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

class WardBasedModelDataGeneratorCSV:
    """
    Comprehensive Toronto Ward-Based Model Data Generator - CSV Version
    Optimized for Gradient Boosting Algorithms (XGBoost, LightGBM, CatBoost)
    Outputs clean CSV files for easy ML model training
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.df: Optional[pd.DataFrame] = None
        self.wards_gdf: Optional[gpd.GeoDataFrame] = None
        self.features_df: Optional[pd.DataFrame] = None
        self.final_df: Optional[pd.DataFrame] = None
        self.feature_metadata: Dict = {}
        self.ward_name_col: Optional[str] = None
        
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
        print("🏗️  COMPREHENSIVE WARD-BASED MODEL DATA GENERATOR - CSV VERSION")
        print("   Optimized for Gradient Boosting Algorithms")
        print("=" * 80)
        
        # Check ward boundaries (optional for CSV version)
        ward_boundaries_available = os.path.exists(self.config['ward_boundaries_file'])
        if not ward_boundaries_available:
            print(f"\n⚠️  Ward boundaries file not found - will use ward names only")
            print(f"   Expected: {self.config['ward_boundaries_file']}")
            print(f"   Continuing without geographic boundaries...")
            
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
            
        # Load ward boundaries if available
        if ward_boundaries_available:
            print(f"\n🗺️  Loading ward boundaries...")
            try:
                self.wards_gdf = gpd.read_file(self.config['ward_boundaries_file'])
                print(f"   ✓ Loaded {len(self.wards_gdf)} ward boundaries")
            except Exception as e:
                print(f"   ⚠️  Error loading ward boundaries: {e}")
                print(f"   Continuing without geographic boundaries...")
                self.wards_gdf = None
        else:
            self.wards_gdf = None
            
        return True
        
    def clean_and_prepare_data(self) -> bool:
        """Advanced data cleaning and preparation."""
        print(f"\n🧹 Advanced data cleaning and preparation...")
        
        # Validate that data was loaded
        if self.df is None:
            print("❌ ERROR: No data loaded. Cannot proceed with cleaning.")
            return False
        
        # Convert dates with multiple format handling
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
        
        # Validate we still have data after cleaning
        if len(self.df) == 0:
            print("❌ ERROR: No data remaining after cleaning. Check your date range and data quality.")
            return False
        
        print(f"   ✓ Cleaned data: {len(self.df):,} records (removed {initial_count - len(self.df):,})")
        print(f"   ✓ Date range: {self.df['year'].min()} to {self.df['year'].max()}")
        print(f"   ✓ Unique wards: {self.df['Ward_Clean'].nunique()}")
        print(f"   ✓ Unique service types: {self.df['Service Request Type'].nunique()}")
        
        # Handle ward boundaries if available
        if self.wards_gdf is not None:
            # Identify ward name column in boundaries
            ward_cols = ['AREA_NAME', 'NAME', 'Ward_Name', 'WARD_NAME', 'ward_name']
            self.ward_name_col = None
            for col in ward_cols:
                if col in self.wards_gdf.columns:
                    self.ward_name_col = col
                    break
                    
            if self.ward_name_col is None:
                print(f"   ⚠️  Could not identify ward name column in boundaries")
                print(f"   Available columns: {list(self.wards_gdf.columns)}")
                self.wards_gdf = None
            else:
                # Normalize ward names in boundaries
                self.wards_gdf[self.ward_name_col] = (self.wards_gdf[self.ward_name_col]
                                                     .str.strip()
                                                     .str.title()
                                                     .str.replace(r'\s+', ' ', regex=True))
                print(f"   ✓ Using '{self.ward_name_col}' as ward name column")
        
        return True
        
    def create_advanced_features(self) -> bool:
        """Create comprehensive features optimized for gradient boosting."""
        print(f"\n🔧 Creating advanced features for gradient boosting...")
        
        # Validate that data was loaded and cleaned
        if self.df is None:
            print("❌ ERROR: No data loaded. Cannot create features.")
            return False
        
        # Get all unique wards from service requests
        service_wards = set(self.df['Ward_Clean'].unique())
        
        # If we have ward boundaries, get matching wards
        if self.wards_gdf is not None:
            boundary_wards = set(self.wards_gdf[self.ward_name_col].unique())
            matched_wards = service_wards.intersection(boundary_wards)
            
            print(f"   ✓ Service requests: {len(service_wards)} wards")
            print(f"   ✓ Boundary data: {len(boundary_wards)} wards")
            print(f"   ✓ Matched wards: {len(matched_wards)}")
            
            if len(matched_wards) < len(service_wards) * 0.8:
                print(f"   ⚠️  Low match rate: {len(matched_wards)}/{len(service_wards)}")
                
            # Use matched wards
            wards_to_process = matched_wards
        else:
            # Use all service request wards
            wards_to_process = service_wards
            print(f"   ✓ Processing all {len(service_wards)} wards from service requests")
            
        # Create comprehensive features for each ward
        ward_features = []
        
        for ward in wards_to_process:
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
        """Create volume-based features with year-by-year breakdown."""
        features = {}
        
        # Overall totals
        features['total_requests'] = len(historical_data)
        features['target_total_requests'] = len(target_data)
        
        # Year-by-year request counts
        yearly_counts = historical_data.groupby('year').size()
        for year in range(self.config['history_start_year'], self.config['history_end_year'] + 1):
            features[f'requests_{year}'] = yearly_counts.get(year, 0)
        
        # Statistical measures across years
        if len(yearly_counts) > 0:
            features['requests_mean_per_year'] = yearly_counts.mean()
            features['requests_std_per_year'] = yearly_counts.std() if len(yearly_counts) > 1 else 0
            features['requests_min_per_year'] = yearly_counts.min()
            features['requests_max_per_year'] = yearly_counts.max()
            features['requests_range_per_year'] = yearly_counts.max() - yearly_counts.min()
        else:
            features['requests_mean_per_year'] = 0
            features['requests_std_per_year'] = 0
            features['requests_min_per_year'] = 0
            features['requests_max_per_year'] = 0
            features['requests_range_per_year'] = 0
        
        # Growth patterns
        if len(yearly_counts) > 1:
            # Year-over-year growth rates
            yearly_growth = yearly_counts.pct_change().fillna(0)
            features['requests_avg_growth_rate'] = yearly_growth.mean()
            features['requests_growth_volatility'] = yearly_growth.std()
            
            # Recent vs early period comparison
            years_list = sorted(yearly_counts.index)
            mid_point = len(years_list) // 2
            early_years = years_list[:mid_point]
            recent_years = years_list[mid_point:]
            
            early_avg = yearly_counts[early_years].mean() if early_years else 0
            recent_avg = yearly_counts[recent_years].mean() if recent_years else 0
            
            features['requests_early_period_avg'] = early_avg
            features['requests_recent_period_avg'] = recent_avg
            features['requests_recent_vs_early_ratio'] = recent_avg / early_avg if early_avg > 0 else 0
        else:
            features['requests_avg_growth_rate'] = 0
            features['requests_growth_volatility'] = 0
            features['requests_early_period_avg'] = 0
            features['requests_recent_period_avg'] = 0
            features['requests_recent_vs_early_ratio'] = 0
        
        return features
        
    def _create_blight_features(self, historical_data: pd.DataFrame, target_data: pd.DataFrame) -> Dict:
        """Create blight-specific features with weighted scores and year-by-year breakdown."""
        features = {}
        
        # Filter blight data
        blight_data = historical_data[
            historical_data['Service Request Type'].isin(list(self.blight_indicators.keys()))
        ]
        
        # Target blight data
        target_blight = target_data[
            target_data['Service Request Type'].isin(list(self.blight_indicators.keys()))
        ]
        
        # Overall blight metrics
        features['blight_requests_total'] = len(blight_data)
        features['target_blight_requests'] = len(target_blight)
        features['blight_rate'] = len(blight_data) / len(historical_data) if len(historical_data) > 0 else 0
        features['target_blight_rate'] = len(target_blight) / len(target_data) if len(target_data) > 0 else 0
        
        # Year-by-year blight counts
        yearly_blight_counts = blight_data.groupby('year').size()
        for year in range(self.config['history_start_year'], self.config['history_end_year'] + 1):
            features[f'blight_requests_{year}'] = yearly_blight_counts.get(year, 0)
        
        # Calculate weighted blight scores by year
        yearly_weighted_scores = {}
        for year in range(self.config['history_start_year'], self.config['history_end_year'] + 1):
            year_blight_data = blight_data[blight_data['year'] == year]
            weighted_score = 0
            for _, row in year_blight_data.iterrows():
                complaint_type = row['Service Request Type']
                weight = self.blight_indicators.get(complaint_type, 0.5)
                weighted_score += weight
            yearly_weighted_scores[year] = weighted_score
            features[f'blight_weighted_score_{year}'] = weighted_score
        
        # Overall weighted metrics
        total_weighted_score = sum(yearly_weighted_scores.values())
        features['blight_weighted_score_total'] = total_weighted_score
        
        # Statistical measures for blight across years
        if len(yearly_blight_counts) > 0:
            features['blight_mean_per_year'] = yearly_blight_counts.mean()
            features['blight_std_per_year'] = yearly_blight_counts.std() if len(yearly_blight_counts) > 1 else 0
            features['blight_min_per_year'] = yearly_blight_counts.min()
            features['blight_max_per_year'] = yearly_blight_counts.max()
            features['blight_range_per_year'] = yearly_blight_counts.max() - yearly_blight_counts.min()
        else:
            features['blight_mean_per_year'] = 0
            features['blight_std_per_year'] = 0
            features['blight_min_per_year'] = 0
            features['blight_max_per_year'] = 0
            features['blight_range_per_year'] = 0
        
        # Blight growth patterns
        if len(yearly_blight_counts) > 1:
            blight_growth = yearly_blight_counts.pct_change().fillna(0)
            features['blight_avg_growth_rate'] = blight_growth.mean()
            features['blight_growth_volatility'] = blight_growth.std()
            
            # Recent vs early blight comparison
            years_list = sorted(yearly_blight_counts.index)
            mid_point = len(years_list) // 2
            early_years = years_list[:mid_point]
            recent_years = years_list[mid_point:]
            
            early_blight_avg = yearly_blight_counts[early_years].mean() if early_years else 0
            recent_blight_avg = yearly_blight_counts[recent_years].mean() if recent_years else 0
            
            features['blight_early_period_avg'] = early_blight_avg
            features['blight_recent_period_avg'] = recent_blight_avg
            features['blight_recent_vs_early_ratio'] = recent_blight_avg / early_blight_avg if early_blight_avg > 0 else 0
        else:
            features['blight_avg_growth_rate'] = 0
            features['blight_growth_volatility'] = 0
            features['blight_early_period_avg'] = 0
            features['blight_recent_period_avg'] = 0
            features['blight_recent_vs_early_ratio'] = 0
        
        # Blight type diversity by year
        yearly_blight_types = blight_data.groupby('year')['Service Request Type'].nunique()
        for year in range(self.config['history_start_year'], self.config['history_end_year'] + 1):
            features[f'blight_type_diversity_{year}'] = yearly_blight_types.get(year, 0)
        
        # Average blight intensity (weighted score per request)
        features['blight_intensity_avg'] = total_weighted_score / len(blight_data) if len(blight_data) > 0 else 0
        
        return features
        
    def _create_temporal_features(self, historical_data: pd.DataFrame) -> Dict:
        """Create comprehensive temporal pattern features with year-by-year breakdown."""
        features = {}
        
        if len(historical_data) == 0:
            # Initialize all temporal features to 0
            for year in range(self.config['history_start_year'], self.config['history_end_year'] + 1):
                features[f'weekday_requests_{year}'] = 0
                features[f'weekend_requests_{year}'] = 0
                features[f'monday_requests_{year}'] = 0
                features[f'tuesday_requests_{year}'] = 0
                features[f'wednesday_requests_{year}'] = 0
                features[f'thursday_requests_{year}'] = 0
                features[f'friday_requests_{year}'] = 0
                features[f'saturday_requests_{year}'] = 0
                features[f'sunday_requests_{year}'] = 0
                
                for month in range(1, 13):
                    features[f'month_{month}_requests_{year}'] = 0
                
                for quarter in range(1, 5):
                    features[f'quarter_{quarter}_requests_{year}'] = 0
                    
                if 'Creation Date' in historical_data.columns:
                    features[f'morning_requests_{year}'] = 0
                    features[f'afternoon_requests_{year}'] = 0
                    features[f'evening_requests_{year}'] = 0
                    features[f'night_requests_{year}'] = 0
                    features[f'peak_hours_requests_{year}'] = 0
                    
            return features
        
        # Year-by-year temporal breakdown
        for year in range(self.config['history_start_year'], self.config['history_end_year'] + 1):
            year_data = historical_data[historical_data['year'] == year]
            
            if len(year_data) > 0:
                # Day of week patterns by year
                features[f'weekday_requests_{year}'] = len(year_data[year_data['day_of_week'] < 5])
                features[f'weekend_requests_{year}'] = len(year_data[year_data['day_of_week'] >= 5])
                
                # Individual day patterns by year
                for day_num, day_name in enumerate(['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
                    features[f'{day_name}_requests_{year}'] = len(year_data[year_data['day_of_week'] == day_num])
                
                # Monthly patterns by year
                for month in range(1, 13):
                    features[f'month_{month}_requests_{year}'] = len(year_data[year_data['month'] == month])
                
                # Quarterly patterns by year
                for quarter in range(1, 5):
                    features[f'quarter_{quarter}_requests_{year}'] = len(year_data[year_data['quarter'] == quarter])
                
                # Hour patterns by year (if available)
                if 'Creation Date' in year_data.columns:
                    year_data_copy = year_data.copy()
                    year_data_copy['hour'] = year_data_copy['Creation Date'].dt.hour
                    
                    features[f'morning_requests_{year}'] = len(year_data_copy[year_data_copy['hour'].between(6, 11)])
                    features[f'afternoon_requests_{year}'] = len(year_data_copy[year_data_copy['hour'].between(12, 17)])
                    features[f'evening_requests_{year}'] = len(year_data_copy[year_data_copy['hour'].between(18, 23)])
                    features[f'night_requests_{year}'] = len(year_data_copy[~year_data_copy['hour'].between(6, 23)])
                    features[f'peak_hours_requests_{year}'] = len(year_data_copy[year_data_copy['hour'].between(9, 17)])
                    
                    # Hour-by-hour breakdown by year
                    for hour in range(24):
                        features[f'hour_{hour}_requests_{year}'] = len(year_data_copy[year_data_copy['hour'] == hour])
                else:
                    features[f'morning_requests_{year}'] = 0
                    features[f'afternoon_requests_{year}'] = 0
                    features[f'evening_requests_{year}'] = 0
                    features[f'night_requests_{year}'] = 0
                    features[f'peak_hours_requests_{year}'] = 0
                    
                    for hour in range(24):
                        features[f'hour_{hour}_requests_{year}'] = 0
            else:
                # No data for this year
                features[f'weekday_requests_{year}'] = 0
                features[f'weekend_requests_{year}'] = 0
                
                for day_name in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
                    features[f'{day_name}_requests_{year}'] = 0
                
                for month in range(1, 13):
                    features[f'month_{month}_requests_{year}'] = 0
                
                for quarter in range(1, 5):
                    features[f'quarter_{quarter}_requests_{year}'] = 0
                    
                features[f'morning_requests_{year}'] = 0
                features[f'afternoon_requests_{year}'] = 0
                features[f'evening_requests_{year}'] = 0
                features[f'night_requests_{year}'] = 0
                features[f'peak_hours_requests_{year}'] = 0
                
                for hour in range(24):
                    features[f'hour_{hour}_requests_{year}'] = 0
        
        # Overall temporal statistics (aggregated from yearly data)
        total = len(historical_data)
        if total > 0:
            features['temporal_weekday_rate_overall'] = len(historical_data[historical_data['day_of_week'] < 5]) / total
            features['temporal_weekend_rate_overall'] = len(historical_data[historical_data['day_of_week'] >= 5]) / total
            
            # Most active day/month/quarter overall
            day_counts = historical_data['day_of_week'].value_counts()
            features['most_active_day_overall'] = day_counts.idxmax() if len(day_counts) > 0 else 0
            features['most_active_day_count_overall'] = day_counts.max() if len(day_counts) > 0 else 0
            
            month_counts = historical_data['month'].value_counts()
            features['most_active_month_overall'] = month_counts.idxmax() if len(month_counts) > 0 else 0
            features['most_active_month_count_overall'] = month_counts.max() if len(month_counts) > 0 else 0
            
            quarter_counts = historical_data['quarter'].value_counts()
            features['most_active_quarter_overall'] = quarter_counts.idxmax() if len(quarter_counts) > 0 else 0
            features['most_active_quarter_count_overall'] = quarter_counts.max() if len(quarter_counts) > 0 else 0
        else:
            features['temporal_weekday_rate_overall'] = 0
            features['temporal_weekend_rate_overall'] = 0
            features['most_active_day_overall'] = 0
            features['most_active_day_count_overall'] = 0
            features['most_active_month_overall'] = 0
            features['most_active_month_count_overall'] = 0
            features['most_active_quarter_overall'] = 0
            features['most_active_quarter_count_overall'] = 0
        
        return features
        
    def _create_diversity_features(self, historical_data: pd.DataFrame) -> Dict:
        """Create comprehensive service type diversity features with year-by-year breakdown."""
        features = {}
        
        if len(historical_data) == 0:
            # Initialize all diversity features to 0
            for year in range(self.config['history_start_year'], self.config['history_end_year'] + 1):
                features[f'service_type_count_{year}'] = 0
                features[f'service_type_diversity_{year}'] = 0
                features[f'division_count_{year}'] = 0
                features[f'division_diversity_{year}'] = 0
                features[f'top_service_dominance_{year}'] = 0
                features[f'top_service_count_{year}'] = 0
                
                # Top 10 service types by year
                for i in range(1, 11):
                    features[f'top_{i}_service_count_{year}'] = 0
                    features[f'top_{i}_service_rate_{year}'] = 0
                    
            return features
        
        # Year-by-year diversity analysis
        for year in range(self.config['history_start_year'], self.config['history_end_year'] + 1):
            year_data = historical_data[historical_data['year'] == year]
            
            if len(year_data) > 0:
                # Service type analysis by year
                service_counts = year_data['Service Request Type'].value_counts()
                features[f'service_type_count_{year}'] = len(service_counts)
                
                # Service type diversity (entropy) by year
                if len(service_counts) > 0:
                    service_probs = service_counts / service_counts.sum()
                    service_entropy = -np.sum(service_probs * np.log2(service_probs + 1e-10))
                    features[f'service_type_diversity_{year}'] = service_entropy
                    
                    # Top service dominance by year
                    features[f'top_service_dominance_{year}'] = service_counts.iloc[0] / len(year_data)
                    features[f'top_service_count_{year}'] = service_counts.iloc[0]
                    
                    # Top 10 service types by year
                    for i in range(1, 11):
                        if i <= len(service_counts):
                            features[f'top_{i}_service_count_{year}'] = service_counts.iloc[i-1]
                            features[f'top_{i}_service_rate_{year}'] = service_counts.iloc[i-1] / len(year_data)
                        else:
                            features[f'top_{i}_service_count_{year}'] = 0
                            features[f'top_{i}_service_rate_{year}'] = 0
                else:
                    features[f'service_type_diversity_{year}'] = 0
                    features[f'top_service_dominance_{year}'] = 0
                    features[f'top_service_count_{year}'] = 0
                    
                    for i in range(1, 11):
                        features[f'top_{i}_service_count_{year}'] = 0
                        features[f'top_{i}_service_rate_{year}'] = 0
                
                # Division analysis by year
                if 'Division' in year_data.columns:
                    division_counts = year_data['Division'].value_counts()
                    features[f'division_count_{year}'] = len(division_counts)
                    
                    if len(division_counts) > 0:
                        division_probs = division_counts / division_counts.sum()
                        division_entropy = -np.sum(division_probs * np.log2(division_probs + 1e-10))
                        features[f'division_diversity_{year}'] = division_entropy
                    else:
                        features[f'division_diversity_{year}'] = 0
                else:
                    features[f'division_count_{year}'] = 0
                    features[f'division_diversity_{year}'] = 0
            else:
                # No data for this year
                features[f'service_type_count_{year}'] = 0
                features[f'service_type_diversity_{year}'] = 0
                features[f'division_count_{year}'] = 0
                features[f'division_diversity_{year}'] = 0
                features[f'top_service_dominance_{year}'] = 0
                features[f'top_service_count_{year}'] = 0
                
                for i in range(1, 11):
                    features[f'top_{i}_service_count_{year}'] = 0
                    features[f'top_{i}_service_rate_{year}'] = 0
        
        # Overall diversity statistics
        service_counts_overall = historical_data['Service Request Type'].value_counts()
        features['service_type_count_overall'] = len(service_counts_overall)
        
        if len(service_counts_overall) > 0:
            service_probs_overall = service_counts_overall / service_counts_overall.sum()
            features['service_type_diversity_overall'] = -np.sum(service_probs_overall * np.log2(service_probs_overall + 1e-10))
            features['top_service_dominance_overall'] = service_counts_overall.iloc[0] / len(historical_data)
            
            # Most common service types overall
            for i in range(1, 11):
                if i <= len(service_counts_overall):
                    features[f'top_{i}_service_count_overall'] = service_counts_overall.iloc[i-1]
                    features[f'top_{i}_service_rate_overall'] = service_counts_overall.iloc[i-1] / len(historical_data)
                else:
                    features[f'top_{i}_service_count_overall'] = 0
                    features[f'top_{i}_service_rate_overall'] = 0
        else:
            features['service_type_diversity_overall'] = 0
            features['top_service_dominance_overall'] = 0
            
            for i in range(1, 11):
                features[f'top_{i}_service_count_overall'] = 0
                features[f'top_{i}_service_rate_overall'] = 0
        
        # Division diversity overall
        if 'Division' in historical_data.columns:
            division_counts_overall = historical_data['Division'].value_counts()
            features['division_count_overall'] = len(division_counts_overall)
            
            if len(division_counts_overall) > 0:
                division_probs_overall = division_counts_overall / division_counts_overall.sum()
                features['division_diversity_overall'] = -np.sum(division_probs_overall * np.log2(division_probs_overall + 1e-10))
            else:
                features['division_diversity_overall'] = 0
        else:
            features['division_count_overall'] = 0
            features['division_diversity_overall'] = 0
        
        # Diversity trends over time
        yearly_diversity = []
        yearly_service_counts = []
        for year in range(self.config['history_start_year'], self.config['history_end_year'] + 1):
            yearly_diversity.append(features[f'service_type_diversity_{year}'])
            yearly_service_counts.append(features[f'service_type_count_{year}'])
        
        if len(yearly_diversity) > 1:
            diversity_series = pd.Series(yearly_diversity)
            service_count_series = pd.Series(yearly_service_counts)
            
            features['diversity_trend_slope'] = diversity_series.diff().mean()
            features['diversity_volatility'] = diversity_series.std()
            features['service_count_trend_slope'] = service_count_series.diff().mean()
            features['service_count_volatility'] = service_count_series.std()
        else:
            features['diversity_trend_slope'] = 0
            features['diversity_volatility'] = 0
            features['service_count_trend_slope'] = 0
            features['service_count_volatility'] = 0
        
        return features
        
    def _create_trend_features(self, historical_data: pd.DataFrame) -> Dict:
        """Create comprehensive trend analysis features with detailed year-by-year insights."""
        features = {}
        
        if len(historical_data) == 0:
            # Initialize all trend features to 0
            for year in range(self.config['history_start_year'], self.config['history_end_year'] + 1):
                features[f'requests_count_{year}'] = 0
                features[f'requests_growth_rate_{year}'] = 0
                features[f'requests_year_over_year_change_{year}'] = 0
                
                # Blight trend by year
                features[f'blight_count_{year}'] = 0
                features[f'blight_growth_rate_{year}'] = 0
                features[f'blight_year_over_year_change_{year}'] = 0
                
            # Overall trend features
            features['trend_overall_slope'] = 0
            features['trend_overall_r2'] = 0
            features['trend_blight_slope'] = 0
            features['trend_blight_r2'] = 0
            features['trend_recent_vs_early'] = 0
            features['trend_acceleration'] = 0
            
            return features
        
        # Year-by-year request counts
        yearly_counts = historical_data.groupby('year').size()
        
        # Year-by-year blight counts
        blight_data = historical_data[historical_data['Service Request Type'].isin(self.blight_indicators.keys())]
        yearly_blight_counts = blight_data.groupby('year').size() if len(blight_data) > 0 else pd.Series(dtype=int)
        
        # Fill in missing years with 0
        for year in range(self.config['history_start_year'], self.config['history_end_year'] + 1):
            features[f'requests_count_{year}'] = yearly_counts.get(year, 0)
            features[f'blight_count_{year}'] = yearly_blight_counts.get(year, 0)
        
        # Calculate year-over-year changes and growth rates
        prev_year_requests = 0
        prev_year_blight = 0
        
        for year in range(self.config['history_start_year'], self.config['history_end_year'] + 1):
            current_requests = features[f'requests_count_{year}']
            current_blight = features[f'blight_count_{year}']
            
            if year > self.config['history_start_year']:
                # Year-over-year change
                features[f'requests_year_over_year_change_{year}'] = current_requests - prev_year_requests
                features[f'blight_year_over_year_change_{year}'] = current_blight - prev_year_blight
                
                # Growth rate
                if prev_year_requests > 0:
                    features[f'requests_growth_rate_{year}'] = (current_requests - prev_year_requests) / prev_year_requests
                else:
                    features[f'requests_growth_rate_{year}'] = 0
                    
                if prev_year_blight > 0:
                    features[f'blight_growth_rate_{year}'] = (current_blight - prev_year_blight) / prev_year_blight
                else:
                    features[f'blight_growth_rate_{year}'] = 0
            else:
                features[f'requests_year_over_year_change_{year}'] = 0
                features[f'requests_growth_rate_{year}'] = 0
                features[f'blight_year_over_year_change_{year}'] = 0
                features[f'blight_growth_rate_{year}'] = 0
            
            prev_year_requests = current_requests
            prev_year_blight = current_blight
        
        # Overall trend analysis for requests
        if len(yearly_counts) > 1:
            years = yearly_counts.index.values
            counts = yearly_counts.values
            
            # Overall trend
            features['trend_overall_slope'] = np.polyfit(years, counts, 1)[0]
            features['trend_overall_r2'] = np.corrcoef(years, counts)[0, 1] ** 2 if len(years) > 1 else 0
            
            # Trend acceleration (second derivative)
            if len(counts) > 2:
                features['trend_acceleration'] = np.polyfit(years, counts, 2)[0]
            else:
                features['trend_acceleration'] = 0
        else:
            features['trend_overall_slope'] = 0
            features['trend_overall_r2'] = 0
            features['trend_acceleration'] = 0
        
        # Overall trend analysis for blight
        if len(yearly_blight_counts) > 1:
            blight_years = yearly_blight_counts.index.values
            blight_counts = yearly_blight_counts.values
            
            # Blight trend
            features['trend_blight_slope'] = np.polyfit(blight_years, blight_counts, 1)[0]
            features['trend_blight_r2'] = np.corrcoef(blight_years, blight_counts)[0, 1] ** 2
            
            # Blight trend acceleration
            if len(blight_counts) > 2:
                features['trend_blight_acceleration'] = np.polyfit(blight_years, blight_counts, 2)[0]
            else:
                features['trend_blight_acceleration'] = 0
        else:
            features['trend_blight_slope'] = 0
            features['trend_blight_r2'] = 0
            features['trend_blight_acceleration'] = 0
        
        # Period-based comparisons
        if len(yearly_counts) > 1:
            years = yearly_counts.index.values
            mid_year = years[len(years)//2]
            early_avg = yearly_counts[yearly_counts.index < mid_year].mean()
            recent_avg = yearly_counts[yearly_counts.index >= mid_year].mean()
            features['trend_recent_vs_early'] = (recent_avg - early_avg) / early_avg if early_avg > 0 else 0
        else:
            features['trend_recent_vs_early'] = 0
        
        # Quarterly and seasonal trend analysis
        quarterly_counts = historical_data.groupby(['year', 'quarter']).size()
        seasonal_counts = historical_data.groupby(['year', 'season']).size()
        
        # Quarterly volatility by year
        for year in range(self.config['history_start_year'], self.config['history_end_year'] + 1):
            year_quarterly = quarterly_counts[quarterly_counts.index.get_level_values(0) == year]
            if len(year_quarterly) > 1:
                features[f'quarterly_volatility_{year}'] = year_quarterly.std()
            else:
                features[f'quarterly_volatility_{year}'] = 0
            
            year_seasonal = seasonal_counts[seasonal_counts.index.get_level_values(0) == year]
            if len(year_seasonal) > 1:
                features[f'seasonal_volatility_{year}'] = year_seasonal.std()
            else:
                features[f'seasonal_volatility_{year}'] = 0
        
        # Peak and trough analysis
        if len(yearly_counts) > 2:
            counts_values = [features[f'requests_count_{year}'] for year in range(self.config['history_start_year'], self.config['history_end_year'] + 1)]
            features['peak_year'] = self.config['history_start_year'] + np.argmax(counts_values)
            features['trough_year'] = self.config['history_start_year'] + np.argmin(counts_values)
            features['peak_to_trough_ratio'] = np.max(counts_values) / np.min(counts_values) if np.min(counts_values) > 0 else 1.0
        else:
            features['peak_year'] = self.config['history_start_year']
            features['trough_year'] = self.config['history_start_year']
            features['peak_to_trough_ratio'] = 1.0
        
        # Momentum indicators
        if len(yearly_counts) >= 3:
            # Simple moving average
            recent_3_year_avg = np.mean([features[f'requests_count_{year}'] for year in range(self.config['history_end_year'] - 2, self.config['history_end_year'] + 1)])
            overall_avg = np.mean([features[f'requests_count_{year}'] for year in range(self.config['history_start_year'], self.config['history_end_year'] + 1)])
            features['momentum_indicator'] = recent_3_year_avg / overall_avg if overall_avg > 0 else 1.0
        else:
            features['momentum_indicator'] = 1.0
        
        return features
        
    def _create_seasonal_features(self, historical_data: pd.DataFrame) -> Dict:
        """Create seasonal pattern features with year-by-year breakdown."""
        if len(historical_data) == 0:
            features = {}
            # Overall seasonal rates
            for season in ['winter', 'spring', 'summer', 'fall']:
                features[f'seasonal_{season}_rate'] = 0
            # Year-by-year seasonal counts
            for year in range(self.config['history_start_year'], self.config['history_end_year'] + 1):
                for season in ['winter', 'spring', 'summer', 'fall']:
                    features[f'seasonal_{season}_{year}'] = 0
            return features
            
        features = {}
        total = len(historical_data)
        
        # Overall seasonal rates
        for season in ['winter', 'spring', 'summer', 'fall']:
            seasonal_data = historical_data[historical_data['season'] == season]
            features[f'seasonal_{season}_rate'] = len(seasonal_data) / total if total > 0 else 0
        
        # Year-by-year seasonal breakdown
        for year in range(self.config['history_start_year'], self.config['history_end_year'] + 1):
            year_data = historical_data[historical_data['year'] == year]
            for season in ['winter', 'spring', 'summer', 'fall']:
                seasonal_year_data = year_data[year_data['season'] == season]
                features[f'seasonal_{season}_{year}'] = len(seasonal_year_data)
        
        # Seasonal consistency metrics
        seasonal_yearly_counts = {}
        for season in ['winter', 'spring', 'summer', 'fall']:
            seasonal_data = historical_data[historical_data['season'] == season]
            yearly_seasonal_counts = seasonal_data.groupby('year').size()
            
            if len(yearly_seasonal_counts) > 0:
                features[f'seasonal_{season}_mean'] = yearly_seasonal_counts.mean()
                features[f'seasonal_{season}_std'] = yearly_seasonal_counts.std() if len(yearly_seasonal_counts) > 1 else 0
                features[f'seasonal_{season}_consistency'] = 1 / (yearly_seasonal_counts.std() + 1) if yearly_seasonal_counts.std() > 0 else 1
            else:
                features[f'seasonal_{season}_mean'] = 0
                features[f'seasonal_{season}_std'] = 0
                features[f'seasonal_{season}_consistency'] = 0
        
        return features
        
    def _create_ratio_features(self, historical_data: pd.DataFrame) -> Dict:
        """Create ratio-based features with year-by-year breakdown."""
        if len(historical_data) == 0:
            features = {}
            # Overall ratios
            for key in ['blight_to_total', 'infrastructure_to_total', 'quality_to_total']:
                features[f'ratio_{key}'] = 0
            # Year-by-year ratios
            for year in range(self.config['history_start_year'], self.config['history_end_year'] + 1):
                for key in ['blight_to_total', 'infrastructure_to_total', 'quality_to_total']:
                    features[f'ratio_{key}_{year}'] = 0
            return features
            
        features = {}
        total = len(historical_data)
        
        # Define categories
        infrastructure_types = ['Road', 'Sidewalk', 'Traffic', 'Bridge']
        quality_types = ['Noise', 'Bylaw', 'Parking', 'Animal']
        
        # Overall ratios
        blight_count = len(historical_data[
            historical_data['Service Request Type'].isin(list(self.blight_indicators.keys()))
        ])
        infrastructure_count = len(historical_data[
            historical_data['Service Request Type'].str.contains('|'.join(infrastructure_types), na=False)
        ])
        quality_count = len(historical_data[
            historical_data['Service Request Type'].str.contains('|'.join(quality_types), na=False)
        ])
        
        features['ratio_blight_to_total'] = blight_count / total if total > 0 else 0
        features['ratio_infrastructure_to_total'] = infrastructure_count / total if total > 0 else 0
        features['ratio_quality_to_total'] = quality_count / total if total > 0 else 0
        
        # Year-by-year ratios
        for year in range(self.config['history_start_year'], self.config['history_end_year'] + 1):
            year_data = historical_data[historical_data['year'] == year]
            year_total = len(year_data)
            
            if year_total > 0:
                year_blight = len(year_data[
                    year_data['Service Request Type'].isin(list(self.blight_indicators.keys()))
                ])
                year_infrastructure = len(year_data[
                    year_data['Service Request Type'].str.contains('|'.join(infrastructure_types), na=False)
                ])
                year_quality = len(year_data[
                    year_data['Service Request Type'].str.contains('|'.join(quality_types), na=False)
                ])
                
                features[f'ratio_blight_to_total_{year}'] = year_blight / year_total
                features[f'ratio_infrastructure_to_total_{year}'] = year_infrastructure / year_total
                features[f'ratio_quality_to_total_{year}'] = year_quality / year_total
            else:
                features[f'ratio_blight_to_total_{year}'] = 0
                features[f'ratio_infrastructure_to_total_{year}'] = 0
                features[f'ratio_quality_to_total_{year}'] = 0
        
        # Ratio trends and consistency
        yearly_ratios = {}
        for category in ['blight', 'infrastructure', 'quality']:
            yearly_ratios[category] = []
            for year in range(self.config['history_start_year'], self.config['history_end_year'] + 1):
                yearly_ratios[category].append(features[f'ratio_{category}_to_total_{year}'])
            
            # Calculate trend metrics
            if len(yearly_ratios[category]) > 1:
                ratios_series = pd.Series(yearly_ratios[category])
                features[f'ratio_{category}_trend_slope'] = ratios_series.diff().mean()
                features[f'ratio_{category}_volatility'] = ratios_series.std()
                features[f'ratio_{category}_consistency'] = 1 / (ratios_series.std() + 1) if ratios_series.std() > 0 else 1
            else:
                features[f'ratio_{category}_trend_slope'] = 0
                features[f'ratio_{category}_volatility'] = 0
                features[f'ratio_{category}_consistency'] = 0
        
        return features
        
    def _create_statistical_features(self, historical_data: pd.DataFrame) -> Dict:
        """Create comprehensive statistical summary features with detailed breakdowns."""
        features = {}
        
        if len(historical_data) == 0:
            # Initialize all statistical features to 0
            for year in range(self.config['history_start_year'], self.config['history_end_year'] + 1):
                features[f'monthly_mean_{year}'] = 0
                features[f'monthly_std_{year}'] = 0
                features[f'monthly_skewness_{year}'] = 0
                features[f'monthly_kurtosis_{year}'] = 0
                features[f'monthly_cv_{year}'] = 0
                features[f'monthly_range_{year}'] = 0
                features[f'monthly_iqr_{year}'] = 0
                
                features[f'quarterly_mean_{year}'] = 0
                features[f'quarterly_std_{year}'] = 0
                features[f'quarterly_range_{year}'] = 0
                
                features[f'weekly_mean_{year}'] = 0
                features[f'weekly_std_{year}'] = 0
                features[f'weekly_range_{year}'] = 0
                
            # Overall statistical features
            features['stats_yearly_std'] = 0
            features['stats_yearly_cv'] = 0
            features['stats_monthly_std'] = 0
            features['stats_monthly_cv'] = 0
            features['stats_peak_month_ratio'] = 0
            features['stats_consistency_score'] = 0
            
            return features
        
        # Year-by-year statistical analysis
        for year in range(self.config['history_start_year'], self.config['history_end_year'] + 1):
            year_data = historical_data[historical_data['year'] == year]
            
            if len(year_data) > 0:
                # Monthly statistics by year
                monthly_counts = year_data.groupby('month').size()
                if len(monthly_counts) > 0:
                    features[f'monthly_mean_{year}'] = monthly_counts.mean()
                    features[f'monthly_std_{year}'] = monthly_counts.std() if len(monthly_counts) > 1 else 0
                    features[f'monthly_skewness_{year}'] = monthly_counts.skew() if len(monthly_counts) > 2 else 0
                    features[f'monthly_kurtosis_{year}'] = monthly_counts.kurtosis() if len(monthly_counts) > 3 else 0
                    features[f'monthly_cv_{year}'] = monthly_counts.std() / monthly_counts.mean() if monthly_counts.mean() > 0 else 0
                    features[f'monthly_range_{year}'] = monthly_counts.max() - monthly_counts.min()
                    features[f'monthly_iqr_{year}'] = monthly_counts.quantile(0.75) - monthly_counts.quantile(0.25)
                else:
                    features[f'monthly_mean_{year}'] = 0
                    features[f'monthly_std_{year}'] = 0
                    features[f'monthly_skewness_{year}'] = 0
                    features[f'monthly_kurtosis_{year}'] = 0
                    features[f'monthly_cv_{year}'] = 0
                    features[f'monthly_range_{year}'] = 0
                    features[f'monthly_iqr_{year}'] = 0
                
                # Quarterly statistics by year
                quarterly_counts = year_data.groupby('quarter').size()
                if len(quarterly_counts) > 0:
                    features[f'quarterly_mean_{year}'] = quarterly_counts.mean()
                    features[f'quarterly_std_{year}'] = quarterly_counts.std() if len(quarterly_counts) > 1 else 0
                    features[f'quarterly_range_{year}'] = quarterly_counts.max() - quarterly_counts.min()
                else:
                    features[f'quarterly_mean_{year}'] = 0
                    features[f'quarterly_std_{year}'] = 0
                    features[f'quarterly_range_{year}'] = 0
                
                # Weekly statistics by year (if we have enough data)
                if 'Creation Date' in year_data.columns:
                    year_data_copy = year_data.copy()
                    year_data_copy['week'] = year_data_copy['Creation Date'].dt.isocalendar().week
                    weekly_counts = year_data_copy.groupby('week').size()
                    
                    if len(weekly_counts) > 0:
                        features[f'weekly_mean_{year}'] = weekly_counts.mean()
                        features[f'weekly_std_{year}'] = weekly_counts.std() if len(weekly_counts) > 1 else 0
                        features[f'weekly_range_{year}'] = weekly_counts.max() - weekly_counts.min()
                    else:
                        features[f'weekly_mean_{year}'] = 0
                        features[f'weekly_std_{year}'] = 0
                        features[f'weekly_range_{year}'] = 0
                else:
                    features[f'weekly_mean_{year}'] = 0
                    features[f'weekly_std_{year}'] = 0
                    features[f'weekly_range_{year}'] = 0
            else:
                # No data for this year
                features[f'monthly_mean_{year}'] = 0
                features[f'monthly_std_{year}'] = 0
                features[f'monthly_skewness_{year}'] = 0
                features[f'monthly_kurtosis_{year}'] = 0
                features[f'monthly_cv_{year}'] = 0
                features[f'monthly_range_{year}'] = 0
                features[f'monthly_iqr_{year}'] = 0
                
                features[f'quarterly_mean_{year}'] = 0
                features[f'quarterly_std_{year}'] = 0
                features[f'quarterly_range_{year}'] = 0
                
                features[f'weekly_mean_{year}'] = 0
                features[f'weekly_std_{year}'] = 0
                features[f'weekly_range_{year}'] = 0
        
        # Overall statistical features across all years
        yearly_counts = historical_data.groupby('year').size()
        yearly_std = yearly_counts.std() if len(yearly_counts) > 1 else 0
        yearly_cv = yearly_std / yearly_counts.mean() if yearly_counts.mean() > 0 else 0
        
        monthly_counts_overall = historical_data.groupby('month').size()
        monthly_std = monthly_counts_overall.std() if len(monthly_counts_overall) > 1 else 0
        monthly_cv = monthly_std / monthly_counts_overall.mean() if monthly_counts_overall.mean() > 0 else 0
        
        peak_month_ratio = monthly_counts_overall.max() / monthly_counts_overall.mean() if monthly_counts_overall.mean() > 0 else 0
        consistency_score = 1 / (yearly_cv + 1) if yearly_cv >= 0 else 0
        
        features.update({
            'stats_yearly_std': yearly_std,
            'stats_yearly_cv': yearly_cv,
            'stats_monthly_std': monthly_std,
            'stats_monthly_cv': monthly_cv,
            'stats_peak_month_ratio': peak_month_ratio,
            'stats_consistency_score': consistency_score
        })
        
        # Service type distribution statistics
        service_counts = historical_data['Service Request Type'].value_counts()
        if len(service_counts) > 0:
            features['service_type_entropy'] = -np.sum((service_counts / service_counts.sum()) * np.log2(service_counts / service_counts.sum() + 1e-10))
            features['service_type_gini'] = 1 - np.sum((service_counts / service_counts.sum()) ** 2)
            features['service_type_max_share'] = service_counts.max() / service_counts.sum()
            features['service_type_min_share'] = service_counts.min() / service_counts.sum()
        else:
            features['service_type_entropy'] = 0
            features['service_type_gini'] = 0
            features['service_type_max_share'] = 0
            features['service_type_min_share'] = 0
        
        # Temporal concentration metrics
        if 'Creation Date' in historical_data.columns:
            historical_data_copy = historical_data.copy()
            historical_data_copy['hour'] = historical_data_copy['Creation Date'].dt.hour
            
            # Hour concentration
            hour_counts = historical_data_copy['hour'].value_counts()
            if len(hour_counts) > 0:
                features['hour_concentration_entropy'] = -np.sum((hour_counts / hour_counts.sum()) * np.log2(hour_counts / hour_counts.sum() + 1e-10))
                features['hour_concentration_gini'] = 1 - np.sum((hour_counts / hour_counts.sum()) ** 2)
                features['peak_hour_share'] = hour_counts.max() / hour_counts.sum()
            else:
                features['hour_concentration_entropy'] = 0
                features['hour_concentration_gini'] = 0
                features['peak_hour_share'] = 0
        else:
            features['hour_concentration_entropy'] = 0
            features['hour_concentration_gini'] = 0
            features['peak_hour_share'] = 0
        
        # Day of week concentration
        dow_counts = historical_data['day_of_week'].value_counts()
        if len(dow_counts) > 0:
            features['dow_concentration_entropy'] = -np.sum((dow_counts / dow_counts.sum()) * np.log2(dow_counts / dow_counts.sum() + 1e-10))
            features['dow_concentration_gini'] = 1 - np.sum((dow_counts / dow_counts.sum()) ** 2)
            features['peak_dow_share'] = dow_counts.max() / dow_counts.sum()
        else:
            features['dow_concentration_entropy'] = 0
            features['dow_concentration_gini'] = 0
            features['peak_dow_share'] = 0
        
        # Seasonal concentration
        season_counts = historical_data['season'].value_counts()
        if len(season_counts) > 0:
            features['season_concentration_entropy'] = -np.sum((season_counts / season_counts.sum()) * np.log2(season_counts / season_counts.sum() + 1e-10))
            features['season_concentration_gini'] = 1 - np.sum((season_counts / season_counts.sum()) ** 2)
            features['peak_season_share'] = season_counts.max() / season_counts.sum()
        else:
            features['season_concentration_entropy'] = 0
            features['season_concentration_gini'] = 0
            features['peak_season_share'] = 0
        
        # Blight-specific statistical features
        blight_data = historical_data[historical_data['Service Request Type'].isin(self.blight_indicators.keys())]
        if len(blight_data) > 0:
            blight_monthly = blight_data.groupby(['year', 'month']).size()
            if len(blight_monthly) > 0:
                features['blight_monthly_mean'] = blight_monthly.mean()
                features['blight_monthly_std'] = blight_monthly.std()
                features['blight_monthly_cv'] = blight_monthly.std() / blight_monthly.mean() if blight_monthly.mean() > 0 else 0
                features['blight_monthly_skewness'] = blight_monthly.skew() if len(blight_monthly) > 2 else 0
                features['blight_monthly_kurtosis'] = blight_monthly.kurtosis() if len(blight_monthly) > 3 else 0
            else:
                features['blight_monthly_mean'] = 0
                features['blight_monthly_std'] = 0
                features['blight_monthly_cv'] = 0
                features['blight_monthly_skewness'] = 0
                features['blight_monthly_kurtosis'] = 0
        else:
            features['blight_monthly_mean'] = 0
            features['blight_monthly_std'] = 0
            features['blight_monthly_cv'] = 0
            features['blight_monthly_skewness'] = 0
            features['blight_monthly_kurtosis'] = 0
        
        return features
        
    def merge_and_finalize(self) -> bool:
        """Merge features and create final dataset."""
        print(f"\n🔗 Finalizing dataset...")
        
        # Start with features dataframe
        self.final_df = self.features_df.copy()
        
        # Add ward information if available
        if self.wards_gdf is not None:
            # Keep geometry for GeoJSON output capability
            ward_info = self.wards_gdf.copy()
            
            # Merge with features
            self.final_df = self.final_df.merge(
                ward_info,
                left_on='ward_name',
                right_on=self.ward_name_col,
                how='left'
            )
            
            # Convert to GeoDataFrame to preserve geometry
            self.final_df = gpd.GeoDataFrame(self.final_df, geometry='geometry')
            
            print(f"   ✓ Merged with ward boundary information (geometry preserved)")
        
        print(f"   ✓ Final dataset: {len(self.final_df)} wards")
        
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
        blight_threshold_75 = self.final_df['target_blight_requests'].quantile(0.75)
        self.final_df['is_high_blight_risk'] = (
            self.final_df['target_blight_requests'] > blight_threshold_75
        ).astype(int)
        
        # Binary extreme-risk classification (90th percentile)
        blight_threshold_90 = self.final_df['target_blight_requests'].quantile(0.90)
        self.final_df['is_extreme_blight_risk'] = (
            self.final_df['target_blight_requests'] > blight_threshold_90
        ).astype(int)
        
        # Multi-class risk levels
        self.final_df['risk_level'] = pd.cut(
            self.final_df['target_blight_requests'],
            bins=[0, 
                  self.final_df['target_blight_requests'].quantile(0.33),
                  self.final_df['target_blight_requests'].quantile(0.67),
                  self.final_df['target_blight_requests'].max()],
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        # Continuous risk score (normalized)
        max_blight = self.final_df['target_blight_requests'].max()
        self.final_df['risk_score'] = (
            self.final_df['target_blight_requests'] / max_blight if max_blight > 0 else 0
        )
        
        print(f"   ✓ High-risk wards (75th percentile): {self.final_df['is_high_blight_risk'].sum()}")
        print(f"   ✓ Extreme-risk wards (90th percentile): {self.final_df['is_extreme_blight_risk'].sum()}")
        
    def _add_feature_metadata(self):
        """Add comprehensive feature metadata."""
        feature_cols = [col for col in self.final_df.columns 
                       if col not in ['ward_name', self.ward_name_col if hasattr(self, 'ward_name_col') else None]]
        
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
        
        # Check for missing values
        missing_counts = self.final_df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"   ⚠️  Missing values found in {(missing_counts > 0).sum()} features")
            
        # Separate numeric and categorical columns
        numeric_cols = self.final_df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.final_df.select_dtypes(include=['category']).columns
        object_cols = self.final_df.select_dtypes(include=['object']).columns
        
        # Handle infinite values in numeric columns
        if len(numeric_cols) > 0:
            inf_counts = np.isinf(self.final_df[numeric_cols]).sum()
            if inf_counts.sum() > 0:
                print(f"   ⚠️  Infinite values found in {(inf_counts > 0).sum()} numeric features")
                # Replace inf with large finite values
                self.final_df[numeric_cols] = self.final_df[numeric_cols].replace([np.inf, -np.inf], [1e10, -1e10])
            
            # Fill missing values in numeric columns
            self.final_df[numeric_cols] = self.final_df[numeric_cols].fillna(0)
            
        # Handle categorical columns
        if len(categorical_cols) > 0:
            print(f"   🔧 Processing {len(categorical_cols)} categorical features...")
            for col in categorical_cols:
                if self.final_df[col].isnull().any():
                    # Convert categorical to string for easier handling
                    self.final_df[col] = self.final_df[col].astype(str)
                    self.final_df[col] = self.final_df[col].replace('nan', 'Unknown')
                    
        # Handle object columns
        if len(object_cols) > 0:
            for col in object_cols:
                self.final_df[col] = self.final_df[col].fillna('Unknown')
        
        print(f"   ✓ Feature validation complete")
        
    def save_results(self) -> bool:
        """Save the final dataset with comprehensive metadata."""
        print(f"\n💾 Saving comprehensive model-ready dataset...")
        
        # Save main CSV (without geometry for ML compatibility)
        if 'geometry' in self.final_df.columns:
            # Save CSV without geometry for machine learning
            csv_df = self.final_df.drop('geometry', axis=1)
            csv_df.to_csv(self.config['output_csv'], index=False)
            print(f"   ✓ ML-ready CSV (no geometry): {self.config['output_csv']}")
            
            # Save GeoJSON with geometry for visualization
            if self.config.get('save_geojson', True):
                geojson_file = self.config['output_csv'].replace('.csv', '.geojson')
                self.final_df.to_file(geojson_file, driver='GeoJSON')
                print(f"   ✓ GeoJSON with geometry: {geojson_file}")
        else:
            # No geometry available, save as regular CSV
            self.final_df.to_csv(self.config['output_csv'], index=False)
            print(f"   ✓ CSV dataset: {self.config['output_csv']}")
        
        # Save feature metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'config': self.config,
            'data_summary': {
                'total_wards': len(self.final_df),
                'total_features': len([col for col in self.final_df.columns 
                                     if col not in ['ward_name', self.ward_name_col if hasattr(self, 'ward_name_col') else None, 'geometry']]),
                'date_range': f"{self.config['history_start_year']}-{self.config['history_end_year']}",
                'target_year': self.config['target_year'],
                'blight_indicators': len(self.blight_indicators),
                'high_risk_wards': int(self.final_df['is_high_blight_risk'].sum()),
                'extreme_risk_wards': int(self.final_df['is_extreme_blight_risk'].sum()),
                'has_geometry': 'geometry' in self.final_df.columns
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
        
        metadata_file = self.config['output_csv'].replace('.csv', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        print(f"   ✓ Metadata: {metadata_file}")
        
        return True
        
    def print_summary(self):
        """Print comprehensive summary of the generated dataset."""
        print(f"\n📈 COMPREHENSIVE DATASET SUMMARY")
        print("=" * 80)
        
        # Basic stats
        print(f"🏢 Dataset Overview:")
        print(f"   • Total wards: {len(self.final_df)}")
        print(f"   • Training period: {self.config['history_start_year']}-{self.config['history_end_year']}")
        print(f"   • Target year: {self.config['target_year']}")
        print(f"   • Total features: {len([col for col in self.final_df.columns if col not in ['ward_name', self.ward_name_col if hasattr(self, 'ward_name_col') else None, 'geometry']])}")
        print(f"   • Geometry included: {'Yes' if 'geometry' in self.final_df.columns else 'No'}")
        
        # Target variable distribution
        print(f"\n🎯 Target Variable Distribution:")
        print(f"   • High-risk wards: {self.final_df['is_high_blight_risk'].sum()}/{len(self.final_df)} ({self.final_df['is_high_blight_risk'].mean():.1%})")
        print(f"   • Extreme-risk wards: {self.final_df['is_extreme_blight_risk'].sum()}/{len(self.final_df)} ({self.final_df['is_extreme_blight_risk'].mean():.1%})")
        
        # Feature categories
        print(f"\n🔧 Feature Categories:")
        for category, features in self.feature_metadata.items():
            if features:
                print(f"   • {category.replace('_', ' ').title()}: {len(features)}")
            
        # Top features by variance (proxy for importance)
        numeric_cols = self.final_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols 
                       if col not in ['ward_name', self.ward_name_col if hasattr(self, 'ward_name_col') else None, 'geometry'] 
                       and not col.startswith('target_')]
        
        if len(feature_cols) > 0:
            variances = self.final_df[feature_cols].var().sort_values(ascending=False)
            print(f"\n📊 Top Features by Variance:")
            for i, (feature, variance) in enumerate(variances.head(10).items()):
                print(f"   {i+1:2d}. {feature}: {variance:.4f}")
                
        print(f"\n✅ SUCCESS! Comprehensive dataset ready for ML and visualization!")
        print(f"   📁 ML-ready CSV: {self.config['output_csv']}")
        if 'geometry' in self.final_df.columns:
            geojson_file = self.config['output_csv'].replace('.csv', '.geojson')
            print(f"   🗺️  GeoJSON with geometry: {geojson_file}")
        print(f"   🎯 Recommended target: 'is_high_blight_risk' for binary classification")
        print(f"   🎯 Alternative targets: 'risk_level' (multi-class) or 'risk_score' (regression)")
        print(f"   🎯 Use GeoJSON for map visualization and CSV for machine learning!")
        
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
    'ward_boundaries_file': 'toronto_ward_data.geojson',  # Required for geometry output
    'output_csv': 'model_ready_data.csv',
    'save_geojson': True,  # Set to True to save GeoJSON with geometry
    'history_start_year': 2014,
    'history_end_year': 2023,
    'target_year': 2024
}

# Run the generator
if __name__ == "__main__":
    generator = WardBasedModelDataGeneratorCSV(CONFIG)
    success = generator.run()
    
    if success:
        print("\n🎉 Model-ready CSV dataset generation completed successfully!")
        print("   Ready for XGBoost, LightGBM, CatBoost, or any gradient boosting algorithm!")
        print("   Perfect for standard ML libraries like scikit-learn, pandas, etc.")
    else:
        print("\n💥 Dataset generation failed. Please check the errors above.") 