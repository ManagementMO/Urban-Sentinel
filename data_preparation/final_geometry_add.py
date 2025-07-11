#!/usr/bin/env python3
"""
Simple Ward Predictions + Geometry Merger
==========================================

This script merges LightGBM predictions with Toronto ward geometry 
and adds risk colors for beautiful visualization.

Usage:
    python add_geometry_final_geojson.py

Input:
    - model_outputs/lightgbm_predictions.csv (your ML predictions)
    - toronto_ward_data.geojson (Toronto ward boundaries)

Output:
    - lightgbm_predictions_with_geometry.geojson (combined data with colors)
"""

import pandas as pd
import json
import os
from typing import Dict, Any

# ============================================================================
# CONFIGURATION
# ============================================================================

# Risk color scheme for visualization
RISK_COLORS = {
    'High Risk': {
        'fill_color': '#d73027',    # Strong red
        'border_color': '#a50026',  # Darker red
        'opacity': 0.8
    },
    'Medium Risk': {
        'fill_color': '#fc8d59',    # Orange  
        'border_color': '#e31a1c',  # Dark orange
        'opacity': 0.7
    },
    'Low Risk': {
        'fill_color': '#4575b4',    # Blue
        'border_color': '#313695',  # Dark blue
        'opacity': 0.6
    },
    'No Data': {
        'fill_color': '#999999',    # Gray
        'border_color': '#666666',  # Dark gray
        'opacity': 0.4
    }
}

# File paths
INPUT_PREDICTIONS = "model_outputs/lightgbm_predictions.csv"
INPUT_WARD_GEOJSON = "toronto_ward_data.geojson"
OUTPUT_GEOJSON = "lightgbm_predictions_with_geometry.geojson"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_ward_name(name: str) -> str:
    """Clean ward name for matching (lowercase, no punctuation)."""
    if not name:
        return ""
    return name.lower().replace("-", " ").replace("'", "").replace(".", "").strip()

def get_risk_colors(risk_category: str) -> Dict[str, Any]:
    """Get color scheme for a risk category."""
    return RISK_COLORS.get(risk_category, RISK_COLORS['No Data'])

# ============================================================================
# DATA LOADING
# ============================================================================

def load_predictions() -> pd.DataFrame:
    """Load the LightGBM predictions CSV."""
    print("📊 Loading predictions...")
    
    if not os.path.exists(INPUT_PREDICTIONS):
        raise FileNotFoundError(f"Predictions file not found: {INPUT_PREDICTIONS}")
    
    df = pd.read_csv(INPUT_PREDICTIONS)
    print(f"   ✅ Loaded {len(df)} ward predictions")
    print(f"   📋 Columns: {list(df.columns)}")
    return df

def load_ward_geometry() -> Dict[str, Any]:
    """Load the Toronto ward GeoJSON with geometry."""
    print("🗺️  Loading ward boundaries...")
    
    if not os.path.exists(INPUT_WARD_GEOJSON):
        raise FileNotFoundError(f"Ward GeoJSON not found: {INPUT_WARD_GEOJSON}")
    
    with open(INPUT_WARD_GEOJSON, 'r', encoding='utf-8') as f:
        geojson = json.load(f)
    
    ward_count = len(geojson.get('features', []))
    print(f"   ✅ Loaded {ward_count} ward boundaries")
    
    # Show sample ward names
    sample_names = []
    for feature in geojson.get('features', [])[:3]:
        name = feature.get('properties', {}).get('AREA_NAME', 'Unknown')
        sample_names.append(name)
    print(f"   📋 Sample wards: {sample_names}")
    
    return geojson

# ============================================================================
# WARD NAME MATCHING
# ============================================================================

def create_name_mapping(predictions_df: pd.DataFrame, ward_geojson: Dict[str, Any]) -> Dict[str, str]:
    """Create mapping between prediction ward names and GeoJSON ward names."""
    print("🔗 Matching ward names...")
    
    # Get all ward names from both sources
    prediction_wards = predictions_df['ward_name'].unique().tolist()
    geojson_wards = [
        feature['properties']['AREA_NAME'] 
        for feature in ward_geojson.get('features', [])
        if feature.get('properties', {}).get('AREA_NAME')
    ]
    
    print(f"   📊 Prediction wards: {len(prediction_wards)}")
    print(f"   🗺️  GeoJSON wards: {len(geojson_wards)}")
    
    # Create mapping by matching cleaned names
    mapping = {}
    for pred_ward in prediction_wards:
        pred_clean = clean_ward_name(pred_ward)
        
        # Find best match in GeoJSON wards
        for geo_ward in geojson_wards:
            geo_clean = clean_ward_name(geo_ward)
            
            if pred_clean == geo_clean:
                mapping[pred_ward] = geo_ward
                print(f"   ✅ '{pred_ward}' → '{geo_ward}'")
                break
        
        if pred_ward not in mapping:
            print(f"   ⚠️  No match for: '{pred_ward}'")
    
    print(f"   🎯 Successfully matched {len(mapping)}/{len(prediction_wards)} wards")
    return mapping

# ============================================================================
# DATA MERGING
# ============================================================================

def merge_data(predictions_df: pd.DataFrame, ward_geojson: Dict[str, Any], name_mapping: Dict[str, str]) -> Dict[str, Any]:
    """Merge predictions with ward geometry and add visualization colors."""
    print("🔀 Merging predictions with geometry...")
    
    # Create lookup: GeoJSON ward name → prediction data
    prediction_lookup = {}
    for _, row in predictions_df.iterrows():
        pred_ward = row['ward_name']
        if pred_ward in name_mapping:
            geo_ward = name_mapping[pred_ward]
            prediction_lookup[geo_ward] = {
                'feature_id': int(row['feature_id']),
                'actual_blight': int(row['actual_blight']),
                'predicted_blight': int(row['predicted_blight']),
                'blight_probability': float(row['blight_probability']),
                'risk_category': str(row['risk_category']),
                'prediction_ward_name': pred_ward
            }
    
    # Create output GeoJSON structure
    output = {
        "type": "FeatureCollection",
        "name": "Toronto Wards with Blight Risk Predictions",
        "crs": ward_geojson.get('crs'),
        "features": []
    }
    
    # Process each ward feature
    matched_count = 0
    for feature in ward_geojson.get('features', []):
        ward_name = feature.get('properties', {}).get('AREA_NAME', '')
        
        # Start with original feature
        new_feature = {
            "type": "Feature",
            "properties": feature.get('properties', {}).copy(),
            "geometry": feature.get('geometry')
        }
        
        # Add prediction data if available
        if ward_name in prediction_lookup:
            pred_data = prediction_lookup[ward_name]
            colors = get_risk_colors(pred_data['risk_category'])
            
            # Add all prediction columns
            new_feature['properties'].update({
                # Original prediction data
                'feature_id': pred_data['feature_id'],
                'actual_blight': pred_data['actual_blight'],
                'predicted_blight': pred_data['predicted_blight'],
                'blight_probability': pred_data['blight_probability'],
                'risk_category': pred_data['risk_category'],
                'prediction_ward_name': pred_data['prediction_ward_name'],
                'has_prediction': True,
                
                # Visualization properties
                'fill_color': colors['fill_color'],
                'fill_opacity': colors['opacity'],
                'stroke_color': colors['border_color'],
                'stroke_width': 2,
                'popup_text': f"{ward_name}: {pred_data['risk_category']} ({pred_data['blight_probability']:.1%})"
            })
            
            matched_count += 1
            print(f"   ✅ {ward_name}: {pred_data['risk_category']} ({pred_data['blight_probability']:.1%})")
            
        else:
            # No prediction data - add default values
            colors = get_risk_colors('No Data')
            new_feature['properties'].update({
                'feature_id': None,
                'actual_blight': None,
                'predicted_blight': None,
                'blight_probability': None,
                'risk_category': 'No Data',
                'prediction_ward_name': None,
                'has_prediction': False,
                'fill_color': colors['fill_color'],
                'fill_opacity': colors['opacity'],
                'stroke_color': colors['border_color'],
                'stroke_width': 1,
                'popup_text': f"{ward_name}: No prediction data"
            })
            print(f"   ⚠️  {ward_name}: No prediction data")
        
        output['features'].append(new_feature)
    
    print(f"   📊 Final result: {matched_count} wards with predictions, {len(output['features']) - matched_count} without")
    return output

# ============================================================================
# FILE SAVING
# ============================================================================

def save_geojson(geojson_data: Dict[str, Any]) -> None:
    """Save the merged GeoJSON with colors."""
    print("💾 Saving final GeoJSON...")
    
    with open(OUTPUT_GEOJSON, 'w', encoding='utf-8') as f:
        json.dump(geojson_data, f, ensure_ascii=False, indent=2)
    
    # Get file info
    file_size = os.path.getsize(OUTPUT_GEOJSON) / 1024  # KB
    feature_count = len(geojson_data.get('features', []))
    
    print(f"   ✅ Saved {feature_count} features ({file_size:.1f} KB)")
    print(f"   📁 Output: {OUTPUT_GEOJSON}")
    
    # Show color distribution
    color_counts = {}
    for feature in geojson_data['features']:
        risk = feature['properties'].get('risk_category', 'Unknown')
        color_counts[risk] = color_counts.get(risk, 0) + 1
    
    print("   🎨 Risk distribution:")
    for risk, count in sorted(color_counts.items()):
        color = RISK_COLORS.get(risk, {}).get('fill_color', '#000000')
        print(f"      • {risk}: {count} wards ({color})")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function - simple and clear workflow."""
    print("🎯 Toronto Ward Predictions + Geometry Merger")
    print("=" * 50)
    
    try:
        # Step 1: Load data
        predictions_df = load_predictions()
        ward_geojson = load_ward_geometry()
        
        # Step 2: Match ward names
        name_mapping = create_name_mapping(predictions_df, ward_geojson)
        
        if not name_mapping:
            print("❌ No ward names could be matched!")
            return
        
        # Step 3: Merge data and add colors
        merged_data = merge_data(predictions_df, ward_geojson, name_mapping)
        
        # Step 4: Save result
        save_geojson(merged_data)
        
        # Success message
        print("\n🎉 SUCCESS! Created ward predictions with geometry and colors!")
        print("🎨 Your GeoJSON includes:")
        print("   • Ward boundaries (MultiPolygon geometry)")
        print("   • Blight risk predictions (probabilities, categories)")
        print("   • Visualization colors (fill_color, stroke_color)")
        print("   • Interactive tooltips (popup_text)")
        print("🗺️  Ready for mapping applications!")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return

if __name__ == "__main__":
    main() 