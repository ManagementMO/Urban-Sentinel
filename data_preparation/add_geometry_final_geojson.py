#!/usr/bin/env python3
"""
Create Predictions GeoJSON with Risk Colors
===========================================

This script merges LightGBM model predictions from CSV with Toronto ward geometry 
from GeoJSON to create a final predictions GeoJSON file with risk color coding
for easy visualization.

Usage:
    python create_predictions_geojson.py

Input files:
    - model_outputs/lightgbm_predictions.csv
    - toronto_ward_data.geojson

Output:
    - model_outputs/ward_predictions_with_geometry.geojson
"""

import pandas as pd
import json
import sys
import os
from typing import Dict, Any

def get_risk_colors(risk_category: str, blight_probability: float = None) -> Dict[str, Any]:
    """
    Get color coding for risk categories for visualization.
    
    Args:
        risk_category: The risk category (High Risk, Medium Risk, Low Risk, No Data)
        blight_probability: The numerical probability (0-1) for gradient coloring
    
    Returns:
        Dictionary with color information including hex, rgb, and opacity
    """
    # Define risk color scheme
    risk_colors = {
        'High Risk': {
            'color': '#d73027',  # Strong red
            'rgb': [215, 48, 39],
            'opacity': 0.8,
            'border_color': '#a50026',
            'description': 'High blight risk area'
        },
        'Medium Risk': {
            'color': '#fc8d59',  # Orange
            'rgb': [252, 141, 89],
            'opacity': 0.7,
            'border_color': '#e31a1c',
            'description': 'Medium blight risk area'
        },
        'Low Risk': {
            'color': '#4575b4',  # Blue
            'rgb': [69, 117, 180],
            'opacity': 0.6,
            'border_color': '#313695',
            'description': 'Low blight risk area'
        },
        'No Data': {
            'color': '#999999',  # Gray
            'rgb': [153, 153, 153],
            'opacity': 0.4,
            'border_color': '#666666',
            'description': 'No prediction data available'
        }
    }
    
    # Get base color info
    base_color = risk_colors.get(risk_category, risk_colors['No Data'])
    
    # Add probability-based intensity for High/Medium/Low risk
    if blight_probability is not None and risk_category != 'No Data':
        # Adjust opacity based on probability
        if risk_category == 'High Risk':
            # High risk: probability 0.5-1.0 maps to opacity 0.6-0.9
            intensity = 0.6 + (blight_probability - 0.5) * 0.6 if blight_probability >= 0.5 else 0.4
        elif risk_category == 'Medium Risk':
            # Medium risk: probability 0.3-0.7 maps to opacity 0.5-0.8
            intensity = 0.5 + (blight_probability - 0.3) * 0.75 if blight_probability >= 0.3 else 0.4
        else:  # Low Risk
            # Low risk: probability 0.0-0.5 maps to opacity 0.3-0.7
            intensity = 0.3 + blight_probability * 0.8 if blight_probability <= 0.5 else 0.7
        
        base_color['opacity'] = min(0.9, max(0.3, intensity))
        base_color['probability_intensity'] = blight_probability
    
    return base_color

def load_predictions_csv(file_path: str) -> pd.DataFrame:
    """Load the predictions CSV file."""
    print(f"📊 Loading predictions from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"   ✅ Loaded {len(df)} predictions")
        print(f"   📋 Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"   ❌ Error loading predictions: {e}")
        sys.exit(1)

def load_ward_geojson(file_path: str) -> Dict[str, Any]:
    """Load the Toronto ward GeoJSON file."""
    print(f"🗺️  Loading ward geometry from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        
        feature_count = len(geojson_data.get('features', []))
        print(f"   ✅ Loaded {feature_count} ward features")
        
        # Show sample ward names
        sample_names = []
        for feature in geojson_data.get('features', [])[:5]:
            name = feature.get('properties', {}).get('AREA_NAME', 'Unknown')
            sample_names.append(name)
        print(f"   📋 Sample ward names: {sample_names}")
        
        return geojson_data
    except Exception as e:
        print(f"   ❌ Error loading GeoJSON: {e}")
        sys.exit(1)

def normalize_ward_name(name: str) -> str:
    """Normalize ward names for better matching."""
    if pd.isna(name):
        return ""
    
    # Convert to lowercase and remove special characters for matching
    normalized = str(name).lower()
    normalized = normalized.replace("-", " ").replace("'", "").replace(".", "")
    normalized = " ".join(normalized.split())  # Remove extra spaces
    return normalized

def create_ward_name_mapping(geojson_data: Dict[str, Any], predictions_df: pd.DataFrame) -> Dict[str, str]:
    """Create a mapping between prediction ward names and GeoJSON ward names."""
    print("🔗 Creating ward name mapping...")
    
    # Get ward names from both sources
    geojson_names = []
    for feature in geojson_data.get('features', []):
        name = feature.get('properties', {}).get('AREA_NAME', '')
        if name:
            geojson_names.append(name)
    
    prediction_names = predictions_df['ward_name'].unique().tolist()
    
    print(f"   🗺️  GeoJSON ward names ({len(geojson_names)}): {geojson_names[:5]}...")
    print(f"   📊 Prediction ward names ({len(prediction_names)}): {prediction_names[:5]}...")
    
    # Create mapping using normalized names
    mapping = {}
    for pred_name in prediction_names:
        if pd.isna(pred_name):
            continue
            
        pred_normalized = normalize_ward_name(pred_name)
        best_match = None
        
        for geo_name in geojson_names:
            geo_normalized = normalize_ward_name(geo_name)
            
            # Direct match
            if pred_normalized == geo_normalized:
                best_match = geo_name
                break
            
            # Partial match (prediction name contained in GeoJSON name)
            if pred_normalized in geo_normalized:
                best_match = geo_name
                break
    
        if best_match:
            mapping[pred_name] = best_match
            print(f"   ✅ Mapped: '{pred_name}' → '{best_match}'")
        else:
            print(f"   ⚠️  No match found for: '{pred_name}'")
    
    return mapping

def merge_predictions_with_geometry(predictions_df: pd.DataFrame, 
                                    geojson_data: Dict[str, Any], 
                                    ward_mapping: Dict[str, str]) -> Dict[str, Any]:
    """Merge predictions with ward geometry and add risk colors."""
    print("🔀 Merging predictions with geometry and adding risk colors...")
    
    # Create a lookup dictionary for predictions by ward name
    predictions_by_ward = {}
    for _, row in predictions_df.iterrows():
        ward_name = row['ward_name']
        if ward_name in ward_mapping:
            mapped_name = ward_mapping[ward_name]
            predictions_by_ward[mapped_name] = {
                'feature_id': row['feature_id'],
                'actual_blight': row['actual_blight'],
                'predicted_blight': row['predicted_blight'],
                'blight_probability': row['blight_probability'],
                'risk_category': row['risk_category'],
                'prediction_ward_name': ward_name
            }
    
    # Create new GeoJSON with merged data
    output_geojson = {
        "type": "FeatureCollection",
        "name": "Ward Predictions with Geometry and Risk Colors",
        "crs": geojson_data.get('crs', {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}),
        "metadata": {
            "description": "Toronto ward boundaries with urban blight risk predictions and visualization colors",
            "model": "LightGBM Urban Blight Predictor",
            "risk_categories": {
                "High Risk": {"threshold": "≥0.5 probability", "color": "#d73027"},
                "Medium Risk": {"threshold": "0.3-0.5 probability", "color": "#fc8d59"},
                "Low Risk": {"threshold": "<0.3 probability", "color": "#4575b4"},
                "No Data": {"threshold": "No prediction", "color": "#999999"}
            }
        },
        "features": []
    }
    
    matched_count = 0
    unmatched_count = 0
    
    for feature in geojson_data.get('features', []):
        ward_name = feature.get('properties', {}).get('AREA_NAME', '')
        
        # Create new feature with original properties
        new_feature = {
            "type": "Feature",
            "properties": feature.get('properties', {}).copy(),
            "geometry": feature.get('geometry', {})
        }
        
        # Add prediction data if available
        if ward_name in predictions_by_ward:
            pred_data = predictions_by_ward[ward_name]
            
            # Get risk colors for this ward
            risk_colors = get_risk_colors(
                pred_data['risk_category'], 
                pred_data['blight_probability']
            )
            
            # Add all prediction and visualization data
            new_feature['properties'].update({
                # Prediction data
                'feature_id': pred_data['feature_id'],
                'actual_blight': pred_data['actual_blight'],
                'predicted_blight': pred_data['predicted_blight'],
                'blight_probability': pred_data['blight_probability'],
                'risk_category': pred_data['risk_category'],
                'prediction_ward_name': pred_data['prediction_ward_name'],
                'has_prediction': True,
                
                # Visualization colors
                'fill_color': risk_colors['color'],
                'fill_opacity': risk_colors['opacity'],
                'stroke_color': risk_colors['border_color'],
                'stroke_width': 2,
                'fill_rgb': risk_colors['rgb'],
                'color_description': risk_colors['description'],
                
                # Additional visualization properties
                'popup_text': f"{ward_name}: {pred_data['risk_category']} ({pred_data['blight_probability']:.1%})",
                'risk_score': pred_data['blight_probability'],
                'risk_level': pred_data['risk_category']
            })
            
            matched_count += 1
            print(f"   ✅ {ward_name}: {pred_data['risk_category']} (Prob: {pred_data['blight_probability']:.3f}, Color: {risk_colors['color']})")
            
        else:
            # No prediction data available
            risk_colors = get_risk_colors('No Data')
            
            new_feature['properties'].update({
                # Empty prediction data
                'feature_id': None,
                'actual_blight': None,
                'predicted_blight': None,
                'blight_probability': None,
                'risk_category': 'No Data',
                'prediction_ward_name': None,
                'has_prediction': False,
                
                # Default visualization colors
                'fill_color': risk_colors['color'],
                'fill_opacity': risk_colors['opacity'],
                'stroke_color': risk_colors['border_color'],
                'stroke_width': 1,
                'fill_rgb': risk_colors['rgb'],
                'color_description': risk_colors['description'],
                
                # Additional visualization properties
                'popup_text': f"{ward_name}: No prediction data available",
                'risk_score': 0,
                'risk_level': 'No Data'
            })
            
            unmatched_count += 1
            print(f"   ⚠️  {ward_name}: No prediction (Color: {risk_colors['color']})")
        
        output_geojson['features'].append(new_feature)
    
    print(f"   📊 Summary: {matched_count} wards with predictions, {unmatched_count} without")
    print("   🎨 Added risk color visualization properties to all features")
    
    return output_geojson

def save_geojson(geojson_data: Dict[str, Any], output_path: str) -> None:
    """Save the merged GeoJSON to file."""
    print(f"💾 Saving merged GeoJSON with risk colors to {output_path}...")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, ensure_ascii=False, indent=2)
        
        file_size = os.path.getsize(output_path) / 1024  # KB
        feature_count = len(geojson_data.get('features', []))
        print(f"   ✅ Saved {feature_count} features ({file_size:.1f} KB)")
        
        # Show color summary
        colors_used = set()
        for feature in geojson_data['features']:
            color = feature['properties'].get('fill_color', '#000000')
            risk = feature['properties'].get('risk_category', 'Unknown')
            colors_used.add(f"{risk}: {color}")
        
        print("   🎨 Risk colors used:")
        for color_info in sorted(colors_used):
            print(f"      • {color_info}")
        
    except Exception as e:
        print(f"   ❌ Error saving GeoJSON: {e}")
        sys.exit(1)

def main():
    """Main function to execute the merging process."""
    print("🎯 Creating Ward Predictions GeoJSON with Risk Colors")
    print("=" * 60)
    
    # File paths
    predictions_csv = "model_outputs/lightgbm_predictions.csv"
    ward_geojson = "toronto_ward_data.geojson"
    output_geojson = "ward_predictions_with_geometry.geojson"
    
    # Check if input files exist
    if not os.path.exists(predictions_csv):
        print(f"❌ Predictions file not found: {predictions_csv}")
        sys.exit(1)
    
    if not os.path.exists(ward_geojson):
        print(f"❌ Ward GeoJSON file not found: {ward_geojson}")
        sys.exit(1)
    
    # Load data
    predictions_df = load_predictions_csv(predictions_csv)
    geojson_data = load_ward_geojson(ward_geojson)
    
    # Create ward name mapping
    ward_mapping = create_ward_name_mapping(geojson_data, predictions_df)
    
    if not ward_mapping:
        print("❌ No ward names could be matched between files!")
        sys.exit(1)
    
    # Merge data with risk colors
    merged_geojson = merge_predictions_with_geometry(predictions_df, geojson_data, ward_mapping)
    
    # Save result
    save_geojson(merged_geojson, output_geojson)
    
    print("\n🎉 Successfully created ward predictions GeoJSON with risk colors!")
    print(f"📁 Output file: {output_geojson}")
    print("🎨 Features include:")
    print("   • fill_color & fill_opacity for choropleth mapping")
    print("   • stroke_color & stroke_width for borders")
    print("   • popup_text for interactive tooltips")
    print("   • RGB values for custom styling")
    print("🗺️  Ready for beautiful risk visualization!")

if __name__ == "__main__":
    main() 