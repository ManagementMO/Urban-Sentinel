import { RiskGridCell } from '../services/api';
import * as wellknown from 'wellknown';

// ============================================================================
// --- GEOMETRY PARSING AND CONVERSION ---
// ============================================================================

// Convert WKT geometry to GeoJSON
export const parseWKTToGeoJSON = (wktString: string): GeoJSON.Geometry => {
  try {
    // Use wellknown library to parse WKT strings
    const geometry = wellknown.parse(wktString);
    if (geometry) {
      return geometry as GeoJSON.Geometry;
    }
    
    // Fallback - try to parse if it's already JSON
    if (wktString.startsWith('{')) {
      return JSON.parse(wktString);
    }
    
    // Basic WKT POLYGON parsing (simplified for common cases)
    if (wktString.startsWith('POLYGON')) {
      return parseWKTPolygon(wktString);
    }
    
    throw new Error('Could not parse WKT geometry string');
  } catch (error) {
    console.error('Failed to parse WKT geometry:', error);
    // Log the problematic string for debugging
    console.error('Problematic WKT string:', wktString.substring(0, 100) + '...');
    throw error;
  }
};

// Simple WKT POLYGON parser (fallback)
const parseWKTPolygon = (wktString: string): GeoJSON.Polygon => {
  // Remove POLYGON(( and ))
  const coordsString = wktString
    .replace(/^POLYGON\s*\(\s*\(\s*/, '')
    .replace(/\s*\)\s*\)$/, '');
  
  // Split coordinates
  const coordPairs = coordsString.split(',').map(pair => {
    const [lng, lat] = pair.trim().split(/\s+/).map(Number);
    return [lng, lat];
  });
  
  return {
    type: 'Polygon',
    coordinates: [coordPairs]
  };
};

// Convert risk data to GeoJSON format for Mapbox
export const convertRiskDataToGeoJSON = (riskData: RiskGridCell[]): GeoJSON.FeatureCollection => {
  return {
    type: 'FeatureCollection',
    features: riskData.map(cell => ({
      type: 'Feature',
      geometry: parseWKTToGeoJSON(cell.geometry),
      properties: {
        cell_id: cell.cell_id,
        risk_score: cell.risk_score,
        risk_level: getRiskLevel(cell.risk_score),
        risk_color: getRiskColor(cell.risk_score),
        total_complaints_mean: cell.total_complaints_mean || 0,
        blight_complaints_mean: cell.blight_complaints_mean || 0,
        total_complaints_std: cell.total_complaints_std || 0,
        blight_complaints_std: cell.blight_complaints_std || 0,
        blight_complaints_recent_mean: cell.blight_complaints_recent_mean || 0,
        blight_complaints_trend: cell.blight_complaints_trend || 0,
        blight_complaints_2019: cell.blight_complaints_2019 || 0,
        blight_complaints_2020: cell.blight_complaints_2020 || 0,
        is_blighted: cell.is_blighted || false,
        target_blight_count: cell.target_blight_count || 0,
        overall_most_common_blight: cell.overall_most_common_blight || 'None',
        recent_most_common_blight: cell.recent_most_common_blight || 'None',
      }
    }))
  };
};

// ============================================================================
// --- RISK VISUALIZATION HELPERS ---
// ============================================================================

// Convert risk score to human-readable level
export const getRiskLevel = (riskScore: number): string => {
  if (riskScore >= 0.8) return 'Very High';
  if (riskScore >= 0.6) return 'High';
  if (riskScore >= 0.4) return 'Medium';
  if (riskScore >= 0.2) return 'Low';
  return 'Very Low';
};

// Get color for risk visualization
export const getRiskColor = (riskScore: number): string => {
  if (riskScore >= 0.8) return '#8B0000'; // Dark red
  if (riskScore >= 0.6) return '#DC143C'; // Red
  if (riskScore >= 0.4) return '#FF4500'; // Red-orange
  if (riskScore >= 0.2) return '#FFA500'; // Orange
  return '#2E8B57'; // Green
};

// Get opacity based on risk score
export const getRiskOpacity = (riskScore: number): number => {
  // Higher risk = higher opacity
  return Math.max(0.3, Math.min(0.8, riskScore));
};

// ============================================================================
// --- MAPBOX LAYER CONFIGURATIONS ---
// ============================================================================

// Create Mapbox fill layer configuration for risk overlay
export const createRiskFillLayer = (sourceId: string, layerId: string = 'risk-fill') => ({
  id: layerId,
  type: 'fill' as const,
  source: sourceId,
  paint: {
    'fill-color': [
      'interpolate',
      ['linear'],
      ['get', 'risk_score'],
      0, '#2E8B57',    // Green for low risk
      0.2, '#FFD700',  // Yellow for medium-low risk
      0.4, '#FFA500',  // Orange for medium risk
      0.6, '#FF4500',  // Red-orange for high risk
      0.8, '#DC143C',  // Red for very high risk
      1, '#8B0000'     // Dark red for maximum risk
    ],
    'fill-opacity': [
      'interpolate',
      ['linear'],
      ['zoom'],
      10, 0.6,
      14, 0.4,
      18, 0.2
    ]
  }
});

// Create Mapbox line layer configuration for grid boundaries
export const createRiskStrokeLayer = (sourceId: string, layerId: string = 'risk-stroke') => ({
  id: layerId,
  type: 'line' as const,
  source: sourceId,
  paint: {
    'line-color': '#000',
    'line-width': [
      'interpolate',
      ['linear'],
      ['zoom'],
      10, 0.5,
      14, 0.8,
      18, 1.2
    ],
    'line-opacity': [
      'interpolate',
      ['linear'],
      ['zoom'],
      10, 0.3,
      14, 0.5,
      18, 0.7
    ]
  }
});

// Create heatmap layer configuration for risk density
export const createRiskHeatmapLayer = (sourceId: string, layerId: string = 'risk-heatmap') => ({
  id: layerId,
  type: 'heatmap' as const,
  source: sourceId,
  paint: {
    'heatmap-weight': [
      'interpolate',
      ['linear'],
      ['get', 'risk_score'],
      0, 0,
      1, 1
    ],
    'heatmap-intensity': [
      'interpolate',
      ['linear'],
      ['zoom'],
      0, 1,
      9, 3
    ],
    'heatmap-color': [
      'interpolate',
      ['linear'],
      ['heatmap-density'],
      0, 'rgba(0, 0, 0, 0)',
      0.2, 'rgba(0, 87, 192, 0.6)',
      0.4, 'rgba(255, 235, 59, 0.7)',
      0.6, 'rgba(255, 152, 0, 0.8)',
      0.8, 'rgba(255, 56, 56, 0.9)',
      1, 'rgba(255, 0, 0, 1)'
    ],
    'heatmap-radius': [
      'interpolate',
      ['linear'],
      ['zoom'],
      0, 20,
      9, 40,
      16, 80
    ],
    'heatmap-opacity': [
      'interpolate',
      ['linear'],
      ['zoom'],
      7, 0.8,
      9, 0.6,
      14, 0.4,
      18, 0.2
    ]
  }
});

// ============================================================================
// --- GEOGRAPHIC CALCULATIONS ---
// ============================================================================

// Calculate the center point of a polygon
export const getPolygonCenter = (polygon: GeoJSON.Polygon): [number, number] => {
  const coords = polygon.coordinates[0];
  let x = 0, y = 0;
  
  for (const coord of coords) {
    x += coord[0];
    y += coord[1];
  }
  
  return [x / coords.length, y / coords.length];
};

// Calculate bounding box of a feature collection
export const getBoundingBox = (featureCollection: GeoJSON.FeatureCollection): mapboxgl.LngLatBoundsLike => {
  let minLng = Infinity, minLat = Infinity;
  let maxLng = -Infinity, maxLat = -Infinity;
  
  featureCollection.features.forEach(feature => {
    if (feature.geometry.type === 'Polygon') {
      const coords = feature.geometry.coordinates[0];
      coords.forEach(coord => {
        minLng = Math.min(minLng, coord[0]);
        maxLng = Math.max(maxLng, coord[0]);
        minLat = Math.min(minLat, coord[1]);
        maxLat = Math.max(maxLat, coord[1]);
      });
    }
  });
  
  return [[minLng, minLat], [maxLng, maxLat]];
};

// ============================================================================
// --- POPUP AND TOOLTIP HELPERS ---
// ============================================================================

// Create HTML content for risk popup
export const createRiskPopupContent = (properties: any): string => {
  return `
    <div class="risk-popup">
      <div class="risk-popup-header">
        <h4 style="margin: 0; color: ${properties.risk_color};">
          ${properties.risk_level} Risk
        </h4>
        <div class="risk-score">
          ${(properties.risk_score * 100).toFixed(1)}%
        </div>
      </div>
      
      <div class="risk-popup-body">
        <div class="risk-detail">
          <strong>Cell ID:</strong> ${properties.cell_id}
        </div>
        
        <div class="risk-detail">
          <strong>Total Complaints (avg):</strong> ${properties.total_complaints_mean?.toFixed(1) || 'N/A'}
        </div>
        
        <div class="risk-detail">
          <strong>Blight Complaints (avg):</strong> ${properties.blight_complaints_mean?.toFixed(1) || 'N/A'}
        </div>
        
        <div class="risk-detail">
          <strong>Recent Blight Complaints:</strong> ${properties.blight_complaints_recent_mean?.toFixed(1) || 'N/A'}
        </div>
        
        <div class="risk-detail">
          <strong>Trend:</strong> ${properties.blight_complaints_trend > 0 ? 'Increasing' : properties.blight_complaints_trend < 0 ? 'Decreasing' : 'Stable'}
        </div>
        
        ${properties.is_blighted ? '<div class="risk-detail blight-status">⚠️ Currently Blighted</div>' : ''}
        
        <div class="risk-detail">
          <strong>Most Common Blight:</strong> ${properties.overall_most_common_blight}
        </div>
      </div>
    </div>
  `;
};

// Create tooltip content for quick info
export const createRiskTooltipContent = (properties: any): string => {
  return `
    <div class="risk-tooltip">
      <strong>Cell ${properties.cell_id}</strong><br/>
      Risk: ${properties.risk_level} (${(properties.risk_score * 100).toFixed(1)}%)
    </div>
  `;
};

// ============================================================================
// --- FILTERING AND ANALYSIS HELPERS ---
// ============================================================================

// Filter features by risk level
export const filterByRiskLevel = (
  featureCollection: GeoJSON.FeatureCollection,
  minRisk: number = 0,
  maxRisk: number = 1
): GeoJSON.FeatureCollection => {
  return {
    type: 'FeatureCollection',
    features: featureCollection.features.filter(feature => {
      const riskScore = feature.properties?.risk_score || 0;
      return riskScore >= minRisk && riskScore <= maxRisk;
    })
  };
};

// Get top risk features
export const getTopRiskFeatures = (
  featureCollection: GeoJSON.FeatureCollection,
  count: number = 10
): GeoJSON.FeatureCollection => {
  const sortedFeatures = [...featureCollection.features].sort((a, b) => {
    const riskA = a.properties?.risk_score || 0;
    const riskB = b.properties?.risk_score || 0;
    return riskB - riskA;
  });
  
  return {
    type: 'FeatureCollection',
    features: sortedFeatures.slice(0, count)
  };
};

// Calculate risk statistics
export const calculateRiskStats = (featureCollection: GeoJSON.FeatureCollection) => {
  const riskScores = featureCollection.features.map(f => f.properties?.risk_score || 0);
  
  if (riskScores.length === 0) {
    return {
      total: 0,
      mean: 0,
      median: 0,
      min: 0,
      max: 0,
      std: 0,
      riskLevels: { 'Very Low': 0, 'Low': 0, 'Medium': 0, 'High': 0, 'Very High': 0 }
    };
  }
  
  const sorted = riskScores.sort((a, b) => a - b);
  const mean = riskScores.reduce((a, b) => a + b, 0) / riskScores.length;
  const median = sorted[Math.floor(sorted.length / 2)];
  const std = Math.sqrt(
    riskScores.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / riskScores.length
  );
  
  const riskLevels = {
    'Very Low': 0,
    'Low': 0,
    'Medium': 0,
    'High': 0,
    'Very High': 0
  };
  
  riskScores.forEach(score => {
    const level = getRiskLevel(score);
    riskLevels[level as keyof typeof riskLevels]++;
  });
  
  return {
    total: riskScores.length,
    mean: Number(mean.toFixed(3)),
    median: Number(median.toFixed(3)),
    min: Number(sorted[0].toFixed(3)),
    max: Number(sorted[sorted.length - 1].toFixed(3)),
    std: Number(std.toFixed(3)),
    riskLevels
  };
};

// ============================================================================
// --- EXPORT UTILITIES ---
// ============================================================================

// Export GeoJSON to downloadable file
export const exportGeoJSON = (featureCollection: GeoJSON.FeatureCollection, filename: string = 'risk_data.geojson') => {
  const dataStr = JSON.stringify(featureCollection, null, 2);
  const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
  
  const exportFileDefaultName = filename;
  const linkElement = document.createElement('a');
  linkElement.setAttribute('href', dataUri);
  linkElement.setAttribute('download', exportFileDefaultName);
  linkElement.click();
};

// Convert to CSV format
export const exportToCSV = (featureCollection: GeoJSON.FeatureCollection, filename: string = 'risk_data.csv') => {
  const features = featureCollection.features;
  if (features.length === 0) return;
  
  const headers = Object.keys(features[0].properties || {});
  const csvContent = [
    headers.join(','),
    ...features.map(feature => 
      headers.map(header => feature.properties?.[header] || '').join(',')
    )
  ].join('\n');
  
  const dataUri = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csvContent);
  const linkElement = document.createElement('a');
  linkElement.setAttribute('href', dataUri);
  linkElement.setAttribute('download', filename);
  linkElement.click();
}; 