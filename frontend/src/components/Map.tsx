import React, { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import './Map.css';
import { RiskGridCell, FeatureImportanceResponse, ApiStats, TopRiskArea } from '../services/api';
import { 
  createRiskPopupContent,
  getBoundingBox 
} from '../utils/geoHelpers';
import '../components/RiskPopup.css';

// You'll need to add your Mapbox token here or via environment variable
mapboxgl.accessToken = 'pk.eyJ1IjoibWFwYm94bW8xMjMiLCJhIjoiY21jcXN5dXdqMDFrODJqcTBhcnppb3QzMSJ9.do-_5cjmbfnW2o-FyMofvA';

interface AppRiskData {
  riskData: RiskGridCell[];
  geoJsonData: GeoJSON.FeatureCollection;
  featureImportance: FeatureImportanceResponse;
  stats: ApiStats;
  topRiskAreas: TopRiskArea[];
}

interface MapProps {
  riskData: GeoJSON.FeatureCollection | null;
  loading: boolean;
  allRiskData: AppRiskData | null;
}

const Map: React.FC<MapProps> = ({ riskData, loading, allRiskData }) => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const popup = useRef<mapboxgl.Popup | null>(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [lng] = useState(-79.3832);
  const [lat] = useState(43.6532);
  const [zoom] = useState(11);
  const [hoveredCell, setHoveredCell] = useState<number | null>(null);
  const [layersAdded, setLayersAdded] = useState(false);

  // Initialize map with performance-optimized settings
  useEffect(() => {
    if (map.current || !mapContainer.current) return;

    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/standard',
      center: [lng, lat],
      zoom: zoom,
      pitch: 60, // Enhanced for beautiful 3D buildings
      bearing: -20, // Better angle for 3D viewing
      antialias: true, // Enable for beautiful 3D buildings
      fadeDuration: 300, // Smooth transitions
      preserveDrawingBuffer: false
    });

    map.current.on('load', () => {
      setMapLoaded(true);
      
      // Mapbox Standard Style already includes beautiful 3D buildings and lighting
      // No need for custom fog or building layers
    });

    // Basic navigation controls
    map.current.addControl(new mapboxgl.NavigationControl(), 'top-right');
    map.current.addControl(new mapboxgl.ScaleControl(), 'bottom-left');
    map.current.addControl(new mapboxgl.FullscreenControl(), 'top-right');

    return () => {
      // Cleanup popup
      if (popup.current) {
        popup.current.remove();
        popup.current = null;
      }
      
      // Cleanup map
      map.current?.remove();
      map.current = null;
      setMapLoaded(false);
      setLayersAdded(false);
    };
  }, [lat, lng, zoom]);

  // Add risk data layers (only once)
  useEffect(() => {
    if (!map.current || !mapLoaded || loading || !riskData || layersAdded) return;

    const addRiskLayers = () => {
      if (!map.current) return;

      // Add source
      map.current.addSource('risk-grid', {
        type: 'geojson',
        data: riskData
      });

      // Add intense inner glow layer for high-risk areas
      map.current.addLayer({
        id: 'risk-inner-glow',
        type: 'fill',
        source: 'risk-grid',
        paint: {
          'fill-color': [
            'interpolate',
            ['linear'],
            ['get', 'risk_score'],
            0, '#66BB6A',       // Bright green for low risk
            0.3, '#FFEB3B',     // Bright yellow
            0.5, '#FFB74D',     // Bright orange
            0.7, '#FF7043',     // Bright orange-red
            0.8, '#EF5350',     // Bright red
            1, '#E53935'        // Bright crimson
          ],
          'fill-opacity': [
            'interpolate',
            ['linear'],
            ['get', 'risk_score'],
            0, 0.02,    // Minimal glow for low risk
            0.3, 0.05,  // Subtle glow
            0.5, 0.1,   // Light glow
            0.7, 0.2,   // Noticeable glow
            0.8, 0.35,  // Strong inner glow
            1, 0.5      // Maximum inner glow
          ]
        }
      });

      // Add outer glow layer (softer, more spread out)
      map.current.addLayer({
        id: 'risk-outer-glow',
        type: 'fill',
        source: 'risk-grid',
        paint: {
          'fill-color': [
            'interpolate',
            ['linear'],
            ['get', 'risk_score'],
            0, '#4CAF50',       // Green glow
            0.3, '#FFC107',     // Gold glow
            0.5, '#FF9800',     // Orange glow
            0.7, '#FF5722',     // Orange-red glow
            0.8, '#F44336',     // Red glow
            1, '#B71C1C'        // Dark red glow
          ],
          'fill-opacity': [
            'interpolate',
            ['linear'],
            ['get', 'risk_score'],
            0, 0.06,    // Very subtle glow for low risk
            0.3, 0.12,  // Slight glow for medium-low risk
            0.5, 0.2,   // Moderate glow for medium risk
            0.7, 0.3,   // More noticeable glow for high risk
            0.8, 0.45,  // Strong glow for very high risk
            1, 0.6      // Maximum glow for extreme risk
          ]
        }
      });

      // Add fill layer
      map.current.addLayer({
        id: 'risk-fill-optimized',
        type: 'fill',
        source: 'risk-grid',
        paint: {
          'fill-color': [
            'interpolate',
            ['linear'],
            ['get', 'risk_score'],
            0, '#4CAF50',       // Softer green
            0.3, '#FFC107',     // Softer gold
            0.5, '#FF9800',     // Softer orange
            0.7, '#FF5722',     // Softer orange red
            0.8, '#F44336',     // Softer red
            1, '#B71C1C'        // Softer dark red
          ],
          'fill-opacity': [
            'interpolate',
            ['linear'],
            ['get', 'risk_score'],
            0, 0.2,
            0.5, 0.3,
            0.8, 0.4,
            1, 0.5
          ]
        }
      });

      // Add stroke layer
      map.current.addLayer({
        id: 'risk-stroke-optimized',
        type: 'line',
        source: 'risk-grid',
        paint: {
          'line-color': [
            'interpolate',
            ['linear'],
            ['get', 'risk_score'],
            0, 'rgba(76, 175, 80, 0.3)',
            0.5, 'rgba(255, 152, 0, 0.4)',
            1, 'rgba(183, 28, 28, 0.6)'
          ],
          'line-width': [
            'interpolate',
            ['linear'],
            ['get', 'risk_score'],
            0, 0.5,
            0.8, 1.5,
            1, 2.0
          ],
          'line-opacity': 0.5
        }
      });

      // Add hover highlight layer (initially hidden)
      map.current.addLayer({
        id: 'risk-highlight',
        type: 'line',
        source: 'risk-grid',
        paint: {
          'line-color': '#ffffff',
          'line-width': 3,
          'line-opacity': 0.9
        },
        filter: ['==', 'cell_id', -1] // Initially hide all
      });

      setLayersAdded(true);
    };

    addRiskLayers();
  }, [riskData, mapLoaded, loading, layersAdded]);

  // Update data when riskData changes (for filters)
  useEffect(() => {
    if (!map.current || !layersAdded || !riskData) return;

    // Update the GeoJSON source with new filtered data
    const source = map.current.getSource('risk-grid') as mapboxgl.GeoJSONSource;
    if (source) {
      source.setData(riskData);
    }
  }, [riskData, layersAdded]);

  // Setup event listeners (only once after layers are added)
  useEffect(() => {
    if (!map.current || !layersAdded) return;

    let hoverTimeout: NodeJS.Timeout;

    // Click handler for popups
    const handleClick = (e: mapboxgl.MapMouseEvent) => {
      if (!e.features || e.features.length === 0) return;

      const feature = e.features[0];
      const properties = feature.properties;

      if (properties) {
        // Remove existing popup
        if (popup.current) {
          popup.current.remove();
        }

        // Create new popup
        popup.current = new mapboxgl.Popup({
          closeButton: true,
          closeOnClick: false,
          maxWidth: '300px'
        });

        popup.current
          .setLngLat(e.lngLat)
          .setHTML(createRiskPopupContent(properties))
          .addTo(map.current!);
      }
    };

    // Mouse enter handler
    const handleMouseEnter = (e: mapboxgl.MapMouseEvent) => {
      if (!map.current) return;
      
      map.current.getCanvas().style.cursor = 'pointer';
      if (e.features && e.features.length > 0) {
        clearTimeout(hoverTimeout);
        hoverTimeout = setTimeout(() => {
          const cellId = e.features![0].properties?.cell_id;
          setHoveredCell(cellId);
        }, 16); // ~60fps throttling
      }
    };

    // Mouse leave handler
    const handleMouseLeave = () => {
      if (!map.current) return;
      
      map.current.getCanvas().style.cursor = '';
      clearTimeout(hoverTimeout);
      setHoveredCell(null);
    };

    // Add event listeners
    map.current.on('click', 'risk-fill-optimized', handleClick);
    map.current.on('mouseenter', 'risk-fill-optimized', handleMouseEnter);
    map.current.on('mouseleave', 'risk-fill-optimized', handleMouseLeave);

    // Cleanup function
    return () => {
      if (map.current) {
        map.current.off('click', 'risk-fill-optimized', handleClick);
        map.current.off('mouseenter', 'risk-fill-optimized', handleMouseEnter);
        map.current.off('mouseleave', 'risk-fill-optimized', handleMouseLeave);
      }
      clearTimeout(hoverTimeout);
    };
  }, [layersAdded]);

  // Update hover highlighting
  useEffect(() => {
    if (!map.current || !layersAdded) return;

    // Update highlight layer filter
    map.current.setFilter('risk-highlight', ['==', 'cell_id', hoveredCell || -1]);

    // Update glow layers for hover effect (enhance glow on hover)
    map.current.setPaintProperty('risk-inner-glow', 'fill-opacity', [
      'case',
      ['==', ['get', 'cell_id'], hoveredCell || -1],
      [
        'interpolate',
        ['linear'],
        ['get', 'risk_score'],
        0, 0.08,
        0.5, 0.2,
        0.8, 0.5,
        1, 0.7
      ],
      [
        'interpolate',
        ['linear'],
        ['get', 'risk_score'],
        0, 0.02,
        0.3, 0.05,
        0.5, 0.1,
        0.7, 0.2,
        0.8, 0.35,
        1, 0.5
      ]
    ]);

    map.current.setPaintProperty('risk-outer-glow', 'fill-opacity', [
      'case',
      ['==', ['get', 'cell_id'], hoveredCell || -1],
      [
        'interpolate',
        ['linear'],
        ['get', 'risk_score'],
        0, 0.15,
        0.5, 0.35,
        0.8, 0.6,
        1, 0.8
      ],
      [
        'interpolate',
        ['linear'],
        ['get', 'risk_score'],
        0, 0.06,
        0.3, 0.12,
        0.5, 0.2,
        0.7, 0.3,
        0.8, 0.45,
        1, 0.6
      ]
    ]);

    // Update fill layer opacity for hover effect
    map.current.setPaintProperty('risk-fill-optimized', 'fill-opacity', [
      'case',
      ['==', ['get', 'cell_id'], hoveredCell || -1],
      0.7,
      [
        'interpolate',
        ['linear'],
        ['get', 'risk_score'],
        0, 0.2,
        0.5, 0.3,
        0.8, 0.4,
        1, 0.5
      ]
    ]);

    // Update stroke layer for hover effect
    map.current.setPaintProperty('risk-stroke-optimized', 'line-color', [
      'case',
      ['==', ['get', 'cell_id'], hoveredCell || -1],
      '#ffffff',
      [
        'interpolate',
        ['linear'],
        ['get', 'risk_score'],
        0, 'rgba(76, 175, 80, 0.3)',
        0.5, 'rgba(255, 152, 0, 0.4)',
        1, 'rgba(183, 28, 28, 0.6)'
      ]
    ]);

    map.current.setPaintProperty('risk-stroke-optimized', 'line-width', [
      'case',
      ['==', ['get', 'cell_id'], hoveredCell || -1],
      2.5,
      [
        'interpolate',
        ['linear'],
        ['get', 'risk_score'],
        0, 0.5,
        0.8, 1.5,
        1, 2.0
      ]
    ]);
  }, [hoveredCell, layersAdded]);

  // Fit bounds when data changes or loads
  useEffect(() => {
    if (!map.current || !layersAdded || !riskData) return;

    try {
      // Use the actual filtered data for bounds fitting
      const bounds = getBoundingBox(riskData);
      map.current.fitBounds(bounds, {
        padding: 50,
        duration: 1000,
        essential: true
      });
    } catch (error) {
      console.warn('Could not fit bounds:', error);
      // Fallback to allRiskData bounds if available
      if (allRiskData) {
        try {
          const bounds = getBoundingBox(allRiskData.geoJsonData);
          map.current.fitBounds(bounds, {
            padding: 50,
            duration: 1000,
            essential: true
          });
        } catch (fallbackError) {
          console.warn('Could not fit fallback bounds:', fallbackError);
        }
      }
    }
  }, [riskData, layersAdded, allRiskData]);

  return (
    <div className="map-wrapper">
      {loading && (
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <p>Loading urban blight risk data...</p>
        </div>
      )}
      
      {!loading && !riskData && (
        <div className="no-data-overlay">
          <div className="no-data-message">
            <h3>No Risk Data Available</h3>
            <p>Please check your connection and try again.</p>
          </div>
        </div>
      )}
      
      <div ref={mapContainer} className="map" />
    </div>
  );
};

export default Map; 