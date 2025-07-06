import React, { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import './Map.css';
import { RiskGridCell, FeatureImportanceResponse, ApiStats, TopRiskArea } from '../services/api';
import { 
  createRiskFillLayer, 
  createRiskStrokeLayer, 
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
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [lng, lat],
      zoom: zoom,
      pitch: 50, // Reduced from 60 for better performance
      bearing: -15, // Simplified angle
      antialias: false, // Disabled for performance
      fadeDuration: 100, // Reduced animation duration
      preserveDrawingBuffer: false // Better memory management
    });

    map.current.on('load', () => {
      setMapLoaded(true);
      
      // Simplified fog effects for performance
      map.current!.setFog({
        color: 'rgb(20, 25, 40)',
        'high-color': 'rgb(10, 15, 25)',
        'horizon-blend': 0.02,
        'space-color': 'rgb(5, 5, 10)',
        'star-intensity': 0.3
      });

      // Simplified 3D buildings - no animation
      const layers = map.current!.getStyle().layers;
      const labelLayerId = layers!.find(
        (layer) => layer.type === 'symbol' && layer.layout && layer.layout['text-field']
      )?.id;

      if (labelLayerId) {
        map.current!.addLayer(
          {
            'id': '3d-buildings',
            'source': 'composite',
            'source-layer': 'building',
            'filter': ['==', 'extrude', 'true'],
            'type': 'fill-extrusion',
            'minzoom': 15,
            'paint': {
              'fill-extrusion-color': [
                'interpolate',
                ['linear'],
                ['get', 'height'],
                0, '#2a2a3a',
                100, '#3a3a4a',
                200, '#4a4a5a'
              ],
              'fill-extrusion-height': ['get', 'height'],
              'fill-extrusion-base': ['get', 'min_height'],
              'fill-extrusion-opacity': 0.7
            }
          },
          labelLayerId
        );
      }

      // Simplified water styling
      map.current!.setPaintProperty('water', 'fill-color', '#1a1a2a');
      map.current!.setPaintProperty('water', 'fill-opacity', 0.8);
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
  }, []);

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
            0, '#228B22',       // Forest green
            0.3, '#FFD700',     // Gold
            0.5, '#FF8C00',     // Dark orange
            0.7, '#FF4500',     // Orange red
            0.8, '#DC143C',     // Crimson
            1, '#8B0000'        // Dark red
          ],
          'fill-opacity': [
            'interpolate',
            ['linear'],
            ['get', 'risk_score'],
            0, 0.4,
            0.5, 0.6,
            0.8, 0.8,
            1, 0.9
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
            0, 'rgba(34, 139, 34, 0.5)',
            0.5, 'rgba(255, 140, 0, 0.7)',
            1, 'rgba(139, 0, 0, 1.0)'
          ],
          'line-width': [
            'interpolate',
            ['linear'],
            ['get', 'risk_score'],
            0, 0.5,
            0.8, 1.5,
            1, 2.0
          ],
          'line-opacity': 0.8
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

    // Update fill layer opacity for hover effect
    map.current.setPaintProperty('risk-fill-optimized', 'fill-opacity', [
      'case',
      ['==', ['get', 'cell_id'], hoveredCell || -1],
      0.9,
      [
        'interpolate',
        ['linear'],
        ['get', 'risk_score'],
        0, 0.4,
        0.5, 0.6,
        0.8, 0.8,
        1, 0.9
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
        0, 'rgba(34, 139, 34, 0.5)',
        0.5, 'rgba(255, 140, 0, 0.7)',
        1, 'rgba(139, 0, 0, 1.0)'
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
  }, [riskData, layersAdded]);

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