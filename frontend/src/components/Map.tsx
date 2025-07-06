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
  const [mapLoaded, setMapLoaded] = useState(false);
  const [lng] = useState(-79.3832);
  const [lat] = useState(43.6532);
  const [zoom] = useState(11);

  // Initialize map
  useEffect(() => {
    if (map.current || !mapContainer.current) return;

    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [lng, lat],
      zoom: zoom,
      pitch: 45,
      bearing: -17.6,
      antialias: true
    });

    map.current.on('load', () => {
      setMapLoaded(true);
      
      // Add 3D buildings
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
              'fill-extrusion-color': '#aaa',
              'fill-extrusion-height': [
                'interpolate',
                ['linear'],
                ['zoom'],
                15,
                0,
                15.05,
                ['get', 'height']
              ],
              'fill-extrusion-base': [
                'interpolate',
                ['linear'],
                ['zoom'],
                15,
                0,
                15.05,
                ['get', 'min_height']
              ],
              'fill-extrusion-opacity': 0.6
            }
          },
          labelLayerId
        );
      }
    });

    // Add navigation controls
    map.current.addControl(new mapboxgl.NavigationControl(), 'top-right');

    return () => {
      map.current?.remove();
      map.current = null;
      setMapLoaded(false);
    };
  }, []); // Remove dependencies to prevent re-initialization

  // Update risk visualizations
  useEffect(() => {
    if (!map.current || !mapLoaded || loading || !riskData) return;

    // Function to add risk overlay
    const addRiskOverlay = () => {
      if (!map.current) return;

      // Remove existing risk layers and sources
      if (map.current.getLayer('risk-fill')) {
        map.current.removeLayer('risk-fill');
      }
      if (map.current.getLayer('risk-stroke')) {
        map.current.removeLayer('risk-stroke');
      }
      if (map.current.getSource('risk-grid')) {
        map.current.removeSource('risk-grid');
      }

      // Add source
      map.current.addSource('risk-grid', {
        type: 'geojson',
        data: riskData
      });

      // Add fill layer
      map.current.addLayer({
        id: 'risk-fill',
        type: 'fill',
        source: 'risk-grid',
        paint: {
          'fill-color': [
            'interpolate',
            ['linear'],
            ['get', 'risk_score'],
            0, '#2e8b57',    // Green for low risk
            0.2, '#ffd700',  // Yellow for medium-low risk
            0.4, '#ffa500',  // Orange for medium risk
            0.6, '#ff4500',  // Red-orange for high risk
            0.8, '#dc143c',  // Red for very high risk
            1, '#8b0000'     // Dark red for maximum risk
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

      // Add stroke layer
      map.current.addLayer({
        id: 'risk-stroke',
        type: 'line',
        source: 'risk-grid',
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

      // Add click handler for popups
      map.current.on('click', 'risk-fill', (e) => {
        if (!e.features || e.features.length === 0) return;

        const feature = e.features[0];
        const properties = feature.properties;

        if (properties) {
          new mapboxgl.Popup()
            .setLngLat(e.lngLat)
            .setHTML(createRiskPopupContent(properties))
            .addTo(map.current!);
        }
      });

      // Change cursor on hover
      map.current.on('mouseenter', 'risk-fill', () => {
        if (map.current) map.current.getCanvas().style.cursor = 'pointer';
      });

      map.current.on('mouseleave', 'risk-fill', () => {
        if (map.current) map.current.getCanvas().style.cursor = '';
      });

      // Fit map to data bounds if this is the first load
      if (allRiskData && riskData.features.length > 0) {
        try {
          const bounds = getBoundingBox(allRiskData.geoJsonData);
          map.current.fitBounds(bounds, {
            padding: 50,
            duration: 1000
          });
        } catch (error) {
          console.warn('Could not fit bounds:', error);
        }
      }
    };

    // Add the risk overlay
    addRiskOverlay();
  }, [riskData, mapLoaded, loading, allRiskData]);

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
      
      {/* Risk Legend */}
      <div className="risk-legend">
        <div className="risk-legend-title">Blight Risk Level</div>
        <div className="risk-legend-item">
          <div className="risk-legend-color" style={{ backgroundColor: '#8b0000' }}></div>
          <div className="risk-legend-label">Very High (80-100%)</div>
        </div>
        <div className="risk-legend-item">
          <div className="risk-legend-color" style={{ backgroundColor: '#dc143c' }}></div>
          <div className="risk-legend-label">High (60-80%)</div>
        </div>
        <div className="risk-legend-item">
          <div className="risk-legend-color" style={{ backgroundColor: '#ff4500' }}></div>
          <div className="risk-legend-label">Medium (40-60%)</div>
        </div>
        <div className="risk-legend-item">
          <div className="risk-legend-color" style={{ backgroundColor: '#ffa500' }}></div>
          <div className="risk-legend-label">Low (20-40%)</div>
        </div>
        <div className="risk-legend-item">
          <div className="risk-legend-color" style={{ backgroundColor: '#2e8b57' }}></div>
          <div className="risk-legend-label">Very Low (0-20%)</div>
        </div>
      </div>
    </div>
  );
};

export default Map; 