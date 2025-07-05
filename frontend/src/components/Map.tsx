import React, { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import './Map.css';
import { DecayData } from '../types';

// You'll need to add your Mapbox token here or via environment variable
mapboxgl.accessToken = process.env.REACT_APP_MAPBOX_TOKEN || 'YOUR_MAPBOX_TOKEN_HERE';

interface MapProps {
  decayData: DecayData[];
  loading: boolean;
}

const Map: React.FC<MapProps> = ({ decayData, loading }) => {
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

  // Update decay visualizations
  useEffect(() => {
    if (!map.current || !mapLoaded || loading) return;

    // Function to add decay layer
    const addDecayLayer = () => {
      if (!map.current) return;

      // Remove existing decay layers and sources
      if (map.current.getLayer('decay-glow')) {
        map.current.removeLayer('decay-glow');
      }
      if (map.current.getSource('decay-points')) {
        map.current.removeSource('decay-points');
      }

      // Create GeoJSON from decay data
      const geojson: GeoJSON.FeatureCollection = {
        type: 'FeatureCollection',
        features: decayData.map(point => ({
          type: 'Feature',
          geometry: {
            type: 'Point',
            coordinates: [point.longitude, point.latitude]
          },
          properties: {
            decay_level: point.decay_level,
            source: point.source
          }
        }))
      };

      // Add source
      map.current.addSource('decay-points', {
        type: 'geojson',
        data: geojson
      });

      // Add heatmap layer for glow effect
      map.current.addLayer({
        id: 'decay-glow',
        type: 'heatmap',
        source: 'decay-points',
        paint: {
          // Increase weight based on decay level
          'heatmap-weight': [
            'interpolate',
            ['linear'],
            ['get', 'decay_level'],
            0, 0,
            0.2, 0.3,
            0.5, 0.6,
            0.8, 0.8,
            1, 1
          ],
          // Increase intensity as zoom level increases
          'heatmap-intensity': [
            'interpolate',
            ['linear'],
            ['zoom'],
            0, 1,
            9, 3
          ],
          // Color ramp for heatmap
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
          // Adjust radius by zoom level and decay level
          'heatmap-radius': [
            'interpolate',
            ['linear'],
            ['zoom'],
            0, [
              'interpolate',
              ['linear'],
              ['get', 'decay_level'],
              0, 10,
              1, 50
            ],
            9, [
              'interpolate',
              ['linear'],
              ['get', 'decay_level'],
              0, 20,
              1, 100
            ],
            16, [
              'interpolate',
              ['linear'],
              ['get', 'decay_level'],
              0, 50,
              1, 200
            ]
          ],
          // Fade out the heatmap at higher zoom levels
          'heatmap-opacity': [
            'interpolate',
            ['linear'],
            ['zoom'],
            7, 0.8,
            9, 0.6
          ]
        }
      });

      // Add click handler for tooltips
      map.current.on('click', 'decay-glow', (e) => {
        if (!e.features || e.features.length === 0) return;

        const coordinates = e.lngLat;
        const feature = e.features[0];
        const decayLevel = feature.properties?.decay_level || 0;
        const source = feature.properties?.source || 'unknown';

        const decayDescription = getDecayDescription(decayLevel);

        new mapboxgl.Popup()
          .setLngLat([coordinates.lng, coordinates.lat])
          .setHTML(`
            <div class="popup-content">
              <h4>${decayDescription}</h4>
              <p><strong>Decay Level:</strong> ${decayLevel.toFixed(2)}</p>
              <p><strong>Source:</strong> ${source}</p>
            </div>
          `)
          .addTo(map.current!);
      });

      // Change cursor on hover
      map.current.on('mouseenter', 'decay-glow', () => {
        if (map.current) map.current.getCanvas().style.cursor = 'pointer';
      });

      map.current.on('mouseleave', 'decay-glow', () => {
        if (map.current) map.current.getCanvas().style.cursor = '';
      });
    };

    // Add the decay layer
    addDecayLayer();
  }, [decayData, mapLoaded, loading]);

  const getDecayDescription = (level: number): string => {
    if (level < 0.2) return 'Minimal Decay';
    if (level < 0.5) return 'Mild Decay';
    if (level < 0.8) return 'Noticeable Decay';
    return 'Severe Decay';
  };

  return (
    <div className="map-wrapper">
      {loading && (
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <p>Loading decay data...</p>
        </div>
      )}
      <div ref={mapContainer} className="map" />
    </div>
  );
};

export default Map; 