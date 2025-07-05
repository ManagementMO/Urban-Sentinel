import React, { useState } from 'react';
import './MapView.css';
import Map from './Map';
import Legend from './Legend';
import { DecayData } from '../types';

interface MapViewProps {
  decayData: DecayData[];
  loading: boolean;
  onBackToHome: () => void;
}

const MapView: React.FC<MapViewProps> = ({ decayData, loading, onBackToHome }) => {
  const [dataSource, setDataSource] = useState<'all' | '311' | 'satellite'>('all');

  const filteredData = dataSource === 'all' 
    ? decayData 
    : decayData.filter(d => d.source === dataSource);

  return (
    <div className="map-view">
      <header className="map-header">
        <button className="back-btn" onClick={onBackToHome}>
          ← Back to Home
        </button>
        <h1 className="map-title">Urban Decay Map</h1>
        <div className="header-spacer"></div>
      </header>
      
      <div className="map-container">
        <Map decayData={filteredData} loading={loading} />
        <Legend />
        <div className="controls">
          <div className="source-toggle">
            <button 
              className={dataSource === 'all' ? 'active' : ''}
              onClick={() => setDataSource('all')}
            >
              All Sources
            </button>
            <button 
              className={dataSource === '311' ? 'active' : ''}
              onClick={() => setDataSource('311')}
            >
              311 Calls
            </button>
            <button 
              className={dataSource === 'satellite' ? 'active' : ''}
              onClick={() => setDataSource('satellite')}
            >
              Satellite
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MapView; 