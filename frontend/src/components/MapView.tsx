import React, { useState } from 'react';
import './MapView.css';
import Map from './Map';
import Legend from './Legend';
import { RiskGridCell, FeatureImportanceResponse, ApiStats, TopRiskArea } from '../services/api';
import { filterByRiskLevel, calculateRiskStats } from '../utils/geoHelpers';
import '../components/RiskPopup.css';

interface AppRiskData {
  riskData: RiskGridCell[];
  geoJsonData: GeoJSON.FeatureCollection;
  featureImportance: FeatureImportanceResponse;
  stats: ApiStats;
  topRiskAreas: TopRiskArea[];
}

interface MapViewProps {
  riskData: AppRiskData | null;
  loading: boolean;
  error: string | null;
  onBackToHome: () => void;
  onRefresh: () => Promise<void>;
}

const MapView: React.FC<MapViewProps> = ({ 
  riskData, 
  loading, 
  error, 
  onBackToHome, 
  onRefresh 
}) => {
  const [riskFilter, setRiskFilter] = useState<'all' | 'very-high' | 'high' | 'medium' | 'low'>('all');
  const [showStats, setShowStats] = useState(false);

  // Filter data based on risk level
  const getFilteredData = () => {
    if (!riskData) return null;
    
    if (riskFilter === 'all') {
      return riskData.geoJsonData;
    }
    
    const riskRanges = {
      'very-high': [0.8, 1.0],
      'high': [0.6, 0.8],
      'medium': [0.4, 0.6],
      'low': [0.0, 0.4]
    };
    
    const [minRisk, maxRisk] = riskRanges[riskFilter];
    return filterByRiskLevel(riskData.geoJsonData, minRisk, maxRisk);
  };

  const filteredData = getFilteredData();
  const riskStats = filteredData ? calculateRiskStats(filteredData) : null;

  if (error) {
    return (
      <div className="map-view">
        <header className="map-header">
          <button className="back-btn" onClick={onBackToHome}>
            ← Back to Home
          </button>
          <h1 className="map-title">Urban Blight Risk Map</h1>
          <div className="header-spacer"></div>
        </header>
        
        <div className="error-container">
          <div className="error-message">
            <h3>Error Loading Risk Data</h3>
            <p>{error}</p>
            <button onClick={onRefresh} className="retry-btn">
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="map-view">
      <header className="map-header">
        <button className="back-btn" onClick={onBackToHome}>
          ← Back to Home
        </button>
        <h1 className="map-title">Urban Blight Risk Map</h1>
        <div className="header-actions">
          <button 
            className="stats-btn" 
            onClick={() => setShowStats(!showStats)}
          >
            {showStats ? 'Hide Stats' : 'Show Stats'}
          </button>
          <button onClick={onRefresh} className="refresh-btn" disabled={loading}>
            {loading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </header>
      
      <div className="map-container">
        <Map 
          riskData={filteredData} 
          loading={loading} 
          allRiskData={riskData}
        />
        <Legend />
        
        {/* Risk Level Controls */}
        <div className="controls">
          <div className="risk-filter">
            <label>Filter by Risk Level:</label>
            <div className="filter-buttons">
              <button 
                className={riskFilter === 'all' ? 'active' : ''}
                onClick={() => setRiskFilter('all')}
              >
                All Levels
              </button>
              <button 
                className={`risk-btn very-high ${riskFilter === 'very-high' ? 'active' : ''}`}
                onClick={() => setRiskFilter('very-high')}
              >
                Very High
              </button>
              <button 
                className={`risk-btn high ${riskFilter === 'high' ? 'active' : ''}`}
                onClick={() => setRiskFilter('high')}
              >
                High
              </button>
              <button 
                className={`risk-btn medium ${riskFilter === 'medium' ? 'active' : ''}`}
                onClick={() => setRiskFilter('medium')}
              >
                Medium
              </button>
              <button 
                className={`risk-btn low ${riskFilter === 'low' ? 'active' : ''}`}
                onClick={() => setRiskFilter('low')}
              >
                Low
              </button>
            </div>
          </div>
        </div>

        {/* Statistics Panel */}
        {showStats && riskStats && (
          <div className="stats-panel">
            <h3>Risk Statistics</h3>
            <div className="stats-grid">
              <div className="stat-item">
                <strong>Total Cells:</strong> {riskStats.total}
              </div>
              <div className="stat-item">
                <strong>Mean Risk:</strong> {(riskStats.mean * 100).toFixed(1)}%
              </div>
              <div className="stat-item">
                <strong>Max Risk:</strong> {(riskStats.max * 100).toFixed(1)}%
              </div>
              <div className="stat-item">
                <strong>Very High Risk:</strong> {riskStats.riskLevels['Very High']}
              </div>
              <div className="stat-item">
                <strong>High Risk:</strong> {riskStats.riskLevels['High']}
              </div>
              <div className="stat-item">
                <strong>Medium Risk:</strong> {riskStats.riskLevels['Medium']}
              </div>
            </div>
          </div>
        )}

        {/* Top Risk Areas Panel */}
        {riskData && (
          <div className="top-risk-panel">
            <h4>Top Risk Areas</h4>
            <div className="risk-list">
              {riskData.topRiskAreas.slice(0, 5).map(area => (
                <div key={area.cell_id} className="risk-item">
                  <span className="cell-id">Cell {area.cell_id}</span>
                  <span className={`risk-level ${area.risk_level.toLowerCase().replace(' ', '-')}`}>
                    {area.risk_level}
                  </span>
                  <span className="risk-percentage">
                    {(area.risk_score * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MapView; 