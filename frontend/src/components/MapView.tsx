import React, { useState } from 'react';
import './MapView.css';
import Map from './Map';
import FilterButtonPanel from './FilterButtonPanel';
import Legend from './Legend';
import StatsBtn from './StatsBtn';
import TopBlightMenu from './TopBlightMenu';
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
  onRetry: () => Promise<void>;
}

const MapView: React.FC<MapViewProps> = ({ 
  riskData, 
  loading, 
  error, 
  onBackToHome, 
  onRetry 
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
            <button onClick={onRetry} className="retry-btn">
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="map-view">
      
      
      
      <div className="map-container">
        <Map 
          riskData={filteredData} 
          loading={loading} 
          allRiskData={riskData}
        />
        
        <Legend />
        <StatsBtn riskStats={riskStats} riskFilter={riskFilter} />
        <FilterButtonPanel riskFilter={riskFilter} setRiskFilter={setRiskFilter} />
        <TopBlightMenu riskData={riskData?.riskData || null} />
        
        
        

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
        
      </div>
    </div>
  );
};

export default MapView; 