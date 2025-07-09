import React, { useEffect, useState } from 'react';
import './App.css';
import LandingPage from './components/LandingPage';
import MapView from './components/MapView';
import LoadingAnimation from './components/LoadingAnimation';
import { fetchAllRiskData, RiskGridCell, FeatureImportanceResponse, ApiStats, TopRiskArea, checkHealth } from './services/api';
import { convertRiskDataToGeoJSON } from './utils/geoHelpers';

type ViewType = 'landing' | 'map';

interface AppRiskData {
  riskData: RiskGridCell[];
  geoJsonData: GeoJSON.FeatureCollection;
  featureImportance: FeatureImportanceResponse;
  stats: ApiStats;
  topRiskAreas: TopRiskArea[];
}

function App() {
  const [currentView, setCurrentView] = useState<ViewType>('landing');
  const [riskData, setRiskData] = useState<AppRiskData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showLoadingAnimation, setShowLoadingAnimation] = useState(true);

  useEffect(() => {
    // This function will ping the backend every 14 minutes to prevent it from sleeping on Render's free tier.
    const keepAlive = async () => {
      try {
        await checkHealth();
        console.log('Backend keep-alive ping sent successfully.');
      } catch (err) {
        console.error('Keep-alive ping failed:', err);
      }
    };

    // Send the first ping immediately on load
    keepAlive();

    const intervalId = setInterval(keepAlive, 14 * 60 * 1000); // 14 minutes in milliseconds

    // Clean up the interval when the component unmounts
    return () => clearInterval(intervalId);
  }, []);

  const loadRiskData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const data = await fetchAllRiskData();
      if (data) {
        setRiskData(data);
      }
    } catch (error) {
      console.error('Error loading risk data:', error);
      setError(error instanceof Error ? error.message : 'Failed to load risk data');
    } finally {
      setLoading(false);
    }
  };

  const handleTryItOut = () => {
    setCurrentView('map');
    if (!riskData) {
      loadRiskData();
    }
  };

  const handleBackToHome = () => {
    setCurrentView('landing');
  };

  const handleAnimationComplete = () => {
    setShowLoadingAnimation(false);
  };

  // Show loading animation on initial load
  if (showLoadingAnimation) {
    return <LoadingAnimation onAnimationComplete={handleAnimationComplete} />;
  }

  return (
    <div className="App">
      {/* Navigation Bar */}
      <nav className="app-nav">
        <button 
          className={`nav-button ${currentView === 'landing' ? 'active' : ''}`}
          onClick={handleBackToHome}
        >
          Home
        </button>
        <button 
          className={`nav-button ${currentView === 'map' ? 'active' : ''}`}
          onClick={handleTryItOut}
        >
          Risk Map
        </button>
      </nav>

      {/* Main Content */}
      {currentView === 'landing' && (
        <LandingPage onTryItOut={handleTryItOut} />
      )}
      {currentView === 'map' && (
        <MapView 
          riskData={riskData} 
          loading={loading} 
          error={error}
          onBackToHome={handleBackToHome}
          onRefresh={loadRiskData}
        />
      )}
    </div>
  );
}

export default App;
