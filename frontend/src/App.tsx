import React, { useState, useEffect } from 'react';
import './App.css';
import LandingPage from './components/LandingPage';
import MapView from './components/MapView';
import LoadingAnimation from './components/LoadingAnimation';
import { AppRiskData } from './types';
import { fetchAllRiskData } from './services/api';
import { keepAliveService } from './services/keepAlive';

type ViewType = 'landing' | 'map';

function App() {
  const [currentView, setCurrentView] = useState<ViewType>('landing');
  const [riskData, setRiskData] = useState<AppRiskData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showLoadingAnimation, setShowLoadingAnimation] = useState(true);
  const [backendStatus, setBackendStatus] = useState<'awake' | 'sleeping' | 'unknown'>('unknown');

  useEffect(() => {
    // Set up the keep-alive service
    keepAliveService.setStatusCallback(setBackendStatus);
    keepAliveService.start();

    // Cleanup on unmount
    return () => {
      keepAliveService.stop();
    };
  }, []);

  const loadRiskData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // If backend is sleeping, try to wake it up first
      if (backendStatus === 'sleeping') {
        console.log('🔄 Backend is sleeping - attempting wake-up before loading data');
        const wakeUpSuccess = await keepAliveService.manualCheck();
        if (!wakeUpSuccess) {
          throw new Error('Backend is not responding. Please try again in a moment.');
        }
      }
      
      const data = await fetchAllRiskData();
      if (data) {
        setRiskData(data);
        setBackendStatus('awake');
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
      {/* Backend Status Indicator (only show if sleeping) */}
      {backendStatus === 'sleeping' && (
        <div style={{
          position: 'fixed',
          top: '10px',
          right: '10px',
          background: 'rgba(255, 193, 7, 0.9)',
          color: 'black',
          padding: '8px 12px',
          borderRadius: '4px',
          fontSize: '12px',
          zIndex: 1000,
          boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
        }}>
          🔄 Backend warming up...
        </div>
      )}

      {/* Service Status Indicator (for development) */}
      {process.env.NODE_ENV === 'development' && (
        <div style={{
          position: 'fixed',
          bottom: '10px',
          right: '10px',
          background: 'rgba(0, 0, 0, 0.7)',
          color: 'white',
          padding: '4px 8px',
          borderRadius: '4px',
          fontSize: '10px',
          zIndex: 999
        }}>
          Keep-Alive: {keepAliveService.isActive() ? '✅' : '❌'} | Status: {backendStatus}
        </div>
      )}

      {currentView === 'landing' && (
        <LandingPage 
          onTryItOut={handleTryItOut}
          loading={loading}
          error={error}
        />
      )}
      
      {currentView === 'map' && (
        <MapView 
          riskData={riskData}
          loading={loading}
          error={error}
          onBackToHome={handleBackToHome}
          onRetry={loadRiskData}
        />
      )}
    </div>
  );
}

export default App;
