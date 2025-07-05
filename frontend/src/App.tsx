import React, { useEffect, useState } from 'react';
import './App.css';
import LandingPage from './components/LandingPage';
import MapView from './components/MapView';
import { DecayData } from './types';
import { fetchDecayData } from './services/api';

type ViewType = 'landing' | 'map';

function App() {
  const [currentView, setCurrentView] = useState<ViewType>('landing');
  const [decayData, setDecayData] = useState<DecayData[]>([]);
  const [loading, setLoading] = useState(false);

  const loadDecayData = async () => {
    try {
      setLoading(true);
      const data = await fetchDecayData();
      setDecayData(data);
    } catch (error) {
      console.error('Error loading decay data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleTryItOut = () => {
    setCurrentView('map');
    loadDecayData();
  };

  const handleBackToHome = () => {
    setCurrentView('landing');
  };

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
          Map
        </button>
      </nav>

      {/* Main Content */}
      {currentView === 'landing' && (
        <LandingPage onTryItOut={handleTryItOut} />
      )}
      {currentView === 'map' && (
        <MapView 
          decayData={decayData} 
          loading={loading} 
          onBackToHome={handleBackToHome}
        />
      )}
    </div>
  );
}

export default App;
