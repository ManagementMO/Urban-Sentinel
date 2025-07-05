import React, { useEffect, useState } from 'react';
import './App.css';
import LandingPage from './components/LandingPage';
import MapView from './components/MapView';
import { DecayData } from './types';
import { fetchDecayData } from './services/api';

function App() {
  const [showMap, setShowMap] = useState(false);
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
    setShowMap(true);
    loadDecayData();
  };

  const handleBackToHome = () => {
    setShowMap(false);
  };

  return (
    <div className="App">
      {showMap ? (
        <MapView 
          decayData={decayData} 
          loading={loading} 
          onBackToHome={handleBackToHome}
        />
      ) : (
        <LandingPage onTryItOut={handleTryItOut} />
      )}
    </div>
  );
}

export default App;
