import React from 'react';
import './LandingPage.css';

interface LandingPageProps {
  onTryItOut: () => void;
}

const LandingPage: React.FC<LandingPageProps> = ({ onTryItOut }) => {
  return (
    <div className="landing-page">
      <header className="landing-header">
        <h1 className="brand-title">URBAN SENTINEL</h1>
        <button className="try-it-out-btn" onClick={onTryItOut}>
          TRY IT OUT
        </button>
      </header>
      
      <main className="landing-main">
        <div className="content-section">
          {/* Subtitle overlays that will be behind the image */}
          <div className="background-text">
            <div className="subtitle-overlay subtitle-overlay-1">
              <span className="subtitle-text">Toronto City</span>
            </div>
            <div className="subtitle-overlay subtitle-overlay-2">
              <span className="subtitle-text1">Toronto City</span>
            </div>
            <div className="subtitle-overlay subtitle-overlay-3">
              <span className="subtitle-text2">Toronto City</span>
            </div>
          </div>
          
          <div className="image-section">
            <div className="toronto-image">
              <img 
                src="https://images.unsplash.com/photo-1517935706615-2717063c2225?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80" 
                alt="Toronto Skyline with CN Tower"
                className="skyline-img"
              />
            </div>
          </div>
          
          <div className="text-section">
            <h1 className="main-title">
              Toronto City
            </h1>
            
            <div className="description">
              <p>
                <strong>MAJORITY</strong> of cities are either to late to recover, and 
                with earlier notice, we could of tackled, <strong>so this is 
                where we come in to tackle this, and save cities before 
                to late</strong>
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default LandingPage; 