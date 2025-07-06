import React from 'react';
import './Legend.css';

const Legend: React.FC = () => {
  return (
    <div className="legend">
      <h3>Blight Risk Level</h3>
      <div className="legend-items">
        <div className="legend-item">
          <div className="legend-color" style={{ background: '#8b0000' }}></div>
          <span>Very High (80-100%)</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ background: '#dc143c' }}></div>
          <span>High (60-80%)</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ background: '#ff4500' }}></div>
          <span>Medium (40-60%)</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ background: '#ffa500' }}></div>
          <span>Low (20-40%)</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ background: '#2e8b57' }}></div>
          <span>Very Low (0-20%)</span>
        </div>
      </div>
      <div className="legend-footer">
        <p>Risk scores indicate the probability of urban blight development based on ML predictions.</p>
      </div>
    </div>
  );
};

export default Legend; 