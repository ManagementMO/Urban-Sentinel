import React from 'react';
import './Legend.css';

const Legend: React.FC = () => {
  return (
    <div className="legend">
      <h3>Decay Levels</h3>
      <div className="legend-items">
        <div className="legend-item">
          <div className="legend-color" style={{ background: 'rgba(0, 87, 192, 0.6)' }}></div>
          <span>Minimal (0.0 - 0.2)</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ background: 'rgba(255, 235, 59, 0.7)' }}></div>
          <span>Mild (0.2 - 0.5)</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ background: 'rgba(255, 152, 0, 0.8)' }}></div>
          <span>Noticeable (0.5 - 0.8)</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ background: 'rgba(255, 56, 56, 0.9)' }}></div>
          <span>Severe (0.8 - 1.0)</span>
        </div>
      </div>
    </div>
  );
};

export default Legend; 