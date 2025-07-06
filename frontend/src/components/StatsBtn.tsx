import React, { useState } from 'react';
import './StatsBtn.css';

const StatsBtn: React.FC = () => {
  const [open, setOpen] = useState(false);

  return (
    <div className="stats-btn-wrapper">
      {/* Toggle button */}
      <button
        className="stats-btn-toggle"
        onClick={() => setOpen((o) => !o)}
        aria-controls="stats-panel"
      >
        {open ? 'Hide Stats ▲' : 'Show Stats ▼'}
      </button>

      {/* Collapsible panel */}
      <div
        id="stats-panel"
        className={`stats-panel ${open ? 'open' : 'closed'}`}
      >
        <h3>Statistics</h3>
        <div className="stats-items">
          {/* Example stat items */}
          <div className="stat-item">Mean Risk: 57%</div>
          <div className="stat-item">Median Risk: 61%</div>
          <div className="stat-item">Active Sensors: 184</div>
          {/* Replace above with real stats as needed */}
        </div>
        <div className="stats-footer">
          <p>
            These statistics represent current data for the selected region.
          </p>
        </div>
      </div>
    </div>
  );
};

export default StatsBtn;
