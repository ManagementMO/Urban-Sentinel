import React, { useState } from 'react';
import './StatsBtn.css';

interface RiskStats {
  total: number;
  mean: number;
  median: number;
  max: number;
  min: number;
  riskLevels: {
    'Very High': number;
    'High': number;
    'Medium': number;
    'Low': number;
    'Very Low': number;
  };
}

interface Props {
  riskStats: RiskStats | null;
  riskFilter: string;
}

const StatsBtn: React.FC<Props> = ({ riskStats, riskFilter }) => {
  const [open, setOpen] = useState(false);

  const getFilterName = () => {
    switch (riskFilter) {
      case 'very-high': return 'Very High Risk';
      case 'high': return 'High Risk';
      case 'medium': return 'Medium Risk';
      case 'low': return 'Low Risk';
      default: return 'All Risk Levels';
    }
  };

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
        <h3>Statistics - {getFilterName()}</h3>
        
        {riskStats ? (
          <div className="stats-items">
            <div className="stat-item">
              Total Cells: {riskStats.total.toLocaleString()}
            </div>
            <div className="stat-item">
              Mean Risk: {(riskStats.mean * 100).toFixed(1)}%
            </div>
            <div className="stat-item">
              Median Risk: {(riskStats.median * 100).toFixed(1)}%
            </div>
            <div className="stat-item">
              Max Risk: {(riskStats.max * 100).toFixed(1)}%
            </div>
            <div className="stat-item">
              Min Risk: {(riskStats.min * 100).toFixed(1)}%
            </div>
            
            <div className="stats-divider"></div>
            <h4>Risk Distribution:</h4>
            <div className="stat-item">
              Very High: {riskStats.riskLevels['Very High']} cells
            </div>
            <div className="stat-item">
              High: {riskStats.riskLevels['High']} cells
            </div>
            <div className="stat-item">
              Medium: {riskStats.riskLevels['Medium']} cells
            </div>
            <div className="stat-item">
              Low: {riskStats.riskLevels['Low']} cells
            </div>
            <div className="stat-item">
              Very Low: {riskStats.riskLevels['Very Low']} cells
            </div>
          </div>
        ) : (
          <div className="stats-items">
            <div className="stat-item">No data available</div>
          </div>
        )}
        
        <div className="stats-footer">
          <p>
            Statistics for {getFilterName().toLowerCase()} in the selected region.
          </p>
        </div>
      </div>
    </div>
  );
};

export default StatsBtn;
