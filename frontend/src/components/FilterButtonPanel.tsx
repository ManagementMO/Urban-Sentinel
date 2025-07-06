import React, { useState } from 'react';
import './FilterButtonPanel.css';

type RiskLevel = 'all' | 'very-high' | 'high' | 'medium' | 'low';

interface Props {
  riskFilter: RiskLevel;
  setRiskFilter: (risk: RiskLevel) => void;
}

const FilterButtonPanel: React.FC<Props> = ({ riskFilter, setRiskFilter }) => {
  const [open, setOpen] = useState(false);

  return (
    <div className="filter-panel-wrapper">
      {/* Toggle */}
      <button
        className="filter-panel-toggle"
        onClick={() => setOpen((v) => !v)}
        aria-controls="filter-panel"
      >
        {open ? 'Hide Filters ▲' : 'Show Filters ▼'}
      </button>
      
      <div id="filter-panel" className={`filter-panel ${open ? 'open' : 'closed'}`}>
        <h3>Risk Levels</h3>
        <div className="filter-buttons">
          <button 
            className={riskFilter === 'all' ? 'active' : ''}
            onClick={() => setRiskFilter('all')}
          >
            All Levels
          </button>
          <button 
            className={`risk-btn very-high ${riskFilter === 'very-high' ? 'active' : ''}`}
            onClick={() => setRiskFilter('very-high')}
          >
            Very High
          </button>
          <button 
            className={`risk-btn high ${riskFilter === 'high' ? 'active' : ''}`}
            onClick={() => setRiskFilter('high')}
          >
            High
          </button>
          <button 
            className={`risk-btn medium ${riskFilter === 'medium' ? 'active' : ''}`}
            onClick={() => setRiskFilter('medium')}
          >
            Medium
          </button>
          <button 
            className={`risk-btn low ${riskFilter === 'low' ? 'active' : ''}`}
            onClick={() => setRiskFilter('low')}
          >
            Low
          </button>
        </div>
      </div>
    </div>
  );
};

export default FilterButtonPanel;
