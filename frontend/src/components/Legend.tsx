// Legend.tsx
import React, { useState } from 'react';
import './Legend.css';

const Legend: React.FC = () => {
  const [open, setOpen] = useState(true);          // ← track state

  return (
    <div className="legend-wrapper">
      {/* toggle button */}
      <button
        className="legend-toggle"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-controls="legend-panel"
      >
        {open ? 'Hide legend ▲' : 'Show legend ▼'}
      </button>

      {/* collapsible panel */}
      <div id="legend-panel" className={`legend ${open ? 'open' : 'closed'}`}>
        <h3>Blight Risk Level</h3>

        <div className="legend-items">
          {[
            { color: '#B71C1C', label: 'Very High (80–100%)' },
            { color: '#F44336', label: 'High (60–80%)' },
            { color: '#FF5722', label: 'Medium (40–60%)' },
            { color: '#FF9800', label: 'Low (20–40%)' },
            { color: '#4CAF50', label: 'Very Low (0–20%)' },
          ].map(({ color, label }) => (
            <div key={label} className="legend-item">
              <span
                className="legend-color"
                style={{ backgroundColor: color }}
              />
              <span>{label}</span>
            </div>
          ))}
        </div>

        <div className="legend-footer">
          <p>
            Risk scores indicate the probability of urban-blight development
            based on ML predictions.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Legend;
