import React from 'react';
import { RiskGridCell } from '../services/api';
import './TopBlightMenu.css';

interface TopBlightMenuProps {
  riskData: RiskGridCell[] | null;
}

interface BlightTypeCount {
  type: string;
  count: number;
  percentage: number;
}

const TopBlightMenu: React.FC<TopBlightMenuProps> = ({ riskData }) => {
  // Calculate top blight types from the data
  const getTopBlightTypes = (): BlightTypeCount[] => {
    if (!riskData || riskData.length === 0) return [];

    const blightCounts: { [key: string]: number } = {};
    let totalBlight = 0;

    // Count occurrences of each blight type
    riskData.forEach(cell => {
      // Use overall_most_common_blight for analysis
      const blightType = cell.overall_most_common_blight;
      
      if (blightType && blightType !== 'None' && blightType !== 'null' && blightType.trim() !== '') {
        blightCounts[blightType] = (blightCounts[blightType] || 0) + 1;
        totalBlight++;
      }
    });

    // Convert to array and sort by count
    const sortedBlights = Object.entries(blightCounts)
      .map(([type, count]) => ({
        type: formatBlightType(type),
        count,
        percentage: Math.round((count / totalBlight) * 100)
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5); // Top 5

    return sortedBlights;
  };

  // Format blight type names for display
  const formatBlightType = (type: string): string => {
    if (!type) return 'Unknown';
    
    // Clean and format the blight type name
    return type
      .replace(/_/g, ' ')
      .toLowerCase()
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Get color for each blight type
  const getBlightColor = (index: number): string => {
    const colors = [
      '#FF6B6B', // Red
      '#4ECDC4', // Teal
      '#45B7D1', // Blue
      '#96CEB4', // Green
      '#FECA57'  // Yellow
    ];
    return colors[index] || '#95A5A6';
  };

  const topBlights = getTopBlightTypes();

  if (!riskData || topBlights.length === 0) {
    return null;
  }

  return (
    <div className="top-blight-menu">
      <div className="blight-header">
        <h3>🏚️ Top Blight Types</h3>
        <span className="blight-subtitle">Most Prevalent in Toronto</span>
      </div>
      
      <div className="blight-list">
        {topBlights.map((blight, index) => (
          <div key={blight.type} className="blight-item">
            <div className="blight-indicator">
              <div 
                className="blight-dot"
                style={{ backgroundColor: getBlightColor(index) }}
              ></div>
              <span className="blight-rank">#{index + 1}</span>
            </div>
            
            <div className="blight-info">
              <div className="blight-name">{blight.type}</div>
              <div className="blight-stats">
                <span className="blight-count">{blight.count.toLocaleString()} areas</span>
                <span className="blight-percentage">{blight.percentage}%</span>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      <div className="blight-footer">
        <small>Based on historical 311 complaint data</small>
      </div>
    </div>
  );
};

export default TopBlightMenu; 