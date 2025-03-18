import React from 'react';
import './GraphSelector.css';

const GraphSelector = ({ graphs, onSelect, selectedGraph }) => {
  if (!graphs || graphs.length === 0) {
    return (
      <div className="graph-selector empty">
        <p>No knowledge graphs available. Please build a new one.</p>
      </div>
    );
  }

  return (
    <div className="graph-selector">
      <h2>Select a Knowledge Graph</h2>
      <div className="graphs-list">
        {graphs.map((graph, index) => (
          <div 
            key={index} 
            className={`graph-item ${graph.topic === selectedGraph ? 'selected' : ''}`}
            onClick={() => onSelect(graph.topic)}
          >
            <h3>{graph.topic}</h3>
            <div className="graph-metadata">
              <span>Created: {new Date(graph.created_at).toLocaleString()}</span>
              <span>Last accessed: {new Date(graph.last_accessed).toLocaleString()}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default GraphSelector;
