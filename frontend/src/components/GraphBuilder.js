import React, { useState } from 'react';
import axios from 'axios';
import './GraphBuilder.css';

const GraphBuilder = ({ onGraphBuilt }) => {
  const [topic, setTopic] = useState('');
  const [depth, setDepth] = useState(1);
  const [building, setBuilding] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!topic.trim()) {
      setError('Please enter a topic');
      return;
    }
    
    setBuilding(true);
    setError(null);
    setProgress('Starting graph build process...');
    
    try {
      setProgress('Building knowledge graph. This may take a few minutes...');
      const response = await axios.post('/api/build', {
        topic,
        depth
      });
      
      setProgress('Knowledge graph built successfully!');
      setTimeout(() => {
        onGraphBuilt(topic);
      }, 1500);
    } catch (err) {
      console.error('Error building knowledge graph:', err);
      setError(err.response?.data?.error || 'An error occurred while building the knowledge graph');
    } finally {
      setBuilding(false);
    }
  };

  return (
    <div className="graph-builder">
      <h2>Build a New Knowledge Graph</h2>
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="topic">Topic</label>
          <input
            type="text"
            id="topic"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="Enter a topic to research (e.g., 'Quantum Computing', 'FPGA')"
            disabled={building}
            required
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="depth">Search Depth</label>
          <select
            id="depth"
            value={depth}
            onChange={(e) => setDepth(Number(e.target.value))}
            disabled={building}
          >
            <option value={1}>Shallow (faster)</option>
            <option value={2}>Deep (more comprehensive)</option>
          </select>
          <small>
            Higher depth will search more sources but takes longer to process
          </small>
        </div>
        
        {error && <div className="error-message">{error}</div>}
        
        {progress && (
          <div className="progress-message">
            <div className="spinner"></div>
            <p>{progress}</p>
          </div>
        )}
        
        <button type="submit" disabled={building || !topic.trim()}>
          {building ? 'Building...' : 'Build Knowledge Graph'}
        </button>
      </form>
      
      <div className="info-box">
        <h3>What happens when you build a graph?</h3>
        <p>
          The system will search for information on your topic, extract entities and facts, 
          and build a structured knowledge graph that you can query using natural language.
        </p>
        <p>
          Building a knowledge graph may take a few minutes depending on the topic 
          and search depth. Once complete, you can ask questions about the topic.
        </p>
      </div>
    </div>
  );
};

export default GraphBuilder;
