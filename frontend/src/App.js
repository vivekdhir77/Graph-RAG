import React, { useState, useEffect } from 'react';
import './App.css';
import axios from 'axios';

function App() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [existingGraphs, setExistingGraphs] = useState([]);
  const [selectedGraph, setSelectedGraph] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [buildingGraph, setBuildingGraph] = useState(false);
  const [vizAvailable, setVizAvailable] = useState(false);
  const [showVisualization, setShowVisualization] = useState(false);
  const [visualizationUrl, setVisualizationUrl] = useState('');

  useEffect(() => {
    fetchExistingGraphs();
  }, []);

  // Check if visualization is available for the selected graph
  useEffect(() => {
    if (selectedGraph) {
      checkVisualizationAvailable(selectedGraph);
      setShowVisualization(false); // Hide visualization when changing topics
    }
  }, [selectedGraph]);

  const fetchExistingGraphs = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/graphs');
      setExistingGraphs(response.data);
      if (response.data.length > 0) {
        setSelectedGraph(response.data[0].topic);
      }
    } catch (err) {
      setError('Failed to fetch existing knowledge graphs');
      console.error(err);
    } finally {
      setLoading(false);
      setBuildingGraph(false);
    }
  };

  // Check if visualization is available for a topic
  const checkVisualizationAvailable = async (topic) => {
    try {
      const response = await axios.head(`/api/visualize/${encodeURIComponent(topic)}`);
      setVizAvailable(response.status === 200);
    } catch (error) {
      setVizAvailable(false);
    }
  };
  
  // Generate visualization and show it in the chat interface
  const requestVisualization = async () => {
    if (!selectedGraph) return;
    
    try {
      setLoading(true);
      setError(null);
      setResponse(`Generating visualization for "${selectedGraph}"...`);

      // Call the API to generate a new visualization
      try {
        // Make a POST request to generate the visualization
        const genResponse = await axios.post(`/api/generate_visualization/${encodeURIComponent(selectedGraph)}`);
        
        if (genResponse.status === 200) {
          console.log('Visualization generation successful:', genResponse.data);
          
          // Now fetch the newly generated visualization
          setVisualizationUrl(`/api/visualize/${encodeURIComponent(selectedGraph)}?v=${new Date().getTime()}`);
          setShowVisualization(true);
          setVizAvailable(true);
          setResponse(`**Knowledge graph visualization for "${selectedGraph}"**\n\nVisualization generated successfully! The image shows the relationships between different entities in your knowledge graph.`);
        } else {
          throw new Error('Failed to generate visualization');
        }
      } catch (error) {
        console.error('Error generating visualization:', error);
        
        // Try to load existing visualization as fallback
        try {
          const checkResponse = await axios.head(`/api/visualize/${encodeURIComponent(selectedGraph)}`);
          if (checkResponse.status === 200) {
            // Existing visualization found, show it
            setVisualizationUrl(`/api/visualize/${encodeURIComponent(selectedGraph)}`);
            setShowVisualization(true);
            setVizAvailable(true);
            setResponse(`**Knowledge graph for "${selectedGraph}"**\n\nCouldn't generate a new visualization, but found an existing one to display.`);
          } else {
            throw new Error('No visualization available');
          }
        } catch (fallbackError) {
          // No visualization available at all, show placeholder
          setResponse(`**Knowledge graph for "${selectedGraph}"**\n\nThis is a simple visualization of your knowledge graph. Since Matplotlib has issues in web servers, we're showing a placeholder visualization.`);
          
          // Generate a simple visualization URL using an online service
          const placeholderUrl = `https://via.placeholder.com/800x600.png?text=${encodeURIComponent(`Knowledge Graph: ${selectedGraph}`)}`;  
          setVisualizationUrl(placeholderUrl);
          setShowVisualization(true);
        }
      }
    } catch (error) {
      console.error('Error in visualization process:', error);
      setError('Failed to generate visualization.');
    } finally {
      setLoading(false);
    }
  };

  // Show visualization directly in the chat interface
  const viewVisualization = () => {
    if (selectedGraph) {
      setVisualizationUrl(`/api/visualize/${encodeURIComponent(selectedGraph)}`);
      setShowVisualization(true);
      setResponse(prev => {
        // If there's already a response, keep it
        if (prev && !prev.includes('Knowledge graph visualization:')) {
          return prev + '\n\n**Knowledge graph visualization:**';
        }
        return '**Knowledge graph visualization:**';
      });
    }
  };

  const buildNewGraph = async () => {
    const topic = prompt('Enter a topic to build a knowledge graph about:');
    if (!topic) return;
    
    try {
      setBuildingGraph(true);
      setLoading(true);
      setError(null);
      setResponse('Building knowledge graph for "' + topic + '". This may take a few minutes...');
      
      console.log('Making request to /api/build with topic:', topic);
      const result = await axios.post('/api/build', {
        topic: topic
      });
      
      console.log('Build result:', result.data);
      setResponse('Knowledge graph built successfully for "' + topic + '"! You can now ask questions about it.');
      setVizAvailable(result.data.visualization_available);
      fetchExistingGraphs();
      setSelectedGraph(topic);
    } catch (error) {
      console.error('Error building knowledge graph:', error);
      
      // More detailed error information
      const errorDetails = {
        message: error.message,
        status: error.response?.status,
        statusText: error.response?.statusText,
        data: error.response?.data,
        config: {
          url: error.config?.url,
          method: error.config?.method,
          headers: error.config?.headers,
          data: error.config?.data
        }
      };
      
      console.error('Detailed error:', JSON.stringify(errorDetails, null, 2));
      
      let errorMessage = 'Failed to build knowledge graph';
      if (error.response?.status === 403) {
        errorMessage += ': API key error or permission denied';
      } else if (error.response?.data?.error) {
        errorMessage += ': ' + error.response.data.error;
      } else if (error.message) {
        errorMessage += ': ' + error.message;
      }
      
      setError(errorMessage);
    } finally {
      setLoading(false);
      setBuildingGraph(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim() || !selectedGraph) return;

    setLoading(true);
    setResponse('');
    setError(null);

    try {
      console.log(`Querying topic "${selectedGraph}" with query: "${query}"`);
      
      const result = await axios.post('/api/query', {
        query: query,
        topic: selectedGraph
      });
      
      console.log('Query result:', result.data);
      
      if (result.data.formatted_response) {
        // Format the response nicely with markdown
        setResponse(result.data.formatted_response);
      } else if (result.data.error) {
        setError(result.data.error);
        if (result.data.status === 'empty_graph') {
          setResponse('This knowledge graph needs to be rebuilt. Please run the Project.py script.');
        } else {
          setResponse('No response from the knowledge graph.');
        }
        
        // Show cypher query and raw results if available
        if (result.data.cypher_query) {
          console.log('Cypher query used:', result.data.cypher_query);
        }
        if (result.data.raw_results) {
          console.log('Raw results:', result.data.raw_results);
        }
      } else {
        setResponse(result.data.response || 'No response from the knowledge graph.');
      }
    } catch (error) {
      console.error('Error querying knowledge graph:', error);
      let errorMessage = 'Error querying the knowledge graph';
      
      if (error.response?.data?.error) {
        errorMessage += ': ' + error.response.data.error;
      } else if (error.message) {
        errorMessage += ': ' + error.message;
      }
      
      setError(errorMessage);
      setResponse('No response from the knowledge graph. Try running Project.py to rebuild the graph.');
      
      // Show stack trace if available
      if (error.response?.data?.stacktrace) {
        console.error('Error stack trace:', error.response.data.stacktrace);
      }
    } finally {
      setLoading(false);
      setBuildingGraph(false);
    }

    // Clear the input after sending
    setQuery('');
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Knowledge Graph Chat</h1>
      </header>

      {loading && !response && <div className="loading">
        {buildingGraph ? 'Building knowledge graph... This may take several minutes...' : 'Loading...'}
      </div>}
      {error && <div className="error">{error}</div>}

      <main className="app-content">
        {existingGraphs.length === 0 ? (
          <div className="no-graphs-message">
            <p>No knowledge graphs available. Please create a new one:</p>
            <button onClick={buildNewGraph} className="build-button">Build New Graph</button>
          </div>
        ) : (
          <>
            <div className="graph-selector">
              <label>Selected Knowledge Graph: </label>
              <select 
                value={selectedGraph} 
                onChange={(e) => setSelectedGraph(e.target.value)}
              >
                {existingGraphs.map((graph) => (
                  <option key={graph.topic} value={graph.topic}>
                    {graph.topic}
                  </option>
                ))}
              </select>
              <div className="graph-actions">
                <button 
                  onClick={buildNewGraph} 
                  className="build-button" 
                  disabled={buildingGraph}
                >
                  {buildingGraph ? 'Building...' : 'Build New Graph'}
                </button>
                {vizAvailable ? (
                  <button 
                    onClick={viewVisualization} 
                    className="viz-button" 
                    disabled={!selectedGraph}
                  >
                    View Graph Visualization
                  </button>
                ) : (
                  <button 
                    onClick={requestVisualization} 
                    className="viz-button viz-button-request" 
                    disabled={!selectedGraph || loading}
                  >
                    Generate Visualization
                  </button>
                )}
              </div>
            </div>
            
            <div className="chat-container">
              {response && (
                <div className="response-area">
                  <div className="response">
                {response}
                {showVisualization && (
                  <div className="visualization-container">
                    <img 
                      src={visualizationUrl} 
                      alt="Knowledge Graph Visualization" 
                      className="visualization-image"
                      onError={(e) => {
                        console.error('Error loading image:', e);
                        setError('Failed to load visualization image.');
                      }}
                    />
                  </div>
                )}
              </div>
                </div>
              )}
              
              <form onSubmit={handleSubmit} className="query-form">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Ask about the knowledge graph..."
                  disabled={loading || !selectedGraph}
                  className="query-input"
                />
                <button 
                  type="submit" 
                  disabled={loading || !query.trim() || !selectedGraph}
                  className="query-button"
                >
                  Send
                </button>
              </form>
              {loading && response && <div className="loading">Thinking...</div>}
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
