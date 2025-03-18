import os
import json
import traceback
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from neo4j.time import DateTime as Neo4jDateTime
from kg import KnowledgeGraph

# Custom JSON encoder to handle Neo4j DateTime objects
class Neo4jDateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Neo4jDateTime):
            # Convert Neo4j DateTime to Python datetime and then to ISO format string
            return datetime(obj.year, obj.month, obj.day, 
                           obj.hour, obj.minute, obj.second, 
                           obj.nanosecond // 1000000).isoformat()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(Neo4jDateTimeEncoder, self).default(obj)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check for required environment variables
required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GROQ_API_KEY", "SERPER_API_KEY"]
for var in required_vars:
    value = os.getenv(var)
    if value:
        # Log that we found the variable (but mask most of the value for security)
        masked_value = value[:5] + "..." if len(value) > 5 else "***"
        logger.info(f"Found {var}: {masked_value}")
    else:
        logger.error(f"Missing required environment variable: {var}")

app = Flask(__name__)

# Configure CORS to allow requests from the frontend
CORS(app)

# Override Flask's default JSON encoder with our custom encoder
app.json_encoder = Neo4jDateTimeEncoder

# Default port
port = int(os.getenv("FLASK_PORT", 5001))

@app.route('/')
def index():
    """Root endpoint that redirects to the frontend"""
    return jsonify({
        "status": "ok",
        "message": "Graph-RAG API is running. Use /api/graphs to list available knowledge graphs and /api/query to query them."
    })

@app.route('/api/graphs', methods=['GET'])
def list_graphs():
    """List all available knowledge graphs"""
    try:
        graphs = KnowledgeGraph.list_existing_graphs()
        
        # Format the response
        formatted_graphs = []
        for topic, metadata in graphs:
            formatted_graphs.append({
                "topic": topic,
                "created_at": metadata["created_at"].isoformat(),
                "last_accessed": metadata["last_accessed"].isoformat()
            })
        
        logger.info(f"Retrieved {len(formatted_graphs)} knowledge graphs")
        return jsonify(formatted_graphs)
    except Exception as e:
        logger.error(f"Error listing knowledge graphs: {str(e)}")
        return jsonify({
            "error": str(e),
            "stacktrace": traceback.format_exc()
        }), 500

@app.route('/api/query', methods=['POST'])
def query_knowledge_graph():
    """Query a knowledge graph with a natural language question"""
    try:
        # Parse request data
        data = request.json
        if not data:
            logger.warning("No JSON data received in request")
            return jsonify({
                "error": "No JSON data received. Please provide 'query' and 'topic' fields."
            }), 400
        
        # Extract query and topic
        query = data.get('query')
        topic = data.get('topic')
        
        if not query:
            logger.warning("No query provided")
            return jsonify({
                "error": "No query provided. Please include a 'query' field."
            }), 400
            
        if not topic:
            logger.warning("No topic provided")
            return jsonify({
                "error": "No topic provided. Please include a 'topic' field."
            }), 400
        
        logger.info(f"Processing query for topic '{topic}': '{query}'")
        
        # Initialize the KnowledgeGraph class
        kg = KnowledgeGraph()
        
        # Check if the requested graph exists
        if not KnowledgeGraph.graph_exists(topic):
            logger.warning(f"Knowledge graph for '{topic}' not found")
            
            # Search for similar topics as suggestions
            with kg.driver.session() as session:
                # Try a fuzzy search to suggest possible topics
                fuzzy_query = "MATCH (t:Topic) WHERE toLower(t.name) CONTAINS toLower($topic_part) RETURN t.name as topic LIMIT 5"
                topic_parts = topic.split()
                
                if topic_parts:
                    first_word = topic_parts[0]
                    suggested_topics = session.run(fuzzy_query, topic_part=first_word).data()
                    
                    if suggested_topics:
                        suggestions = [t['topic'] for t in suggested_topics]
                        return jsonify({
                            'error': f"Knowledge graph for '{topic}' not found. Did you mean one of these topics: {', '.join(suggestions)}?",
                            'status': 'topic_suggestions',
                            'suggestions': suggestions
                        }), 404
            
            kg.close()
            return jsonify({
                'error': f"Knowledge graph for '{topic}' not found. Please build it first using Project.py.",
                'status': 'not_found'
            }), 404
                
        # Verify graph has data in Neo4j
        with kg.driver.session() as session:
            # Use a more flexible query that doesn't require exact topic name match
            check_query = "MATCH (t:Topic) WHERE toLower(t.name) CONTAINS toLower($topic) RETURN count(t) as count"
            result = session.run(check_query, topic=topic).single()
            
            if result and result["count"] == 0:
                logger.warning(f"Topic '{topic}' not found in Neo4j")
                kg.close()
                return jsonify({
                    'error': f"Knowledge graph for '{topic}' exists in metadata but has no data in the database. Please run 'python Project.py' to rebuild it.",
                    'status': 'empty_graph'
                }), 404
            
            # For Neo4j query preparation, also set a context variable
            kg.current_topic = topic
            logger.info(f"Set current_topic to '{topic}' for querying")
        
        # Use the proper Project.py method for querying the knowledge graph
        logger.info(f"Executing query using query_knowledge_graph method: '{query}'")
        
        # First, check that we have sample entities to verify graph content
        with kg.driver.session() as session:
            # Use a more flexible query that doesn't require exact topic name match
            entities_query = "MATCH (t:Topic) WHERE toLower(t.name) CONTAINS toLower($topic) "
            entities_query += "MATCH (e:Entity)-[:RELATED_TO]->(t) RETURN e.name AS name, e.type AS type LIMIT 10"
            entities = session.run(entities_query, topic=topic).data()
            logger.info(f"Sample entities for '{topic}': {entities}")
            
            if not entities:
                logger.warning(f"No entities found for topic '{topic}'")
                
                # Try a more lenient query as a fallback
                fallback_query = "MATCH (t:Topic) WHERE toLower(t.name) CONTAINS toLower($topic_part) "
                fallback_query += "RETURN t.name as topic LIMIT 5"
                
                # Try with first word of the topic
                topic_parts = topic.split()
                if topic_parts:
                    first_word = topic_parts[0]
                    suggested_topics = session.run(fallback_query, topic_part=first_word).data()
                    
                    if suggested_topics:
                        suggestions = [t['topic'] for t in suggested_topics]
                        return jsonify({
                            'error': f"Knowledge graph for '{topic}' not found. Did you mean one of these topics: {', '.join(suggestions)}?",
                            'status': 'topic_suggestions',
                            'suggestions': suggestions
                        }), 404
                
                return jsonify({
                    'error': f"Knowledge graph for '{topic}' appears to be empty. Please run 'python Project.py' to rebuild it.",
                    'status': 'empty_graph'
                }), 404
        
        # Load the graph first if needed
        if topic and hasattr(kg, 'load_graph'):
            logger.info(f"Loading graph for topic: '{topic}'")
            
            # Set current topic before querying
            kg.current_topic = topic
            kg.load_graph(topic)
        
        # Now use the Project.py query_knowledge_graph method which uses Groq for both query generation and response formatting
        try:
            result = kg.query_knowledge_graph(query)
            logger.info(f"Query completed with result keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
            
            # Log the generated cypher query
            logger.debug(f"Cypher query: {result.get('cypher_query', '') if isinstance(result, dict) else ''}")
            
            # Handle Neo4j DateTime objects in raw results
            if isinstance(result, dict) and 'raw_results' in result:
                # Convert raw_results with our custom JSON encoder
                try:
                    # First serialize to JSON string with custom encoder
                    serialized = json.dumps(result['raw_results'], cls=Neo4jDateTimeEncoder)
                    # Then deserialize back to Python objects
                    result['raw_results'] = json.loads(serialized)
                    logger.info("Successfully processed Neo4j DateTime objects in results")
                except Exception as e:
                    logger.error(f"Error processing raw results: {str(e)}")
                    # If serialization fails, provide empty raw results
                    result['raw_results'] = []
        except Exception as e:
            logger.error(f"Error in query_knowledge_graph: {str(e)}")
            result = {'error': str(e)}
        
        kg.close()
        
        # If no formatted response, provide a clear message
        if not isinstance(result, dict) or 'formatted_response' not in result or not result['formatted_response']:
            logger.warning(f"No formatted response returned for query: '{query}'")
            return jsonify({
                'error': 'No information found for this query. The knowledge graph may not contain relevant data.',
                'raw_results': result.get('raw_results', []) if isinstance(result, dict) else [],
                'cypher_query': result.get('cypher_query', '') if isinstance(result, dict) else ''
            })
            
        logger.info(f"Successfully retrieved response for query: '{query}'")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Failed to query knowledge graph: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'stacktrace': traceback.format_exc()
        }), 500

@app.route('/api/generate_visualization/<topic>', methods=['POST'])
def generate_visualization(topic):
    """Generate a visualization for a knowledge graph"""
    try:
        if not topic:
            return jsonify({
                "error": "No topic provided. Please include a topic in the URL."
            }), 400
            
        logger.info(f"Generating visualization for topic: {topic}")
        
        # Check if the requested graph exists using the KnowledgeGraph class
        if not KnowledgeGraph.graph_exists(topic):
            return jsonify({
                'error': f"Knowledge graph for '{topic}' not found. Please build it first.",
                'status': 'not_found'
            }), 404
        
        # Initialize the KnowledgeGraph and generate visualization directly
        try:
            # Initialize KnowledgeGraph
            kg = KnowledgeGraph()
            
            # Load the graph
            kg.load_graph(topic)
            
            # Create graphs directory if it doesn't exist
            graphs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
            if not os.path.exists(graphs_dir):
                os.makedirs(graphs_dir)
            
            # Generate the visualization
            viz_path = os.path.join(graphs_dir, f"{topic.replace(' ', '_')}_viz.png")
            
            # Call the visualize_graph method directly
            filename = kg.visualize_graph(topic)
            
            # If visualization was generated with a different name, rename it
            if os.path.exists(filename) and filename != viz_path:
                os.rename(filename, viz_path)
                logger.info(f"Renamed visualization from {filename} to {viz_path}")
            
            # Close the KnowledgeGraph connection
            kg.close()
            
            # Check if the file exists at the expected location
            if os.path.exists(viz_path):
                logger.info(f"Visualization generated successfully at: {viz_path}")
                return jsonify({
                    'success': True,
                    'message': f"Visualization generated for '{topic}'",
                    'path': f"graphs/{topic.replace(' ', '_')}_viz.png"
                })
            else:
                logger.warning(f"Visualization file not found at expected path: {viz_path}")
                return jsonify({
                    'error': "Visualization was generated but file not found at expected location.",
                    'status': 'missing_file'
                }), 500
            
        except Exception as viz_error:
            logger.error(f"Error generating visualization: {str(viz_error)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': f"Error generating visualization: {str(viz_error)}",
                'stacktrace': traceback.format_exc()
            }), 500
            
    except Exception as e:
        logger.error(f"Failed to generate visualization: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'stacktrace': traceback.format_exc()
        }), 500


@app.route('/api/visualize/<topic>', methods=['GET', 'HEAD'])
def visualize_graph(topic):
    """Serve a visualization of a knowledge graph"""
    try:
        if not topic:
            return jsonify({
                "error": "No topic provided. Please include a topic in the URL."
            }), 400
            
        logger.info(f"Serving visualization for topic: {topic}")
        
        # Check if visualization file exists
        viz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs", f"{topic.replace(' ', '_')}_viz.png")
        
        if not os.path.exists(viz_path):
            # If this is a HEAD request, just return 404
            if request.method == 'HEAD':
                return "", 404
                
            logger.warning(f"Visualization file not found for: {topic}")
            return jsonify({
                'error': f"Visualization for '{topic}' not found. Please generate it using the generate_visualization endpoint or script.",
                'status': 'not_found'
            }), 404
        
        # If this is a HEAD request, return 200 to indicate the file exists
        if request.method == 'HEAD':
            return "", 200
            
        return send_file(viz_path, mimetype='image/png')
    except Exception as e:
        logger.error(f"Failed to serve visualization: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'stacktrace': traceback.format_exc()
        }), 500

@app.route('/api/stats/<topic>', methods=['GET'])
def get_graph_stats(topic):
    """Get statistics about a knowledge graph"""
    try:
        if not topic:
            return jsonify({
                "error": "No topic provided. Please include a topic in the URL."
            }), 400
            
        logger.info(f"Getting statistics for topic: {topic}")
        
        # Initialize the KnowledgeGraph class
        kg = KnowledgeGraph()
        
        # Check if the requested graph exists
        if not KnowledgeGraph.graph_exists(topic):
            kg.close()
            return jsonify({
                'error': f"Knowledge graph for '{topic}' not found. Please build it first using Project.py.",
                'status': 'not_found'
            }), 404
        
        # Load the graph and get statistics
        kg.load_graph(topic)
        stats = kg.get_graph_statistics()
        
        kg.close()
        
        return jsonify({
            'topic': topic,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"Failed to get knowledge graph statistics: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'stacktrace': traceback.format_exc()
        }), 500

@app.route('/api/build', methods=['POST'])
def build_knowledge_graph():
    """Build a new knowledge graph for a topic"""
    try:
        # Parse request data
        data = request.json
        if not data:
            logger.warning("No JSON data received in request")
            return jsonify({
                "error": "No JSON data received. Please provide a 'topic' field."
            }), 400
        
        topic = data.get('topic')
        if not topic:
            logger.warning("No topic provided")
            return jsonify({
                "error": "No topic provided. Please include a 'topic' field."
            }), 400
            
        logger.info(f"Building knowledge graph for topic: '{topic}'")
        
        # Initialize the KnowledgeGraph class
        kg = KnowledgeGraph()
        
        # Add more detailed logging to debug the issue
        logger.info(f"KnowledgeGraph initialized, driver connected: {kg.driver is not None}")
        
        # Check if the graph already exists
        if KnowledgeGraph.graph_exists(topic):
            logger.info(f"Knowledge graph for '{topic}' already exists")
            
            # Return success with a message and the graph stats
            kg.load_graph(topic)
            stats = kg.get_graph_statistics()
            kg.close()
            
            return jsonify({
                'message': f"Knowledge graph for '{topic}' already exists",
                'status': 'exists',
                'topic': topic,
                'stats': stats
            })
        
        # Build the knowledge graph
        logger.info(f"Starting build process for '{topic}'")
        
        try:
            # Add timeout handling to prevent long-running requests
            stats = kg.build_knowledge_graph(topic)
            logger.info(f"Build completed with stats: {stats}")
        except Exception as build_error:
            logger.error(f"Error during build_knowledge_graph: {str(build_error)}")
            logger.error(traceback.format_exc())
            raise build_error
        
        # Don't generate visualization directly from API - it causes threading issues with Matplotlib
        # Instead provide a URL where the visualization could be generated separately
        visualization_url = f"/api/visualize/{topic}"
        visualization_available = False  # Set to False by default
        
        logger.info(f"Knowledge graph built successfully. Visualization needs to be generated separately.")
        
        kg.close()
        
        return jsonify({
            'message': f"Knowledge graph for '{topic}' built successfully",
            'status': 'success',
            'topic': topic,
            'stats': stats,
            'visualization_available': visualization_available,
            'visualization_url': visualization_url
        })
    except Exception as e:
        logger.error(f"Failed to build knowledge graph: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'stacktrace': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=port)
