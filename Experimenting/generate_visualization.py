#!/usr/bin/env python
"""
Generate a visualization for a knowledge graph.
Run this from the command line with the topic as an argument.
e.g., python generate_visualization.py "Neural Networks"
"""

import sys
import os
import logging
from Project import KnowledgeGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_visualization(topic):
    """Generate visualization for a knowledge graph topic"""
    print(f"Generating visualization for topic: {topic}")
    
    # Initialize KnowledgeGraph
    kg = KnowledgeGraph()
    
    # Check if topic exists
    if not KnowledgeGraph.graph_exists(topic):
        print(f"Error: Knowledge graph for '{topic}' not found")
        return False
    
    # Load the graph
    kg.load_graph(topic)
    
    # Create graphs directory if it doesn't exist
    graphs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)
    
    # Generate visualization
    try:
        filename = kg.visualize_graph(topic)
        
        # Rename to match the expected format for the API
        viz_path = os.path.join(graphs_dir, f"{topic.replace(' ', '_')}_viz.png")
        
        # If the visualization was generated with a different name, rename it
        if os.path.exists(filename) and filename != viz_path:
            # The file might be in the current directory, not the graphs directory
            if not os.path.dirname(filename):
                source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
            else:
                source_path = filename
                
            print(f"Renaming visualization file from {source_path} to {viz_path}")
            if os.path.exists(source_path):
                # Copy the file to the new location
                import shutil
                shutil.copy2(source_path, viz_path)
                print(f"Copied visualization to {viz_path}")
        
        print(f"Visualization generated: {viz_path}")
        
        # Close the KnowledgeGraph connection
        kg.close()
        return True
    except Exception as e:
        print(f"Error generating visualization: {e}")
        kg.close()
        return False

if __name__ == "__main__":
    # Get topic from command line arguments or use default
    topic = sys.argv[1] if len(sys.argv) > 1 else "Diffusion Models"
    generate_visualization(topic)
