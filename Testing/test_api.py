import os
import requests
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Get API keys
serper_api_key = os.getenv("SERPER_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

print(f"SERPER_API_KEY: {'✓ Found' if serper_api_key else '✗ Missing'}")
print(f"GROQ_API_KEY: {'✓ Found' if groq_api_key else '✗ Missing'}")
print(f"NEO4J_URI: {'✓ Found' if neo4j_uri else '✗ Missing'}")
print(f"NEO4J_USERNAME: {'✓ Found' if neo4j_username else '✗ Missing'}")
print(f"NEO4J_PASSWORD: {'✓ Found' if neo4j_password else '✗ Missing'}")

# Test Serper API
print("\nTesting Serper API...")
try:
    response = requests.post(
        "https://google.serper.dev/search",
        headers={
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        },
        json={"q": "test query"}
    )
    
    if response.status_code == 200:
        print(f"✅ Serper API test succeeded with status code {response.status_code}")
    else:
        print(f"❌ Serper API test failed with status code {response.status_code}")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"❌ Serper API test error: {str(e)}")

# Test Neo4j connection
print("\nTesting Neo4j connection...")
try:
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) as count")
        count = result.single()["count"]
        print(f"✅ Neo4j connection test succeeded. Database has {count} nodes.")
    driver.close()
except Exception as e:
    print(f"❌ Neo4j connection test error: {str(e)}")

# Test Groq API
print("\nTesting Groq API...")
try:
    groq = ChatGroq(
        api_key=groq_api_key,
        model_name="llama3-70b-8192"
    )
    response = groq.invoke("Hello, are you working?")
    print(f"✅ Groq API test succeeded with response: {response.content[:50]}...")
except Exception as e:
    print(f"❌ Groq API test error: {str(e)}")

# API endpoint - Updated port from 5000 to 5001
api_url = "http://localhost:5001/api/query"

# Test query
test_data = {
    "topic": "Diffusion Models",
    "query": "What are diffusion models?"
}

print(f"Sending query to API: {test_data}")

try:
    # Make POST request to the API
    response = requests.post(api_url, json=test_data)
    
    # Print status code
    print(f"Status code: {response.status_code}")
    
    # Print response
    if response.status_code == 200:
        result = response.json()
        print("\nAPI Response:")
        print(json.dumps(result, indent=2))
        
        # Check if there's a formatted response
        if "formatted_response" in result:
            print("\nFormatted Response:")
            print(result["formatted_response"])
        else:
            print("\nNo formatted response in the result")
            
        # Check if there's a cypher query generated
        if "cypher_query" in result:
            print("\nCypher Query:")
            print(result["cypher_query"])
        
        # Check if there are raw results
        if "raw_results" in result:
            print(f"\nRaw Results Count: {len(result['raw_results'])}")
            if len(result['raw_results']) > 0:
                print("First few raw results:")
                for i, res in enumerate(result['raw_results'][:3]):
                    print(f"Result {i+1}:")
                    print(json.dumps(res, indent=2))
    else:
        print(f"Error: {response.text}")
        
except Exception as e:
    print(f"Error making API request: {e}")
    print("Make sure the API server is running (run ./run_backend.sh)")

print("\nDone testing APIs")
