import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()


uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

print(f"Connecting to Neo4j database at: {uri}")
driver = GraphDatabase.driver(uri, auth=(username, password))

def check_connection():
    try:
        with driver.session() as session:
            result = session.run("RETURN 'Connected to Neo4j' AS message")
            message = result.single()["message"]
            print(message)
            return True
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        return False

def check_topic_exists(topic):
    try:
        with driver.session() as session:
            query = "MATCH (t:Topic {name: $topic}) RETURN t.name AS name"
            result = session.run(query, topic=topic)
            record = result.single()
            if record:
                print(f"Topic '{record['name']}' exists in the database")
                return True
            else:
                print(f"Topic '{topic}' not found in the database")
                return False
    except Exception as e:
        print(f"Error querying topic: {e}")
        return False

def check_topic_entities(topic):
    try:
        with driver.session() as session:
            query = """
            MATCH (t:Topic {name: $topic})
            OPTIONAL MATCH (e:Entity)-[:RELATED_TO]->(t)
            RETURN count(e) AS entity_count
            """
            result = session.run(query, topic=topic)
            record = result.single()
            if record:
                count = record["entity_count"]
                print(f"Found {count} entities related to topic '{topic}'")
                
                if count > 0:
                    # Get sample entities
                    entity_query = """
                    MATCH (e:Entity)-[:RELATED_TO]->(t:Topic {name: $topic})
                    RETURN e.name AS name, e.type AS type LIMIT 10
                    """
                    entities = session.run(entity_query, topic=topic).data()
                    print("Sample entities:")
                    for entity in entities:
                        print(f"  - {entity['name']} ({entity['type']})")
                
                return count
            else:
                print(f"No entity count data returned for topic '{topic}'")
                return 0
    except Exception as e:
        print(f"Error querying entities: {e}")
        return 0

def check_webpage_sources(topic):
    try:
        with driver.session() as session:
            query = """
            MATCH (t:Topic {name: $topic})
            OPTIONAL MATCH (t)-[:HAS_SOURCE]->(w:Webpage)
            RETURN count(w) AS webpage_count
            """
            result = session.run(query, topic=topic)
            record = result.single()
            if record:
                count = record["webpage_count"]
                print(f"Found {count} webpage sources for topic '{topic}'")
                
                if count > 0:
                    # Get sample webpages
                    webpage_query = """
                    MATCH (t:Topic {name: $topic})-[:HAS_SOURCE]->(w:Webpage)
                    RETURN w.title AS title, w.url AS url LIMIT 5
                    """
                    webpages = session.run(webpage_query, topic=topic).data()
                    print("Sample webpages:")
                    for webpage in webpages:
                        print(f"  - {webpage.get('title', 'No title')}: {webpage.get('url', 'No URL')}")
                
                return count
            else:
                print(f"No webpage count data returned for topic '{topic}'")
                return 0
    except Exception as e:
        print(f"Error querying webpages: {e}")
        return 0

def check_facts(topic):
    try:
        with driver.session() as session:
            query = """
            MATCH (t:Topic {name: $topic})
            OPTIONAL MATCH (e:Entity)-[:RELATED_TO]->(t)
            OPTIONAL MATCH (e)-[:HAS_FACT]->(f:Fact)
            RETURN count(f) AS fact_count
            """
            result = session.run(query, topic=topic)
            record = result.single()
            if record:
                count = record["fact_count"]
                print(f"Found {count} facts for topic '{topic}'")
                
                if count > 0:
                    # Get sample facts
                    fact_query = """
                    MATCH (t:Topic {name: $topic})
                    MATCH (e:Entity)-[:RELATED_TO]->(t)
                    MATCH (e)-[:HAS_FACT]->(f:Fact)
                    RETURN f.subject AS subject, f.predicate AS predicate, 
                           f.object AS object LIMIT 5
                    """
                    facts = session.run(fact_query, topic=topic).data()
                    print("Sample facts:")
                    for fact in facts:
                        print(f"  - {fact.get('subject', 'No subject')} {fact.get('predicate', 'No predicate')} {fact.get('object', 'No object')}")
                
                return count
            else:
                print(f"No fact count data returned for topic '{topic}'")
                return 0
    except Exception as e:
        print(f"Error querying facts: {e}")
        return 0

if __name__ == "__main__":
    topic = "Diffusion Models"
    
    print("\n=== Testing Neo4j Connection ===")
    if check_connection():
        print("\n=== Checking Topic Existence ===")
        if check_topic_exists(topic):
            print("\n=== Checking Topic Entities ===")
            entity_count = check_topic_entities(topic)
            
            print("\n=== Checking Webpage Sources ===")
            webpage_count = check_webpage_sources(topic)
            
            print("\n=== Checking Facts ===")
            fact_count = check_facts(topic)
            
            print("\n=== Summary ===")
            print(f"Topic: {topic}")
            print(f"Entity Count: {entity_count}")
            print(f"Webpage Source Count: {webpage_count}")
            print(f"Fact Count: {fact_count}")
            
            if entity_count == 0 and webpage_count == 0 and fact_count == 0:
                print("\nDiagnosis: The knowledge graph metadata exists but contains no data.")
                print("Recommendation: Rebuild the knowledge graph by running: python Project.py")
        
    # Close connection
    driver.close() 