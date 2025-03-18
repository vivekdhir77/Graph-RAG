import os
import json
import requests
import pickle
import datetime
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import networkx as nx
import matplotlib.pyplot as plt
from pydantic import BaseModel
from tqdm import tqdm

# Load environment variables
load_dotenv()

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        # Check if it's a Neo4j DateTime object
        if hasattr(obj, 'year') and hasattr(obj, 'month') and hasattr(obj, 'day') and hasattr(obj, 'hour') and hasattr(obj, 'minute') and hasattr(obj, 'second') and hasattr(obj, 'nanosecond'):
            # Convert Neo4j DateTime to ISO format
            try:
                return datetime(obj.year, obj.month, obj.day, 
                                obj.hour, obj.minute, obj.second, 
                                obj.nanosecond // 1000000).isoformat()
            except Exception:
                return str(obj)  # Fallback to string representation
        # Handle regular Python datetime
        if isinstance(obj, datetime):
            return obj.isoformat()  # Convert datetime to ISO 8601 string
        return super(DateTimeEncoder, self).default(obj)

# Define Pydantic models for graph-maker style extraction
class Document(BaseModel):
    text: str
    metadata: dict

class Ontology(BaseModel):
    labels: List[Union[str, Dict]]
    relationships: List[str]

class EntityNode(BaseModel):
    label: str
    name: str

class GraphRelation(BaseModel):
    node_1: EntityNode
    node_2: EntityNode
    relationship: str
    metadata: dict = {}

class KnowledgeGraph:
    def __init__(self):
        # Neo4j connection
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        
        # Groq API setup
        self.groq = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY", ""),
            model_name="llama3-70b-8192"
        )
        
        # Topic being researched
        self.current_topic = None
        
        # Initialize graph with constraints
        with self.driver.session() as session:
            # First, try to clean up any problematic nodes
            try:
                # Remove any Entity nodes with empty names
                session.run("MATCH (e:Entity) WHERE e.name = '' DETACH DELETE e")
                # Remove any Empty constraints first to avoid conflicts
                session.run("DROP CONSTRAINT entity_name IF EXISTS")
            except Exception as e:
                print(f"Warning: Error during cleanup: {e}")
            
            # Create constraints and indexes
            try:
                # Create constraints with better error handling
                session.run("CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE")
                session.run("CREATE CONSTRAINT webpage_url IF NOT EXISTS FOR (w:Webpage) REQUIRE w.url IS UNIQUE")
                session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE AND e.name <> ''")
                session.run("CREATE CONSTRAINT fact_id IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE")
            except Exception as e:
                print(f"Warning: Error creating constraints: {e}")
                print("Continuing without some constraints...")
    
    def search_topic(self, query):
        """Search for a topic using Serper API"""
        print(f"Searching for information about: {query}")
        conn = requests.post(
            "https://google.serper.dev/search",
            headers={
                'X-API-KEY': os.getenv("SERPER_API_KEY"),
                'Content-Type': 'application/json'
            },
            json={"q": query}
        )
        return conn.json()
    
    def extract_links(self, results, limit=3):
        """Extract top N links from Serper API results"""
        extracted_links = []
        
        # Extract organic search results
        if "organic" in results:
            for result in results["organic"][:limit]:
                extracted_links.append({
                    "title": result.get("title", ""),
                    "link": result.get("link", "")
                })
                
        print(f"Found {len(extracted_links)} relevant links")
        return extracted_links
    
    def scrape_webpage(self, url):
        """Scrape content from a webpage"""
        try:
            print(f"Scraping: {url}")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract text content
                elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5'])
                
                print(f"Found {len(elements)} text elements on page")
                text_parts = []
                for element in tqdm(elements, desc="Extracting text", unit="element", leave=False):
                    text_parts.append(element.get_text())
                
                text = ' '.join(text_parts)
                
                # Use text splitter from LangChain for better chunking
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )
                chunks = text_splitter.split_text(text)
                
                return {
                    "url": url,
                    "title": soup.title.string if soup.title else "",
                    "text": text,
                    "chunks": chunks
                }
            return None
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
    
    def summarize_text(self, text):
        """Generate a summary of the given text"""
        prompt = ChatPromptTemplate.from_template("""
        Summarize the following text concisely:
        
        {text}
        
        Provide only the summary, no introductory or concluding remarks.
        """)
        
        chain = prompt | self.groq | StrOutputParser()
        
        try:
            summary = chain.invoke({"text": text})
            return summary
        except Exception as e:
            print(f"Error summarizing text: {e}")
            return ""
    
    def extract_entities_and_facts_with_graph_maker(self, text, context):
        """Extract entities and facts using the graph-maker approach"""
        # Define the ontology for extraction
        ontology = {
            "labels": [
                {"Person": "A person, character, or individual."},
                {"Organization": "A company, institution, or group."},
                {"Concept": "An abstract idea or notion."},
                {"Event": "A significant occurrence or happening."},
                {"Place": "A location, area, or geographical entity."},
                {"Object": "A physical item or thing."},
                {"Document": "Any written, printed, or digital material."},
                {"Action": "An activity, process, or operation."},
                {"Miscellaneous": "Any important entity that doesn't fit other categories."}
            ],
            "relationships": [
                "Relation between any pair of entities"
            ]
        }
        
        prompt = ChatPromptTemplate.from_template("""
        You are an expert knowledge graph builder. Analyze the following text and extract entities and relationships.
        
        Context: {context}
        
        Text: {text}
        
        Extract entities based on these categories:
        {ontology}
        
        Format your response as a JSON array with objects having this structure:
        [
            {{
                "node_1": {{"label": "EntityType1", "name": "Entity1Name"}},
                "node_2": {{"label": "EntityType2", "name": "Entity2Name"}},
                "relationship": "Describe how Entity1 relates to Entity2",
                "metadata": {{"confidence": 0.9, "source": "The source of this information"}}
            }}
        ]
        
        Only return the JSON array, nothing else.
        """)
        
        chain = prompt | self.groq | StrOutputParser()
        
        try:
            result = chain.invoke({
                "text": text, 
                "context": context,
                "ontology": json.dumps(ontology, indent=2)
            })
            extracted_data = json.loads(result)
            
            # Convert to a format compatible with our existing code
            converted_data = {
                "entities": [],
                "facts": []
            }
            
            # Track unique entities to avoid duplicates
            entities_added = set()
            
            for item in extracted_data:
                node1 = item["node_1"]
                node2 = item["node_2"]
                
                # Add entities if not already added
                if node1["name"] not in entities_added:
                    converted_data["entities"].append({
                        "name": node1["name"],
                        "type": node1["label"],
                        "attributes": {}
                    })
                    entities_added.add(node1["name"])
                
                if node2["name"] not in entities_added:
                    converted_data["entities"].append({
                        "name": node2["name"],
                        "type": node2["label"],
                        "attributes": {}
                    })
                    entities_added.add(node2["name"])
                
                # Add the fact/relationship
                confidence = item.get("metadata", {}).get("confidence", 0.9)
                converted_data["facts"].append({
                    "subject": node1["name"],
                    "predicate": item["relationship"],
                    "object": node2["name"],
                    "confidence": confidence
                })
            
            return converted_data
        except Exception as e:
            print(f"Error extracting entities with graph-maker approach: {e}")
            return {"entities": [], "facts": []}
    
    def extract_entities_and_facts(self, text, context):
        """Use Groq to extract entities and facts from text"""
        # Use the graph-maker approach for better extraction
        return self.extract_entities_and_facts_with_graph_maker(text, context)

    def build_knowledge_graph(self, topic):
        """Build a knowledge graph for the given topic"""
        print(f"Building knowledge graph for '{topic}'...")
        self.current_topic = topic
        
        # Step 1: Search for the topic
        search_results = self.search_topic(topic)
        
        # Step 2: Extract top 3 links
        links = self.extract_links(search_results, limit=3)
        
        # Step 3: Create the main topic node
        with self.driver.session() as session:
            session.run(
                """
                MERGE (t:Topic {name: $name})
                ON CREATE SET t.created_at = datetime()
                """,
                name=topic
            )
        
        # Step 4: Scrape and process each link
        print("Scraping and processing webpages...")
        for link_info in tqdm(links, desc="Processing webpages", unit="page"):
            scraped_data = self.scrape_webpage(link_info["link"])
            if not scraped_data:
                continue
            
            # Add webpage node
            domain = urlparse(scraped_data["url"]).netloc
            with self.driver.session() as session:
                session.run(
                    """
                    MERGE (w:Webpage {url: $url})
                    ON CREATE SET w.title = $title, w.domain = $domain, w.created_at = datetime(), w.summary = $summary
                    WITH w
                    MATCH (t:Topic {name: $topic})
                    MERGE (t)-[:HAS_SOURCE]->(w)
                    """,
                    url=scraped_data["url"],
                    title=scraped_data["title"],
                    domain=domain,
                    topic=topic,
                    summary=self.summarize_text(scraped_data["text"][:5000])  # Summarize for better context
                )
            
            # Process each chunk of text
            print(f"Processing {len(scraped_data['chunks'])} text chunks from {domain}...")
            for chunk in tqdm(scraped_data["chunks"], desc=f"Analyzing {domain}", unit="chunk", leave=False):
                context = f"This text is from a webpage about {topic}. The page title is '{scraped_data['title']}'."
                
                # Extract entities and facts
                extracted_data = self.extract_entities_and_facts(chunk, context)
                
                # Add entities and facts to the graph
                self.add_to_neo4j(topic, scraped_data, extracted_data)
        
        # Save graph metadata for future loading
        self.save_graph_metadata(topic)
        
        print("Knowledge graph built successfully!")
        return self.get_graph_statistics()
    
    def add_to_neo4j(self, main_topic, webpage_info, extracted_data):
        """Add extracted information to Neo4j database"""
        with self.driver.session() as session:
            # Add entities
            if extracted_data.get("entities"):
                for entity in tqdm(extracted_data.get("entities", []), desc="Adding entities", unit="entity", leave=False):
                    # Skip entities with empty names
                    if not entity["name"] or entity["name"].strip() == "":
                        continue
                    
                    # Create entity node
                    session.run(
                        """
                        MERGE (e:Entity {name: $name})
                        ON CREATE SET e.type = $type, e.created_at = datetime()
                        """,
                        name=entity["name"],
                        type=entity["type"]
                    )
                    
                    # Connect entity to webpage and topic
                    session.run(
                        """
                        MATCH (e:Entity {name: $name})
                        MATCH (w:Webpage {url: $url})
                        MERGE (e)-[:MENTIONED_IN]->(w)
                        """,
                        name=entity["name"],
                        url=webpage_info["url"]
                    )
                    
                    session.run(
                        """
                        MATCH (e:Entity {name: $name})
                        MATCH (t:Topic {name: $topic})
                        MERGE (e)-[:RELATED_TO]->(t)
                        """,
                        name=entity["name"],
                        topic=main_topic
                    )
            
            # Add facts/relationships between entities
            if extracted_data.get("facts"):
                for fact in tqdm(extracted_data.get("facts", []), desc="Adding facts", unit="fact", leave=False):
                    # Skip facts with empty subject or object
                    if not fact["subject"] or not fact["object"] or fact["subject"].strip() == "" or fact["object"].strip() == "":
                        continue
                    
                    # Create unique ID for fact
                    fact_id = f"{fact['subject']}_{fact['predicate']}_{fact['object']}".replace(" ", "_")
                    
                    # Sanitize relationship type for Neo4j
                    relationship_type = self.sanitize_relationship_type(fact["predicate"])
                    
                    # Add fact with relationship to source
                    try:
                        # First ensure both entities exist (they might not if they were filtered)
                        session.run(
                            """
                            MERGE (subj:Entity {name: $subject})
                            ON CREATE SET subj.type = 'Unknown', subj.created_at = datetime()
                            
                            MERGE (obj:Entity {name: $object})
                            ON CREATE SET obj.type = 'Unknown', obj.created_at = datetime()
                            """,
                            subject=fact["subject"],
                            object=fact["object"]
                        )
                        
                        # Now create the relationship and fact
                        session.run(
                            """
                            MATCH (subj:Entity {name: $subject})
                            MATCH (obj:Entity {name: $object})
                            MERGE (subj)-[r:`{relationship}`]->(obj)
                            ON CREATE SET r.confidence = $confidence, r.created_at = datetime()
                            ON MATCH SET r.confidence = CASE WHEN r.confidence < $confidence THEN $confidence ELSE r.confidence END
                            WITH subj, obj
                            MATCH (w:Webpage {url: $url})
                            MERGE (f:Fact {id: $id})
                            ON CREATE SET f.subject = $subject, f.predicate = $predicate, f.object = $object, 
                                          f.confidence = $confidence, f.created_at = datetime()
                            MERGE (f)-[:FROM_SOURCE]->(w)
                            MERGE (subj)-[:HAS_FACT]->(f)
                            MERGE (obj)-[:IN_FACT]->(f)
                            """.replace("{relationship}", relationship_type),
                            subject=fact["subject"],
                            predicate=fact["predicate"],
                            object=fact["object"],
                            confidence=fact["confidence"],
                            url=webpage_info["url"],
                            id=fact_id
                        )
                    except Exception as e:
                        print(f"Error adding fact: {e}")
    
    def sanitize_relationship_type(self, relationship):
        """Sanitize relationship type for Neo4j compatibility"""
        # Replace any character that isn't alphanumeric or underscore with underscore
        sanitized = re.sub(r'[^\w]', '_', relationship)
        # If it starts with a number, prefix with 'rel_'
        if sanitized and sanitized[0].isdigit():
            sanitized = 'rel_' + sanitized
        # Ensure it's not empty
        if not sanitized:
            return 'has_relation'
        return sanitized[:50]  # Limit length
    
    def get_graph_statistics(self):
        """Get statistics about the knowledge graph"""
        with self.driver.session() as session:
            stats = {}
            
            # Count entities
            result = session.run("MATCH (e:Entity) RETURN count(e) as count")
            stats["entity_count"] = result.single()["count"]
            
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            stats["relationship_count"] = result.single()["count"]
            
            # Count webpages
            result = session.run("MATCH (w:Webpage) RETURN count(w) as count")
            stats["webpage_count"] = result.single()["count"]
            
            # Count facts
            result = session.run("MATCH (f:Fact) RETURN count(f) as count")
            stats["fact_count"] = result.single()["count"]
            
            return stats
    
    def query_knowledge_graph(self, query):
        """Query the knowledge graph for information (RAG implementation)"""
        # First use Groq to parse the natural language query into a Cypher query
        query_prompt = ChatPromptTemplate.from_template("""
        Convert the following natural language query into a Cypher query for Neo4j.
        
        The graph has the following schema:
        - (Topic) node represents the main topic
        - (Entity) nodes represent entities with properties: name, type
        - (Webpage) nodes represent web pages with properties: url, title, domain, summary
        - (Fact) nodes represent facts with properties: subject, predicate, object, confidence
        - Relationships: 
          * (Topic)-[:HAS_SOURCE]->(Webpage)
          * (Entity)-[:MENTIONED_IN]->(Webpage) 
          * (Entity)-[:RELATED_TO]->(Topic)
          * (Entity)-[varies]->(Entity) (dynamic relationships based on facts)
          * (Entity)-[:HAS_FACT]->(Fact)
          * (Fact)-[:FROM_SOURCE]->(Webpage)
        
        Query: {query}
        
        Current topic being researched: {topic}
        
        Important Neo4j Cypher syntax rules:
        1. Define patterns in the MATCH clause, not in the RETURN clause
        2. For returning complex paths, use named paths in the MATCH clause like:
           MATCH path = (node1)-[rel]->(node2) RETURN path
        3. Use WITH clause for intermediate results when needed
        4. Use quotes consistently for string values
        5. When matching topic names, use case-insensitive matching with toLower() function
        6. For topic name matching, use the CONTAINS operator for partial matches
        
        Examples of valid Cypher queries:
        - MATCH (t:Topic) WHERE toLower(t.name) CONTAINS toLower("machine learning") MATCH (t)-[:HAS_SOURCE]->(w:Webpage) RETURN t.name, w.title
        - MATCH (e:Entity)-[r]->(e2:Entity) WHERE toLower(e.name) CONTAINS toLower("ai") RETURN e.name, type(r), e2.name
        - MATCH p = (t:Topic)-[:HAS_SOURCE]->(w:Webpage) WHERE toLower(t.name) CONTAINS toLower("quantum") RETURN p
        - MATCH (e:Entity)-[:RELATED_TO]->(t:Topic) WHERE toLower(t.name) CONTAINS toLower("diffusion") RETURN e.name, e.type
        
        Return only the Cypher query without any explanation, markdown formatting, or backticks. Do not use triple backticks (```) around the query.
        """)
        
        cypher_chain = query_prompt | self.groq | StrOutputParser()
        
        try:
            # Generate the Cypher query
            raw_cypher_query = cypher_chain.invoke({
                "query": query, 
                "topic": self.current_topic or ""
            }).strip()
            
            # Clean up the query - remove any markdown formatting or backticks
            cypher_query = re.sub(r'^```[\w]*\n', '', raw_cypher_query)  # Remove opening ```cypher
            cypher_query = re.sub(r'\n```$', '', cypher_query)           # Remove closing ```
            cypher_query = cypher_query.replace('`', '')                 # Remove any remaining backticks
            cypher_query = cypher_query.strip()                          # Strip extra whitespace
            
            # Make topic name matching more flexible by replacing direct equals with CONTAINS and toLower()
            if self.current_topic:
                # Make topic name matching case-insensitive and fuzzy
                cypher_query = re.sub(
                    r'(t|topic):Topic\s*{{\s*name\s*:\s*[\'"]([^\'"]+)[\'"]\s*}}', 
                    r'\1:Topic WHERE toLower(\1.name) CONTAINS toLower("\2")', 
                    cypher_query
                )
            
            print(f"Generated Cypher query: {cypher_query}")
            
            # Execute the query
            with self.driver.session() as session:
                try:
                    result = session.run(cypher_query)
                    records = [record.data() for record in result]
                    
                    # If no results, try an alternative query that's more flexible with topic matching
                    if not records and self.current_topic:
                        # Try a more forgiving query with fuzzy matching
                        print("No results found, trying alternative query...")
                        alt_query = self._generate_alternative_query(query)
                        print(f"Alternative query: {alt_query}")
                        result = session.run(alt_query)
                        records = [record.data() for record in result]
                    
                    # If still no results, fall back to the simplest query
                    if not records:
                        print("No results from alternative query, trying fallback query...")
                        fallback_query = self._generate_fallback_query(query)
                        print(f"Fallback query: {fallback_query}")
                        result = session.run(fallback_query)
                        records = [record.data() for record in result]
                    
                except Exception as e:
                    print(f"Error executing Cypher query: {e}")
                    # Fall back to a simpler query if the generated one fails
                    fallback_query = self._generate_fallback_query(query)
                    print(f"Error in query execution. Trying fallback query: {fallback_query}")
                    result = session.run(fallback_query)
                    records = [record.data() for record in result]
                
                # Format the results to be JSON-serializable
                try:
                    # First convert to a JSON string using our custom encoder
                    json_string = json.dumps(records, indent=2, cls=DateTimeEncoder)
                    # Then parse back to Python objects (this ensures all objects are serializable)
                    safe_records = json.loads(json_string)
                except Exception as e:
                    print(f"Error serializing records: {e}")
                    # Fallback to a simpler representation
                    safe_records = []
                    for record in records:
                        safe_record = {}
                        for key, value in record.items():
                            try:
                                # Try to serialize each value individually
                                json.dumps({key: value}, cls=DateTimeEncoder)
                                safe_record[key] = value
                            except (TypeError, json.JSONDecodeError):
                                # If serialization fails, convert to string
                                safe_record[key] = str(value)
                        safe_records.append(safe_record)
                
                # Generate a response using RAG
                format_prompt = ChatPromptTemplate.from_template("""
                The following is the result of a knowledge graph query in JSON format:
                
                {results}
                
                Based on these results, provide a well-formatted answer to the original query: "{query}"
                
                Format your response with the following structure:
                1. Start with "**Question:** {query}"
                2. Then include "**Answer:**" followed by your comprehensive response
                3. Organize the information logically with headings and bullet points where appropriate
                4. Include the source of information where available
                
                If the results are empty or insufficient, kindly mention that the knowledge graph doesn't contain enough information to answer the query.
                """)
                
                format_chain = format_prompt | self.groq | StrOutputParser()
                formatted_response = format_chain.invoke({
                    "results": json.dumps(safe_records, indent=2),
                    "query": query
                })
                
                return {
                    "cypher_query": cypher_query,
                    "raw_results": safe_records,
                    "formatted_response": formatted_response
                }
        except Exception as e:
            print(f"Error querying knowledge graph: {e}")
            return {"error": str(e)}
    
    def _generate_alternative_query(self, query):
        """Generate a more flexible alternative query for better matches"""
        if not self.current_topic:
            return self._generate_fallback_query(query)
        
        # Extract key terms from the query for better matching
        query_terms = [word.lower() for word in re.findall(r'\w+', query) if len(word) > 3]
        
        # Create a query that checks entities and facts related to the topic
        # and also looks for entities that match the query terms
        return f"""
        MATCH (t:Topic)
        WHERE toLower(t.name) CONTAINS toLower("{self.current_topic}")
        WITH t
        MATCH (e:Entity)-[:RELATED_TO]->(t)
        OPTIONAL MATCH (e)-[r]->(e2:Entity)
        OPTIONAL MATCH (e)-[:HAS_FACT]->(f:Fact)
        OPTIONAL MATCH (e)-[:MENTIONED_IN]->(w:Webpage)
        RETURN e.name as entity, e.type as type, 
               type(r) as relationship, e2.name as related_entity,
               f.predicate as fact_predicate, f.object as fact_object,
               w.title as source_title, w.url as source_url
        LIMIT 50
        """
    
    def _generate_fallback_query(self, query):
        """Generate a simple fallback query if the main query fails"""
        # If we have a current topic, search for entities related to that topic
        if self.current_topic:
            # Create a more informative query that returns entity relationships and facts
            return f"""
            MATCH (t:Topic)
            WHERE toLower(t.name) CONTAINS toLower("{self.current_topic}")
            WITH t
            MATCH (e:Entity)-[:RELATED_TO]->(t)
            OPTIONAL MATCH (e)-[:HAS_FACT]->(f:Fact)
            RETURN e.name as entity, e.type as type, 
                   f.predicate as predicate, f.object as object
            LIMIT 50
            """
        else:
            # If no current topic, look for relevant entities based on query words
            query_words = [word.lower() for word in re.findall(r'\w+', query) if len(word) > 3]
            if query_words:
                # Build a CASE expression for scoring relevance
                case_expr = " + ".join([f"CASE WHEN toLower(e.name) CONTAINS '{word}' THEN 1 ELSE 0 END" for word in query_words])
                return f"""
                MATCH (e:Entity)
                WITH e, {case_expr} as score
                WHERE score > 0
                OPTIONAL MATCH (e)-[:HAS_FACT]->(f:Fact)
                OPTIONAL MATCH (e)-[:MENTIONED_IN]->(w:Webpage)
                RETURN e.name as entity, e.type as type, 
                       f.predicate as predicate, f.object as object,
                       w.title as source, score as relevance
                ORDER BY score DESC
                LIMIT 50
                """
            else:
                # Last resort - just return some informative entities
                return """
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[:HAS_FACT]->(f:Fact)
                RETURN e.name as entity, e.type as type,
                       f.predicate as predicate, f.object as object
                LIMIT 30
                """
    
    def save_graph_metadata(self, topic):
        """Save metadata about a built knowledge graph"""
        metadata_file = "knowledge_graphs.pkl"
        
        # Create or load existing metadata
        if os.path.exists(metadata_file):
            with open(metadata_file, "rb") as f:
                graphs = pickle.load(f)
        else:
            graphs = {}
        
        # Add or update this graph's metadata
        graphs[topic] = {
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "topic": topic
        }
        
        # Save metadata back to file
        with open(metadata_file, "wb") as f:
            pickle.dump(graphs, f)
    
    def load_graph(self, topic):
        """Load an existing graph for the given topic"""
        self.current_topic = topic
        
        # Update the access time
        self.update_access_time(topic)
        
        print(f"Loaded knowledge graph for topic: {topic}")
        return self.get_graph_statistics()
    
    def update_access_time(self, topic):
        """Update the last accessed time for a topic"""
        metadata_file = "knowledge_graphs.pkl"
        if os.path.exists(metadata_file):
            with open(metadata_file, "rb") as f:
                graphs = pickle.load(f)
            
            if topic in graphs:
                graphs[topic]["last_accessed"] = datetime.now()
                
                with open(metadata_file, "wb") as f:
                    pickle.dump(graphs, f)
    
    def visualize_graph(self, topic=None, limit=50):
        """Visualize the knowledge graph using NetworkX and Matplotlib"""
        topic = topic or self.current_topic
        if not topic:
            print("No topic specified for visualization")
            return
        
        print(f"Generating visualization for topic: {topic}")
        
        # Query Neo4j to get nodes and relationships
        with self.driver.session() as session:
            # Get a limited number of nodes and relationships connected to the topic
            query = """
            MATCH (t:Topic {name: $topic})-[:HAS_SOURCE]->(w:Webpage)<-[:MENTIONED_IN]-(e:Entity)
            WITH e, t LIMIT $limit
            MATCH (e)-[r]->(e2:Entity)
            RETURN e.name as source, type(r) as relationship, e2.name as target, e.type as source_type, e2.type as target_type
            """
            
            result = session.run(query, topic=topic, limit=limit)
            
            # Create a directed graph
            G = nx.DiGraph()
            
            print("Building graph visualization structure...")
            # Add edges with relationship types as labels
            for record in tqdm(result, desc="Processing relationships", unit="relationship"):
                source = record["source"]
                target = record["target"]
                rel = record["relationship"]
                source_type = record["source_type"]
                target_type = record["target_type"]
                
                # Add nodes with types
                if not G.has_node(source):
                    G.add_node(source, node_type=source_type)
                if not G.has_node(target):
                    G.add_node(target, node_type=target_type)
                
                # Add the edge with the relationship as an attribute
                G.add_edge(source, target, relationship=rel)
            
            if not G.nodes():
                print("No relationships found to visualize. The graph may be empty or disconnected.")
                return
            
            # Create a figure
            plt.figure(figsize=(15, 10))
            
            # Define node colors based on types
            node_colors = []
            node_types = {node: data.get('node_type', 'Unknown') for node, data in G.nodes(data=True)}
            unique_types = set(node_types.values())
            color_map = plt.cm.get_cmap('tab20', len(unique_types))
            type_to_color = {t: color_map(i) for i, t in enumerate(unique_types)}
            
            for node in G.nodes():
                node_type = node_types[node]
                node_colors.append(type_to_color[node_type])
            
            # Draw the graph
            print("Generating layout and rendering visualization...")
            pos = nx.spring_layout(G, seed=42)  # Position nodes using force-directed layout
            nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, alpha=0.8)
            nx.draw_networkx_labels(G, pos, font_size=10)
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrows=True)
            
            # Add a legend for node types
            legend_patches = []
            for node_type, color in type_to_color.items():
                legend_patches.append(plt.Line2D([0], [0], marker='o', color='w', 
                                              markerfacecolor=color, markersize=10, label=node_type))
            plt.legend(handles=legend_patches, loc='upper right')
            
            # Save the visualization
            filename = f"{topic.replace(' ', '_')}_graph.png"
            plt.title(f"Knowledge Graph for {topic}")
            plt.axis('off')  # Turn off axis
            plt.tight_layout()
            plt.savefig(filename, format="PNG", dpi=300)
            plt.close()
            
            print(f"Visualization saved to {filename}")
            return filename
    
    def close(self):
        """Close the Neo4j driver connection"""
        self.driver.close()

    @staticmethod
    def list_existing_graphs():
        """List all previously built knowledge graphs"""
        metadata_file = "knowledge_graphs.pkl"
        if os.path.exists(metadata_file):
            with open(metadata_file, "rb") as f:
                graphs = pickle.load(f)
            
            # Sort by last accessed time (most recent first)
            sorted_graphs = sorted(
                graphs.items(), 
                key=lambda x: x[1]["last_accessed"], 
                reverse=True
            )
            
            return sorted_graphs
        else:
            return []
    
    @staticmethod
    def graph_exists(topic):
        """Check if a knowledge graph for the given topic exists"""
        metadata_file = "knowledge_graphs.pkl"
        if os.path.exists(metadata_file):
            with open(metadata_file, "rb") as f:
                graphs = pickle.load(f)
            return topic in graphs
        return False


def main():
    # Check for environment variables
    required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GROQ_API_KEY", "SERPER_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing environment variables: {', '.join(missing_vars)}")
        print("Please add them to your .env file")
        return
    
    print("Knowledge Graph Builder")
    print("======================")
    
    # Initialize the knowledge graph
    try:
        kg = KnowledgeGraph()
    except Exception as e:
        print(f"Error initializing KnowledgeGraph: {e}")
        print("Trying with a clean database...")
        # Try with a clean approach if first initialization fails
        try:
            with GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))) as driver:
                with driver.session() as session:
                    # Complete cleanup
                    print("Cleaning database...")
                    session.run("MATCH (n) DETACH DELETE n")
            # Now try again with the clean database
            kg = KnowledgeGraph()
        except Exception as e2:
            print(f"Fatal error: Could not initialize knowledge graph: {e2}")
            return
    
    try:
        # Check for existing knowledge graphs
        existing_graphs = KnowledgeGraph.list_existing_graphs()
        
        if existing_graphs:
            print("\nExisting knowledge graphs:")
            for i, (topic, metadata) in enumerate(existing_graphs, 1):
                created = metadata["created_at"].strftime("%Y-%m-%d %H:%M")
                last_accessed = metadata["last_accessed"].strftime("%Y-%m-%d %H:%M")
                print(f"{i}. {topic} (created: {created}, last accessed: {last_accessed})")
            
            choice = input("\nDo you want to (q)uery an existing graph, (b)uild a new one, or (v)isualize an existing graph? (q/b/v): ").lower()
            
            if choice == 'q':
                # Query existing graph
                graph_idx = int(input("Enter the number of the graph to query: ")) - 1
                if 0 <= graph_idx < len(existing_graphs):
                    topic = existing_graphs[graph_idx][0]
                    print(f"\nLoading existing knowledge graph for '{topic}'")
                    
                    stats = kg.load_graph(topic)
                    
                    # Display statistics
                    print(f"\nKnowledge graph stats:")
                    print(f"- {stats['entity_count']} entities")
                    print(f"- {stats['relationship_count']} relationships")
                    print(f"- {stats['webpage_count']} webpages")
                    print(f"- {stats['fact_count']} facts")
                    
                    # Query the knowledge graph
                    while True:
                        query = input("\nEnter a question (or 'exit' to quit): ")
                        if query.lower() == 'exit':
                            break
                            
                        print("Querying knowledge graph...")
                        results = kg.query_knowledge_graph(query)
                        
                        if "error" in results:
                            print(f"Error: {results['error']}")
                        else:
                            print("\nAnswer:")
                            print(results["formatted_response"])
                else:
                    print("Invalid selection. Building a new graph instead.")
                    choice = 'b'
            
            elif choice == 'v':
                # Visualize existing graph
                graph_idx = int(input("Enter the number of the graph to visualize: ")) - 1
                if 0 <= graph_idx < len(existing_graphs):
                    topic = existing_graphs[graph_idx][0]
                    print(f"\nVisualizing knowledge graph for '{topic}'")
                    
                    # Load the graph first
                    kg.load_graph(topic)
                    
                    # Visualize the graph
                    kg.visualize_graph(topic)
                    
                    # Ask if they want to query the graph after visualization
                    if input("\nDo you want to query this graph? (y/n): ").lower() == 'y':
                        # Query the knowledge graph
                        while True:
                            query = input("\nEnter a question (or 'exit' to quit): ")
                            if query.lower() == 'exit':
                                break
                                
                            print("Querying knowledge graph...")
                            results = kg.query_knowledge_graph(query)
                            
                            if "error" in results:
                                print(f"Error: {results['error']}")
                            else:
                                print("\nAnswer:")
                                print(results["formatted_response"])
                else:
                    print("Invalid selection. Building a new graph instead.")
                    choice = 'b'
        else:
            print("No existing knowledge graphs found. You'll need to build a new one.")
            choice = 'b'
        
        # Build a new knowledge graph if needed
        if choice == 'b':
            # Step 1: Ask user for a topic
            topic = input("Enter a topic to research: ")
            
            # Step 2-4: Search, scrape, and build the knowledge graph
            print(f"Building knowledge graph for '{topic}'... This may take a while.")
            stats = kg.build_knowledge_graph(topic)
            
            # Display statistics
            print(f"\nKnowledge graph built with:")
            print(f"- {stats['entity_count']} entities")
            print(f"- {stats['relationship_count']} relationships")
            print(f"- {stats['webpage_count']} webpages")
            print(f"- {stats['fact_count']} facts")
            
            # Ask if user wants to visualize the graph
            if input("\nDo you want to visualize this graph? (y/n): ").lower() == 'y':
                kg.visualize_graph(topic)
            
            # Step 5: Query the knowledge graph
            print("\nYou can now query the knowledge graph.")
            while True:
                query = input("\nEnter a question (or 'exit' to quit): ")
                if query.lower() == 'exit':
                    break
                    
                print("Querying knowledge graph...")
                results = kg.query_knowledge_graph(query)
                
                if "error" in results:
                    print(f"Error: {results['error']}")
                else:
                    print("\nAnswer:")
                    print(results["formatted_response"])
    
    finally:
        # Close the connection
        if 'kg' in locals():
            kg.close()


if __name__ == "__main__":
    main()
