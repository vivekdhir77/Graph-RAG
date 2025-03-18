import os
import json
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from typing import Dict, List, TypedDict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pickle
import os.path
from datetime import datetime


# Load environment variables
load_dotenv()

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()  # Convert datetime to ISO 8601 string
        return super(DateTimeEncoder, self).default(obj)

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
        
        # Initialize graph with constraints
        with self.driver.session() as session:
            # Create constraints and indexes
            session.run("CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE")
            session.run("CREATE CONSTRAINT webpage_url IF NOT EXISTS FOR (w:Webpage) REQUIRE w.url IS UNIQUE")
            session.run("CREATE CONSTRAINT fact_id IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE")
            session.run("CREATE INDEX webpage_domain IF NOT EXISTS FOR (w:Webpage) ON (w.domain)")
    
    def clear_graph(self):
        """Clear all nodes and relationships from the graph"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def search_topic(self, query):
        """Search for a topic using Serper API"""
        conn = requests.post(
            "https://google.serper.dev/search",
            headers={
                'X-API-KEY': os.getenv("SERPER_API_KEY"),
                'Content-Type': 'application/json'
            },
            json={"q": query}
        )
        return conn.json()
    
    def extract_info_from_serper(self, results):
        """Extract relevant information from Serper API results"""
        extracted_info = {
            "main_topic": results.get("searchParameters", {}).get("q", ""),
            "knowledge_panel": results.get("knowledgeGraph", {}),
            "organic_results": []
        }
        
        # Extract organic search results
        if "organic" in results:
            for result in results["organic"]:
                extracted_info["organic_results"].append({
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                })
                
        return extracted_info
    
    def scrape_webpage(self, url):
        """Scrape content from a webpage"""
        try:
            headers = {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                        }
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract text content
                text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5'])])
                
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
    
    def extract_entities_and_facts(self, text, context):
        """Use Groq to extract entities and facts from text"""
        prompt = ChatPromptTemplate.from_template("""
        You are an expert knowledge graph builder. Analyze the following text and extract:
        1. Key entities (people, organizations, concepts, etc.)
        2. Important facts about these entities
        3. Relationships between entities

        For each fact, assign a confidence score from 0-1.
        
        Context: {context}
        
        Text: {text}
        
        Format your response as a JSON object with the following structure:
        {{
            "entities": [
                {{ "name": "Entity Name", "type": "person|organization|concept|etc", "attributes": {{"key": "value"}} }}
            ],
            "facts": [
                {{ "subject": "Entity1", "predicate": "relationship", "object": "Entity2", "confidence": 0.9 }}
            ]
        }}
        Only return the JSON object, nothing else.
        """)
        
        chain = prompt | self.groq | StrOutputParser()
        
        try:
            result = chain.invoke({"text": text, "context": context})
            return json.loads(result)
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return {"entities": [], "facts": []}

    def add_to_neo4j(self, main_topic, webpage_info, extracted_data):
        """Add extracted information to Neo4j database"""
        with self.driver.session() as session:
            # Add main topic node if it doesn't exist
            session.run(
                """
                MERGE (t:Topic {name: $name})
                ON CREATE SET t.created_at = datetime()
                """,
                name=main_topic
            )
            
            # Add webpage node
            domain = urlparse(webpage_info["url"]).netloc
            session.run(
                """
                MERGE (w:Webpage {url: $url})
                ON CREATE SET w.title = $title, w.domain = $domain, w.created_at = datetime()
                WITH w
                MATCH (t:Topic {name: $topic})
                MERGE (t)-[:HAS_SOURCE]->(w)
                """,
                url=webpage_info["url"],
                title=webpage_info["title"],
                domain=domain,
                topic=main_topic
            )
            
            # Add entities
            for entity in extracted_data.get("entities", []):
                entity_props = entity.get("attributes", {})
                entity_props["name"] = entity["name"]
                entity_props["type"] = entity["type"]
                
                # Create entity node
                entity_result = session.run(
                    """
                    MERGE (e:Entity {name: $name, type: $type})
                    ON CREATE SET e += $props, e.created_at = datetime()
                    RETURN e
                    """,
                    name=entity["name"],
                    type=entity["type"],
                    props=entity_props
                ).single()
                
                # Connect entity to webpage
                session.run(
                    """
                    MATCH (e:Entity {name: $name, type: $type})
                    MATCH (w:Webpage {url: $url})
                    MERGE (e)-[:MENTIONED_IN]->(w)
                    """,
                    name=entity["name"],
                    type=entity["type"],
                    url=webpage_info["url"]
                )
                
                # Connect entity to topic
                session.run(
                    """
                    MATCH (e:Entity {name: $name, type: $type})
                    MATCH (t:Topic {name: $topic})
                    MERGE (e)-[:RELATED_TO]->(t)
                    """,
                    name=entity["name"],
                    type=entity["type"],
                    topic=main_topic
                )
            
            # Add facts/relationships between entities
            for fact in extracted_data.get("facts", []):
                # Create unique ID for fact
                fact_id = f"{fact['subject']}_{fact['predicate']}_{fact['object']}".replace(" ", "_")
                
                # Add fact with relationship to source
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
                    """.replace("{relationship}", fact["predicate"]),
                    subject=fact["subject"],
                    predicate=fact["predicate"],
                    object=fact["object"],
                    confidence=fact["confidence"],
                    url=webpage_info["url"],
                    id=fact_id
                )

    def build_knowledge_graph(self, query, depth=1):
        """Build a knowledge graph based on the search query with optimized performance"""
        print(f"Starting knowledge graph construction for '{query}'...")
        
        # Step 1: Search for the topic
        print("Searching for information...")
        search_results = self.search_topic(query)
        
        # Step 2: Extract info from search results
        extracted_info = self.extract_info_from_serper(search_results)
        
        # Add main topic node first (outside the loop)
        with self.driver.session() as session:
            session.run(
                """
                MERGE (t:Topic {name: $name})
                ON CREATE SET t.created_at = datetime()
                """,
                name=query
            )
        
        # Collect URLs to scrape with basic info
        pages_to_scrape = []
        for i, result in enumerate(extracted_info["organic_results"]):
            if i >= depth * 3:  # Limit number of pages to scrape
                break
            pages_to_scrape.append({
                "title": result.get("title", ""),
                "link": result.get("link", "")
            })
        
        print(f"Found {len(pages_to_scrape)} pages to analyze")
        
        # Parallel web scraping
        def scrape_single_page(page_info):
            return {
                "page_info": page_info,
                "scraped_data": self.scrape_webpage(page_info["link"])
            }
        
        # Use ThreadPoolExecutor for parallel scraping with tqdm
        print("Scraping web pages in parallel...")
        scraped_pages = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(scrape_single_page, page): page for page in pages_to_scrape}
            
            # Add tqdm progress bar for web scraping
            for future in tqdm(as_completed(futures), total=len(futures), desc="Web scraping", unit="page"):
                try:
                    result = future.result()
                    original_page = futures[future]  # Get the original page info
                    if result["scraped_data"]:
                        print(f"✓ Scraped: {original_page['title']}")
                        scraped_pages.append(result)
                    else:
                        print(f"✗ Failed to scrape: {original_page['title']}")
                except Exception as e:
                    original_page = futures[future]  # Get the original page info
                    print(f"✗ Error scraping {original_page['title']}: {str(e)}")
        
        # Process content into chunks
        all_chunks = []
        for page in scraped_pages:
            if not page["scraped_data"]:
                continue
            
            page_title = page["page_info"]["title"]
            
            for chunk in page["scraped_data"]["chunks"]:
                all_chunks.append({
                    "chunk": chunk,
                    "context": f"This text is from a webpage about {query}. The page title is '{page_title}'.",
                    "page_data": page["scraped_data"]
                })
        
        print(f"Extracted {len(all_chunks)} text chunks for analysis")
        
        # Process chunks in batches to reduce API calls
        batch_size = 5  # Adjust based on token limits
        batches = [all_chunks[i:i + batch_size] for i in range(0, len(all_chunks), batch_size)]
        
        # Batch entity extraction using a more efficient prompt
        def process_batch(batch):
            batch_prompt = ChatPromptTemplate.from_template("""
            You are an expert knowledge graph builder. Analyze each of the following text chunks and extract entities and relationships.
            For each chunk, provide the extracted data in JSON format.
            
            {batch_data}
            
            Format your response as a JSON array with one object per chunk:
            [
                {{
                    "chunk_index": 0,
                    "entities": [
                        {{ "name": "Entity Name", "type": "person|organization|concept|etc", "attributes": {{"key": "value"}} }}
                    ],
                    "facts": [
                        {{ "subject": "Entity1", "predicate": "relationship", "object": "Entity2", "confidence": 0.9 }}
                    ]
                }},
                // more chunk results...
            ]
            Only return the JSON array, nothing else.
            """)
            
            # Format the batch data for the prompt
            batch_text = ""
            for i, item in enumerate(batch):
                batch_text += f"CHUNK {i}:\nContext: {item['context']}\nText: {item['chunk']}\n\n"
            
            try:
                chain = batch_prompt | self.groq | StrOutputParser()
                result = chain.invoke({"batch_data": batch_text})
                return json.loads(result)
            except Exception as e:
                print(f"Error batch processing: {e}")
                return []
        
        # Process all batches with tqdm
        print("Analyzing text chunks with Groq AI...")
        all_extracted_data = []
        for batch in tqdm(batches, desc="LLM analysis", unit="batch"):
            batch_results = process_batch(batch)
            for i, result in enumerate(batch_results):
                if i < len(batch):  # Safety check
                    all_extracted_data.append({
                        "extracted_data": result,
                        "page_data": batch[i]["page_data"]
                    })
        
        # Count entities and facts for reporting
        entity_count = 0
        fact_count = 0
        for item in all_extracted_data:
            entity_count += len(item["extracted_data"].get("entities", []))
            fact_count += len(item["extracted_data"].get("facts", []))
        
        print(f"Extracted {entity_count} entities and {fact_count} facts")
        print("Building Neo4j knowledge graph...")
        
        # Optimize Neo4j writes with transactions and batching
        with self.driver.session() as session:
            # Create a single transaction for all writes
            tx = session.begin_transaction()
            try:
                # Add all webpage nodes in one go
                print("Adding webpage nodes...")
                for page in tqdm(scraped_pages, desc="Adding webpages", unit="page"):
                    if not page["scraped_data"]:
                        continue
                    
                    scraped_data = page["scraped_data"]
                    domain = urlparse(scraped_data["url"]).netloc
                    
                    # Add webpage and connect to topic
                    tx.run(
                        """
                        MERGE (w:Webpage {url: $url})
                        ON CREATE SET w.title = $title, w.domain = $domain, w.created_at = datetime()
                        WITH w
                        MATCH (t:Topic {name: $topic})
                        MERGE (t)-[:HAS_SOURCE]->(w)
                        """,
                        url=scraped_data["url"],
                        title=scraped_data["title"],
                        domain=domain,
                        topic=query
                    )
                
                # Process all entities and facts
                print("Collecting entities and facts...")
                all_entities = set()
                entity_webpage_pairs = []
                all_facts = []
                
                for item in all_extracted_data:
                    extracted_data = item["extracted_data"]
                    page_data = item["page_data"]
                    
                    # Collect entities
                    for entity in extracted_data.get("entities", []):
                        all_entities.add((entity["name"], entity["type"]))
                        entity_webpage_pairs.append((entity["name"], entity["type"], page_data["url"]))
                    
                    # Collect facts
                    for fact in extracted_data.get("facts", []):
                        all_facts.append({
                            "subject": fact["subject"],
                            "predicate": fact["predicate"],
                            "object": fact["object"],
                            "confidence": fact["confidence"],
                            "url": page_data["url"]
                        })
                
                # Batch create all entities
                print(f"Adding {len(all_entities)} entities to database...")
                for entity_name, entity_type in tqdm(all_entities, desc="Creating entities", unit="entity"):
                    tx.run(
                        """
                        MERGE (e:Entity {name: $name, type: $type})
                        ON CREATE SET e.created_at = datetime()
                        """,
                        name=entity_name,
                        type=entity_type
                    )
                
                # Batch connect entities to webpages and topics
                print(f"Creating {len(entity_webpage_pairs)} entity-webpage connections...")
                for entity_name, entity_type, url in tqdm(entity_webpage_pairs, desc="Connecting entities", unit="conn"):
                    tx.run(
                        """
                        MATCH (e:Entity {name: $name, type: $type})
                        MATCH (w:Webpage {url: $url})
                        MERGE (e)-[:MENTIONED_IN]->(w)
                        """,
                        name=entity_name,
                        type=entity_type,
                        url=url
                    )
                    
                    tx.run(
                        """
                        MATCH (e:Entity {name: $name, type: $type})
                        MATCH (t:Topic {name: $topic})
                        MERGE (e)-[:RELATED_TO]->(t)
                        """,
                        name=entity_name,
                        type=entity_type,
                        topic=query
                    )
                
                # Batch create all facts and relationships
                print(f"Adding {len(all_facts)} facts to knowledge graph...")
                for fact in tqdm(all_facts, desc="Creating facts", unit="fact"):
                    # Create unique ID for fact
                    fact_id = f"{fact['subject']}_{fact['predicate']}_{fact['object']}".replace(" ", "_")
                    
                    tx.run(
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
                        """.replace("{relationship}", fact["predicate"]),
                        subject=fact["subject"],
                        predicate=fact["predicate"],
                        object=fact["object"],
                        confidence=fact["confidence"],
                        url=fact["url"],
                        id=fact_id
                    )
                
                # Commit the transaction
                print("Committing all changes to database...")
                tx.commit()
                print("Transaction completed successfully!")
            except Exception as e:
                tx.rollback()
                print(f"Error in transaction: {e}")
                print("Changes rolled back - no data was saved.")
        
        print(f"Knowledge graph built for topic: {query}")
        return self.get_graph_statistics()
    
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
        """Query the knowledge graph for information"""
        # Use Groq to parse the natural language query
        query_prompt = ChatPromptTemplate.from_template("""
        Convert the following natural language query into a Cypher query for Neo4j.
        
        The graph has the following schema:
        - (Topic) node represents the main topic
        - (Entity) nodes represent entities with properties: name, type
        - (Webpage) nodes represent web pages with properties: url, title, domain
        - (Fact) nodes represent facts with properties: subject, predicate, object, confidence
        - Relationships: 
          * (Topic)-[:HAS_SOURCE]->(Webpage)
          * (Entity)-[:MENTIONED_IN]->(Webpage) 
          * (Entity)-[:RELATED_TO]->(Topic)
          * (Entity)-[varies]->(Entity) (dynamic relationships based on facts)
          * (Entity)-[:HAS_FACT]->(Fact)
          * (Fact)-[:FROM_SOURCE]->(Webpage)
        
        Query: {query}
        
        Return only the Cypher query without any explanation or additional text. Do not include markdown formatting or backticks.
        """)
        
        cypher_chain = query_prompt | self.groq | StrOutputParser()
        
        try:
            cypher_query = cypher_chain.invoke({"query": query})
            cypher_query = cypher_query.strip()
            
            # Open a session
            with self.driver.session() as session:
                # Run the query
                result = session.run(cypher_query)
                records = [record.data() for record in result]
                
                # Format results with Groq
                format_prompt = ChatPromptTemplate.from_template("""
                The following is the result of a knowledge graph query in JSON format:
                
                {results}
                
                Based on these results, provide a concise, well-formatted answer to the original query: "{query}"
                
                Organize the information logically and include the source of information where available.
                """)
                
                format_chain = format_prompt | self.groq | StrOutputParser()
                formatted_response = format_chain.invoke({
                    "results": json.dumps(records, indent=2, cls=DateTimeEncoder),
                    "query": query
                })
                
                # Add exploration query for debugging
                explore_query = """
                MATCH (e:Entity)-[:RELATED_TO]->(t:Topic {name: $topic})
                RETURN e.name, e.type LIMIT 20
                """
                exploration = session.run(explore_query, topic=query).data()
                print(f"Sample entities: {exploration}")
                
                return {
                    "cypher_query": cypher_query,
                    "raw_results": records,
                    "formatted_response": formatted_response
                }
        except Exception as e:
            print(f"Error querying knowledge graph: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Close the Neo4j driver connection"""
        self.driver.close()

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


# LangGraph State Management
class GraphState(TypedDict):
    topic: str
    depth: int
    current_step: str
    results: Dict[str, Any]
    error: str


# Define LangGraph agents
def search_agent(state: GraphState) -> GraphState:
    """Agent that performs the search"""
    topic = state["topic"]
    try:
        kg = KnowledgeGraph()
        search_results = kg.search_topic(topic)
        kg.close()
        state["results"]["search"] = search_results
        state["current_step"] = "extraction"
    except Exception as e:
        state["error"] = f"Search error: {str(e)}"
        state["current_step"] = "error_handler"
    return state

def extraction_agent(state: GraphState) -> GraphState:
    """Agent that extracts information from search results"""
    try:
        kg = KnowledgeGraph()
        extracted_info = kg.extract_info_from_serper(state["results"]["search"])
        kg.close()
        state["results"]["extraction"] = extracted_info
        state["current_step"] = "graph_building"
    except Exception as e:
        state["error"] = f"Extraction error: {str(e)}"
        state["current_step"] = "error_handler"
    return state

def graph_building_agent(state: GraphState) -> GraphState:
    """Agent that builds the knowledge graph"""
    try:
        kg = KnowledgeGraph()
        stats = kg.build_knowledge_graph(state["topic"], depth=state["depth"])
        kg.close()
        state["results"]["graph_stats"] = stats
        state["current_step"] = "complete"
    except Exception as e:
        state["error"] = f"Graph building error: {str(e)}"
        state["current_step"] = "error_handler"
    return state

# Build the LangGraph workflow
def create_workflow():
    """Create a LangGraph workflow for building knowledge graphs"""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("search", search_agent)
    workflow.add_node("extraction", extraction_agent)
    workflow.add_node("graph_building", graph_building_agent)
    workflow.add_node("complete", lambda x: x)
    workflow.add_node("error_handler", lambda x: x)
    
    # Add edges
    workflow.add_edge("search", "extraction")
    workflow.add_edge("extraction", "graph_building")
    workflow.add_edge("graph_building", "complete")
    
    # Add conditional edges for error handling
    workflow.add_conditional_edges(
        "search",
        lambda x: "error_handler" if x["error"] else "extraction"
    )
    workflow.add_conditional_edges(
        "extraction",
        lambda x: "error_handler" if x["error"] else "graph_building"
    )
    workflow.add_conditional_edges(
        "graph_building",
        lambda x: "error_handler" if x["error"] else "complete"
    )
    
    # Set entry point
    workflow.set_entry_point("search")
    
    return workflow.compile()

# Example usage
if __name__ == "__main__":
    # Ensure the .env file exists with Neo4j and Groq credentials
    required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GROQ_API_KEY", "SERPER_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing environment variables: {', '.join(missing_vars)}")
        print("Please add them to your .env file")
        exit(1)
    
    print("Knowledge Graph Builder with Neo4j, Groq, LangChain, and LangGraph")
    print("=================================================================")
    
    # Check for existing knowledge graphs
    existing_graphs = KnowledgeGraph.list_existing_graphs()
    if existing_graphs:
        print("\nExisting knowledge graphs:")
        for i, (topic, metadata) in enumerate(existing_graphs, 1):
            created = metadata["created_at"].strftime("%Y-%m-%d %H:%M")
            print(f"{i}. {topic} (created: {created})")
        
        choice = input("\nDo you want to (q)uery an existing graph or (b)uild a new one? (q/b): ").lower()
        
        if choice == 'q':
            # Query existing graph
            graph_idx = int(input("Enter the number of the graph to query: ")) - 1
            if 0 <= graph_idx < len(existing_graphs):
                topic = existing_graphs[graph_idx][0]
                print(f"\nQuerying existing knowledge graph for '{topic}'")
                
                kg = KnowledgeGraph()
                kg.update_access_time(topic)
                
                try:
                    # Query the knowledge graph
                    print("\nYou can now query the knowledge graph.")
                    while True:
                        query = input("\nEnter a question (or 'exit' to quit): ")
                        if query.lower() == 'exit':
                            break
                            
                        results = kg.query_knowledge_graph(query)
                        
                        if "error" in results:
                            print(f"Error: {results['error']}")
                        else:
                            print("\nAnswer:")
                            print(results["formatted_response"])
                finally:
                    kg.close()
                    
                # Exit after querying
                exit(0)
            else:
                print("Invalid selection. Building a new graph instead.")
    else:
        print("No existing knowledge graphs found. You'll need to build a new one.")
    
    # Option to use LangGraph workflow or direct KnowledgeGraph class
    use_langgraph = input("Use LangGraph workflow? (y/n): ").lower() == 'y'
    
    # Get topic regardless of approach
    topic = input("Enter a topic to research: ")
    # Set depth to 1 by default without asking
    depth = 1
    kg = None
    try:
        if use_langgraph:
            # Using LangGraph workflow
            workflow = create_workflow()
            result = workflow.invoke({
                "topic": topic,
                "depth": depth,
                "current_step": "search",
                "results": {},
                "error": ""
            })
            
            if result["error"]:
                print(f"Error: {result['error']}")
                exit(1)
            
            print(f"\nKnowledge graph built successfully!")
            print(f"Statistics: {json.dumps(result['results']['graph_stats'], indent=2)}")
            
            # Save graph metadata
            temp_kg = KnowledgeGraph()
            temp_kg.save_graph_metadata(topic)
            temp_kg.close()
            
            # Now allow querying the knowledge graph
            kg = KnowledgeGraph()
            
        else:
            # Using KnowledgeGraph class directly
            kg = KnowledgeGraph()
            
            # Build knowledge graph
            print(f"\nBuilding knowledge graph for: {topic}")
            stats = kg.build_knowledge_graph(topic, depth=depth)
            
            print(f"\nKnowledge graph built with:")
            print(f"- {stats['entity_count']} entities")
            print(f"- {stats['relationship_count']} relationships")
            print(f"- {stats['webpage_count']} webpages")
            print(f"- {stats['fact_count']} facts")
            
            # Save graph metadata
            kg.save_graph_metadata(topic)
        
        # Make sure we have a valid KnowledgeGraph instance before querying
        if kg is not None:
            print("\nYou can now query the knowledge graph.")
            # Query the knowledge graph
            while True:
                query = input("\nEnter a question (or 'exit' to quit): ")
                if query.lower() == 'exit':
                    break
                    
                results = kg.query_knowledge_graph(query)
                
                if "error" in results:
                    print(f"Error: {results['error']}")
                else:
                    print("\nAnswer:")
                    print(results["formatted_response"])

    finally:
        # Make sure we close the connection if it exists
        if kg is not None:
            kg.close()