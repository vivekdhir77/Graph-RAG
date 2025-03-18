# Knowledge Graph RAG System

## Overview
This project is an intelligent research assistant that builds knowledge graphs from web content and allows users to query them using natural language. It combines web search, information extraction, and graph-based reasoning to provide comprehensive answers to complex questions.

## How It Works
1. **Web Search**: Searches the internet for information on a topic
2. **Content Extraction**: Scrapes and processes web pages
3. **Knowledge Graph Construction**: Identifies entities and relationships from text
4. **Query Processing**: Translates natural language queries to graph database queries
5. **Answer Generation**: Creates comprehensive responses from the graph data

## Key Features
- Automated knowledge graph construction from any topic
- Natural language querying using an LLM-powered interface
- Graph visualization capabilities
- API for integration with other applications
- Persistent storage of graph data for reuse

## LLM Prompts Used in the System
The system relies on carefully engineered prompts to extract information and process queries:

### Text Summarization Prompt
```
Summarize the following text concisely:

{text}

Provide only the summary, no introductory or concluding remarks.
```

### Graph-Maker Entity Extraction Prompt
```
You are an expert knowledge graph builder. Analyze the following text and extract entities and relationships.

Context: {context}

Text: {text}

Extract entities based on these categories:
{ontology}

Format your response as a JSON array with objects having this structure:
[
    {
        "node_1": {"label": "EntityType1", "name": "Entity1Name"},
        "node_2": {"label": "EntityType2", "name": "Entity2Name"},
        "relationship": "Describe how Entity1 relates to Entity2",
        "metadata": {"confidence": 0.9, "source": "The source of this information"}
    }
]

Only return the JSON array, nothing else.
```

### Standard Entity Extraction Prompt
```
You are an expert knowledge graph builder. Analyze the following text and extract:
1. Key entities (people, organizations, concepts, etc.)
2. Important facts about these entities
3. Relationships between entities

For each fact, assign a confidence score from 0-1.

Context: {context}

Text: {text}

Format your response as a JSON object with the following structure:
{
    "entities": [
        { "name": "Entity Name", "type": "person|organization|concept|etc", "attributes": {"key": "value"} }
    ],
    "facts": [
        { "subject": "Entity1", "predicate": "relationship", "object": "Entity2", "confidence": 0.9 }
    ]
}
Only return the JSON object, nothing else.
```

### Batch Entity Extraction Prompt
```
You are an expert knowledge graph builder. Analyze each of the following text chunks and extract entities and relationships.
For each chunk, provide the extracted data in JSON format.

{batch_data}

Format your response as a JSON array with one object per chunk:
[
    {
        "chunk_index": 0,
        "entities": [
            { "name": "Entity Name", "type": "person|organization|concept|etc", "attributes": {"key": "value"} }
        ],
        "facts": [
            { "subject": "Entity1", "predicate": "relationship", "object": "Entity2", "confidence": 0.9 }
        ]
    },
    // more chunk results...
]
Only return the JSON array, nothing else.
```

### Advanced Cypher Query Generation Prompt
```
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
```

### Query Results Formatting Prompt
```
The following is the result of a knowledge graph query in JSON format:

{results}

Based on these results, provide a well-formatted answer to the original query: "{query}"

Format your response with the following structure:
1. Start with "**Question:** {query}"
2. Then include "**Answer:**" followed by your comprehensive response
3. Organize the information logically with headings and bullet points where appropriate
4. Include the source of information where available

If the results are empty or insufficient, kindly mention that the knowledge graph doesn't contain enough information to answer the query.
```

## Architecture
The system uses several key components:
- **Neo4j**: Graph database storing the knowledge graph
- **LangChain**: Framework for working with LLMs and building chains
- **LangGraph**: Workflow orchestration for multi-step processes
- **Groq LLM**: Powers entity extraction and natural language understanding
- **Flask API**: Provides RESTful access to the knowledge graph
- **Serper**: API for web search functionality

## Setup Requirements
- Python 3.8+
- Neo4j database
- Groq API key
- Serper API key
- Required Python packages (see requirements.txt)

## Usage
1. Create a virtual environment and activate it:
   ```bash
   fill in the environment variables in environment.env and rename it in .env
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the API:
   ```bash
   python api.py
   ```

4. Set up the frontend:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```