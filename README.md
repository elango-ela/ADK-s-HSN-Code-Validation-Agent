# HSN Code Validation and Suggestion Agent

## Overview

This project implements an intelligent agent using Google’s Agent Developer Kit (ADK) framework. The agent validates Harmonized System Nomenclature (HSN) codes based on a master dataset and suggests relevant HSN codes based on user queries about goods or services.

#### The agent supports:

- Validating single or multiple HSN codes.

- Providing hierarchical descriptions for valid HSN codes.

- Suggesting HSN codes from product or service descriptions via semantic search.


<!--                     -->


## Approach

Retrieval-Augmented Generation (RAG) This agent leverages a Retrieval-Augmented Generation (RAG) approach to combine the power of pretrained language models with a domain-specific knowledge base.

#### Why RAG
The agent allows users to ask naturally, finding relevant HSN codes by meaning rather than exact keywords. For instance, if a user asks for the HSN code for “car,” the agent can also find results related to “vehicle,” because it understands the semantic similarity between these cars and vehicles. This helps users get accurate answers even if their query wording differs from the dataset.
<!--                                  -->

## Architecture and Components
Built with the Google ADK Framework, leveraging:

### Agent: 
The Agent acts as the central orchestrator that manages user input, invokes appropriate tools, and formats the output response.

```python

root_agent = Agent(
    model="gemini-2.0-flash",
    name="rag_excel_agent",
    instruction=("You are a smart assistant that helps users search HSN code data using retrieval")
    tools=[retrieve_from_docs, hsn_hierarchy_search],
)

```

**root_agent = Agent( model="gemini-2.0-flash",
tools=[retrieve_from_docs, hsn_hierarchy_search],)**
### Tools Implemented for the Agent:
####       <-RAG-based Retrieval Tool (retrieve_from_docs)
- Implements a Retrieval-Augmented Generation (RAG) approach for semantic search.

- The master HSN dataset is embedded into a vector database (ChromaDB) using the all-MiniLM-L6-v2 sentence transformer model.

- When a user inputs a natural language query, this tool performs a semantic search — finding relevant HSN codes and descriptions based on the meaning of the query, not just keyword matching.

- For example, a query mentioning "car" can also retrieve documents related to "vehicle" due to semantic similarity.

- This enables flexible and accurate retrieval of HSN data even if the user phrasing differs from the dataset text.

####       <-HSN Hierarchy Search Tool (hsn_hierarchy_search)
- Validates and explains the hierarchical structure of an HSN code provided by the user.

- Checks if the input code is valid in format (numeric and correct length: 2, 4, 6, or 8 digits).

- Looks up the code and its parent codes in the dataset (e.g., for "01011010", parents "010110", "0101", and "01").

 - Returns a clean, aligned, and deduplicated description hierarchy for easy understanding of the product classification.

- Helps users verify the validity and context of HSN codes at multiple classification levels.

### Data Handling and Model Initialization

- At startup, the master HSN Excel file is loaded and converted into text lines (combining HSN code and description).

- The text data is embedded using the all-MiniLM-L6-v2 Sentence Transformer model to create vector embeddings.

- These embeddings are stored in ChromaDB, which supports fast semantic retrieval queries.

- Embeddings are batched during insertion to optimize performance for large datasets.

- This setup allows the agent to efficiently handle large, structured HSN datasets and perform semantic search dynamically.


## Code Highlights
### Key Functions
- load_and_embed_once(): Loads and embeds the static HSN master Excel file, Finally stores in ChromaDb.
  ```python

  def load_and_embed_once():

  ```

- retrieve_from_docs(query): Retrieves documents semantically related to the query.
  ```python

  def retrieve_from_docs(query: str) -> dict:

  ```

- hsn_hierarchy_search(hsn_code): Validates and returns the hierarchical description for a given HSN code.
   ```python

   def hsn_hierarchy_search(hsn_code: str) -> dict:

   ```

### Agent Initialization
```python

root_agent = Agent(
    model="gemini-2.0-flash",
    name="rag_excel_agent",
    instruction="You are a smart assistant that helps users search HSN code data using retrieval, "
                "show the output in aligned manner, and explore the hierarchy of HSN codes.",
    tools=[retrieve_from_docs, hsn_hierarchy_search],
)

```


### Robustness
- Handles edge cases such as invalid formats, missing codes, and duplicate descriptions gracefully.
### Performance
- Pre-embedding the static dataset improves query and reduces response time.



