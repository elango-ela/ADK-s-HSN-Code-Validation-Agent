# -----------------------------
# Imports and Configuration
# -----------------------------
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

# Load environment variables from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # ← Add this in your .env

# File path and model setup
EXCEL_FILE = "/home/elango/Documents/Google ADK/multi_tool_agent/HSN_SAC.xlsx"
MODEL_NAME = "all-MiniLM-L6-v2"

# -----------------------------
# Model and Vector DB Init
# -----------------------------
model = SentenceTransformer(MODEL_NAME)  # Load sentence transformer model
chroma_client = chromadb.Client()        # Init ChromaDB client

# Create or get Chroma collection
collection_name = "docs"
try:
    collection = chroma_client.get_collection(collection_name)
except Exception:
    collection = chroma_client.create_collection(collection_name)

# -----------------------------
# Initial Embedding (One-Time Load)
# -----------------------------
def load_and_embed_once():
    """Embed Excel data into vector DB at startup."""
    df = pd.read_excel(EXCEL_FILE)  # Load Excel
    texts = df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()
    embeddings = model.encode(texts, convert_to_numpy=True).tolist()
    ids = [str(i) for i in range(len(texts))]

    # Add data in batches to Chroma
    BATCH_SIZE = 5000
    for i in range(0, len(texts), BATCH_SIZE):
        collection.add(
            documents=texts[i:i + BATCH_SIZE],
            embeddings=embeddings[i:i + BATCH_SIZE],
            ids=ids[i:i + BATCH_SIZE]
        )

load_and_embed_once()  # Run once during startup

# -----------------------------
# Tool 1: Semantic Retrieval
# -----------------------------
def retrieve_from_docs(query: str) -> dict:
    """Perform semantic search on embedded Excel data."""
    query_embedding = model.encode([query], convert_to_numpy=True)[0].tolist()  # Encode query
    results = collection.query(query_embeddings=[query_embedding], n_results=2)  # Search Chroma
    docs = results["documents"][0]
    if not docs:
        return {"status": "error", "error_message": "No relevant documents found."}
    return {"status": "success", "context": "\n".join(docs)}

# -----------------------------
# Tool 2: HSN Hierarchy Search
# -----------------------------
def hsn_hierarchy_search(hsn_code: str) -> dict:
    """Return aligned hierarchy for HSN code without duplicate descriptions."""
    code = str(hsn_code).strip()
    if not code.isdigit() or len(code) not in [2, 4, 6, 8]:
        return {
            "HSNCode": code,
            "status": "error",
            "error_message": "Invalid HSN code format. Must be numeric and 2, 4, 6, or 8 digits."
        }

    # Load and clean CSV
    df = pd.read_excel(EXCEL_FILE)
    df.columns = df.columns.str.strip()
    df["HSNCode"] = df["HSNCode"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip()

    # Helper: get description for code
    def find_desc(code_lookup):
        match = df[df["HSNCode"] == code_lookup]
        return match.iloc[0]["Description"] if not match.empty else None

    levels = []
    seen_descriptions = set()
    for digits in [2, 4, 6, 8]:
        sliced = code[:digits]
        desc = find_desc(sliced)
        if desc and desc not in seen_descriptions:
            seen_descriptions.add(desc)
            levels.append({
                "level": f"{digits}-digit",
                "code": sliced,
                "description": desc
            })

    if not levels:
        return {"HSNCode": code, "status": "error", "error_message": "No matching codes found."}

    # Format output
    output_lines = [f"HSN Code: {code}", "", "Hierarchy:"]
    for lvl in levels:
        output_lines.append(f"{lvl['level']:<10} : {lvl['code']:<8} - {lvl['description']}")

    return {
        "HSNCode": code,
        "status": "success",
        "formatted_output": "\n".join(output_lines)
    }


'''
HSN Code: 30049011

Hierarchy:
2-digit   : 30       - Pharmaceutical products
4-digit   : 3004     - Medicaments
6-digit   : 300490   - Other Medicaments
8-digit   : 30049011 - Paracetamol Tablets

'''

# -----------------------------
# Agent Definition
# -----------------------------
root_agent = Agent(
    model="gemini-2.0-flash",
    name="rag_excel_agent",
    instruction=(
        "You are a smart assistant that helps users search HSN code data using retrieval. "
        "Show the output in aligned manner, not messy. "
        "Use `hsn_hierarchy_search` to get hierarchy for a given HSN code. "
        "Use `excel_update_tool` when user wants to upload a new Excel. "
        "Accept Excel URLs if valid and admin password is correct."
    ),
    tools=[retrieve_from_docs, hsn_hierarchy_search],
)
