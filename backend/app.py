
# Import necessary libraries
from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Import the RAG entry point and the hybrid search
from rag_utils import retrieve, rag
from aurora_retrieval import get_retriever


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

EMBED_MODEL = "HarishMaths/Hotel-Policy-Embedding"
VECTORSTORE_PATH = "faiss_index"


# ---------------------------------------------------------
# Initialize the Flask application
# ---------------------------------------------------------

rag_api = Flask("Aurora Stay API")


# ---------------------------------------------------------
# Load the embedding model
# ---------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


# ---------------------------------------------------------
# Load the saved FAISS vector store
# ---------------------------------------------------------

vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print(f"Vectors stored: {vectorstore.index.ntotal}")

# Build the hybrid retriever (BM25 index and property list) once at startup, not on the first request
print(f"Properties found: {len(get_retriever(vectorstore).properties)}")


# ---------------------------------------------------------
# Default route
# ---------------------------------------------------------

@rag_api.get("/")
def home():
    """
    Handles GET requests to the root URL.
    """
    return "Welcome to the Aurora Stay RAG API!"


# ---------------------------------------------------------
# Relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/relevant_chunks")
def relevant_chunks():
    """
    Handles POST requests to /v1/relevant_chunks.

    Example request:
    {
        "query": "What is the pet fee at ASH-DEN Union, and how many pets can I bring?",
        "k": 4
    }

    Returns the top-k relevant chunks from one hybrid search
    (FAISS + BM25) with the whole query.
    """

    # Get JSON data from request body
    data = request.get_json()

    # Extract parameters
    query = data["query"]
    k = data.get("k", 4)  # default = evaluated setting (4 chunks per part of the question)

    # Retrieve relevant chunks
    chunks = retrieve(
        query=query,
        k=k,
        vectorstore=vectorstore
    )

    # Return the retrieved chunks
    return jsonify({
        "query": query,
        "k": k,
        "relevant_chunks": chunks
    })


# ---------------------------------------------------------
# Answer with relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer_with_relevant_chunks")
def answer_with_relevant_chunks():
    """
    Handles POST requests to /v1/answer_with_relevant_chunks.

    Example request:
    {
        "query": "What is the pet fee at ASH-DEN Union, and how many pets can I bring?",
        "k": 4,
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512
    }

    Returns the generated answer, the evidence chunks it is
    based on, and the label (route, reason, note for staff).
    """

    # Get JSON data from request body
    data = request.get_json()

    # Extract parameters
    query = data["query"]
    k = data.get("k", 4)  # default = evaluated setting (4 chunks per part of the question)
    model_name = data.get("model_name", "gpt-4o-mini")
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    max_tokens = data.get("max_tokens", 512)

    # Run the RAG agent
    answer, chunks, label = rag(
        query=query,
        k=k,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        vectorstore=vectorstore
    )

    # Return answer, chunks, and parameters used
    return jsonify({
        "query": query,
        "k": k,
        "model_name": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "relevant_chunks": chunks,
        "answer": answer,
        "label": label
    })


# ---------------------------------------------------------
# Answer endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer")
def answer():
    """
    Handles POST requests to /v1/answer.

    Example request:
    {
        "query": "What is the pet fee at ASH-DEN Union, and how many pets can I bring?",
        "k": 4,
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512
    }

    Returns the generated answer and its label.
    """

    # Get JSON data from request body
    data = request.get_json()

    # Extract parameters
    query = data["query"]
    k = data.get("k", 4)  # default = evaluated setting (4 chunks per part of the question)
    model_name = data.get("model_name", "gpt-4o-mini")
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    max_tokens = data.get("max_tokens", 512)

    # Run the RAG agent
    generated_answer, _, label = rag(
        query=query,
        k=k,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        vectorstore=vectorstore
    )

    # Return the answer and its label (kept separate from the answer text)
    return jsonify({
        "answer": generated_answer,
        "label": label
    })


# ---------------------------------------------------------
# Run the Flask application
# ---------------------------------------------------------

if __name__ == "__main__":
    rag_api.run(debug=True)
