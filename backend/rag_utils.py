# Entry point of the backend: the same rag() function that was evaluated in the notebook.
# The agent modules (aurora_*.py) are copied next to this file. The API credentials come from the environment
# variables OPENAI_API_KEY and OPENAI_API_BASE (Codespace secrets), which ChatOpenAI reads directly.
from aurora_retrieval import retrieve
from aurora_agent import run_agent

def rag(
    query: str,
    k: int = 4,                  # evaluated setting: 4 chunks searched for each part of the question
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Run the RAG agent (guard, sub-agent, answer, checks, route)
    result = run_agent(
        query,
        vectorstore,
        k=k,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )

    # The label is kept apart from the answer: it is shown in the UI, never part of the answer text
    label = {
        "route": result["route"],
        "route_reason": result["route_reason"],
        "review_note": result["review_note"]
    }

    # Return the answer, the evidence chunks it is based on, and the label
    return result["answer"], result["evidence"], label
