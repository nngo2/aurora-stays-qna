
import streamlit as st
import requests


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BACKEND_URL = "http://localhost:7860"


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Aurora Stays",
    page_icon="🏨",
    layout="wide"
)


# ---------------------------------------------------------
# Sidebar  -  Model parameters
# ---------------------------------------------------------

with st.sidebar:

    st.header("⚙️ Parameters")

    k = st.slider(
        "k (chunks per part of the question)",
        min_value=1,
        max_value=10,
        value=4,
        help="Number of chunks searched for each part of the question. Default 4, the setting used in evaluation."
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Higher values make the output more creative; lower values make it more focused."
    )

    max_tokens = st.slider(
        "Max tokens",
        min_value=64,
        max_value=2048,
        value=512,
        step=64,
        help="Maximum number of tokens in the generated answer."
    )


# ---------------------------------------------------------
# Display helper
# ---------------------------------------------------------

def show(text):
    # Escape "$" so dollar amounts are not rendered as LaTeX math by Streamlit's Markdown
    st.markdown(text.replace("$", "\\$"))


# How each label is shown: the box type, an icon, and what the label tells staff
LABEL_STYLES = {
    "ANSWER": (st.success, "✅", "ready to use"),
    "STAFF REVIEW": (st.warning, "⚠️", "check before using"),
    "HAND TO PERSON": (st.error, "🚨", "escalate"),
    "DECLINE": (st.info, "ℹ️", "not answered"),
}


def show_label(label):
    # The agent's label is kept apart from the answer: a read-only Label field, then a coloured box with
    # the note for staff (or the reason when there is no note)
    route = label.get("route", "")
    st.text_input("Label", value=route, disabled=True)

    box, icon, meaning = LABEL_STYLES.get(route, (st.info, "ℹ️", ""))
    detail = label.get("review_note") or label.get("route_reason", "")
    box(f"**{route}**: {meaning}. {detail}".replace("$", "\\$"), icon=icon)


# ---------------------------------------------------------
# Main area
# ---------------------------------------------------------

st.title("🏨 Aurora Stays")
st.write("Ask a question about Aurora Stays policies.")


# ---------------------------------------------------------
# Response type selection (two options only)
# ---------------------------------------------------------

response_type = st.radio(
    "Response mode",
    ["Answer only", "Answer + Relevant Chunks"],
    horizontal=True
)


# ---------------------------------------------------------
# Query input
# ---------------------------------------------------------

query = st.text_input(
    "Your question",
    placeholder="e.g., What is the pet fee at ASH-DEN Union, and how many pets can I bring?"
)


# ---------------------------------------------------------
# Submit
# ---------------------------------------------------------

if st.button("Submit", type="primary"):

    if not query.strip():
        st.warning("Please enter a question.")

    else:

        # Pick endpoint
        if response_type == "Answer only":
            endpoint = "/v1/answer"
        else:
            endpoint = "/v1/answer_with_relevant_chunks"

        # Build request payload with parameters
        payload = {
            "query": query,
            "k": k,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            with st.spinner("Thinking…"):
                response = requests.post(
                    f"{BACKEND_URL}{endpoint}",
                    json=payload
                )

            if response.status_code == 200:

                result = response.json()

                # -- Answer only --------------------------------
                if response_type == "Answer only":

                    st.subheader("Answer")
                    show(result["answer"])
                    show_label(result.get("label", {}))

                # -- Answer + Relevant Chunks -------------------
                else:

                    st.subheader("Answer")
                    show(result["answer"])
                    show_label(result.get("label", {}))

                    st.subheader("Relevant Chunks")
                    for i, chunk in enumerate(result["relevant_chunks"], start=1):
                        with st.expander(f"Chunk {i}"):
                            show(chunk)

            else:
                st.error(
                    f"Unable to get a response from the RAG API. "
                    f"Status code: {response.status_code}"
                )

        except requests.exceptions.RequestException:
            st.error(
                "Unable to connect to the RAG API. "
                "Please make sure the backend service is running."
            )
