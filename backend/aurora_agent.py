# RAG agent module: the fixed LangGraph pipeline Guard -> RAG sub-agent -> Answer -> Checks -> Route.
# The order is decided by code, not by the LLM; injection and out-of-scope questions skip straight to Route.
# Every question gets a reply; the Route step adds a label, the reason, and a note for staff.
import operator
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from aurora_guard import guard
from aurora_subagent import build_subagent, K_PER_PART
from aurora_generation import generate
from aurora_checks import run_checks

class AgentState(TypedDict, total=False):
    question: str
    intent: str                 # POLICY_QUESTION, OUT_OF_SCOPE, NEEDS_PERSON, UNCLEAR, or INJECTION
    intent_reason: str
    escalation_owner: str
    sub_questions: list
    coverage: list              # one {"part", "covered", "missing"} per sub-question
    evidence: list              # the chunks the answer is based on
    answer: str
    unsupported: list           # claims the grounding check could not find in the evidence
    policy_issues: list         # hotel policy rules broken by the answer
    checks_passed: bool
    route: str                  # ANSWER, STAFF REVIEW, HAND TO PERSON, or DECLINE
    route_reason: str
    review_note: str
    trajectory: Annotated[list, operator.add]   # the steps taken, appended by every node

DECLINE_INJECTION = ("This message could not be processed because it looks like an attempt to change the "
                     "assistant's instructions. Please rephrase it as a question about AuroraStay policies.")
DECLINE_OUT_OF_SCOPE = ("This question is not covered by the AuroraStay policies, so the assistant cannot answer "
                        "it. Please ask about bookings, rooms and properties, charges and refunds, or guest conduct.")

def route(state):
    # The single place that decides the reply label, in priority order: DECLINE > HAND TO PERSON >
    # STAFF REVIEW > ANSWER. Every path through the graph ends here, so no flagged case can skip it.
    intent = state.get("intent")
    if intent == "INJECTION":
        return {"route": "DECLINE", "route_reason": state["intent_reason"], "review_note": "",
                "answer": DECLINE_INJECTION}
    if intent == "OUT_OF_SCOPE":
        return {"route": "DECLINE", "route_reason": state["intent_reason"], "review_note": "",
                "answer": DECLINE_OUT_OF_SCOPE}

    # What staff should verify: parts the search did not cover and checks that still failed after the redraft
    verify = [f'Not covered by the policies found: "{c["part"]}".' if c.get("searched", True) else
              f'Not searched (the question has too many parts), check separately: "{c["part"]}".'
              for c in state.get("coverage", []) if not c["covered"]]
    verify += [f"Claim not found in the policies: {c}" for c in state.get("unsupported", [])]
    verify += [f"Policy rule: the answer {p}." for p in state.get("policy_issues", [])]
    if intent == "UNCLEAR":
        verify.insert(0, "The question is unclear; confirm what the guest is asking.")

    if intent == "NEEDS_PERSON":
        note = f"Involve: {state['escalation_owner']}. " + " ".join(verify)
        return {"route": "HAND TO PERSON", "route_reason": state["intent_reason"], "review_note": note.strip()}
    if verify:
        return {"route": "STAFF REVIEW", "route_reason": verify[0], "review_note": " ".join(verify)}
    return {"route": "ANSWER", "route_reason": "Every part is covered and the answer passed the checks.",
            "review_note": ""}

def build_agent(vectorstore, model, checker, k=K_PER_PART):
    # model writes the answer; checker makes the small decisions (intent, plan, coverage, grounding)

    def guard_node(state):
        result = guard(state["question"], checker)
        return {**result, "trajectory": [f"guard: {result['intent']} ({result['intent_reason']})"]}

    def answer_node(state):
        # Tell the answer which parts the search could not cover, so it says so instead of guessing
        missing = [c["part"] for c in state["coverage"] if not c["covered"] and c.get("searched", True)]
        skipped = [c["part"] for c in state["coverage"] if not c.get("searched", True)]
        guidance = ("The search found no policy passage for these parts; say clearly that the retrieved "
                    "policies do not state them: " + "; ".join(missing)) if missing else ""
        if skipped:
            guidance += ("\nThese parts were not searched separately because the question has too many parts: "
                         + "; ".join(skipped) + ". Answer one only if the passages above state it; otherwise say it "
                         "was not looked up and staff should check it separately (do not say the policies do not "
                         "state it).")
        answer = generate(state["question"], state["evidence"], model, guidance).content
        return {"answer": answer, "trajectory": ["answer: draft written"]}

    def checks_node(state):
        result = run_checks(state["question"], state["evidence"], state["answer"], model, checker)
        return {"answer": result["answer"], "unsupported": result["unsupported"],
                "policy_issues": result["policy_issues"], "checks_passed": result["checks_passed"],
                "trajectory": result["steps"]}

    def route_node(state):
        result = route(state)
        return {**result, "trajectory": [f"route: {result['route']}"]}

    def after_guard(state):
        # Injection and out-of-scope questions skip the search; all other questions are searched and answered
        return "route" if state["intent"] in ("INJECTION", "OUT_OF_SCOPE") else "rag_subagent"

    graph = StateGraph(AgentState)
    graph.add_node("guard", guard_node)
    graph.add_node("rag_subagent", build_subagent(vectorstore, checker, k))  # a subgraph used as one node
    graph.add_node("answer", answer_node)
    graph.add_node("checks", checks_node)
    graph.add_node("route", route_node)
    graph.add_edge(START, "guard")
    graph.add_conditional_edges("guard", after_guard, ["rag_subagent", "route"])
    graph.add_edge("rag_subagent", "answer")
    graph.add_edge("answer", "checks")
    graph.add_edge("checks", "route")
    graph.add_edge("route", END)
    return graph.compile()

def run_agent(question, vectorstore, k=K_PER_PART, model_name="gpt-4o-mini", temperature=0.0, top_p=1.0,
              max_tokens=512):
    # The generator uses the given settings; the checker is the same model at temperature 0 with room for
    # longer structured outputs (for example a grounding check that lists every claim). The timeout stops a
    # call that hangs, so it is retried instead of blocking the reply.
    model = ChatOpenAI(model=model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens, timeout=60)
    checker = ChatOpenAI(model=model_name, temperature=0, max_tokens=1024, timeout=60)
    state = build_agent(vectorstore, model, checker, k).invoke({"question": question, "trajectory": []})
    return {key: state.get(key, default) for key, default in [
        ("question", question), ("answer", ""), ("route", ""), ("route_reason", ""), ("review_note", ""),
        ("intent", ""), ("sub_questions", []), ("coverage", []), ("evidence", []), ("trajectory", []),
    ]}

def format_reply(result):
    # The reply shown to staff: the answer, followed by the label and the note when staff need to act
    if result["route"] in ("ANSWER", "DECLINE") or not result["review_note"]:
        return result["answer"]
    return f"{result['answer']}\n\n[{result['route']}] {result['review_note']}"
