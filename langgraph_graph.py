import os
import uuid
from typing import Literal, Optional, TypedDict

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver


# Load env vars if running locally
load_dotenv()

# ---------------------------
# 1. Global config
# ---------------------------

llm = ChatOpenAI(
    model="gpt-4.1-mini",  # or gpt-4o-mini / gpt-4.1 depending on your access
    temperature=0.4,
)


# ---------------------------
# 2. State definition
# ---------------------------

class EssayState(TypedDict, total=False):
    """Shared state passed around the LangGraph workflow."""

    # User input & mode
    user_input: str
    mode: Literal["essay", "open_question"]

    # Parsed topic / constraints
    topic: str
    instructions: str  # tone, style, length, etc.

    # Clarified understanding (HITL 1)
    clarification_questions: str          # questions à poser à l'humain
    clarification_answers: str            # réponses humaines
    clarifications_used: bool             # True si on les a prises en compte

    # Planning & research
    plan: str
    plan_feedback: str                    # feedback humain sur le plan (HITL 2)
    plan_validated: bool                  # plan validé ou pas
    research_notes: str                   # collected references / notes

    # Drafting & critique
    draft: str
    draft_feedback_human: str             # feedback humain sur le draft (HITL 3)
    draft_approved: bool                  # décision humaine sur draft
    critique: str

    # Save to DB
    saved: bool                           # marquer un état « figé »

    # Final answer
    final_draft: str                      # version finale interne
    final_feedback: str                   # feedback humain de finition (HITL 4)
    final_approved: bool                  # validation finale
    answer: str                           # renvoyé au front

    # Optional legacy human feedback (for backward compat)
    human_feedback: str

    # --- NEW: per-step skip flags for "let AI decide" ---
    skip_clarification: bool
    skip_plan_review: bool
    skip_draft_review: bool


# ---------------------------
# 3. Helper: call LLM with simple prompt
# ---------------------------

def call_llm(system_prompt: str, user_prompt: str) -> str:
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    resp = llm.invoke(messages)
    return resp.content


# ---------------------------
# 4. Nodes
# ---------------------------

def classify_intent(state: EssayState) -> EssayState:
    """
    Decide whether the user wants an essay or just an open question answer.
    This is your 'router' node.
    """

    # If mode is already set (e.g. from the API), do nothing
    if "mode" in state and state["mode"] in ("essay", "open_question"):
        return {}

    user_input = state["user_input"]

    system = (
        "You are an intent classifier for an essay assistant.\n"
        "Decide if the user wants: (1) an ESSAY, or (2) a simple OPEN_QUESTION answer.\n"
        "Return exactly one word: 'essay' or 'open_question'."
    )
    result = call_llm(system, user_input).strip().lower()

    if "essay" in result:
        mode: Literal["essay", "open_question"] = "essay"
    elif "open" in result:
        mode = "open_question"
    else:
        # default slightly in favor of essay, since it's the core of the project
        mode = "essay"

    return {"mode": mode}


def analyze_topic(state: EssayState) -> EssayState:
    """
    Normalize the topic and extract constraints (style, length, etc.).
    Also produce clarification questions (HITL 1).
    """
    user_input = state["user_input"]

    system = (
        "You are an assistant that extracts a clean essay TOPIC and INSTRUCTIONS "
        "(tone, length, audience, constraints) from a user request.\n"
        "You also propose clarification questions for the human.\n\n"
        "Return the result as:\n"
        "TOPIC: ...\n"
        "INSTRUCTIONS: ...\n"
        "CLARIFICATION_QUESTIONS:\n"
        "- ...\n- ...\n- ..."
    )
    analysis = call_llm(system, user_input)

    topic = ""
    instructions = ""
    clarification_questions = ""

    current_section: Optional[str] = None

    for line in analysis.splitlines():
        upper = line.upper().strip()
        if upper.startswith("TOPIC:"):
            current_section = "topic"
            topic = line.split(":", 1)[1].strip()
        elif upper.startswith("INSTRUCTIONS:"):
            current_section = "instructions"
            instructions = line.split(":", 1)[1].strip()
        elif upper.startswith("CLARIFICATION_QUESTIONS"):
            current_section = "clarifications"
        else:
            if current_section == "instructions" and line.strip():
                # allow multi-line instructions
                instructions += " " + line.strip()
            elif current_section == "clarifications" and line.strip():
                clarification_questions += line + "\n"

    if not topic:
        topic = user_input.strip()

    return {
        "topic": topic,
        "instructions": instructions,
        "clarification_questions": clarification_questions.strip(),
    }


def plan_essay(state: EssayState) -> EssayState:
    """
    Produce a bullet-point outline for the essay.
    """
    topic = state.get("topic", "")
    instructions = state.get("instructions", "")

    system = (
        "You are an expert essay planner. "
        "Create a clear bullet-point OUTLINE for the essay.\n"
        "Use 3–6 main sections with short explanations."
    )
    user = f"Topic: {topic}\nInstructions: {instructions}\n\nCreate the outline."

    plan = call_llm(system, user)
    return {"plan": plan}


def plan_human_review(state: EssayState) -> EssayState:
    """
    Plan review node (HITL 2).

    - If plan_feedback is provided -> revise the plan.
    - Otherwise keep current plan.
    In both cases we consider the plan 'validated' for this run.
    """
    plan = state.get("plan", "")
    feedback = (state.get("plan_feedback") or "").strip()

    if feedback:
        system = (
            "You are helping to revise an outline based on HUMAN FEEDBACK.\n"
            "Improve the plan accordingly, while keeping it clear and structured."
        )
        user = (
            f"CURRENT PLAN:\n{plan}\n\n"
            f"HUMAN FEEDBACK:\n{feedback}\n\n"
            "Return the revised plan."
        )
        improved_plan = call_llm(system, user)
        return {
            "plan": improved_plan,
            "plan_validated": True,
        }

    # No feedback: we just mark it validated
    return {"plan_validated": True}


def research_agentic(state: EssayState) -> EssayState:
    """
    Agentic web search step.

    Prefer Tavily (real web search).
    If Tavily or API key is missing, fall back to LLM-only notes.

    Uses clarification_answers if present.
    """
    topic = state.get("topic", "")
    plan = state.get("plan", "")
    clarification_answers = state.get("clarification_answers", "")

    query = (
        f"Essay topic: {topic}. Use this outline to guide research: {plan}. "
        f"Take into account these clarifications from the human (if any): {clarification_answers}"
    )

    clarifications_used = bool((clarification_answers or "").strip())

    # Try Tavily web search first
    try:
        try:
            # recommended import
            from langchain_community.tools.tavily_search.tool import TavilySearchResults
        except ImportError:
            from langchain_community.tools.tavily_search import (  # type: ignore
                TavilySearchResults,
            )

        tavily_tool = TavilySearchResults(
            max_results=5,
            include_answer=True,
            include_raw_content=False,
        )

        tavily_result = tavily_tool.invoke({"query": query})

        notes = (
            "Web research (Tavily) results:\n\n"
            f"Query: {query}\n\n"
            f"{tavily_result}"
        )

    except Exception as e:
        # Fallback: use LLM-only 'research' when Tavily fails
        system = (
            "You are a research assistant. Based on the topic, outline, and any clarifications, "
            "produce a short set of research notes (facts, arguments, references). "
            "Do NOT write the full essay, just notes."
        )
        user = (
            f"Topic: {topic}\n\nOutline:\n{plan}\n\n"
            f"Clarification answers (may be empty): {clarification_answers}\n\n"
            "We could not use a web search API (Tavily). "
            "Just rely on your own knowledge to produce research notes."
        )

        notes_llm = call_llm(system, user)
        notes = (
            f"{notes_llm}\n\n[Note: Tavily web search failed or is not configured. "
            f"Error: {type(e).__name__}: {e}]"
        )

    return {
        "research_notes": notes,
        "clarifications_used": clarifications_used,
    }


def write_draft(state: EssayState) -> EssayState:
    """
    Write a draft essay using topic, instructions, plan, and research notes.

    Can optionally integrate draft_feedback_human on later runs.
    """
    topic = state.get("topic", "")
    instructions = state.get("instructions", "")
    plan = state.get("plan", "")
    research_notes = state.get("research_notes", "")
    previous_draft = state.get("draft", "")
    draft_feedback_human = state.get("draft_feedback_human", "")

    system = (
        "You are a senior essay writer. Write a coherent essay following the outline.\n"
        "If there is a previous version, improve it; otherwise, draft from scratch.\n"
        "If there is HUMAN FEEDBACK on the draft, use it to guide your improvements.\n"
        "Aim for clarity, structure, and good academic style."
    )
    user = (
        f"Topic: {topic}\n"
        f"Instructions: {instructions}\n\n"
        f"Outline:\n{plan}\n\n"
        f"Research notes:\n{research_notes}\n\n"
        f"Previous draft (if any):\n{previous_draft}\n\n"
        f"Human feedback on draft (if any):\n{draft_feedback_human}"
    )

    draft = call_llm(system, user)
    return {"draft": draft}


def critic_node(state: EssayState) -> EssayState:
    """
    Critic / reflection node (no loop inside the graph).
    Produces a critique; any improvement loop is done across runs by the human.
    """
    draft = state.get("draft", "")
    topic = state.get("topic", "")
    instructions = state.get("instructions", "")

    system = (
        "You are an essay critic. Evaluate the draft against the topic and instructions.\n"
        "1) List strengths and weaknesses.\n"
        "2) Give a quality score between 0 and 10."
    )
    user = (
        f"Topic: {topic}\nInstructions: {instructions}\n\nDraft:\n{draft}"
    )

    critique = call_llm(system, user)
    return {"critique": critique}


def save_to_db(state: EssayState) -> EssayState:
    """
    Conceptual 'Save to DB' node.

    The SqliteSaver already persists the whole state.
    This node just marks that we've reached a 'frozen' intermediate result.
    """
    return {"saved": True}


def finalize_essay(state: EssayState) -> EssayState:
    """
    Finalize answer for essay mode.

    Interprets 'final_feedback' (or legacy 'human_feedback') as
    HITL final polish, and produces final_draft + answer.
    """
    draft = state.get("draft", "")
    critique = state.get("critique", "")
    # 'final_feedback' is the new name; keep 'human_feedback' as fallback
    final_feedback = state.get("final_feedback") or state.get("human_feedback", "")

    if final_feedback:
        system = (
            "You are improving an essay based on critic comments and HUMAN FINAL FEEDBACK.\n"
            "Apply the requested changes while keeping quality high."
        )
        user = (
            f"Original draft:\n{draft}\n\n"
            f"Critique:\n{critique}\n\n"
            f"Human final feedback:\n{final_feedback}\n\n"
            "Produce the final improved essay."
        )
        final_draft = call_llm(system, user)
    else:
        # no additional human feedback – just use the last draft
        final_draft = draft

    return {
        "final_draft": final_draft,
        "final_approved": True,
        "answer": final_draft,
    }


def basic_llm_response(state: EssayState) -> EssayState:
    """
    Simple one-shot LLM answer for open questions (non-essay mode).
    """
    user_input = state["user_input"]

    system = (
        "You are a helpful assistant. Answer the user question directly.\n"
        "If the user asks for an essay-like answer, you can write a short structured reply, "
        "but do NOT overcomplicate it."
    )
    answer = call_llm(system, user_input)
    return {"answer": answer}


# --- STOP NODES (HITL breakpoints, but DAG: no cycles) ---

def stop_after_analyze(state: EssayState) -> EssayState:
    """HITL gate: stop after analyze until clarifications are answered or skipped."""
    return {}


def stop_after_plan_review(state: EssayState) -> EssayState:
    """HITL gate: stop after plan review until plan is validated or skipped."""
    return {}


def stop_after_critic(state: EssayState) -> EssayState:
    """HITL gate: stop after critic until draft is approved or skipped."""
    return {}


# ---------------------------
# 5. Build the graph (DAG, no loops)
# ---------------------------

builder = StateGraph(EssayState)

# Nodes
builder.add_node("classify", classify_intent)
builder.add_node("analyze", analyze_topic)
builder.add_node("plan", plan_essay)
builder.add_node("plan_review", plan_human_review)
builder.add_node("research", research_agentic)
builder.add_node("write", write_draft)
builder.add_node("critic", critic_node)
builder.add_node("save", save_to_db)
builder.add_node("finalize", finalize_essay)
builder.add_node("basic_reply", basic_llm_response)

# HITL stop nodes
builder.add_node("stop_after_analyze", stop_after_analyze)
builder.add_node("stop_after_plan_review", stop_after_plan_review)
builder.add_node("stop_after_critic", stop_after_critic)

# Entry: START -> classify
builder.add_edge(START, "classify")


def route_from_classify(state: EssayState) -> str:
    mode = state.get("mode", "open_question")
    if mode == "essay":
        return "analyze"
    else:
        return "basic_reply"


builder.add_conditional_edges(
    "classify",
    route_from_classify,
    {
        "analyze": "analyze",
        "basic_reply": "basic_reply",
    },
)

# --- Gate 1: after analyze ---

def route_from_analyze(state: EssayState) -> str:
    """
    If we don't yet have clarification_answers and user didn't choose to skip,
    stop to let the human answer.
    Otherwise continue to plan.
    """
    answers = (state.get("clarification_answers") or "").strip()
    skip = bool(state.get("skip_clarification", False))

    if not answers and not skip:
        return "stop_after_analyze"
    return "plan"


builder.add_conditional_edges(
    "analyze",
    route_from_analyze,
    {
        "plan": "plan",
        "stop_after_analyze": "stop_after_analyze",
    },
)

# --- Plan & Gate 2: plan_review ---

builder.add_edge("plan", "plan_review")


def route_from_plan_review(state: EssayState) -> str:
    """
    If user did not choose to skip and plan_feedback is still empty,
    stop here and wait for feedback. Otherwise go to research.
    """
    feedback = (state.get("plan_feedback") or "").strip()
    skip = bool(state.get("skip_plan_review", False))

    if not feedback and not skip:
        return "stop_after_plan_review"
    return "research"


builder.add_conditional_edges(
    "plan_review",
    route_from_plan_review,
    {
        "research": "research",
        "stop_after_plan_review": "stop_after_plan_review",
    },
)

# Research -> write -> critic
builder.add_edge("research", "write")
builder.add_edge("write", "critic")

# --- Gate 3: after critic ---

def route_from_critic(state: EssayState) -> str:
    """
    If user did not choose to skip and draft_approved is neither True nor False
    (i.e., no explicit human decision yet), stop here.
    Otherwise go to save (then finalize).
    """
    skip = bool(state.get("skip_draft_review", False))
    draft_approved = state.get("draft_approved", None)

    if not skip and draft_approved is None:
        return "stop_after_critic"
    return "save"


builder.add_conditional_edges(
    "critic",
    route_from_critic,
    {
        "save": "save",
        "stop_after_critic": "stop_after_critic",
    },
)

# Save -> finalize
builder.add_edge("save", "finalize")

# Endpoints
builder.add_edge("basic_reply", END)
builder.add_edge("finalize", END)
builder.add_edge("stop_after_analyze", END)
builder.add_edge("stop_after_plan_review", END)
builder.add_edge("stop_after_critic", END)


# ---------------------------
# 6. Compile with persistence
# ---------------------------

CHECKPOINTS_DB = "checkpoints.sqlite"

# Create a real sqlite3 connection and pass it to SqliteSaver
_conn = sqlite3.connect(CHECKPOINTS_DB, check_same_thread=False)
checkpointer = SqliteSaver(_conn)

graph = builder.compile(checkpointer=checkpointer)


# ---------------------------
# 7. Helper function for FastAPI
# ---------------------------

def run_essay_graph(
    user_input: str,
    thread_id: Optional[str] = None,
    *,
    clarification_answers: Optional[str] = None,
    plan_feedback: Optional[str] = None,
    draft_feedback_human: Optional[str] = None,
    draft_approved: Optional[bool] = None,
    final_feedback: Optional[str] = None,
    skip_clarification: bool = False,
    skip_plan_review: bool = False,
    skip_draft_review: bool = False,
) -> EssayState:
    """
    Convenience wrapper to run the graph from FastAPI.

    True HITL step-by-step:

    - Call 1: just user_input
        -> classify + analyze, then STOP_AFTER_ANALYZE (clarification questions).
    - Call 2: same thread_id + clarification_answers OR skip_clarification=True
        -> plan + plan_review, then STOP_AFTER_PLAN_REVIEW (plan shown).
    - Call 3: same thread_id + plan_feedback OR skip_plan_review=True
        -> research + write + critic, then STOP_AFTER_CRITIC (draft+critique).
    - Call 4: same thread_id + draft_approved=True OR skip_draft_review=True
        -> save + finalize, returns final answer.

    You can also give feedback multiple times across runs; state is persisted.
    """
    # Treat empty string as "no thread"
    if not thread_id:
        thread_id = str(uuid.uuid4())

    initial_state: EssayState = {
        "user_input": user_input,
    }

    if clarification_answers:
        initial_state["clarification_answers"] = clarification_answers
    if plan_feedback:
        initial_state["plan_feedback"] = plan_feedback
    if draft_feedback_human:
        initial_state["draft_feedback_human"] = draft_feedback_human
    if draft_approved is not None:
        initial_state["draft_approved"] = draft_approved
    if final_feedback:
        initial_state["final_feedback"] = final_feedback

    # Skip flags
    initial_state["skip_clarification"] = skip_clarification
    initial_state["skip_plan_review"] = skip_plan_review
    initial_state["skip_draft_review"] = skip_draft_review

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 50,
    }
    result = graph.invoke(initial_state, config=config)
    result["thread_id"] = thread_id
    return result
