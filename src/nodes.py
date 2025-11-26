from typing import Optional

from .state import EssayState
from .llm_utils import call_llm


# ---------------------------
# 1. Core Nodes
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
        mode = "essay"
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
            from langchain_community.tools.tavily_search.tool import (
                TavilySearchResults,
            )
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
# 2. Router Functions for Conditional Edges
# ---------------------------

def route_from_classify(state: EssayState) -> str:
    mode = state.get("mode", "open_question")
    if mode == "essay":
        return "analyze"
    else:
        return "basic_reply"


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
