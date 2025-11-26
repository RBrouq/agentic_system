import uuid
from typing import Optional

from .state import EssayState
from .graph_builder import graph
from .config import DEFAULT_RECURSION_LIMIT


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
    Convenience wrapper to run the graph.

    HITL step-by-step:

    - Call 1: just user_input
        -> classify + analyze, then STOP_AFTER_ANALYZE (clarification questions).
    - Call 2: same thread_id + clarification_answers OR skip_clarification=True
        -> plan + plan_review, then STOP_AFTER_PLAN_REVIEW (plan shown).
    - Call 3: same thread_id + plan_feedback OR skip_plan_review=True
        -> research + write + critic, then STOP_AFTER_CRITIC (draft+critique).
    - Call 4: same thread_id + draft_approved=True OR skip_draft_review=True
        -> save + finalize, returns final answer.
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
        "recursion_limit": DEFAULT_RECURSION_LIMIT,
    }
    result: EssayState = graph.invoke(initial_state, config=config)  # type: ignore[assignment]
    result["thread_id"] = thread_id  # type: ignore[index]
    return result
