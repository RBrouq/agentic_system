from typing import Literal, TypedDict


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

    # --- Per-step skip flags for "let AI decide" ---
    skip_clarification: bool
    skip_plan_review: bool
    skip_draft_review: bool
