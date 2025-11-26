import sqlite3
from typing import Tuple

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import EssayState
from .config import CHECKPOINTS_DB
from .nodes import (
    classify_intent,
    analyze_topic,
    plan_essay,
    plan_human_review,
    research_agentic,
    write_draft,
    critic_node,
    save_to_db,
    finalize_essay,
    basic_llm_response,
    stop_after_analyze,
    stop_after_plan_review,
    stop_after_critic,
    route_from_classify,
    route_from_analyze,
    route_from_plan_review,
    route_from_critic,
)


def build_graph() -> Tuple[object, SqliteSaver]:
    """Build the LangGraph graph and attach a SqliteSaver checkpointer."""
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

    # From classify: route depending on mode
    builder.add_conditional_edges(
        "classify",
        route_from_classify,
        {
            "analyze": "analyze",
            "basic_reply": "basic_reply",
        },
    )

    # --- Gate 1: after analyze ---
    builder.add_conditional_edges(
        "analyze",
        route_from_analyze,
        {
            "plan": "plan",
            "stop_after_analyze": "stop_after_analyze",
        },
    )

    # Plan & Gate 2: plan_review
    builder.add_edge("plan", "plan_review")

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

    # --- Compile with persistence (SqliteSaver) ---
    conn = sqlite3.connect(CHECKPOINTS_DB, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = builder.compile(checkpointer=checkpointer)
    return graph, checkpointer


# Build once at import time
graph, checkpointer = build_graph()
