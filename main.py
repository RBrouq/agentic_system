from src import run_essay_graph


def main() -> None:
    # Exemple simple : essai en 1 shot avec skip de tous les HITL
    user_input = "Write a structured essay about agentic AI systems in education."

    result = run_essay_graph(
        user_input=user_input,
        skip_clarification=True,
        skip_plan_review=True,
        skip_draft_review=True,
    )

    print("\n=== ANSWER ===\n")
    print(result.get("answer", "No answer in state."))
    print("\n=== METADATA ===")
    print("Thread ID:", result.get("thread_id"))
    print("Mode:", result.get("mode"))
    print("Saved:", result.get("saved"))


if __name__ == "__main__":
    main()
