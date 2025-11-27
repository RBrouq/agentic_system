# Essay Agent — LangGraph HITL Workflow (FastAPI UI)

This project provides a full **Agentic Essay Writing System** powered by:

- **LangGraph** (workflow orchestration)  
- **OpenAI + LangChain** (LLM reasoning & generation)  
- **FastAPI** (backend + UI)  
- **Human-in-the-loop (HITL)** checkpoints  
- **DOCX & PDF export**  

It supports multi-step essay generation with optional human feedback at every stage.

---

## Features

### Multi-stage Essay Workflow
- Intent classification (essay vs open question)
- Topic extraction
- Clarification questions (HITL)
- Outline creation + human review (HITL)
- Agentic research (Tavily or LLM fallback)
- Draft writing + critique (HITL)
- Finalization & polishing

### Human-In-The-Loop (HITL)
The essay writing process has 3 main HITL checkpoints:

1. **Clarification of instructions** (after topic analysis)  
2. **Plan/outline feedback** (after planning)  
3. **Draft approval** (after critique)

For each step you can either:

- Provide human inputs (clarifications / feedback / approval), or  
- Skip the step and let the AI decide everything.

### Persistent Sessions
Each run uses a **thread_id**, and the whole state is saved using a `SqliteSaver`.  
You can resume the same essay workflow by calling the API again with the same `thread_id`.

### Export Tools
- Export final essay as **DOCX**
- Export final essay as **PDF**

---

## Project Structure

```text
agentic_system/
├── main.py                    # FastAPI app (UI + API)
├── src/
│   ├── __init__.py
│   ├── config.py              # Loads .env and configures OpenAI LLM
│   ├── state.py               # EssayState definition
│   ├── llm_utils.py           # call_llm helper
│   ├── nodes.py               # All workflow nodes
│   ├── graph_builder.py       # Builds graph + persistence
│   └── runner.py              # run_essay_graph() public API
├── checkpoints.sqlite         # LangGraph persistent memory
├── static/                    # CSS/JS (for frontend)
├── templates/
│   └── index.html             # Basic UI
├── .env                       # OPENAI_API_KEY
└── README.md
```

---

## Installation

### 1️⃣ Clone the project

```bash
git clone https://github.com/RBrouq/agentic_system
cd agentic_system
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add your `.env`

At the root of the project:

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Make sure this is a valid OpenAI API key with access to the Chat Completions API.

---

## ▶️ Running the App

From the project root:

```bash
python -m uvicorn app:app --reload
```

Then open your browser at:

http://127.0.0.1:8000/

---

## 🧠 The HITL Workflow Explained

The LangGraph DAG for essay mode roughly follows this structure:

```text
START
  ↓
classify_intent          # essay vs open_question
  ↓
analyze_topic ───────→ stop_after_analyze        (HITL #1: clarifications)
  ↓
plan_essay
  ↓
plan_review ─────────→ stop_after_plan_review   (HITL #2: outline feedback)
  ↓
research_agentic
  ↓
write_draft
  ↓
critic_node ─────────→ stop_after_critic        (HITL #3: draft approval)
  ↓
save_to_db
  ↓
finalize
  ↓
END
```

For **open question** mode, the workflow simply runs a one-shot LLM reply via `basic_reply` and ends.

At each HITL stop, the graph returns the current state and **does not proceed further**.  
You can then call the API again with the same `thread_id` and the additional human inputs to continue.

---

## 📡 API Endpoints

### POST `/api/run`

Runs the next step of the essay workflow.

**Form fields**:

| Field                 | Type      | Description                                    |
|-----------------------|-----------|------------------------------------------------|
| `prompt`              | string    | User message / essay request                   |
| `thread_id`           | string?   | Continue an existing thread                    |
| `clarification_answers` | string? | Answers to clarification questions             |
| `plan_feedback`       | string?   | Feedback on the outline                        |
| `draft_feedback_human`| string?   | Feedback on the draft                          |
| `draft_approved`      | checkbox? | Mark draft as approved (HITL #3)              |
| `final_feedback`      | string?   | Optional final improvements                    |
| `skip_clarification`  | checkbox? | Skip HITL #1                                   |
| `skip_plan_review`    | checkbox? | Skip HITL #2                                   |
| `skip_draft_review`   | checkbox? | Skip HITL #3                                   |

**Response**: JSON containing the current `EssayState`, including:

- `thread_id`
- `mode`
- `topic`
- `instructions`
- `clarification_questions`
- `clarification_answers`
- `plan`
- `plan_validated`
- `research_notes`
- `draft`
- `critique`
- `final_draft`
- `answer`
- `saved`
- `final_approved`

You can inspect these fields to know **where** in the workflow you currently are and what to show in the UI.

---

### POST `/api/export/docx`

Exports a DOCX file from the final answer.

**Form fields**:

- `answer` — essay text  
- `topic` — title for the document (default: `"Essay"`)

**Response**: `.docx` file via `FileResponse`.

The file name is derived from `topic` (slugified), e.g. `My_essay.docx`.

---

### POST `/api/export/pdf`

Exports a PDF file from the final answer.

**Form fields**:

- `answer` — essay text  
- `topic` — title on the first page

**Response**: `.pdf` file via `FileResponse`.

The PDF is generated using ReportLab with basic line wrapping and multiple-page support.

---

## 🧪 Testing the Workflow in Python

You can directly use the core function from Python (outside of FastAPI):

```python
from src.runner import run_essay_graph

# Simple full-automatic example (AI skips all HITL steps)
result = run_essay_graph(
    "Write an essay about agentic AI systems in education.",
    skip_clarification=True,
    skip_plan_review=True,
    skip_draft_review=True,
)

print("Mode:", result.get("mode"))
print("Topic:", result.get("topic"))
print("Answer:
", result.get("answer"))
```

To use the HITL steps, call `run_essay_graph` multiple times with the same `thread_id` and the appropriate feedback fields.

---

## 🔧 Tech Stack

| Component   | Technology                          |
|------------|--------------------------------------|
| Backend    | FastAPI                              |
| Workflow   | LangGraph                            |
| LLM        | LangChain + OpenAI (`ChatOpenAI`)    |
| Templates  | Jinja2                               |
| Exports    | `python-docx`, `reportlab`           |
| Persistence| SQLite via `LangGraph SqliteSaver`   |

