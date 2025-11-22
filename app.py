from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from agents.analyzer import Analyzer
from agents.planner import Planner
from agents.writer import Writer
from agents.critic import Critic
from tools.search_tool import WebSearcher
from tools.export_tool import save_docx, save_pdf

import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialisation des agents
analyzer = Analyzer()
planner = Planner()
writer = Writer()
critic = Critic()
searcher = WebSearcher()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate")
def generate(request: Request, topic: str = Form(...), use_search: str = Form("no")):
    # Étape 1 : analyse du sujet
    analysis = analyzer.run(topic)

    # Étape 2 : recherche web (optionnelle)
    sources = []
    if use_search.lower() == "yes":
        q = f"{topic} overview"
        sources = searcher.search(q)  # list of dicts {title,url,snippet}

    # Étape 3 : plan de l'essai
    plan = planner.run(analysis, sources=sources)

    # Étape 4 : génération du brouillon
    draft = writer.run(plan, sources=sources)

    # Étape 5 : critique automatique (non appliquée pour réécriture)
    critique = critic.run(draft)

    # Retourner la page avec le brouillon et le formulaire de validation humaine
    return templates.TemplateResponse("result.html", {
        "request": request,
        "topic": topic,
        "analysis": analysis,
        "plan": plan,
        "draft": draft,
        "critique": critique,
        "sources": sources
    })


@app.post("/finalize")
def finalize(
    request: Request,
    topic: str = Form(...),
    plan: str = Form(...),
    draft: str = Form(...),
    hints: str = Form("")
):
    # Générer la version finale avec les suggestions humaines
    final_draft = writer.run(plan, sources=None, hints=hints)
    critique = critic.run(final_draft)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "topic": topic,
        "analysis": "Validation humaine appliquée",
        "plan": plan,
        "draft": final_draft,
        "critique": critique,
        "sources": []
    })


@app.post("/download/docx")
def download_docx(topic: str = Form(...), content: str = Form(...)):
    out = save_docx(topic, content)
    return FileResponse(
        out,
        filename=os.path.basename(out),
        media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    )


@app.post("/download/pdf")
def download_pdf(topic: str = Form(...), content: str = Form(...)):
    out = save_pdf(topic, content)
    return FileResponse(
        out,
        filename=os.path.basename(out),
        media_type='application/pdf'
    )


@app.get("/health")
def health():
    return {"status": "ok"}

