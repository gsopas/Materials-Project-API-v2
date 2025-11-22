import os
import json
from typing import Optional, List, Any, Dict

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# --------- CONFIG ---------

MP_API_KEY = os.getenv("MP_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not MP_API_KEY:
    raise RuntimeError("MP_API_KEY is not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

MP_BASE_URL = "https://api.materialsproject.org"

FRONTEND_ORIGIN = "https://gsopas.github.io"

# --------- APP ---------

app = FastAPI(title="Materials Explainer API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- MODELS ---------

class MaterialsQuery(BaseModel):
    chemsys: Optional[str] = None
    formula: Optional[str] = None
    limit: int = 20

class ExplainRequest(BaseModel):
    raw_data: dict
    question: Optional[str] = None

# --------- HELPERS ---------

def mp_request(endpoint: str, params: Dict[str, Any]) -> List[dict]:
    headers = {"X-API-KEY": MP_API_KEY}
    url = f"{MP_BASE_URL}{endpoint}"
    r = requests.get(url, params=params, headers=headers, timeout=20)
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    data = r.json()
    if "data" not in data:
        raise HTTPException(status_code=500, detail="Unexpected response from Materials Project")
    return data["data"]

def search_by_chemsys(chemsys: str, limit: int = 20) -> List[dict]:
    params = {
        "chemsys": chemsys,
        "limit": limit,
        "fields": (
            "material_id,formula_pretty,chemsys,band_gap,density,"
            "is_stable,energy_above_hull,nelements"
        ),
    }
    return mp_request("/materials/summary/", params)

def search_by_formula(formula: str, limit: int = 20) -> List[dict]:
    params = {
        "formula": formula,
        "limit": limit,
        "fields": (
            "material_id,formula_pretty,chemsys,band_gap,density,"
            "is_stable,energy_above_hull,nelements"
        ),
    }
    return mp_request("/materials/summary/", params)

# --------- ROUTES ---------

@app.get("/")
def root():
    return {"status": "ok", "message": "Materials Explainer backend running"}

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/api/materials")
def get_materials(q: MaterialsQuery):
    if not q.chemsys and not q.formula:
        raise HTTPException(status_code=400, detail="Provide chemsys or formula")
    try:
        if q.chemsys:
            data = search_by_chemsys(q.chemsys, q.limit)
        else:
            data = search_by_formula(q.formula, q.limit)
        return {"data": data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/explain")
def explain_material(req: ExplainRequest):
    system_prompt = (
        "You are a materials scientist. "
        "Explain Materials Project JSON data to a smart non-expert. "
        "Use clear language, but keep the scientific meaning accurate. "
        "Briefly explain what the key properties imply (e.g. band gap, stability). "
        "If the user asks a follow-up question, answer using the JSON context."
    )

    user_prompt = "Here is a materials JSON entry:\n\n"
    user_prompt += json.dumps(req.raw_data, indent=2)

    if req.question:
        user_prompt += f"\n\nUser question: {req.question}\n"
    else:
        user_prompt += (
            "\n\nExplain the most important properties, "
            "their approximate numerical values, and what they mean."
        )

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.4,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = completion.choices[0].message.content
    return {"answer": answer}
