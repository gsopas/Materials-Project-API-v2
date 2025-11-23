import os
import json
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from mp_api.client import MPRester

# -------------------------------------------------------------------
# CONFIG / CLIENTS
# -------------------------------------------------------------------

MP_API_KEY = os.getenv("MP_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not MP_API_KEY:
    raise RuntimeError("MP_API_KEY is not set in environment variables")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment variables")

# Materials Project v2 client
mpr = MPRester(MP_API_KEY)

# OpenAI client
oa_client = OpenAI(api_key=OPENAI_API_KEY)

FRONTEND_ORIGIN = "https://gsopas.github.io"  # GitHub Pages root

# -------------------------------------------------------------------
# FASTAPI APP
# -------------------------------------------------------------------

app = FastAPI(
    title="Materials Explainer API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------

class MaterialsQuery(BaseModel):
    chemsys: Optional[str] = None  # e.g. "Li-Fe-O"
    formula: Optional[str] = None  # e.g. "LiFePO4"
    limit: int = 20


class ExplainRequest(BaseModel):
    raw_data: Dict[str, Any]
    question: Optional[str] = None


# -------------------------------------------------------------------
# Helpers using MPRester (Materials Project v2)
# -------------------------------------------------------------------

def search_by_chemsys(chemsys: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Query Materials Project by chemical system, return a list of plain dicts.
    """
    try:
        docs = mpr.materials.summary.search(
            chemsys=chemsys,
            fields=[
                "material_id",
                "formula_pretty",
                "chemsys",
                "band_gap",
                "density",
                "is_stable",
            ],
        )
        return [d.model_dump() for d in docs[:limit]]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Materials Project error: {exc}")


def search_by_formula(formula: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Query Materials Project by formula, return a list of plain dicts.
    """
    try:
        docs = mpr.materials.summary.search(
            formula=formula,
            fields=[
                "material_id",
                "formula_pretty",
                "chemsys",
                "band_gap",
                "density",
                "is_stable",
            ],
        )
        return [d.model_dump() for d in docs[:limit]]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Materials Project error: {exc}")


# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "message": "Materials Explainer backend running"}


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/api/materials")
def get_materials(q: MaterialsQuery):
    """
    POST body: { "chemsys": "Li-Fe-O", "formula": null, "limit": 20 }
    Either 'chemsys' or 'formula' must be provided.
    """
    if not q.chemsys and not q.formula:
        raise HTTPException(status_code=400, detail="Provide 'chemsys' or 'formula'")

    try:
        if q.chemsys:
            data = search_by_chemsys(q.chemsys, q.limit)
        else:
            data = search_by_formula(q.formula, q.limit)
        return {"data": data}
    except HTTPException:
        # re-raise HTTPException from helpers
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/explain")
def explain_material(req: ExplainRequest):
    """
    Given a single materials JSON entry (from /api/materials) and an optional
    natural-language question, return an explanation from the LLM.
    """
    system_prompt = (
        "You are a materials scientist. "
        "You are given JSON data from the Materials Project v2 API. "
        "Explain it to a smart non-expert. "
        "Use clear language but keep the scientific meaning accurate. "
        "Highlight formula, chemical system, band gap (insulator vs semiconductor vs metal), "
        "density, and stability, and briefly relate these to possible applications."
    )

    user_prompt = "Here is one materials JSON entry:\n\n"
    user_prompt += json.dumps(req.raw_data, indent=2)

    if req.question:
        user_prompt += f"\n\nUser question: {req.question}\n"
    else:
        user_prompt += (
            "\n\nExplain the most important properties, their approximate values, "
            "and what they imply physically and technologically."
        )

    try:
        completion = oa_client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.4,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = completion.choices[0].message.content
        return {"answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {exc}")
