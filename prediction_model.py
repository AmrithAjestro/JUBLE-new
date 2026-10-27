import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from dataset_loader import LABELS, get_law_provisions, get_past_cases
from module.llm import chat, extract_json

MODEL_PATH = Path(__file__).resolve().parent / "models" / "legal_classifier.pkl"


# -----------------------
# ML model
# -----------------------
@lru_cache(maxsize=1)
def _load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run `python legal_classifier_train.py` first.")
    return joblib.load(MODEL_PATH)


def predict_with_ml(scenario_text: str) -> dict[str, Any]:
    """Returns {"predicted_decision": str, "confidence": float} from the trained classifier."""
    pipeline = _load_pipeline()
    probs = pipeline.predict_proba([scenario_text])[0]
    best = int(probs.argmax())
    label = pipeline.classes_[best]
    return {
        "predicted_decision": LABELS.get(label, str(label)),
        "confidence": round(float(probs[best]) * 100.0, 2),
    }


# -----------------------
# Prompt
# -----------------------
STYLE_GUIDE = """STYLE_GUIDE:
- Write like a senior Indian legal analyst advising a client after reviewing the facts.
- Use clear, assertive, professional language; avoid speculative filler.
- Do not mention any system, model, AI, or prompt.
- Explicitly connect: facts → legal ingredients of the applicable provision → likely outcome.

INDIAN LAW INTEGRATION:
- Cite only provisions that genuinely apply to these facts: the Constitution of India, criminal law, procedural law,
  evidence law, or the relevant special/civil statute.
- Criminal law: for offences committed on or after 1 July 2024 cite the Bharatiya Nyaya Sanhita (BNS),
  Bharatiya Nagarik Suraksha Sanhita (BNSS) and Bharatiya Sakshya Adhiniyam (BSA); for earlier offences cite the
  IPC, CrPC and Indian Evidence Act. If the date is unknown, cite the IPC section with its BNS equivalent.
- For writ / fundamental-rights matters, rely on the relevant Articles (e.g. 14, 19, 21, 22, 32, 226).
- Never cite foreign law or invent section numbers. If unsure of an exact section, name the Act without a number.

ANALYSIS:
- Identify the primary legal issue or offence suggested by the facts.
- Explain which essential ingredients are satisfied or missing, and the strongest supporting/opposing evidence.
- Use the similar past cases where they are genuinely on point; ignore ones that are not.
- Give practical next steps (bail, writ petition, appeal, discharge, evidence to gather) with the governing provision.

WRITING CONSTRAINTS:
- summary: 1–2 sentences. prediction: 1–2 decisive sentences.
- 3 reasons, each starting with one of: Considers, Relies, Notes, Finds, Observes, Highlights.
- Do not repeat the same point across reasons; the final reason must be the actionable next step.
- confidence must reflect how strongly the facts fit the law, not just the classifier score.

STRICT_OUTPUT_SCHEMA:
{
  "summary": "<concise factual summary>",
  "prediction": "<decisive likely outcome with the legal consequence>",
  "confidence": <integer 0-100>,
  "reasons": ["<reason>", "<reason>", "<actionable next step>"]
}

EXAMPLE_GOOD_JSON:
{
  "summary": "Accused was found with stolen property shortly after the theft and failed to prove lawful possession.",
  "prediction": "Court is likely to convict under Section 411 IPC (Section 317 BNS), as unexplained recent possession satisfies the ingredients of dishonestly receiving stolen property.",
  "confidence": 76,
  "reasons": [
    "Considers unexplained recent possession sufficient for Section 411 IPC, supported by the presumption under Section 114(a) of the Indian Evidence Act.",
    "Notes the defence can rebut the presumption with proof of ownership or by challenging the seizure procedure.",
    "Highlights the need to seek bail under Section 437 CrPC and assemble ownership documents before charges are framed."
  ]
}"""


def build_prompt(scenario: str, ml_result: dict[str, Any] | None,
                 num_cases: int = 5, num_laws: int = 5) -> str:
    """User message for the LLM: ML signal + scenario-relevant precedents and Articles + style guide."""
    cases = get_past_cases(num_cases, query=scenario)
    laws = get_law_provisions(num_laws, query=scenario)

    case_text = "\n".join(
        f"- {c['case_title']} | Issues: {c['issues'] or 'n/a'} | Outcome: {c['decision']}" for c in cases
    ) or "- (No similar cases found)"
    law_text = "\n".join(
        f"- Article {l['article']}: {l['title']} — {l['description']}" for l in laws
    ) or "- (No matching provisions found)"

    if ml_result:
        ml_block = (
            "Classifier signal (trained on past Supreme Court judgments; treat as a directional hint and "
            "deviate when the facts clearly point the other way):\n"
            f"- Predicted outcome: {ml_result['predicted_decision']}\n"
            f"- Classifier confidence: {ml_result['confidence']}%"
        )
    else:
        ml_block = "Classifier signal: unavailable — rely on the facts and law alone."

    return f"""Draft a precise, structured judgment prediction for the scenario below in strict JSON.

{ml_block}

Similar past cases (retrieved by textual similarity):
{case_text}

Possibly relevant Constitution of India provisions:
{law_text}

Scenario:
{scenario}

{STYLE_GUIDE}

Return ONLY the JSON object — no markdown, commentary, or extra text."""


SYSTEM_PROMPT = (
    "You are Jublie, a precise Indian legal analyst. "
    "Respond ONLY with strict JSON (no extra text). Keep it concise, plain-language, and legally sound."
)


# -----------------------
# Hybrid ML + LLM prediction
# -----------------------
def _clean(s) -> str:
    return re.sub(r"\s{2,}", " ", str(s or "").strip())


def predict_judgment(scenario: str) -> dict[str, Any]:
    """
    1) ML predicts outcome + confidence (skipped with a warning if the model is missing)
    2) LLM writes the structured analysis from the ML signal + retrieved cases and laws
    Returns {summary, prediction, confidence, reasons, ml, warnings}
    """
    warnings: list[str] = []
    try:
        ml_result = predict_with_ml(scenario)
    except Exception as e:
        ml_result = None
        warnings.append(f"ML classifier unavailable ({e}); prediction is based on the LLM analysis only.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(scenario, ml_result)},
    ]
    try:
        raw = chat(messages, temperature=0.15, max_tokens=2000)
        llm = extract_json(raw) or {
            "summary": "Unable to parse structured output from the model.",
            "prediction": raw[:400],
        }
    except Exception as e:
        if not ml_result:
            raise
        warnings.append(str(e))
        llm = {
            "summary": "Detailed reasoning could not be generated right now.",
            "prediction": ml_result["predicted_decision"],
            "reasons": [
                f"Classifier predicted '{ml_result['predicted_decision']}' with {ml_result['confidence']}% confidence.",
                "Consult the full statute text and recent precedents to confirm.",
            ],
        }

    try:
        confidence = float(llm.get("confidence"))
    except (TypeError, ValueError):
        confidence = ml_result["confidence"] if ml_result else 0
    reasons = llm.get("reasons") or []
    if not isinstance(reasons, list):
        reasons = [reasons]

    return {
        "summary": _clean(llm.get("summary")),
        "prediction": _clean(llm.get("prediction")),
        "confidence": max(0, min(100, int(round(confidence)))),
        "reasons": [_clean(r) for r in reasons if str(r).strip()],
        "ml": ml_result,
        "warnings": warnings,
    }


if __name__ == "__main__":
    text = "A citizen was detained without trial for 6 months under preventive detention laws."
    print(json.dumps(predict_judgment(text), indent=2, ensure_ascii=False))
