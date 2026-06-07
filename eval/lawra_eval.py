#!/usr/bin/env python3
"""
Lawra RAGAS Evaluation Harness
================================
Evaluates retrieval + generation quality against a curated golden Q&A dataset.
Uses local Ollama for both the LLM judge and embeddings — no OpenAI required.

Pipeline per question:
  1. Embed question  →  Ollama (nomic-embed-text)
  2. Retrieve chunks →  Qdrant (top-5, cosine)
  3. Generate answer →  Ollama (llama3.1:8b)
  4. Score           →  Ragas (faithfulness, answer_relevancy,
                                context_precision, context_recall)

Outputs:
  • Terminal: coloured summary table
  • JSON:     <output_dir>/lawra_eval_<timestamp>.json
  • Symlink:  frontend/eval_results.json → latest run  (for the UI Evaluate tab)

Honest caveats:
  - llama3.1:8b judging llama3.1:8b is a weak signal. Use a different/stronger
    judge model if available (pass --judge-model gemma4:e2b).
  - Full run (15 Qs × 4 metrics) takes 15–45 min locally on M4.
    Use --quick (5 Qs) for a fast sanity check.
  - Ground truths in golden_dataset.json must be verified against SSO.

Usage:
  python lawra_eval.py                          # all 15 questions
  python lawra_eval.py --quick                  # first 5 questions
  python lawra_eval.py --questions 8            # first 8 questions
  python lawra_eval.py --judge-model gemma4:e2b # different judge LLM
  python lawra_eval.py --no-context-recall      # skip slow recall metric
  python lawra_eval.py --output-dir ../results  # custom output location

Requirements:
  pip install ragas langchain-ollama datasets qdrant-client requests
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Config (override via CLI or env) ─────────────────────────────────────────
OLLAMA_URL   = os.environ.get("OLLAMA_URL",   "http://localhost:11434")
QDRANT_URL   = os.environ.get("QDRANT_URL",   "http://localhost:6333")
EMBED_MODEL  = os.environ.get("EMBED_MODEL",  "nomic-embed-text")
LLM_MODEL    = os.environ.get("LLM_MODEL",    "llama3.1:8b")
COLLECTION   = os.environ.get("QDRANT_COLLECTION", "lawra")
TOP_K        = 5
TEMPERATURE  = 0.1

GOLDEN_FILE   = Path(__file__).parent / "golden_dataset.json"
OUTPUT_DIR    = Path(__file__).parent / "results"
FRONTEND_LINK = Path(__file__).parent.parent / "frontend" / "eval_results.json"

SYSTEM_PROMPT = (
    "You are Lawra, a Singapore road traffic law assistant. "
    "Answer the question using ONLY the legal text provided. "
    "Cite the source using [Document, Section]. "
    "If the answer is not in the text, say exactly: "
    "\"I cannot find a definitive answer in the provided legislation.\""
)

# ── Colours ───────────────────────────────────────────────────────────────────
RED   = "\033[91m"
GRN   = "\033[92m"
YEL   = "\033[93m"
BLU   = "\033[94m"
DIM   = "\033[2m"
BOLD  = "\033[1m"
RST   = "\033[0m"


def score_color(v: float) -> str:
    if v >= 0.8:  return f"{GRN}{v:.3f}{RST}"
    if v >= 0.6:  return f"{YEL}{v:.3f}{RST}"
    return f"{RED}{v:.3f}{RST}"


# ── Ollama helpers ────────────────────────────────────────────────────────────
def embed(text: str, ollama_url: str = OLLAMA_URL) -> list[float]:
    import requests
    r = requests.post(f"{ollama_url}/api/embeddings",
                      json={"model": EMBED_MODEL, "prompt": text}, timeout=120)
    r.raise_for_status()
    return r.json()["embedding"]


def generate(question: str, context: str, ollama_url: str = OLLAMA_URL,
             model: str = LLM_MODEL) -> str:
    import requests
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_ctx": 4096},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Question: {question}\n\nRetrieved legal text:\n{context}"},
        ],
    }
    r = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"]


# ── Qdrant retrieval ──────────────────────────────────────────────────────────
def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    import requests
    vec = embed(question)
    payload = {"vector": vec, "limit": top_k, "with_payload": True, "with_vector": False}
    r = requests.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
                      json=payload, timeout=30)
    r.raise_for_status()
    return r.json().get("result", [])


def build_context(hits: list[dict]) -> tuple[str, list[str]]:
    """Return (formatted_context_for_prompt, list_of_chunk_texts_for_ragas)."""
    ctx_str, ctx_list = "", []
    for i, h in enumerate(hits, 1):
        p = h.get("payload", {})
        chunk = p.get("text", "")
        label = p.get("section_label", "")
        title = p.get("title", "")
        ctx_str  += f"\n\n--- Source {i} [{title} — {label}] ---\n{chunk}"
        ctx_list.append(chunk)
    return ctx_str.strip(), ctx_list


# ── Ragas setup ───────────────────────────────────────────────────────────────
def make_ragas_components(judge_model: str):
    """Return (llm_wrapper, emb_wrapper) for Ragas, using local Ollama."""
    try:
        from langchain_ollama import OllamaLLM, OllamaEmbeddings
    except ImportError:
        print(f"{RED}Install langchain-ollama: pip install langchain-ollama{RST}")
        sys.exit(1)

    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError:
        print(f"{RED}Install ragas: pip install ragas{RST}")
        sys.exit(1)

    llm = LangchainLLMWrapper(
        OllamaLLM(model=judge_model, base_url=OLLAMA_URL, temperature=0.0,
                  timeout=600))    # 10 min — local 7-8B models are slow on long prompts
    emb = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL))
    return llm, emb


def run_ragas(samples: list[dict], judge_model: str,
              skip_recall: bool = False) -> dict[str, float]:
    """Run Ragas evaluate() and return metric scores dict."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset
    except ImportError as e:
        print(f"{RED}Import error: {e}{RST}")
        print("Run: pip install ragas datasets")
        sys.exit(1)

    llm, emb = make_ragas_components(judge_model)

    metrics = [faithfulness, answer_relevancy, context_precision]
    if not skip_recall:
        try:
            from ragas.metrics import context_recall
            metrics.append(context_recall)
        except ImportError:
            print(f"{YEL}context_recall not available in this Ragas version — skipping{RST}")

    # Configure all metrics with local LLM + embeddings
    for m in metrics:
        m.llm = llm
        if hasattr(m, "embeddings"):
            m.embeddings = emb

    # Build HuggingFace Dataset — try both old and new Ragas schemas
    try:
        # Ragas ≥0.2 schema
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        dataset = EvaluationDataset(samples=[
            SingleTurnSample(
                user_input   = s["question"],
                response     = s["answer"],
                retrieved_contexts = s["contexts"],
                reference    = s["ground_truth"],
            )
            for s in samples
        ])
    except ImportError:
        # Ragas 0.1.x schema
        data = {
            "question":     [s["question"]    for s in samples],
            "answer":       [s["answer"]      for s in samples],
            "contexts":     [s["contexts"]    for s in samples],
            "ground_truths":[[s["ground_truth"]] for s in samples],
        }
        dataset = Dataset.from_dict(data)

    print(f"\n{BLU}Running Ragas evaluation (judge: {judge_model})…{RST}")
    print(f"{DIM}This may take {len(samples) * len(metrics) * 3 // 60 + 1}–"
          f"{len(samples) * len(metrics) * 15 // 60 + 5} minutes.{RST}\n")

    try:
        from ragas import RunConfig
        run_cfg = RunConfig(timeout=600, max_retries=2, max_wait=120)
    except ImportError:
        run_cfg = None

    result = evaluate(dataset, metrics=metrics,
                      **({'run_config': run_cfg} if run_cfg else {}))

    # Normalise result to dict[str, float]
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        scores = {col: float(df[col].mean()) for col in df.columns
                  if col not in ("question","answer","contexts","ground_truths",
                                 "user_input","response","retrieved_contexts","reference")}
    else:
        scores = dict(result)

    # Per-question scores
    per_q = []
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        metric_cols = [c for c in df.columns
                       if c not in ("question","answer","contexts","ground_truths",
                                    "user_input","response","retrieved_contexts","reference")]
        for i, row in df.iterrows():
            q = samples[i] if i < len(samples) else {}
            per_q.append({
                "question":     q.get("question",""),
                "answer":       q.get("answer",""),
                "ground_truth": q.get("ground_truth",""),
                "sources":      q.get("sources",[]),
                **{col: float(row[col]) if row[col] == row[col] else None for col in metric_cols},
            })

    return scores, per_q


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(golden: list[dict], judge_model: str,
                 skip_recall: bool) -> list[dict]:
    """Run embed→retrieve→generate for each question. Returns samples list."""
    samples = []
    total = len(golden)

    for i, item in enumerate(golden, 1):
        q   = item["question"]
        gt  = item["ground_truth"]
        qid = item.get("id", f"q{i:02d}")

        print(f"{BOLD}[{i}/{total}]{RST} {q[:80]}")
        t0 = time.time()

        try:
            hits             = retrieve(q)
            ctx_str, ctx_list = build_context(hits)
            answer            = generate(q, ctx_str, model=LLM_MODEL)
            elapsed           = time.time() - t0

            sources = [{
                "rank":  j,
                "score": round(h.get("score", 0), 4),
                "title": h.get("payload", {}).get("title", ""),
                "section_label": h.get("payload", {}).get("section_label", ""),
                "source_url": h.get("payload", {}).get("source_url", ""),
            } for j, h in enumerate(hits, 1)]

            print(f"  {DIM}Retrieved {len(hits)} chunks · "
                  f"Generated in {elapsed:.1f}s{RST}")

            samples.append({
                "id":          qid,
                "question":    q,
                "answer":      answer,
                "contexts":    ctx_list,
                "ground_truth": gt,
                "sources":     sources,
                "latency_s":   round(elapsed, 2),
            })
        except Exception as e:
            print(f"  {RED}Error: {e}{RST}")
            samples.append({
                "id": qid, "question": q, "answer": f"ERROR: {e}",
                "contexts": [], "ground_truth": gt,
                "sources": [], "latency_s": 0,
            })

    return samples


# ── Output ────────────────────────────────────────────────────────────────────
def save_results(samples: list[dict], scores: dict[str, float],
                 per_q: list[dict], judge_model: str,
                 output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = output_dir / f"lawra_eval_{ts}.json"

    payload = {
        "run_id":      ts,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "llm_model":   LLM_MODEL,
        "judge_model": judge_model,
        "embed_model": EMBED_MODEL,
        "collection":  COLLECTION,
        "num_questions": len(samples),
        "scores":      {k: round(float(v), 4) for k, v in scores.items()},
        "per_question": per_q if per_q else [{
            "question":     s["question"],
            "answer":       s["answer"],
            "ground_truth": s["ground_truth"],
            "sources":      s["sources"],
        } for s in samples],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{GRN}Results saved → {out_path}{RST}")

    # Symlink for the UI Evaluate tab
    try:
        if FRONTEND_LINK.exists() or FRONTEND_LINK.is_symlink():
            FRONTEND_LINK.unlink()
        FRONTEND_LINK.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(out_path, FRONTEND_LINK)
        print(f"{GRN}UI results updated → {FRONTEND_LINK}{RST}")
    except Exception as e:
        print(f"{YEL}Could not update frontend link: {e}{RST}")

    return out_path


def print_summary(scores: dict[str, float], samples: list[dict]) -> None:
    avg_latency = sum(s.get("latency_s", 0) for s in samples) / max(len(samples), 1)
    print(f"\n{'─'*52}")
    print(f"{BOLD}{'METRIC':<28} {'SCORE':>8}  {'INTERPRETATION'}{RST}")
    print(f"{'─'*52}")
    labels = {
        "faithfulness":       "Hallucination resistance",
        "answer_relevancy":   "Answer addresses question",
        "context_precision":  "Best chunks ranked first",
        "context_recall":     "Relevant chunks retrieved",
    }
    for metric, label in labels.items():
        score = scores.get(metric)
        if score is not None:
            print(f"  {metric:<26} {score_color(score):>8}  {DIM}{label}{RST}")
    print(f"{'─'*52}")
    print(f"  {'Questions evaluated':<26} {len(samples):>8}")
    print(f"  {'Avg retrieval+gen time':<26} {avg_latency:>7.1f}s")
    print(f"{'─'*52}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Lawra Ragas evaluation harness")
    ap.add_argument("--quick",         action="store_true", help="Run first 5 questions only")
    ap.add_argument("--questions",     type=int,  default=None, help="Run first N questions")
    ap.add_argument("--judge-model",   default=LLM_MODEL, help="Ollama model to use as judge")
    ap.add_argument("--no-context-recall", action="store_true", help="Skip context_recall (slowest metric)")
    ap.add_argument("--golden",        type=Path, default=GOLDEN_FILE, help="Path to golden_dataset.json")
    ap.add_argument("--output-dir",    type=Path, default=OUTPUT_DIR,  help="Directory for result JSON files")
    ap.add_argument("--pipeline-only", action="store_true", help="Run retrieve+generate only, skip Ragas scoring")
    args = ap.parse_args()

    if not args.golden.exists():
        print(f"{RED}Golden dataset not found: {args.golden}{RST}")
        sys.exit(1)

    golden = json.loads(args.golden.read_text(encoding="utf-8"))

    n = 5 if args.quick else (args.questions or len(golden))
    golden = golden[:n]

    print(f"\n{BOLD}Lawra Evaluation{RST}")
    print(f"  Questions   : {len(golden)}")
    print(f"  LLM         : {LLM_MODEL}")
    print(f"  Judge       : {args.judge_model}")
    print(f"  Embeddings  : {EMBED_MODEL}")
    print(f"  Collection  : {COLLECTION}")
    if args.quick:
        print(f"  {YEL}Quick mode — running {len(golden)} questions{RST}")
    print()

    # Step 1: Run the pipeline
    print(f"{BOLD}Step 1/2 — Running pipeline{RST}")
    samples = run_pipeline(golden, args.judge_model, args.no_context_recall)

    if args.pipeline_only:
        out = args.output_dir / f"pipeline_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(samples, indent=2, ensure_ascii=False))
        print(f"\n{GRN}Pipeline output saved → {out}{RST}")
        return

    # Step 2: Ragas scoring
    print(f"\n{BOLD}Step 2/2 — Ragas scoring{RST}")
    scores, per_q = run_ragas(samples, args.judge_model, args.no_context_recall)

    # Print + save
    print_summary(scores, samples)
    save_results(samples, scores, per_q, args.judge_model, args.output_dir)


if __name__ == "__main__":
    main()
