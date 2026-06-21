"""RAG evaluation runner.

Usage:
    python -m eval.run                     # retrieval-only metrics (no LLM cost)
    python -m eval.run --with-generation   # also generate answers + answer metrics
    python -m eval.run --judge             # + LLM-judged faithfulness/relevance

The retrieval stage uses only the local embedding model and vector store, so the
default run is free and fast. Generation and judging call the configured LLM and
are therefore opt-in.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from app.prompts import INSUFFICIENT_CONTEXT_MESSAGE
from app.rag import _build_context, _is_insufficient_answer, answer_question
from app.vector_store import search_documents

from . import judge as judge_mod
from . import metrics as M

logger = logging.getLogger(__name__)

_EVAL_DIR = Path(__file__).resolve().parent


def _ranked_sources(results: List[Dict[str, Any]]) -> List[str]:
    """De-duplicate retrieved chunk sources into a ranked, best-first list."""
    ranked: List[str] = []
    for r in results:
        src = str((r.get("metadata") or {}).get("source", "unknown"))
        if src not in ranked:
            ranked.append(src)
    return ranked


def _evaluate_case(case: Dict[str, Any], k: int, do_generation: bool, do_judge: bool) -> Dict[str, Any]:
    question = case["question"]
    expected = case.get("expected_sources", [])
    should_answer = bool(case.get("should_answer", True))

    # ── Retrieval stage (no LLM) ──────────────────────────────────────────────
    results = search_documents(question, n_results=k)
    retrieved = _ranked_sources(results)

    row: Dict[str, Any] = {
        "id": case["id"],
        "question": question,
        "should_answer": should_answer,
        "expected_sources": expected,
        "retrieved_sources": retrieved,
    }
    if should_answer:
        row["hit"] = M.hit_at_k(retrieved, expected)
        row["recall"] = M.recall_at_k(retrieved, expected)
        row["mrr"] = M.reciprocal_rank(retrieved, expected)

    # ── Generation stage (LLM) ────────────────────────────────────────────────
    if do_generation:
        t0 = time.perf_counter()
        resp = answer_question(question)
        row["latency_s"] = round(time.perf_counter() - t0, 2)
        answer = str(resp.get("answer", ""))
        row["answer"] = answer
        row["confidence"] = resp.get("confidence")
        row["confidence_score"] = resp.get("confidence_score")

        abstained = _is_insufficient_answer(answer)
        if should_answer:
            row["keyword_coverage"] = M.keyword_coverage(answer, case.get("expected_keywords", []))
            row["answered"] = not abstained  # an answerable question should NOT abstain
        else:
            # For out-of-scope questions, correctly abstaining is the win.
            row["correct_abstention"] = abstained

        if do_judge:
            context = _build_context(results)
            verdict = judge_mod.judge(question, context, answer)
            row["faithful"] = verdict["faithful"]
            row["relevant"] = verdict["relevant"]
            row["judge_reason"] = verdict["reason"]

    return row


def _aggregate(rows: List[Dict[str, Any]], do_generation: bool, do_judge: bool) -> Dict[str, Any]:
    answerable = [r for r in rows if r["should_answer"]]
    oos = [r for r in rows if not r["should_answer"]]

    summary: Dict[str, Any] = {
        "cases_total": len(rows),
        "answerable": len(answerable),
        "out_of_scope": len(oos),
        "retrieval": {
            "hit_at_k": M.mean([r.get("hit") for r in answerable]),
            "recall_at_k": M.mean([r.get("recall") for r in answerable]),
            "mrr": M.mean([r.get("mrr") for r in answerable]),
        },
    }

    if do_generation:
        summary["generation"] = {
            "keyword_coverage": M.mean([r.get("keyword_coverage") for r in answerable]),
            "answer_rate": M.mean([1.0 if r.get("answered") else 0.0 for r in answerable]),
            "abstention_accuracy": M.mean([1.0 if r.get("correct_abstention") else 0.0 for r in oos]) if oos else None,
            "avg_latency_s": M.mean([r.get("latency_s") for r in rows]),
        }
        if do_judge:
            judged = [r for r in rows if "faithful" in r]
            summary["judge"] = {
                "faithfulness": M.mean([r.get("faithful") for r in judged]),
                "relevance": M.mean([r.get("relevant") for r in judged]),
            }

    return summary


def _print_report(rows: List[Dict[str, Any]], summary: Dict[str, Any], do_generation: bool, do_judge: bool) -> None:
    print("\n" + "=" * 72)
    print("  HealthAI RAG — Evaluation Report")
    print("=" * 72)

    # Per-case retrieval table
    print(f"\n  {'ID':<24}{'hit':>5}{'rr':>6}{'recall':>8}   sources")
    print("  " + "-" * 68)
    for r in rows:
        if not r["should_answer"]:
            continue
        hit = "✓" if r.get("hit") else "✗"
        rr = f"{r.get('mrr', 0):.2f}"
        rec = M.pct(r.get("recall"))
        print(f"  {r['id']:<24}{hit:>5}{rr:>6}{rec:>8}   {','.join(r['retrieved_sources'][:3])}")

    ret = summary["retrieval"]
    mrr_str = "—" if ret["mrr"] is None else f"{ret['mrr']:.2f}"
    print("\n  Retrieval (answerable cases):")
    print(f"    Hit@k     : {M.pct(ret['hit_at_k'])}")
    print(f"    Recall@k  : {M.pct(ret['recall_at_k'])}")
    print(f"    MRR       : {mrr_str}")

    if do_generation:
        gen = summary["generation"]
        lat_str = "—" if gen["avg_latency_s"] is None else f"{gen['avg_latency_s']:.2f}s"
        print("\n  Generation:")
        print(f"    Answer rate (answerable did NOT abstain): {M.pct(gen['answer_rate'])}")
        print(f"    Keyword coverage                        : {M.pct(gen['keyword_coverage'])}")
        print(f"    Abstention accuracy (out-of-scope)      : {M.pct(gen['abstention_accuracy'])}")
        print(f"    Avg latency                             : {lat_str}")
        if do_judge:
            j = summary["judge"]
            print("\n  LLM judge:")
            print(f"    Faithfulness : {M.pct(j['faithfulness'])}")
            print(f"    Relevance    : {M.pct(j['relevance'])}")

    print("\n" + "=" * 72 + "\n")


def _write_reports(payload: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    s = payload["summary"]
    ret = s["retrieval"]
    mrr_str = "—" if ret["mrr"] is None else f"{ret['mrr']:.2f}"
    lines = [
        "# HealthAI RAG — Evaluation Report",
        "",
        f"_Generated: {payload['generated_at']} · k={payload['k']} · {s['cases_total']} cases_",
        "",
        "## Retrieval (no LLM)",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Hit@k | {M.pct(ret['hit_at_k'])} |",
        f"| Recall@k | {M.pct(ret['recall_at_k'])} |",
        f"| MRR | {mrr_str} |",
    ]
    if "generation" in s:
        g = s["generation"]
        lat_str = "—" if g["avg_latency_s"] is None else f"{g['avg_latency_s']:.2f}s"
        lines += [
            "",
            "## Generation",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Answer rate | {M.pct(g['answer_rate'])} |",
            f"| Keyword coverage | {M.pct(g['keyword_coverage'])} |",
            f"| Abstention accuracy | {M.pct(g['abstention_accuracy'])} |",
            f"| Avg latency | {lat_str} |",
        ]
    if "judge" in s:
        j = s["judge"]
        lines += [
            "",
            "## LLM judge",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Faithfulness | {M.pct(j['faithfulness'])} |",
            f"| Relevance | {M.pct(j['relevance'])} |",
        ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the HealthAI RAG pipeline.")
    parser.add_argument("--testset", default=str(_EVAL_DIR / "testset.json"), help="Path to the labeled test set JSON.")
    parser.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve (matches the app default).")
    parser.add_argument("--with-generation", action="store_true", help="Also generate answers and score them (uses the LLM).")
    parser.add_argument("--judge", action="store_true", help="Add LLM-judged faithfulness/relevance (implies --with-generation).")
    parser.add_argument("--out", default=str(_EVAL_DIR), help="Directory for report.json / report.md.")
    args = parser.parse_args()

    # Windows consoles default to cp1252, which can't encode the report glyphs.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s - %(message)s")

    do_judge = args.judge
    do_generation = args.with_generation or do_judge

    testset = json.loads(Path(args.testset).read_text(encoding="utf-8"))
    cases = testset["cases"]

    mode = "retrieval-only" if not do_generation else ("generation+judge" if do_judge else "generation")
    print(f"Running {len(cases)} cases · mode={mode} · k={args.k} …")

    rows: List[Dict[str, Any]] = []
    for i, case in enumerate(cases, start=1):
        print(f"  [{i}/{len(cases)}] {case['id']}", flush=True)
        rows.append(_evaluate_case(case, args.k, do_generation, do_judge))

    summary = _aggregate(rows, do_generation, do_judge)
    _print_report(rows, summary, do_generation, do_judge)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "k": args.k,
        "mode": mode,
        "summary": summary,
        "cases": rows,
    }
    _write_reports(payload, Path(args.out))
    print(f"Wrote {Path(args.out) / 'report.json'} and report.md")


if __name__ == "__main__":
    main()
