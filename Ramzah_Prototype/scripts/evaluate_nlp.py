"""Evaluate model outputs on held-out semantic sets, including abstentions."""

import argparse
import json
import time
from pathlib import Path
from statistics import median

import sacrebleu

from ramzah.nlp import GlossRealizer
from ramzah.semantics import validate_candidate

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="checkpoints/nlp/contextual_v3")
parser.add_argument("--output", default="reports/nlp/contextual_evaluation.json")
args = parser.parse_args()
records = [
    json.loads(line)
    for line in Path("data/nlp/test.jsonl").read_text(encoding="utf8").splitlines()
]
by_group = {}
for row in records:
    by_group.setdefault(row["group_id"], row)
model = GlossRealizer(args.checkpoint)
details = []
for group, row in sorted(by_group.items()):
    t = time.perf_counter()
    try:
        output = model.generate(row["glosses"])
    except ValueError as error:
        output = dict(
            input=row["glosses"],
            candidates=[],
            rejected_count=0,
            uncertainty=True,
            error=str(error),
        )
    elapsed = time.perf_counter() - t
    first = output["candidates"][0] if output["candidates"] else ""
    review = validate_candidate(row["glosses"], first, model.concepts)
    details.append(
        dict(
            group_id=group,
            glosses=row["glosses"],
            reference=row["target"],
            first=first,
            candidates=output["candidates"],
            accepted=bool(first),
            concept_preserved=review["valid"],
            missing=review["missing"],
            added=review["added"],
            rejected_generations=output["rejected_count"],
            latency_seconds=elapsed,
            failure=output.get("error"),
        )
    )
    print(
        group,
        json.dumps(dict(first=first, seconds=round(elapsed, 2)), ensure_ascii=False),
        flush=True,
    )
valid = [r for r in details if r["accepted"]]
chrf = (
    sacrebleu.corpus_chrf(
        [r["first"] for r in valid], [[r["reference"] for r in valid]]
    ).score
    if valid
    else 0
)
summary = dict(
    n_groups=len(details),
    candidate_coverage=sum(r["accepted"] for r in details) / len(details),
    concept_preservation_all=sum(r["concept_preserved"] for r in details)
    / len(details),
    missing_slot_rate=sum(bool(r["missing"]) for r in details) / len(details),
    added_slot_rate=sum(bool(r["added"]) for r in details) / len(details),
    chrf_among_covered=chrf,
    median_latency_seconds=median(r["latency_seconds"] for r in details),
    p95_latency_seconds=sorted(r["latency_seconds"] for r in details)[
        int(0.95 * (len(details) - 1))
    ],
    first_choice_user_acceptance="NOT MEASURED: requires real user choices",
    semantic_similarity="NOT MEASURED: no independent Arabic sentence-embedding validation",
    human_meaning_preservation="NOT MEASURED: requires Deaf-user/clinical review",
    note="Synthetic held-out exact input-order/target pairs; automated concept-preservation is narrower than semantic equivalence",
)
out = Path(args.output)
out.parent.mkdir(exist_ok=True, parents=True)
out.write_text(
    json.dumps(dict(summary=summary, samples=details), ensure_ascii=False, indent=2),
    encoding="utf8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
