"""Fine-tuned AraT5 inference; output is always filtered for retained concepts."""

import json
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from .semantics import validate_candidate

ROOT = Path(__file__).resolve().parents[1]


class GlossRealizer:
    def __init__(self, checkpoint=None):
        root = Path(checkpoint or ROOT / "checkpoints/nlp/contextual_v3")
        self.tokenizer = T5Tokenizer.from_pretrained(root, legacy=True)
        self.model = T5ForConditionalGeneration.from_pretrained(root).eval()
        self.ids = json.loads((root / "retained_ids.json").read_text(encoding="utf8"))
        self.reverse = {original: compact for compact, original in enumerate(self.ids)}
        self.concepts = json.loads(
            (ROOT / "data/nlp/concepts.json").read_text(encoding="utf8")
        )

    def encode(self, text):
        old = self.tokenizer(text, add_special_tokens=True)["input_ids"]
        missing = [x for x in old if x not in self.reverse]
        if missing:
            raise ValueError(
                "Unsupported gloss spelling/token; no safe NLP interpretation"
            )
        return torch.tensor([[self.reverse[x] for x in old]])

    @torch.inference_mode()
    def generate(self, recognized_words, max_candidates=3):
        """Always return neural text; validation ranks outputs but never writes text."""
        if not recognized_words or any(
            w not in self.concepts for w in recognized_words
        ):
            raise ValueError("No supported gloss sequence")
        candidates = []
        attempts = []
        model_input = " ".join(recognized_words)
        input_ids = self.encode(
            model_input
        )  # tokenizer output, remapped to compact model vocabulary
        beam_ids = self.model.generate(
            input_ids,
            max_new_tokens=48,
            num_beams=8,
            num_return_sequences=8,
            no_repeat_ngram_size=3,
            early_stopping=True,
            use_cache=False,
        )
        sampled_ids = self.model.generate(
            input_ids,
            max_new_tokens=48,
            do_sample=True,
            top_p=0.92,
            temperature=0.8,
            num_return_sequences=8,
            no_repeat_ngram_size=3,
            use_cache=False,
        )
        for sequence in list(beam_ids) + list(sampled_ids):
            original = [self.ids[int(x)] for x in sequence]
            candidate = self.tokenizer.decode(
                original, skip_special_tokens=True
            ).strip()
            if not candidate:
                continue
            review = validate_candidate(recognized_words, candidate, self.concepts)
            attempts.append(dict(candidate=candidate, **review))
            if review["valid"] and candidate not in candidates:
                candidates.append(candidate)
            if len(candidates) >= max_candidates:
                break
        strictly_valid = bool(candidates)
        if not candidates:
            # The validator is a ranker here, never a fallback sentence generator.
            # Prefer candidates with the fewest missing/added concepts and no hard-risk flags.
            hard = {
                "unsupported_clinical_claim",
                "invented_number",
                "unsupported_negation",
                "malformed_generation",
            }

            def penalty(attempt):
                return (
                    100 * bool(hard.intersection(attempt["reasons"]))
                    + 12 * len(attempt["missing"])
                    + 12 * len(attempt["added"])
                    + 4 * len(attempt["repeated"])
                    + len(attempt["candidate"]) / 1000
                )

            for attempt in sorted(attempts, key=penalty):
                if attempt["candidate"] not in candidates:
                    candidates.append(attempt["candidate"])
                if len(candidates) >= max_candidates:
                    break
        return dict(
            input=recognized_words,
            candidates=candidates,
            requires_confirmation=True,
            rejected_count=sum(not a["valid"] for a in attempts),
            uncertainty=not strictly_valid,
            strictly_validated=strictly_valid,
            candidate_reviews=[
                next(a for a in attempts if a["candidate"] == c) for c in candidates
            ],
            generation_method="fine_tuned_seq2seq",
        )


_realizer = None


def generate_sentence_candidates(glosses):
    global _realizer
    if _realizer is None:
        _realizer = GlossRealizer()
    return _realizer.generate(glosses)


def generate_sentence(recognized_words):
    """Return the first decoded model candidate, or None when the model abstains."""
    result = generate_sentence_candidates(recognized_words)
    return result["candidates"][0] if result["candidates"] else None
