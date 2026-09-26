# Evaluation reports

These reports document local prototype measurements. They do not establish clinical safety or real-world Saudi Sign Language accuracy.

## Computer vision

- `dataset/report.json` audits the local 12-class video subset.
- `cv/evaluation.json` measures the adapted classifier on the held-out local signer split.
- `cv/end_to_end_latency.json` measures complete video-to-gloss latency.
- `cv/ood_smoke.json` demonstrates the limits of softmax confidence on an unsupported sign.

The controlled backgrounds and small signer count can inflate accuracy. The reported results must not be treated as evidence of generalization to new users or environments.

## Arabic generation

- `nlp/contextual_evaluation.json` evaluates the active `contextual_v3` AraT5 checkpoint.
- `nlp/requested_examples.json` records decoded examples and word-order variations.

All candidates are produced by `model.generate()` and decoded by the tokenizer. Semantic validation can rank or reject a candidate, but it never constructs a replacement sentence. If no candidate passes strict validation, the API returns the best neural candidates with an uncertainty flag so the user can review them.

The NLP dataset is synthetic and domain-limited. Automatic metrics and concept checks cannot replace review by Arabic-language, clinical, and Deaf-community experts.

## Integration

- `integration/appointment.json` traces an editable gloss sequence through neural generation and confirmation.
- `integration/smoke.json` covers the API workflow from video recognition through sentence confirmation and TTS.
- `tts/sample.wav` is a local Arabic speech-synthesis sample.

The user must confirm recognized words and the final sentence before it is shared or spoken.
