# Arabic gloss-to-sentence model selection

Ramzah needs a sequence-to-sequence model that converts short Arabic gloss sequences into natural Arabic sentences. It is a constrained reception assistant, not a diagnosis or triage system.

| Model | Strengths | Trade-offs | Decision |
|---|---|---|---|
| [AraT5-msa-small](https://huggingface.co/UBC-NLP/AraT5-msa-small) | Arabic-specific T5 pretraining, direct seq2seq training, practical local inference | Research-oriented model license must be reviewed before commercial deployment | Selected for the prototype |
| [AraT5v2-base-1024](https://huggingface.co/UBC-NLP/AraT5v2-base-1024) | Newer and larger Arabic checkpoint | More memory and slower CPU inference | Candidate for a future measured upgrade |
| [mT5-small](https://huggingface.co/google/mt5-small) | Mature multilingual T5 implementation and Apache 2.0 license | Larger multilingual vocabulary and less Arabic specialization | Licensing-friendly alternative |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Strong instruction following and Arabic support | Heavier inference and more prompt-sensitive behavior than needed | Not selected for this narrow task |

## Final design

The submitted checkpoint is `checkpoints/nlp/contextual_v3`, fine-tuned from AraT5-msa-small on paired gloss and sentence examples. Dataset splitting is grouped by semantic concept set so reordered inputs and paraphrases of the same meaning do not leak across train, validation, and test sets.

Inference joins the recognized glosses, tokenizes the text, calls `model.generate()`, and decodes up to three candidates. Semantic checks score preservation of important concepts and flag uncertainty. They do not build, repair, or substitute sentences. Even when strict validation finds no perfect candidate, the returned text remains neural model output and requires user review.

The dataset is synthetic and limited to the project vocabulary. Evaluation therefore measures prototype behavior only; it does not demonstrate clinical correctness or generalization to unrestricted conversation.
