# Ramzah


<img width="297" height="261" alt="image" src="https://github.com/user-attachments/assets/a344a51d-305e-44e4-a2fc-ab5278d94b49" />

Ramzah is a research prototype that converts isolated Saudi sign-language videos into reviewed Arabic sentences and speech for hospital reception. The submitted system keeps the supplied clean interface and connects it to local computer-vision, Arabic NLP, and text-to-speech models.

## Project structure

```text
Ramza_Final_Version/
├── Ramzah_Interface_Clean/   # Original Next.js interface
├── Ramzah_Prototype/         # FastAPI backend, models, data, tests
├── setup_ramzah.cmd          # One-time dependency installation
├── start_ramzah.cmd          # Starts the complete system
└── README.md
```

Important backend directories:

```text
Ramzah_Prototype/
├── checkpoints/
│   ├── cv/best/              # Active sign classifier
│   ├── nlp/contextual_v3/    # Active fine-tuned AraT5 checkpoint
│   └── tts/mms-ara/          # Arabic speech model
├── data/                     # Labels, vocabulary, and NLP splits
├── notebooks/                # CV, NLP, and inference notebooks
├── ramzah/                   # Runtime Python package
├── reports/                  # Final evaluation evidence
├── scripts/                  # Training, evaluation, and smoke tests
└── tests/
```

## Computer vision

`ramzah/cv.py` decodes one isolated-sign video, extracts MediaPipe landmarks, and applies the adapted CV classifier. The packaged model supports 12 health-related glosses listed in `data/ramzah_vocabulary.csv`. It does not perform continuous sign-language recognition.

## Arabic NLP

`ramzah/nlp.py` joins the confirmed gloss sequence, tokenizes it, calls the fine-tuned AraT5 model with `model.generate()`, and decodes Arabic candidates. `ramzah/semantics.py` reviews generated candidates for missing, added, repeated, or unsafe concepts; it never constructs a sentence. The active checkpoint is `checkpoints/nlp/contextual_v3`.

The dataset is stored as `data/nlp/train.jsonl`, `validation.jsonl`, and `test.jsonl`. It contains manually authored synthetic supervision with context-specific Arabic targets and word-order variation.

## Backend and interface

`ramzah/api.py` provides session, video recognition, gloss editing, sentence generation, confirmation, employee display, and speech endpoints. `Ramzah_Interface_Clean` is the original Next.js interface. Its `/api/*` requests are proxied to the FastAPI backend at `127.0.0.1:8000`.

## Requirements

- Windows 10 or 11
- Python 3.10+
- Node.js LTS
- Approximately 2 GB of free disk space for dependencies
- A modern browser with camera permission for live capture

The model checkpoints are included. Internet access is needed only for the initial dependency installation.

## Installation

```powershell
cd "C:\Users\esrra\Downloads\Ramza_Final_Version"
.\setup_ramzah.cmd
```

This creates `Ramzah_Prototype/.venv`, installs the Python requirements, and installs the clean interface dependencies.

## Run the complete system

```powershell
cd "C:\Users\esrra\Downloads\Ramza_Final_Version"
.\start_ramzah.cmd
```

Open `http://localhost:3000`. The launcher starts the backend in the background and the interface in the visible terminal. Press `Ctrl+C` to stop both.

## Run each part separately

Backend:

```powershell
cd Ramzah_Prototype
.\.venv\Scripts\python.exe -m uvicorn ramzah.api:app --host 127.0.0.1 --port 8000
```

Interface, in another terminal:

```powershell
cd Ramzah_Interface_Clean
pnpm.cmd dev
```

Backend health check: `http://127.0.0.1:8000/api/health`.

## Full pipeline

```text
Isolated sign video
→ MediaPipe landmarks
→ CV gloss prediction
→ reviewed gloss sequence
→ AraT5 tokenizer
→ fine-tuned AraT5 generation
→ semantic review and user confirmation
→ employee display
→ Arabic MMS text-to-speech
```
## Workflow

<img width="1280" height="364" alt="image" src="https://github.com/user-attachments/assets/865d797f-16c4-4b8c-9562-2d6dc2cf9cf4" />


  
## Verification

```powershell
cd Ramzah_Prototype
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.evaluate_nlp
```

```powershell
cd ..\Ramzah_Interface_Clean
pnpm.cmd exec tsc --noEmit
pnpm.cmd build
```


## Limitations

- This is a research prototype, not an emergency or clinically validated communication system.
- The CV model recognizes one isolated sign per video and supports only 12 packaged classes.
- The NLP corpus is synthetic and has not been reviewed by Deaf Saudi signers or clinicians.
- Arbitrary or ambiguous gloss combinations may produce uncertain text and require user review.
- The validator flags model output but does not guarantee medical correctness.
- AraT5 and some source assets have research-use licensing restrictions. Review `Ramzah_Prototype/THIRD_PARTY.md` before redistribution or deployment.
