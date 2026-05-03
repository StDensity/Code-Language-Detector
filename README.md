# Code Language Detector

Detect the programming language of a code snippet using two inference paths:

- **TF-IDF + Logistic Regression** baseline
- **PyTorch character-level CNN** classifier

The project uses CodeSearchNet from Hugging Face, extracts clean `code` and
`language` fields, trains models, and serves inference through FastAPI.

## Project Structure

```text
app/
  main.py                 # FastAPI routes
  schemas.py              # Request/response models
  services.py             # Loads trained models for inference

language_detector/
  data_utils.py           # JSONL loading and snippet sampling
  tfidf_baseline.py       # TF-IDF + Logistic Regression
  char_cnn.py             # PyTorch char-level CNN

data/
  setup_data.py           # Downloads/extracts dataset to JSONL
  manual_stress_tests.jsonl

train.py                  # Train TF-IDF or CNN
predict.py                # CLI prediction
compare_models.py         # Compare TF-IDF and CNN on validation data
stress_test.py            # Manual edge-case tests
```

## Setup

```powershell
uv sync
```

If FastAPI dependencies are not installed yet:

```powershell
uv add "fastapi[standard]"
```

## Prepare Data

Create clean JSONL files from CodeSearchNet:

```powershell
uv run python data/setup_data.py
```

This writes:

```text
data/processed/train.jsonl
data/processed/validation.jsonl
data/processed/test.jsonl
```

Each row looks like:

```json
{"code": "def hello(): print('hi')", "language": "python"}
```

Use the full dataset:

```powershell
uv run python data/setup_data.py --limit -1
```

## Train Models

Train the TF-IDF baseline:

```powershell
uv run python train.py --model tfidf
```

Train the char-level CNN:

```powershell
uv run python train.py --model cnn
```

The CNN uses:

- character-level vocabulary
- max sequence length `1000`
- train/validation split from the processed dataset
- PyTorch checkpoint saving

## CLI Prediction

TF-IDF:

```powershell
uv run python predict.py --model tfidf "def hello(): print('hi')"
```

CNN:

```powershell
uv run python predict.py --model cnn "def hello(): print('hi')"
```

Compare both:

```powershell
uv run python predict.py --model compare "def hello(): print('hi')"
```

## FastAPI Inference

Run the API in development mode:

```powershell
uv run fastapi dev app/main.py
```

Open the docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
GET /health
```

Single-model prediction:

```text
POST /predict
```

Example body:

```json
{
  "code": "def hello(): print('hi')",
  "model": "tfidf"
}
```

Example response:

```json
{
  "model": "tfidf",
  "language": "python",
  "confidence": 0.59,
  "inference_time_ms": 11.29
}
```

Compare available models:

```text
POST /compare
```

Example body:

```json
{
  "code": "def hello(): print('hi')"
}
```

## Model Intuition

The TF-IDF baseline turns code into character-pattern scores, then Logistic
Regression learns which patterns point to each language.

The CNN turns characters into IDs, embeds them, applies convolution filters over
the character sequence, and predicts the language from learned syntax-like
patterns.

Both models return:

- predicted language
- confidence
- inference time
