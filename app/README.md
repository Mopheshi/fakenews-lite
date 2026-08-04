# Web demo

A small Flask front end for the trained ensemble. Paste an article, or give it a
news URL and it will extract the body itself.

## Running it

The demo serves a model directory produced by the training pipeline, so train
one first:

```bash
fakenews build-cache              # once
python experiments/run_main.py    # writes artifacts/models/seed5
python app/app.py
```

Then open <http://127.0.0.1:5000>.

## Configuration

All optional, read from the environment:

| Variable | Default | Purpose |
| --- | --- | --- |
| `FAKENEWS_MODEL_DIR` | `artifacts/models/seed5` | Which trained model to serve |
| `FAKENEWS_HOST` | `127.0.0.1` | Bind address |
| `FAKENEWS_PORT` | `5000` | Port |
| `FAKENEWS_DEBUG` | unset | Set to `1` for Flask's reloader and debugger |

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | The UI |
| `POST` | `/analyse` | Classify an article |
| `GET` | `/health` | Readiness probe; reports whether a model is loaded |

`POST /analyse` takes either form:

```jsonc
{"input_type": "text", "title": "optional", "text": "the article body"}
{"input_type": "url",  "url": "https://example.com/story"}
```

and answers:

```jsonc
{
  "success": true,
  "prediction": "FAKE",          // or "LEGIT"
  "confidence": 99.9,            // percent, for the predicted class
  "fake_probability": 99.9,
  "legit_probability": 0.1,
  "model_type": "Ensemble (soft voting)",
  "title": "...",
  "text_preview": "..."
}
```

Errors return `success: false` with a message, and a 4xx or 5xx status.

## Deployment notes

This is a demo, not a hardened service. Before exposing it:

- Run it behind a real WSGI server (`gunicorn`, `waitress`) rather than Flask's
  development server, and leave `FAKENEWS_DEBUG` unset.
- URL fetching issues an outbound request to a user-supplied address. On a
  network where that matters, restrict it — the current code has a timeout and a
  response size cap, but no allowlist and no SSRF protection.
- Add rate limiting. A single prediction is inexpensive, but URL fetching is not.

## Accuracy in context

The model was trained on WELFake, an English corpus of mostly 2016–2017 US
political news. Treat its output as a signal, not a verdict: performance on
recent articles, other languages, other domains or AI-generated text is not
characterised by the reported figures. See the limitations section of the paper.
