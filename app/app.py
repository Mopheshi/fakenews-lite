"""Flask demo: classify a pasted article or a news URL.

The model directory is produced by ``python experiments/run_main.py``; set
``FAKENEWS_MODEL_DIR`` to serve a different one.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request

from fakenews.config import DEFAULT_SEED, MODELS_DIR
from fakenews.detector import FakeNewsDetector

logger = logging.getLogger(__name__)

app = Flask(__name__)

#: Populated by :func:`load_detector` at start-up.
detector: FakeNewsDetector | None = None

REQUEST_TIMEOUT_SECONDS = 10
MAX_ARTICLE_CHARS = 100_000
USER_AGENT = "Mozilla/5.0 (compatible; fakenews-lite/1.0)"

ARTICLE_SELECTORS = (
    "article",
    '[class*="article"]',
    '[class*="content"]',
    '[class*="post"]',
    '[class*="story"]',
    "main",
    ".entry-content",
    "#content",
)


def model_directory() -> Path:
    override = os.environ.get("FAKENEWS_MODEL_DIR")
    return Path(override) if override else MODELS_DIR / f"seed{DEFAULT_SEED}"


def load_detector() -> bool:
    """Load the saved ensemble. Returns False when no model is available."""
    global detector

    directory = model_directory()
    if not (directory / "manifest.json").exists():
        logger.error(
            "No model found at %s. Train one with `python experiments/run_main.py`.", directory
        )
        return False

    try:
        detector = FakeNewsDetector.load(directory)
    except Exception:
        logger.exception("Failed to load the model from %s", directory)
        return False

    logger.info("Loaded model from %s", directory)
    return True


def extract_article_from_url(url: str) -> tuple[str, str, str | None]:
    """Fetch a URL and pull out a title and body. Returns ``(title, text, error)``."""
    try:
        response = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as error:
        return "", "", f"Could not fetch the URL: {error}"

    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as error:  # pragma: no cover - malformed markup
        return "", "", f"Could not parse the page: {error}"

    heading = soup.find("h1") or soup.find("title")
    title = heading.get_text(" ", strip=True) if heading else ""

    text = ""
    for selector in ARTICLE_SELECTORS:
        container = soup.select_one(selector)
        if not container:
            continue
        paragraphs = container.find_all("p")
        if paragraphs:
            text = " ".join(paragraph.get_text(" ", strip=True) for paragraph in paragraphs)
            if len(text) > 200:
                break

    if len(text) < 200:
        text = " ".join(paragraph.get_text(" ", strip=True) for paragraph in soup.find_all("p"))

    text = re.sub(r"\s+", " ", text).strip()[:MAX_ARTICLE_CHARS]

    if not title and not text:
        return "", "", "No article content could be extracted from this page."

    return title, text, None


@app.route("/")
def index():
    return render_template("index.html", model_loaded=detector is not None)


@app.route("/analyse", methods=["POST"])
def analyse():
    if detector is None:
        return jsonify({"success": False, "error": "No model loaded. Train one first."}), 503

    payload = request.get_json(silent=True) or {}

    if payload.get("input_type") == "url":
        url = (payload.get("url") or "").strip()
        if not url:
            return jsonify({"success": False, "error": "Please provide a URL."}), 400

        title, text, error = extract_article_from_url(url)
        if error:
            return jsonify({"success": False, "error": error}), 400
    else:
        title = (payload.get("title") or "").strip()
        text = (payload.get("text") or "").strip()[:MAX_ARTICLE_CHARS]
        if not text:
            return jsonify({"success": False, "error": "Please provide the article text."}), 400

    try:
        result = detector.predict_article(title, text)
    except Exception as error:  # pragma: no cover - surfaced to the user
        logger.exception("Prediction failed")
        return jsonify({"success": False, "error": f"Analysis failed: {error}"}), 500

    return jsonify(
        {
            "success": True,
            "prediction": result["prediction"],
            "confidence": round(result["confidence"] * 100, 1),
            "fake_probability": round(result["fake_probability"] * 100, 1),
            "legit_probability": round(result["legit_probability"] * 100, 1),
            "title": title[:200],
            "text_preview": text[:500],
            "model_type": result["decided_by"],
        }
    )


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok" if detector is not None else "degraded",
            "model_loaded": detector is not None,
            "model_directory": str(model_directory()),
        }
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s")

    if not load_detector():
        print("\nNo trained model available. Run:  python experiments/run_main.py\n")
        return 1

    host = os.environ.get("FAKENEWS_HOST", "127.0.0.1")
    port = int(os.environ.get("FAKENEWS_PORT", "5000"))
    debug = os.environ.get("FAKENEWS_DEBUG", "").lower() in {"1", "true", "yes"}

    print(f"\nServing on http://{host}:{port}\n")
    app.run(host=host, port=port, debug=debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
