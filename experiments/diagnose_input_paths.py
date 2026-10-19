"""Why does the same article score differently when pasted than when fetched?

The web app runs identical model code on both paths: ``predict_article(title, text)``.
Any score difference therefore comes from the strings those two arguments hold.
This compares them side by side for one article and prints what actually differs.

    python experiments/diagnose_input_paths.py --url <URL> --pasted pasted.txt

``pasted.txt`` is whatever you copied out of the browser, saved as UTF-8. Omit
``--pasted`` to inspect the fetched text alone.

Run it with the project virtualenv (``.venv/Scripts/python.exe`` on Windows,
``.venv/bin/python`` elsewhere); it needs flask, requests, bs4 and the trained
model. The import block below also works without an editable install.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# `fakenews` lives under src/. Add it explicitly so this runs whether or not the
# package happens to be installed in the active interpreter.
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_app_module():
    """Load app/app.py by path.

    Importing it by name is ambiguous: the directory app/ sits next to the file
    app/app.py, so `import app` can resolve to either depending on sys.path
    order. Loading from an explicit location removes the guesswork.
    """
    app_path = ROOT / "app" / "app.py"
    if not app_path.exists():
        raise SystemExit(f"Could not find {app_path}")
    spec = importlib.util.spec_from_file_location("fakenews_webapp", app_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["fakenews_webapp"] = module
    spec.loader.exec_module(module)
    return module


def describe(label, title, text, detector):
    cleaned = detector._clean(title, text)
    result = detector.predict_article(title, text)
    print(
        f"{label:<26} title={len(title):>4}c  text={len(text):>6}c  "
        f"cleaned={len(cleaned.split()):>5} tok  ->  {result['prediction']:<5} "
        f"fake={result['fake_probability'] * 100:5.1f}%"
    )
    return result["fake_probability"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--pasted", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()

    webapp = _load_app_module()
    from fakenews.config import MODELS_DIR
    from fakenews.detector import FakeNewsDetector

    directory = args.model_dir or (MODELS_DIR / "seed5")
    if not directory.exists():
        raise SystemExit(f"No model at {directory}. Pass --model-dir.")
    detector = FakeNewsDetector.load(directory)

    title, text, error = webapp.extract_article_from_url(args.url)
    if error:
        raise SystemExit(f"Fetch failed: {error}")

    print()
    url_both = describe("URL (title + body)", title, text, detector)
    describe("URL (body only)", "", text, detector)

    if args.pasted:
        if not args.pasted.exists():
            raise SystemExit(f"No such file: {args.pasted.resolve()}")
        pasted = args.pasted.read_text(encoding="utf-8", errors="replace").strip()
        paste_both = describe("Pasted (title + body)", title, pasted, detector)
        describe("Pasted (body only)", "", pasted, detector)

        fetched_words = set(detector._clean("", text).split())
        pasted_words = set(detector._clean("", pasted).split())
        only_pasted = sorted(pasted_words - fetched_words)
        only_fetched = sorted(fetched_words - pasted_words)

        print()
        print(f"Paste is {len(pasted) / max(len(text), 1):.2f}x the length of the fetch.")
        print(f"Score moves {(paste_both - url_both) * 100:+.1f} points, title held constant.")
        print()
        print(f"Cleaned tokens only in the paste ({len(only_pasted)}):")
        print("   ", only_pasted[:40])
        print(f"Cleaned tokens only in the fetch ({len(only_fetched)}):")
        print("   ", only_fetched[:40])

    print()
    print("Rows 1 vs 2 isolate the title. Rows 1 vs 3 isolate the body with the")
    print("title held constant: that difference is the one worth explaining.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
