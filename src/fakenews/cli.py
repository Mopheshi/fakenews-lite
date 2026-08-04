"""Command line entry point.

Every subcommand is a thin wrapper over a library function, and each handler
imports lazily so that ``fakenews --help`` does not pay TensorFlow's import cost.
"""

from __future__ import annotations

import argparse
import logging
import sys

from fakenews.config import DEFAULT_SEED


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity < 0 else logging.INFO if verbosity == 0 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # TensorFlow's own logger is noisy at INFO.
    logging.getLogger("tensorflow").setLevel(logging.ERROR)


def _cmd_build_cache(args: argparse.Namespace) -> int:
    from fakenews.data.cache import build_cache
    from fakenews.nltk_resources import ensure_nltk_resources

    ensure_nltk_resources()
    path = build_cache(n_jobs=args.jobs, overwrite=args.overwrite)
    print(f"Preprocessing cache: {path}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from fakenews.config import ExperimentConfig
    from fakenews.pipeline import run_pipeline

    config = ExperimentConfig(
        seed=args.seed,
        use_pos_features=not args.no_pos,
        strip_source_markers=args.blocklist != "none",
    )
    result = run_pipeline(
        config,
        blocklist_preset=args.blocklist,
        save_models_to=args.save_models,
        run_name=args.name,
    )
    print(result.summary_table())
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from fakenews.detector import FakeNewsDetector

    detector = FakeNewsDetector.load(args.model)
    text = args.text if args.text is not None else sys.stdin.read()
    result = detector.predict_article(args.title or "", text)
    print(
        f"{result['prediction']}  (confidence {result['confidence']:.1%}; "
        f"fake {result['fake_probability']:.1%} / legit {result['legit_probability']:.1%})"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fakenews",
        description="Lightweight fake news detection with an XGBoost + TextCNN ensemble.",
    )
    parser.add_argument("-q", "--quiet", dest="verbosity", action="store_const", const=-1, default=0)
    parser.add_argument("-v", "--verbose", dest="verbosity", action="store_const", const=1)

    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser(
        "build-cache",
        help="Clean and POS-tag the corpus once, caching the result as Parquet.",
    )
    cache.add_argument("-j", "--jobs", type=int, default=-1, help="Worker processes (-1 = all cores).")
    cache.add_argument("--overwrite", action="store_true", help="Rebuild even if the cache exists.")
    cache.set_defaults(func=_cmd_build_cache)

    train = subparsers.add_parser("train", help="Train and evaluate the full pipeline.")
    train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train.add_argument("--no-pos", action="store_true", help="Exclude the six POS features.")
    train.add_argument(
        "--blocklist",
        choices=("none", "wire", "extended"),
        default="none",
        help="Source-marker blocklist applied to the text.",
    )
    train.add_argument("--save-models", metavar="DIR", help="Directory to persist trained models to.")
    train.add_argument("--name", default=None, help="Run name used for the results filename.")
    train.set_defaults(func=_cmd_train)

    predict = subparsers.add_parser("predict", help="Classify a single article.")
    predict.add_argument("--model", required=True, metavar="DIR", help="Directory of a saved model.")
    predict.add_argument("--title", default="")
    predict.add_argument("--text", default=None, help="Article body; reads stdin when omitted.")
    predict.set_defaults(func=_cmd_predict)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbosity)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
