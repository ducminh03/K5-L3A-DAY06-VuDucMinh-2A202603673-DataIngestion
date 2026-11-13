"""Command-line entry point for the Day 06 ingestion pipeline."""
import argparse
import csv
import os
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .models import Chunk
from .store import PgVectorStore

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_OVERLAP = 150
DEFAULT_EMBEDDING_DIMENSIONS = 3072
MIN_CONTENT_PAGES = 30
CHUNKING_STRATEGIES = ("fixed", "paragraph")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_LOG_FIELDS = (
    "strategy",
    "chunking",
    "chunk_size",
    "overlap",
    "metadata",
    "filter",
    "vectorizer",
    "hypothesis",
    "dev_hit1",
    "dev_hit3",
    "test_hit1",
    "test_hit3",
    "notes",
)
QUERY_FIELDS = (
    "id",
    "split",
    "question",
    "gold_source_id",
    "gold_page",
    "evidence_quote",
)


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def _nonnegative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return result


def _add_embedding_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
        help="Gemini embedding model (default: EMBEDDING_MODEL or gemini-embedding-001)",
    )
    parser.add_argument(
        "--dimensions",
        type=_positive_int,
        default=os.getenv("EMBEDDING_DIMENSIONS", str(DEFAULT_EMBEDDING_DIMENSIONS)),
        metavar="N",
        help=f"embedding vector dimensions (default: {DEFAULT_EMBEDDING_DIMENSIONS})",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 06 data ingestion pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="read, chunk, embed, and store the corpus")
    ingest.add_argument(
        "--strategy",
        choices=CHUNKING_STRATEGIES,
        default="fixed",
        help="chunking strategy (default: fixed)",
    )
    ingest.add_argument(
        "--chunk-size",
        type=_positive_int,
        default=DEFAULT_CHUNK_SIZE,
        metavar="CHARS",
        help=f"maximum chunk size in characters (default: {DEFAULT_CHUNK_SIZE})",
    )
    ingest.add_argument(
        "--overlap",
        type=_nonnegative_int,
        default=None,
        metavar="CHARS",
        help="fixed-size overlap in characters (default: 150; fixed strategy only)",
    )
    _add_embedding_options(ingest)

    query = commands.add_parser("query", help="search the top three matching chunks")
    query.add_argument(
        "question",
        nargs="?",
        help="question to search for; prompts if omitted",
    )
    query.add_argument(
        "--strategy",
        choices=CHUNKING_STRATEGIES,
        required=True,
        help="chunking strategy used during ingestion",
    )
    _add_embedding_options(query)

    test = commands.add_parser(
        "test",
        help="run the query CSV against ingested chunking strategies",
    )
    test.add_argument(
        "--query-file",
        type=Path,
        default=REPOSITORY_ROOT / "benchmark" / "query.csv",
        help="semicolon-delimited benchmark CSV",
    )
    test.add_argument(
        "--strategy",
        choices=(*CHUNKING_STRATEGIES, "all"),
        default="all",
        help="strategy to test (default: all)",
    )
    test.add_argument(
        "--chunk-size",
        type=_positive_int,
        default=DEFAULT_CHUNK_SIZE,
        metavar="CHARS",
        help=f"chunk size used during ingestion (default: {DEFAULT_CHUNK_SIZE})",
    )
    test.add_argument(
        "--overlap",
        type=_nonnegative_int,
        default=DEFAULT_OVERLAP,
        metavar="CHARS",
        help=f"fixed-size overlap used during ingestion (default: {DEFAULT_OVERLAP})",
    )
    test.add_argument(
        "--log-file",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "experiment_log.csv",
        help="CSV file to append the aggregated metrics to",
    )
    _add_embedding_options(test)
    return parser


def _read_test_questions(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as question_file:
        reader = csv.DictReader(question_file, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError(f"query file {path} is empty")
        missing_fields = set(QUERY_FIELDS) - set(reader.fieldnames)
        if missing_fields:
            raise ValueError(
                f"query file {path} is missing columns: {', '.join(sorted(missing_fields))}"
            )

        questions: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            question = {
                field: (row.get(field) or "").strip()
                for field in QUERY_FIELDS
            }
            if any(not value for value in question.values()):
                raise ValueError(
                    f"query file {path} has an empty required value on line {row_number}"
                )
            if question["split"] not in {"dev", "test"}:
                raise ValueError(
                    f"query file {path} has invalid split {question['split']!r} "
                    f"on line {row_number}"
                )
            try:
                question["gold_page"] = str(int(question["gold_page"]))
            except ValueError as error:
                raise ValueError(
                    f"query file {path} has an invalid gold_page on line {row_number}"
                ) from error
            questions.append(question)

    if not questions:
        raise ValueError(f"query file {path} has no questions")
    if not any(question["split"] == "dev" for question in questions):
        raise ValueError(f"query file {path} has no dev questions")
    if not any(question["split"] == "test" for question in questions):
        raise ValueError(f"query file {path} has no test questions")
    return questions


def _format_hit_rate(hits: int, total: int) -> str:
    return f"{hits}/{total} ({hits / total:.1%})"


def _append_experiment_results(
    path: Path,
    strategies: Sequence[str],
    metrics: dict[str, dict[str, dict[str, int]]],
    chunk_size: int,
    overlap: int,
    model: str,
    dimensions: int,
    query_file: Path,
) -> None:
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as existing_file:
            fieldnames = next(csv.reader(existing_file), None)
        if fieldnames is None:
            raise ValueError(f"experiment log {path} is empty")
        missing_fields = set(EXPERIMENT_LOG_FIELDS) - set(fieldnames)
        if missing_fields:
            raise ValueError(
                f"experiment log {path} is missing columns: "
                f"{', '.join(sorted(missing_fields))}"
            )
        mode = "a"
        if path.stat().st_size:
            with path.open("rb") as existing_file:
                existing_file.seek(-1, os.SEEK_END)
                if existing_file.read(1) != b"\n":
                    with path.open("ab") as log_file:
                        log_file.write(b"\n")
    else:
        fieldnames = list(EXPERIMENT_LOG_FIELDS)
        mode = "w"

    with path.open(mode, encoding="utf-8", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        for strategy in strategies:
            dev = metrics[strategy]["dev"]
            test = metrics[strategy]["test"]
            writer.writerow(
                {
                    "strategy": strategy,
                    "chunking": "fixed-size" if strategy == "fixed" else "paragraph",
                    "chunk_size": chunk_size,
                    "overlap": overlap if strategy == "fixed" else "",
                    "metadata": "manifest",
                    "filter": "none",
                    "vectorizer": model,
                    "hypothesis": "",
                    "dev_hit1": _format_hit_rate(dev["hit1"], dev["total"]),
                    "dev_hit3": _format_hit_rate(dev["hit3"], dev["total"]),
                    "test_hit1": _format_hit_rate(test["hit1"], test["total"]),
                    "test_hit3": _format_hit_rate(test["hit3"], test["total"]),
                    "notes": (
                        f"query_file={query_file}; model={model}; dimensions={dimensions}; "
                        f"splits=dev:{dev['total']}/test:{test['total']}; "
                        f"evidence match requires gold document, page, and quoted text"
                    ),
                }
            )


def _run_test(
    query_file: Path,
    selected_strategy: str,
    log_file: Path,
    chunk_size: int,
    overlap: int,
    model: str,
    dimensions: int,
    database_url: str,
) -> None:
    questions = _read_test_questions(query_file)
    strategies = (
        CHUNKING_STRATEGIES
        if selected_strategy == "all"
        else (selected_strategy,)
    )

    embedder = MyEmbedder(model, dimensions)
    store = PgVectorStore(database_url, dimensions)
    traces: list[dict[str, object]] = []
    for question in questions:
        query_vector = embedder.embed_query(question["question"])
        for strategy in strategies:
            matches = store.search(strategy, query_vector, top_k=3)
            trace = evaluate_query(question, matches)
            trace["strategy"] = strategy
            traces.append(trace)
    metrics = summarize(traces)

    _append_experiment_results(
        log_file,
        strategies,
        metrics,
        chunk_size,
        overlap,
        model,
        dimensions,
        query_file,
    )
    for strategy in strategies:
        dev = metrics[strategy]["dev"]
        test = metrics[strategy]["test"]
        print(
            f"{strategy}: "
            f"dev Hit@1 {_format_hit_rate(dev['hit1'], dev['total'])}, "
            f"Hit@3 {_format_hit_rate(dev['hit3'], dev['total'])}; "
            f"test Hit@1 {_format_hit_rate(test['hit1'], test['total'])}, "
            f"Hit@3 {_format_hit_rate(test['hit3'], test['total'])}"
        )
    print(f"Appended experiment metrics to {log_file}")


def _run_query(
    question: str,
    strategy: str,
    model: str,
    dimensions: int,
    database_url: str,
) -> None:
    embedder = MyEmbedder(model, dimensions)
    query_vector = embedder.embed_query(question)
    store = PgVectorStore(database_url, dimensions)
    matches = store.search(strategy, query_vector, top_k=3)
    if not matches:
        print("No matching chunks found.")
        return

    for rank, match in enumerate(matches, start=1):
        print(
            f"{rank}. score={match['score']:.4f} | "
            f"document={match['document_id']} | "
            f"{match['source']} page {match['page']}"
        )
        print(match["text"])
        print(f"metadata: {match['metadata']}")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the requested pipeline command."""
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    parser = _build_parser()
    args = parser.parse_args(argv)

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        parser.error("DATABASE_URL is missing; configure it in starter/.env")
    if not args.model.strip():
        parser.error("--model must not be empty")

    if args.command == "test":
        _run_test(
            args.query_file,
            args.strategy,
            args.log_file,
            args.chunk_size,
            args.overlap,
            args.model,
            args.dimensions,
            database_url,
        )
        return

    if args.command == "query":
        question = args.question
        if question is None:
            question = input("Enter a question: ")
        question = question.strip()
        if not question:
            parser.error("question must not be empty")
        _run_query(
            question,
            args.strategy,
            args.model,
            args.dimensions,
            database_url,
        )
        return

    if args.strategy == "paragraph" and args.overlap is not None:
        parser.error("--overlap is only supported with --strategy fixed")

    pages = load_pdf()
    if len(pages) < MIN_CONTENT_PAGES:
        parser.error(
            f"the corpus has {len(pages)} non-empty pages; "
            f"at least {MIN_CONTENT_PAGES} content pages are required"
        )

    chunks: list[Chunk] = []
    overlap = DEFAULT_OVERLAP if args.overlap is None else args.overlap
    for page in pages:
        if args.strategy == "fixed":
            chunks.extend(fixed_size(page, args.chunk_size, overlap))
        else:
            chunks.extend(by_paragraph(page, args.chunk_size))
    if not chunks:
        parser.error("chunking produced no non-empty chunks")

    store = PgVectorStore(database_url, args.dimensions)
    store.create_schema()

    embedder = MyEmbedder(args.model, args.dimensions)
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"embedding returned {len(vectors)} vectors for {len(chunks)} chunks"
        )

    store.upsert(args.strategy, chunks, vectors)
    print(
        f"Ingested {len(pages)} pages as {len(chunks)} chunks "
        f"(strategy={args.strategy}, chunk_size={args.chunk_size}, "
        f"overlap={overlap if args.strategy == 'fixed' else 'n/a'}, "
        f"model={args.model}, dimensions={args.dimensions})."
    )


if __name__ == "__main__":
    main()
