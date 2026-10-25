"""Terminal interface:  python cli.py path/to/file.pdf"""
import argparse
import os

from rag.pipeline import PDFQAPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about a PDF (RAG).")
    parser.add_argument("pdf", nargs="+", help="path(s) to PDF file(s)")
    args = parser.parse_args()

    pipeline = PDFQAPipeline()
    for path in args.pdf:
        with open(path, "rb") as f:
            n = pipeline.add_pdf(f, os.path.basename(path))
        print(f"Indexed {path}: {n} chunks")

    history: list[dict] = []
    print("\nAsk a question (type 'exit' to quit).")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", ""}:
            break
        ans = pipeline.ask(q, history)
        print(f"\nAssistant: {ans.text}")
        for i, (chunk, score) in enumerate(ans.sources, start=1):
            print(f"  [{i}] {chunk.source}, page {chunk.page} (score {score:.2f})")
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": ans.text}]


if __name__ == "__main__":
    main()
