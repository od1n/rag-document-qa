#!/usr/bin/env python3
"""Command-line interface.

    python cli.py ingest sample/            # index every PDF/TXT/MD in a folder (or single files)
    python cli.py ask "What is the refund policy?"
    python cli.py ask "..." --top-k 8 --source contract.pdf
    python cli.py list                      # indexed documents
    python cli.py reset                     # drop the index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag import config
from rag.ingest import ingest_files, list_sources
from rag.query import ask
from rag.store import reset_collection


def _collect(paths: list[str]) -> list[Path]:
    out = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out += sorted(x for x in path.iterdir() if x.suffix.lower() in {".pdf", ".txt", ".md"})
        else:
            out.append(path)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest"); s.add_argument("paths", nargs="+")
    s = sub.add_parser("ask"); s.add_argument("question"); s.add_argument("--top-k", type=int, default=config.TOP_K); s.add_argument("--source", action="append")
    sub.add_parser("list"); sub.add_parser("reset")
    args = ap.parse_args(argv)

    print(f"mode: embeddings={config.EMBEDDING_PROVIDER}, llm={'openai:' + config.CHAT_MODEL if config.llm_available() else 'none (extractive answers)'}")
    if args.cmd == "ingest":
        rep = ingest_files(_collect(args.paths))
        print(f"indexed {rep['files']} file(s), {rep['chunks']} chunks")
        for s_ in rep["skipped"]:
            print("skipped:", s_)
    elif args.cmd == "ask":
        ans = ask(args.question, top_k=args.top_k, sources=args.source)
        print("\n" + ans.text + "\n")
        if ans.mode == "llm":
            print("sources:")
            for p in ans.passages:
                print(f"  [{p.rank}] {p.source} p.{p.page} (similarity {p.score})")
    elif args.cmd == "list":
        for s_ in list_sources():
            print(s_)
    elif args.cmd == "reset":
        reset_collection(); print("index cleared")
    return 0


if __name__ == "__main__":
    sys.exit(main())
