

import os
import logging
from rag_utils import (
    ingest_transcript,
    process_transcript,
    build_and_get_rag_chain,
    ask_questions_loop
)

# Configurer le logger global
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("podcast_ai_app")


def main():
    print("\n=== Podcast RAG Test (Console) ===\n")

    # 1) Ingestion : Youtube / audio local / .txt
    raw_text = ingest_transcript()
    if not raw_text:
        return

    # Aperçu du transcript
    print("\n----- Aperçu du transcript (200 premiers caractères) -----")
    print(raw_text[:200].replace("\n", " "))
    print("…\n")

    # 2) Prétraitement : chunking + indexation Chroma
    persist_dir = "data/vectorstores/chunks"
    process_transcript(raw_text, persist_dir=persist_dir)

    # 3) Construction de la chaîne RAG
    try:
        chain, retriever = build_and_get_rag_chain(persist_dir=persist_dir)
    except FileNotFoundError as e:
        print(str(e))
        return

    # 4) Boucle interactive de Q&A
    ask_questions_loop(chain, retriever)


if __name__ == "__main__":
    main()
