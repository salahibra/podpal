
import os
import logging
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

def get_vectorstore(
    text_chunks: List[str],
    persist_dir: str = "data/vectorstores/chunks"
) -> Chroma:
    """
    Crée (ou recharge) un Chroma DB VectorStore à partir d'une liste de chunks de texte.

    - text_chunks : liste de segments (strings) à indexer.
    - persist_dir  : dossier où stocker (ou charger) l'index Chroma.
    """
    logger.info("Création du VectorStore Chroma pour les chunks…")

    # 1) Instancier l’embedder HuggingFace (all-MiniLM)
    hf_emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # 2) On s’assure que le dossier de persistance existe
    os.makedirs(persist_dir, exist_ok=True)

    # 3) Créer la base (ou la recharger si elle existe déjà)
    vectordb = Chroma.from_texts(
        texts=text_chunks,
        embedding=hf_emb,
        persist_directory=persist_dir
    )

    
    logger.info(f"VectorStore Chroma persistant dans : {persist_dir}")

    return vectordb
