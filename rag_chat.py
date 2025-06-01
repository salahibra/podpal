
import os
import logging
from typing import Any
from langchain.schema.runnable import RunnableLambda
from langchain.prompts import PromptTemplate

from hf_router import HuggingFaceRouterLLM

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_chroma import Chroma
logger = logging.getLogger(__name__)


def get_rag_chain(
    persist_dir: str = "data/vectorstores/chunks"
) -> Any:
    """
    Charge le VectorStore Chroma depuis `persist_dir` et
    construit le pipeline RAG (Runnable). Retourne (chain, retriever).
    """
    if not os.path.isdir(persist_dir):
        raise FileNotFoundError(
            f"Le dossier '{persist_dir}' est introuvable. "
            "Veuillez d’abord exécuter get_vectorstore() pour indexer vos chunks."
        )

    logger.info(f"Chargement du VectorStore depuis : {persist_dir}")

    # 1) Ré-instancier l’embedder identique à get_vectorstore()
    hf_emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # 2) Recharger la base Chroma : on passe l’embedder ici aussi,
    #    afin qu’il sache comment créer l’embedding de la query.
    vectordb = Chroma(
         embedding_function=hf_emb,
        persist_directory=persist_dir
    )
    logger.info("VectorStore Chroma rechargé avec succès (embedding fourni).")

    # 3) Construire le retriever (top‐k similarité)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    logger.info("Retriever top‐k construit.")

    # 4) Instancier ensuite le LLM
    llm = HuggingFaceRouterLLM()
    logger.info("LLM HuggingFaceRouterLLM instancié.")

    # 5) Préparer le prompt pour la génération de réponse
    qa_prompt = PromptTemplate.from_template(
        """You are an AI assistant for a podcast transcript. Use the context below to answer the question as clearly and concisely as possible.

Context:
{context}

Question: {question}

Answer:"""
    )

    # 6) Créer un Runnable pour extraire uniquement la partie "question" de l’input
    question_extractor = RunnableLambda(lambda data: data["question"])

    # 7) Construire la chaîne de traitement RAG
    #    - "context": (question → retriever → liste de docs)
    #    - "question": question pure
    #    → on injecte ces deux champs dans `qa_prompt`, puis dans le LLM
    chain = (
        {
            "context": question_extractor | retriever,
            "question": question_extractor
        }
        | qa_prompt
        | llm
    )

    logger.info("Pipeline RAG (Runnable) construit avec embedding_function.")
    return chain, retriever


def ask_loop(chain: Any, retriever: Any) -> None:
    """
    Boucle interactive en console pour poser des questions en RAG :
      - on lit la question utilisateur
      - on récupère les documents pertinents
      - on invoque la chaîne RAG pour générer la réponse
      - on affiche la réponse + extraits sources
    Tapez 'exit' pour quitter.
    """
    print("=== Mode RAG Chat (console) ===")
    print("Entrez votre question (ou 'exit' pour quitter) :")
    while True:
        question = input("> ").strip()
        if question.lower() in ("exit", "quit"):
            print("Sortie du mode RAG Chat.")
            break

        logger.info(f"Question reçue : {question}")

        # 1) Récupérer les documents pertinents
        docs = retriever.get_relevant_documents(question)
        logger.info(f"{len(docs)} document(s) récupéré(s).")

        # 2) Générer la réponse via la chaîne RAG
        result = chain.invoke({"question": question})
        print("\n=== Réponse ===")
        print(result)

        # 3) Afficher les extraits sources
        print("\n=== Extraits Sources ===")
        for doc in docs:
            snippet = doc.page_content.strip().replace("\n", " ")
            print(f"- {snippet[:200]}…")
        print("\n------------------------\n")
