
import os
import logging
from typing import List, Any

# 1) Importer vos modules de transcription
from transcription import download_audio_from_youtube, transcribe_file

# 2) Importer vos modules de prétraitement (chunking + vectorstore)
from chunking import get_text_chunks
from vectorstore import get_vectorstore

# 3) Importer la fonction qui construit la chaîne RAG
from rag_chat import get_rag_chain

# Configurer un logger local
logger = logging.getLogger(__name__)


def choose_source() -> str:
    """
    Affiche un menu en console et retourne le choix (1, 2 ou 3) sous forme de chaîne.
    """
    print("Étape 1) Choisissez la source :")
    print(" 1. URL YouTube")
    print(" 2. Fichier audio local (.mp3/.wav)")
    print(" 3. Fichier transcript (.txt existant)")
    choix = input("Choix (1/2/3) : ").strip()
    return choix


def ingest_from_youtube() -> str:
    """
    Télécharge et transcrit un podcast à partir d'une URL YouTube.
    Retourne le texte complet transcrit, ou '' en cas d'échec.
    """
    yt_url = input("Entrez l'URL YouTube du podcast : ").strip()
    if not yt_url.lower().startswith(("http://", "https://")):
        print("URL invalide, sortie.")
        return ""

    logger.info(f"Téléchargement audio depuis YouTube : {yt_url}")
    try:
        wav_path = download_audio_from_youtube(yt_url)
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement YouTube : {e}")
        print("Impossible de télécharger l’audio. Vérifiez l’URL ou votre connexion.")
        return ""

    logger.info(f"Transcription du fichier WAV : {wav_path}")
    try:
        raw_text = transcribe_file(wav_path, beam_size=5)
    except Exception as e:
        logger.error(f"Erreur lors de la transcription FastWhisper : {e}")
        print("Impossible de transcrire l’audio.")
        return ""

    return raw_text


def ingest_from_local_audio() -> str:
    """
    Transcrit un podcast à partir d'un fichier audio local (.mp3 ou .wav).
    Retourne le texte complet transcrit, ou '' en cas d'échec.
    """
    chemin_audio = input("Chemin du fichier audio local : ").strip()
    if not os.path.isfile(chemin_audio):
        print(f"Le fichier '{chemin_audio}' n'existe pas, sortie.")
        return ""

    logger.info(f"Transcription du fichier audio local : {chemin_audio}")
    try:
        raw_text = transcribe_file(chemin_audio, beam_size=5)
    except Exception as e:
        logger.error(f"Erreur lors de la transcription FastWhisper : {e}")
        print("Impossible de transcrire l’audio.")
        return ""

    return raw_text


def ingest_from_text_file() -> str:
    """
    Charge un fichier .txt contenant déjà la transcription.
    Retourne le contenu intégral, ou '' si le fichier est introuvable.
    """
    chemin_txt = input("Chemin du fichier transcript (.txt) : ").strip()
    if not os.path.isfile(chemin_txt):
        print(f"Le fichier '{chemin_txt}' n'existe pas, sortie.")
        return ""

    with open(chemin_txt, "r", encoding="utf-8") as f:
        raw_text = f.read()

    logger.info(f"Transcript chargé depuis '{chemin_txt}'")
    return raw_text


def ingest_transcript() -> str:
    """
    Wrapper principal d’ingestion. 
    Affiche le menu, appelle la fonction appropriée, retourne le texte complet (ou '' si erreur).
    """
    choix = choose_source()
    if choix == "1":
        return ingest_from_youtube()
    elif choix == "2":
        return ingest_from_local_audio()
    elif choix == "3":
        return ingest_from_text_file()
    else:
        print("Choix invalide, sortie.")
        return ""


def process_transcript(raw_text: str, persist_dir: str = "data/vectorstores/chunks") -> None:
    """
    Découpe le texte (raw_text) en chunks puis crée ou recharge le VectorStore Chroma 
    dans le dossier `persist_dir`. Affiche un message si raw_text est vide.
    """
    if not raw_text:
        print("Aucun texte à traiter. Veuillez ingérer un podcast d’abord.")
        return

    print("\nÉtape 2) Découpage en chunks & indexation du VectorStore Chroma…")
    # 1) Découper en chunks
    chunks = get_text_chunks(raw_text)
    logger.info(f"{len(chunks)} chunks générés.")

    # 2) Création (ou recharge) du VectorStore Chroma
    os.makedirs(persist_dir, exist_ok=True)
    vectordb = get_vectorstore(chunks, persist_dir=persist_dir)
    logger.info(f"VectorStore Chroma persistant dans : {persist_dir}")


def build_and_get_rag_chain(persist_dir: str = "data/vectorstores/chunks") -> Any:
    """
    Recharge le VectorStore Chroma existant (créé par process_transcript) puis 
    construit la pipeline RAG (Runnable). Retourne (chain, retriever).
    Lève FileNotFoundError si `persist_dir` n'existe pas.
    """
    if not os.path.isdir(persist_dir):
        raise FileNotFoundError(
            f"Le dossier '{persist_dir}' est introuvable. "
            "Veuillez exécuter d’abord process_transcript() pour indexer vos chunks."
        )

    print("\nÉtape 3) Construction de la chaîne RAG…")
    chain, retriever = get_rag_chain(persist_dir=persist_dir)
    return chain, retriever


def ask_questions_loop(chain: Any, retriever: Any) -> None:
    """
    Boucle interactive de questions/réponses (RAG).
    Lit la question, récupère les docs pertinents, invoque la chaîne RAG et affiche la réponse + sources.
    Tapez 'exit' ou 'quit' pour quitter.
    """
    print("\nPipeline RAG prêt. Posez vos questions (tapez 'exit' pour quitter).\n")
    while True:
        question = input("Votre question : ").strip()
        if question.lower() in ("exit", "quit"):
            print("Fin de la session RAG. À bientôt !")
            break
        if not question:
            continue

        logger.info(f"Question reçue : {question}")
        docs = retriever.get_relevant_documents(question)
        logger.info(f"{len(docs)} document(s) récupéré(s).")

        answer = chain.invoke({"question": question})
        print("\n--- Réponse ---")
        print(answer)

        print("\n--- Extraits Sources ---")
        for doc in docs:
            snippet = doc.page_content.strip().replace("\n", " ")
            print(f"- {snippet[:200]}…")
        print("\n-----------------------------------\n")
