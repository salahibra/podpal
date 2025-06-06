# app.py

import json
import os
import uuid
import logging
from flask import (
    Flask,
    request,
    session,
    render_template,
    send_from_directory,
    jsonify,
    redirect,
    url_for
)
from werkzeug.utils import secure_filename

# Vos utilitaires RAG (process_transcript, build_and_get_rag_chain)
from rag_utils import process_transcript, build_and_get_rag_chain

# Fonction de découpages textuels (chunks) si nécessaire
# Removed unused import
# Fonction de segmentation en chapitres
from chaptering import segment_by_topic
from model import summarize_chapters_and_global


app = Flask(__name__)
app.secret_key = "une_clé_quelconque_pour_la_session"

# Dossier pour stocker les fichiers uploadés
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Dossier persistant pour Chroma
VECTORDIR = os.path.join(os.getcwd(), "data", "vectorstores", "chunks")
os.makedirs(VECTORDIR, exist_ok=True)

# Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("podcast_ai_app")


# -----------------------------
#   Stockage d’état par utilisateur
# -----------------------------
# On va garder un dictionnaire en mémoire, indexé par session["uid"].
# _STORED[uid] = {
#    "raw_text": None or str,
#    "audio_filename": None or str,
#    "rag_ready": bool,
#    "chain": objet RAG,
#    "retriever": objet RAG,
# }
_STORED = {}


def _get_user_state():
    """
    Retourne le dictionnaire d’état propre à l’utilisateur courant.
    Si la session ne contient pas d’UID, on en génère un nouveau.
    """
    if "uid" not in session:
        uid = str(uuid.uuid4())
        session["uid"] = uid
        _STORED[uid] = {
            "raw_text": None,
            "audio_filename": None,
            "rag_ready": False,
            "chain": None,
            "retriever": None
        }
    else:
        uid = session["uid"]
        if uid not in _STORED:
            # Si on avait perdu l’état côté serveur, on le recrée
            _STORED[uid] = {
                "raw_text": None,
                "audio_filename": None,
                "rag_ready": False,
                "chain": None,
                "retriever": None
            }
    return _STORED[session["uid"]]


# -----------------------------
#   Route principale (import / transcription / RAG build)
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    """
    GET  : affiche la page d’accueil (formulaire d’import + onglets + lecteur audio).  
    POST : traite l’import (YouTube / audio local / texte), transcrit, 
           stocke raw_text et audio_filename dans l’état, 
           construit l’index Chroma + pipeline RAG (chain + retriever).
    """
    state = _get_user_state()
    error = None

    # Lecture de l’état pour le template
    raw_text = state["raw_text"]
    audio_filename = state["audio_filename"]
    # Removed unused variable

    if request.method == "POST":
        source_type = request.form.get("source_type", "")
        raw_text = ""
        audio_filename = None

        # --- Ingestion YouTube --- #
        if source_type == "youtube":
            yt_url = request.form.get("youtube_url", "").strip()
            if not yt_url.startswith(("http://", "https://")):
                error = "URL YouTube invalide."
            else:
                try:
                    logger.info(f"Ingestion depuis YouTube : {yt_url}")
                    from transcription import download_audio_from_youtube, transcribe_file

                    # 1) Download the audio to a temporary path (e.g. /tmp/xyz.wav)
                    wav_path = download_audio_from_youtube(yt_url)
                    basename = os.path.basename(wav_path)

                    # 2) Move it into UPLOAD_FOLDER so send_from_directory can find it
                    new_path = os.path.join(app.config["UPLOAD_FOLDER"], basename)
                    os.replace(wav_path, new_path)
                    # (If you prefer to copy instead of move, you could use `shutil.copy(wav_path, new_path)`)

                    # 3) Transcribe the file now located at new_path
                    raw_text = transcribe_file(new_path, beam_size=5)

                    # 4) Remember only the filename (Flask will serve /uploads/<audio_filename>)
                    audio_filename = basename

                except Exception as e:
                    logger.error(f"Erreur ingestion YouTube : {e}")
                    error = "Échec de la récupération ou de la transcription de l’audio YouTube."

        # --- Ingestion fichier audio local --- #
        elif source_type == "audio_file":
            audio_file = request.files.get("audio_upload")
            if not audio_file or audio_file.filename == "":
                error = "Veuillez sélectionner un fichier audio."
            else:
                try:
                    filename = secure_filename(audio_file.filename)
                    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                    audio_file.save(filepath)
                    logger.info(f"Ingestion depuis fichier audio local : {filepath}")

                    from transcription import transcribe_file
                    raw_text = transcribe_file(filepath, beam_size=5)
                    audio_filename = filename
                except Exception as e:
                    logger.error(f"Erreur transcription audio local : {e}")
                    error = "Échec de la transcription du fichier audio."

        # --- Ingestion fichier texte (.txt) --- #
        elif source_type == "text_file":
            txt_file = request.files.get("text_upload")
            if not txt_file or txt_file.filename == "":
                error = "Veuillez sélectionner un fichier texte."
            else:
                try:
                    txt_filename = secure_filename(txt_file.filename)
                    txt_path = os.path.join(app.config["UPLOAD_FOLDER"], txt_filename)
                    txt_file.save(txt_path)
                    logger.info(f"Ingestion depuis fichier texte local : {txt_path}")
                    with open(txt_path, "r", encoding="utf-8") as f:
                        raw_text = f.read()
                    # No audio file to play in this branch
                    audio_filename = None
                except Exception as e:
                    logger.error(f"Impossible de lire le fichier texte : {e}")
                    error = "Erreur lors de la lecture du fichier texte."


        else:
            error = "Type de source inconnu."

        # --- Si pas d’erreur, on stocke raw_text / audio_filename puis on build la pipeline RAG --- #
        if not error:
            if not raw_text.strip():
                error = "Échec de l’ingestion/transcription : texte vide."
            else:
                # Stocker l’état utilisateur
                state["raw_text"] = raw_text
                state["audio_filename"] = audio_filename

                # Pre-traitement : chunking + indexation Chroma
                try:
                    process_transcript(raw_text, persist_dir=VECTORDIR)
                except Exception as e:
                    logger.error(f"Erreur pendant process_transcript : {e}")
                    error = "Échec du prétraitement (chunking/indexation)."

                # Construire la pipeline RAG (chain + retriever)
                if not error:
                    try:
                        chain, retriever = build_and_get_rag_chain(persist_dir=VECTORDIR)
                        state["chain"] = chain
                        state["retriever"] = retriever
                        state["rag_ready"] = True
                        # Removed unused variable
                        logger.info("Pipeline RAG initialisée avec succès.")
                    except Exception as e:
                        logger.error(f"Erreur build_and_get_rag_chain : {e}")
                        error = "Échec de la construction de la pipeline RAG."

        # Après le POST, on redirige en GET pour afficher le formulaire + données
        return redirect(url_for("index"))

    # En GET, on passe le state au template
    return render_template(
        "index2.html",  # <--- votre template
        raw_text=state["raw_text"],
        audio_filename=state["audio_filename"],
        error=error,
        rag_ready=state["rag_ready"]
    )


# -----------------------------
#   Sert les fichiers uploadés (audio / texte)
# -----------------------------
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# -----------------------------
#   GET /get_chapters
# -----------------------------
@app.route("/get_chapters")
def get_chapters():
    state = _get_user_state()
    raw_text = state["raw_text"] or ""

    if not raw_text:
        # Pas d’erreur, mais liste vide si aucune transcription
        return jsonify({ "chapters": [] }), 200

    # Segmente en chapitres
    chapters_list = segment_by_topic(raw_text, threshold=0.25)
    # create title list
    titles  = ["Chapter "+str(i+1) for i in range(len(chapters_list))]
    # save titles to titles.json
    with open("data/titles.json", "w", encoding="utf-8") as f:
        json.dump({"titles": titles}, f, ensure_ascii=False, indent=4)

    # save chapters_list to chapters.json
    with open("data/chapters.json", "w", encoding="utf-8") as f:
        json.dump({"chapters": chapters_list}, f, ensure_ascii=False, indent=4)

    json_chapters = []
    for idx, chap_text in enumerate(chapters_list):
        title = titles[idx]
        json_chapters.append({ "index": idx, "title": title })

    return jsonify({ "chapters": json_chapters }), 200

# -----------------------------
#   GET /get_summaries
# -----------------------------
@app.route("/get_summaries", methods=["GET"])
def get_summaries():
    state = _get_user_state()
    raw_text = state["raw_text"] or ""
    if not raw_text:
        return jsonify({ "error": "Aucune transcription en mémoire." }), 400

    chapters_list = segment_by_topic(raw_text, threshold=0.25)
    if not chapters_list:
        return jsonify({ "error": "Aucun chapitre à résumer." }), 400

    summaries = summarize_chapters_and_global(chapters_list, model_path=os.getenv("MODEL_PATH"),
                                        output_path="data/summaries.json")["chapter_summaries"]
    # get titles from data/titles
    with open("data/titles.json", "r") as f:
        title_list = json.load(f)["titles"]
    json_summaries = []
    for idx, summary in enumerate(summaries):
        json_summaries.append({
            "index": idx,
            "title": title_list[idx],
            "summary": summary
        })

    return jsonify({ "summaries": json_summaries }), 200


# -----------------------------
#   GET /get_chapter_content/<index>
# -----------------------------
@app.route("/get_chapter_content/<int:index>")
def get_chapter_content(index):
    state = _get_user_state()
    raw_text = state["raw_text"] or ""
    if not raw_text:
        return jsonify({ "error": "Aucune transcription en mémoire." }), 400

    # get chapters from data/chapters.json
    with open("data/chapters.json", "r") as f:
        chapters_list = json.load(f)["chapters"]
    if index < 0 or index >= len(chapters_list):
        return jsonify({ "error": "Index de chapitre invalide." }), 400

    return jsonify({ "content": chapters_list[index] }), 200



# -----------------------------
#   GET /get_summary_content/<index>
# -----------------------------
@app.route("/get_summary_content/<int:index>")
def get_summary_content(index):
    state = _get_user_state()
    raw_text = state["raw_text"] or ""
    if not raw_text:
        return jsonify({ "error": "Aucune transcription en mémoire." }), 400

    # get summaries from data/summaries.json
    with open("data/summaries.json", "r") as f:
        summaries_list = json.load(f)["chapter_summaries"]
    

    if index < 0 or index >= len(summaries_list):
        return jsonify({ "error": "Index de résumé invalide." }), 400

    return jsonify({
        "title": "Résumé " + str(index + 1),
        "summary": summaries_list[index]
    }), 200


# -----------------------------
#   GET /get_global_summary
# -----------------------------
@app.route("/get_global_summary", methods=["GET"])
def get_global_summary():
    state = _get_user_state()
    raw_text = state["raw_text"] or ""
    if not raw_text:
        return jsonify({ "error": "Aucune transcription en mémoire." }), 400
    
    chapters_list = segment_by_topic(raw_text, threshold=0.25)
    if not chapters_list:
        return jsonify({ "error": "Aucun chapitre à résumer." }), 400

    global_summary = summarize_chapters_and_global(chapters_list,
                                                   model_path=os.getenv("MODEL_PATH"),
                                                   output_path="data/summaries.json")["global_summary"]
    return jsonify({ "global_summary": global_summary }), 200


# -----------------------------
#   POST /rag_chat
# -----------------------------
@app.route("/rag_chat", methods=["POST"])
def rag_chat():
    state = _get_user_state()
    if not state["rag_ready"]:
        return jsonify({ "error": "Pipeline RAG non initialisée." }), 400

    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({ "error": "Aucune question fournie." }), 400

    chain = state["chain"]
    retriever = state["retriever"]
    if chain is None or retriever is None:
        return jsonify({ "error": "Pipeline introuvable (chain/retriever)." }), 500

    logger.info(f"Requête RAG reçue : {question}")

    docs = retriever.get_relevant_documents(question)
    answer = chain.invoke({ "question": question })

    sources = []
    for doc in docs:
        snippet = doc.page_content.replace("\n", " ").strip()
        sources.append(snippet[:200] + "…")

    return jsonify({
        "answer": answer,
        "sources": sources
    }), 200

import recommandation
from build_podcast_vectorstore import build_podcast_chroma_index
@app.route("/build_index_podcasts", methods=["GET"])
def build_index_podcasts():
    """
    Cette route appelle directement la fonction `build_podcast_chroma_index`
    qui se trouve dans `recommandation.py`. Elle reconstruit l’index Chroma
    des épisodes de podcast stockés dans './podcast_dataset/podcast_epds_dataset.json'.
    """
    try:
        # Appel à votre fonction pré-existante
        build_podcast_chroma_index(persist_dir="data/vectorstores/podcast_eps")
        return jsonify({"status": "Index Chroma des podcasts reconstruit avec succès."}), 200

    except Exception as e:
        logger.error(f"Erreur lors de build_podcast_chroma_index : {e}")
        return jsonify({"error": f"Échec de la construction de l’index : {str(e)}"}), 500

@app.route("/get_recommendations", methods=["GET"])
def get_recommendations():
    """
    Cette route appelle directement la fonction `run_recommendation_from_summary_chroma`
    qui se trouve dans `recommandation.py`. Elle renvoie en JSON la liste des recommandations.

    ATTENTION :
      - run_recommendation_from_summary_chroma affiche par défaut les résultats au format console.
      - Pour la retourner en JSON, on adapte un peu son appel en interceptant la liste qu'elle génère.
    """

    try:
        # run_recommendation_from_summary_chroma attend par défaut top_k=5 et persiste dans "data/vectorstores/podcast_eps"
        # Mais pour récupérer ses recommandations au lieu de juste les afficher, on modifie légèrement son code
        # pour qu'il renvoie la liste. On crée un wrapper temporaire ici.

        # On duplique l'essentiel de run_recommendation_from_summary_chroma pour récupérer la liste :
        from recommandation import load_global_summary
        from langchain.vectorstores import Chroma
        from langchain.embeddings import HuggingFaceEmbeddings

        # 1) Charger le résumé global
        global_summary = load_global_summary(filepath="data/summaries.json")

        # 2) Préparer l’embedder + recharger l’index Chroma (persisté précédemment)
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        persist_dir = "data/vectorstores/podcast_eps"
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)

        # 3) Lancer la recherche de similarité (on prend top_k=5)
        top_k = 5
        results = vectorstore.similarity_search_with_relevance_scores(global_summary, k=top_k)

        # 4) Construire la liste de recommandations sous forme de dict
        recommendations = []
        for rank, (doc, score) in enumerate(results, start=1):
            meta = doc.metadata or {}
            recommendations.append({
                "rank": rank,
                "podcast_title": meta.get("podcast_title", "Unknown"),
                "episode_title": meta.get("episode_title", "Unknown"),
                "description": doc.page_content,
                "episode_link": meta.get("episode_link", "N/A"),
                "score": float(score)
            })

        return jsonify({"recommendations": recommendations}), 200

    except FileNotFoundError as fnf:
        logger.error(f"Fichier introuvable : {fnf}")
        return jsonify({"error": f"Fichier non trouvé : {str(fnf)}"}), 500

    except Exception as e:
        logger.error(f"Erreur lors de la récupération des recommandations : {e}")
        return jsonify({"error": f"Échec des recommandations : {str(e)}"}), 500

# -----------------------------
#   Lancement de l’application
# -----------------------------
if __name__ == "__main__":
    
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)
