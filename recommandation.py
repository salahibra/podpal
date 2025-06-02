from sentence_transformers import SentenceTransformer
import json
from sklearn.metrics.pairwise import cosine_similarity
from embedding import get_embeddings 
import numpy as np
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os



def load_global_summary(filepath='./data/summaries.json'):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['global_summary']

def run_recommendation_from_summary_chroma(top_k=5, persist_dir="data/vectorstores/podcast_eps"):
    query_text = load_global_summary()

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)

    results = vectorstore.similarity_search_with_relevance_scores(query_text, k=top_k)

    print(f"\n📌 Résultats pour le résumé global :\n\"{query_text}\"\n")
    for i, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        print(f"--- Recommandation {i} ---")
        print(f"🎙️ Podcast       : {meta.get('podcast_title', 'Unknown')}")
        print(f"🎧 Épisode       : {meta.get('episode_title', 'Unknown')}")
        print(f"📝 Description   : {doc.page_content}")
        print(f"🔗 Lien          : {meta.get('episode_link', 'N/A')}")
        print()

if __name__ == "__main__":
    run_recommendation_from_summary_chroma()



