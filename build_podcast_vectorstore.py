from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import json
import os

def load_local_podcasts(filepath='./podcast_dataset/podcast_epds_dataset.json'):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_podcast_chroma_index(persist_dir="data/vectorstores/podcast_eps"):
    episodes = load_local_podcasts()
    descriptions = [ep['episode_description'] for ep in episodes]
    metadata = episodes

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    docs = [Document(page_content=desc, metadata=meta) for desc, meta in zip(descriptions, metadata)]

    vectorstore = Chroma.from_documents(docs, embedding=embedding_model, persist_directory=persist_dir)
    vectorstore.persist()
    print("✅ Podcast dataset indexé et stocké dans Chroma.")

if __name__ == "__main__":
    build_podcast_chroma_index()

