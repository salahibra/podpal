import json
import numpy as np
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

def load_test_summaries(filepath='./podcast_dataset/episode_descriptions.json'):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['summaries']

def embed_text(text, embedding_model):
    return np.array(embedding_model.embed_query(text)).reshape(1, -1)

def run_tests(persist_dir="data/vectorstores/podcast_eps", threshold=0.6, top_k=5):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)

    test_summaries = load_test_summaries()

    total_valid_recos = 0
    total_recos = len(test_summaries) * top_k

    for i, summary in enumerate(test_summaries, 1):
        print(f"\n=== Test Summary {i} ===")
        print(f"Query: {summary}\n")
        results = vectorstore.similarity_search_with_relevance_scores(summary, k=top_k)

        summary_emb = embed_text(summary, embedding_model)
        valid_recos = 0

        for rank, (doc, score) in enumerate(results, 1):
            doc_emb = embed_text(doc.page_content, embedding_model)
            cos_sim = cosine_similarity(summary_emb, doc_emb)[0][0]
            is_valid = cos_sim >= threshold
            if is_valid:
                valid_recos += 1

            meta = doc.metadata
            print(f"Reco {rank}: {meta.get('episode_title', 'Unknown')}")
            print(f" Description: {doc.page_content[:150]}...")
            print(f" Cosine Similarity: {cos_sim:.3f} - {'VALID' if is_valid else 'INVALID'}\n")

        print(f"Summary {i}: {valid_recos}/{top_k} recommendations passed the threshold of {threshold}")
        total_valid_recos += valid_recos

    avg_valid_per_summary = total_valid_recos / len(test_summaries)
    valid_percentage = (total_valid_recos / total_recos) * 100

    print("\n=============================")
    print(f"Total summaries tested: {len(test_summaries)}")
    print(f"Total recommendations checked: {total_recos}")
    print(f"Total valid recommendations (cosine >= {threshold}): {total_valid_recos}")
    print(f"Average valid recommendations per summary: {avg_valid_per_summary:.2f} / {top_k}")
    print(f"Overall valid recommendations percentage: {valid_percentage:.2f}%")
    print("=============================\n")

if __name__ == "__main__":
    run_tests()
