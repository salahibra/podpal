from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def segment_by_topic(text, threshold=0.5):
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = ' '.join(text.split())
    
    chunks = []
    current_chunk = ""
    for sentence in text.split('.'):
        if len(current_chunk) + len(sentence) + 1 <= 500:
            current_chunk += sentence + "."        
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + "."
    if current_chunk:
        chunks.append(current_chunk.strip())

    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    
    chapters = []
    current_chapter = chunks[0]
    for i in range(1, len(chunks)):
        similarity = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
        if similarity < threshold:
            chapters.append(current_chapter.strip())
            current_chapter = chunks[i]
        else:
            current_chapter += " " + chunks[i]
    
    if current_chapter:
        chapters.append(current_chapter.strip())
    
    return chapters

# Example usage
chapters = segment_by_topic(result["text"], threshold=0.2)
for i, chapter in enumerate(chapters, 1):
    print(f"Chapter {i}:\n{chapter}\n")
print(f"Total number of chapters: {len(chapters)}")