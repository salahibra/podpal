

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import logging
from typing import List

logger = logging.getLogger(__name__)

def get_embeddings(texts: List[str]) -> List:
    """
    Retourne la liste d’array numpy d’embeddings pour chaque texte via all-MiniLM-L6-v2.
    
    """
    logger.info(f"Calcul des embeddings pour {len(texts)} textes")
    hf_emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    # Si vous voulez uniquement recupérer les vecteurs dans un np.ndarray :
    return hf_emb.embed_documents(texts)


