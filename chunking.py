
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

def get_text_chunks(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None
) -> List[str]:
    
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    logger.info(f"Découpage du texte en chunks (size={chunk_size}, overlap={chunk_overlap})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len
    )
    chunks = splitter.split_text(text)
    logger.info(f"{len(chunks)} chunks générés (size={chunk_size}, overlap={chunk_overlap})")
    return chunks

