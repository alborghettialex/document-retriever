def register_all():
    """
    Dynamically imports all modules that register components 
    (e.g., chunkers, embedders, vector stores).
    """
    from src.core.text.chunk import chunking  
    from src.core.embeddings import algorithms 
    from src.core.storage import database