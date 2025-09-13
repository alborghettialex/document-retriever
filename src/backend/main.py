import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.bootstrap_registry import register_all
from src.backend.routes import api_router
from src.utils.config import load_config
from src.core.embeddings.registry import get_embedder
from src.core.text.chunk.registry import get_chunker
from src.core.storage.registry import get_vectordb

@asynccontextmanager
async def lifespan(app: FastAPI):
    register_all()

    config = load_config("config/config.yaml")

    app.state.embedder_config = config["embedder"]
    app.state.chunker_config = config["chunker"]
    app.state.db_config = config["db"]

    app.state.embedder = get_embedder(
        **app.state.embedder_config
    )
    if app.state.chunker_config["mode"] == "semantic":
        app.state.chunker_config["params"]["embedder"] = app.state.embedder

    app.state.chunker = get_chunker(**app.state.chunker_config)
    app.state.db = get_vectordb(app.state.db_config["name"], **app.state.db_config["params"])

    yield


app = FastAPI(title="Document Retriever API", lifespan=lifespan)

app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run(
        "src.backend.main:app",  
        host="0.0.0.0",      
        port=8000,
        reload=True          
    )