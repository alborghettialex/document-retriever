from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pathlib import Path
from src.core.text.text_extraction import get_text_extractor
from src.backend.utils.populate import process_document

router = APIRouter()

@router.post("/upload")
async def upload_documents(request: Request, files: list[UploadFile] = File(...)):

    for file in files:
        save_path = f"archive/uploaded/{file.filename}"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        text_extractor = get_text_extractor(save_path)

        try:
            process_document(
                file_path=save_path,
                text_extractor=text_extractor,
                chunker=request.app.state.chunker,
                embedder=request.app.state.embedder,
                db=request.app.state.db,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    return {"message": "Completed!", "filename": file.filename}