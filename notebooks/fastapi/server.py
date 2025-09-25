from fastapi import FastAPI, UploadFile
from notebooks.fastapi.fastvlm import load_model_and_tokenizer, describe_image
import shutil
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator


models = {}

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    models["vlm"] = load_model_and_tokenizer()
    yield
    # Run cleanup code here
    models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/describe-image/")
async def describe_uploaded_image(file: UploadFile):
    # Save the uploaded file temporarily
    temp_file_path = f"/tmp/{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Call the describe_image function
        tok, model = models["vlm"]
        description = describe_image(temp_file_path, tok, model)
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)
    
    return {"description": description}
