from typing import Union
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from pydantic import BaseModel
from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from app.image_classifier import ImageClassifier
from app.digit_generator import DigitGenerator

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. It tells the story of Edmond Dantes, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel()
image_classifier = ImageClassifier()
digit_generator = DigitGenerator()

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class EmbeddingRequest(BaseModel):
    word: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = embedding_model.get_embedding(request.word)
    return {"word": request.word, "embedding": embedding}

@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    if file.content_type is None:
        raise HTTPException(
            status_code=400,
            detail="Missing file type.",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    try:
        prediction = image_classifier.predict(image_bytes)
        return prediction

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be processed.",
        ) from error
        
@app.get("/generate-digits")
def generate_digits(num_images: int = 16):
    if num_images < 1 or num_images > 64:
        raise HTTPException(
            status_code=400,
            detail="num_images must be between 1 and 64.",
        )

    image_bytes = digit_generator.generate(
        num_images=num_images
    )

    return Response(
        content=image_bytes,
        media_type="image/png",
    )