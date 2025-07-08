from fastapi import APIRouter, Depends
from . import schemas, services

router = APIRouter()

@router.post("/generate-question", response_model=schemas.QuestionResponse)
def generate_question(request: schemas.QuestionRequest):
    """
    Generates a medical question based on the provided topic.
    """
    return services.generate_question_service(request.topic)

@router.get("/similar-images/{topic}", response_model=schemas.SimilarImagesResponse)
def get_similar_images(topic: str):
    """
    Retrieves similar images based on the provided topic.
    """
    return services.get_similar_images_service(topic)
