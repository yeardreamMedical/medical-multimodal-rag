from pydantic import BaseModel, Field
from typing import List, Optional

class TopicAnalysis(BaseModel):
    estimated_topic: str
    difficulty_level: str
    clinical_relevance: str

class ImageSelection(BaseModel):
    selected_image_type: str
    korean_name: str
    reason: str

class SelectedImage(BaseModel):
    image_path: str
    score: float
    labels: List[str]

class QuestionResponse(BaseModel):
    question: str
    options: List[str]
    answer: int
    explanation: str
    topic_analysis: TopicAnalysis
    image_selection: ImageSelection
    selected_images: List[SelectedImage]

class SimilarImage(BaseModel):
    id: str
    title: str
    description: str
    url: str
    similarity: int
    labels: List[str]

class SimilarImagesResponse(BaseModel):
    topic: str
    total_count: int
    images: List[SimilarImage]

class QuestionRequest(BaseModel):
    topic: str = Field(..., example="Pneumonia")
