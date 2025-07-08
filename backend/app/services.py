from . import schemas
from .config import settings
from pinecone import Pinecone, Index

import google.generativeai as genai
import openai
import json

# Initialize Pinecone
pc = Pinecone(api_key=settings.PINECONE_API_KEY, environment=settings.PINECONE_ENVIRONMENT)
text_index: Index = pc.Index(settings.PINECONE_TEXT_INDEX_NAME)
image_index: Index = pc.Index(settings.PINECONE_IMAGE_INDEX_NAME) # Add image index initialization

# Configure OpenAI API
openai.api_key = settings.OPENAI_API_KEY

# Configure Gemini API
genai.configure(api_key=settings.GEMINI_API_KEY)

# System Prompt
SYSTEM_PROMPT = """
당신은 존스 홉킨스 의학부에서 한달 1000000 USD를 받고 학생들을 가르치는 의대 교수입니다.
주어진 '참조 컨텍스트'와 '질문'을 바탕으로 학생을 가르치듯 정확하고 상세하게 답변하십시오.
컨텍스트에 명시적으로 없는 내용은 답변하지 마십시오.
만약 주어진 정보만으로는 답변하기 어렵다면, '주어진 정보만으로는 답변하기 어렵습니다.'라고 말하십시오.
"""

# Initialize Gemini Model
gemini_model = genai.GenerativeModel(model_name='gemini-1.5-pro', system_instruction=SYSTEM_PROMPT)



def generate_question_service(topic: str) -> schemas.QuestionResponse:
    # 1. Get embedding for the topic
    response = openai.embeddings.create(
        input=topic,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding

    # 2. Query Pinecone text index
    search_results = text_index.query(
        vector=query_embedding,
        top_k=5,  # Retrieve top 5 relevant text chunks
        include_metadata=True
    )

    retrieved_texts = []
    for match in search_results.matches:
        if 'chunk_text' in match.metadata:
            retrieved_texts.append(match.metadata['chunk_text'])
    
    context_str = "\n---\n".join(retrieved_texts)
    print(f"Retrieved texts for '{topic}':\n" + context_str)

    # 3. Construct User Prompt for Gemini
    user_prompt = f"""
[참조 컨텍스트]
{context_str}

[사용자 질문]
{topic}과 관련된 의대생 수준의 객관식 문제를 다음 JSON 형식으로 생성해줘:
{{
    "question": "{topic}과 관련된 다음 환자 사례를 보고 가장 적절한 답을 선택하세요. 환자의 병력과 신체검사, 검사 소견을 종합할 때 가장 가능성이 높은 진단은?",
    "options": [
        "{topic}의 초기 단계",
        "{topic}의 전형적인 형태",
        "{topic}과 유사한 다른 질환",
        "{topic}의 합병증",
        "{topic}과 무관한 질환"
    ],
    "answer": 1,
    "explanation": "...",
    "topic_analysis": {{
        "estimated_topic": "...",
        "difficulty_level": "...",
        "clinical_relevance": "..."
    }},
    "image_selection": {{
        "selected_image_type": "...",
        "korean_name": "...",
        "reason": "..."
    }},
    "selected_images": []
}}
"""

    # 4. Call Gemini API
    try:
        response = gemini_model.generate_content(user_prompt, generation_config={'response_mime_type': 'application/json'})
        if not response.text:
            raise ValueError("Gemini returned an empty response.")
        response_json = json.loads(response.text)
        print(f"Gemini Response: {response_json}")
        return schemas.QuestionResponse(**response_json)
    except Exception as e:
        print(f"Error generating content with Gemini: {e}")
        # Fallback to mock data or raise an HTTPException
        return schemas.QuestionResponse(
            question=f"문제 생성 중 오류가 발생했습니다: {e}",
            options=["오류", "오류", "오류", "오류", "오류"],
            answer=0,
            explanation="문제 생성 중 오류가 발생했습니다. 다시 시도해주세요.",
            topic_analysis=schemas.TopicAnalysis(
                estimated_topic="오류",
                difficulty_level="오류",
                clinical_relevance="오류"
            ),
            image_selection=schemas.ImageSelection(
                selected_image_type="None",
                korean_name="이미지 없음",
                reason="오류 발생"
            ),
            selected_images=[]
        )

def get_similar_images_service(topic: str) -> schemas.SimilarImagesResponse:
    # 1. Get embedding for the topic
    response = openai.embeddings.create(
        input=topic,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding

    # 2. Query image index to find a representative image for the topic
    #    We'll use top_k=1 to get the most relevant image as a source for similarity search
    source_image_results = image_index.query(
        vector=query_embedding,
        top_k=1,
        include_metadata=True,
        include_values=True # We need the vector of the source image
    )

    source_image_vector = None
    source_image_id = None
    if source_image_results.matches:
        source_image_match = source_image_results.matches[0]
        source_image_vector = source_image_match.values
        source_image_id = source_image_match.id
        print(f"Found source image for topic '{topic}': {source_image_match.metadata.get('image_path')}")
    else:
        print(f"No source image found for topic '{topic}'. Returning empty list.")
        return schemas.SimilarImagesResponse(
            topic=topic,
            total_count=0,
            images=[]
        )

    # 3. Query image index for similar images using the source image's vector
    similar_images_results = image_index.query(
        vector=source_image_vector,
        top_k=6, # Get top 6 to exclude the source image itself
        include_metadata=True
    )

    similar_images = []
    for match in similar_images_results.matches:
        if match.id != source_image_id: # Exclude the source image itself
            similar_images.append(schemas.SimilarImage(
                id=match.id,
                title=match.metadata.get('primary_label', 'N/A'), # Use primary_label as title
                description=match.metadata.get('all_descriptions', 'N/A'), # Use all_descriptions as description
                url=match.metadata.get('image_path', 'N/A'), # Use image_path as URL
                similarity=int(match.score * 100), # Convert score to percentage
                labels=match.metadata.get('labels', [])
            ))
    
    print(f"Found {len(similar_images)} similar images for topic '{topic}'.")
    return schemas.SimilarImagesResponse(
        topic=topic,
        total_count=len(similar_images),
        images=similar_images
    )