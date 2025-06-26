
import os
import sys
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO
from pinecone import Pinecone
from openai import OpenAI

# Load environment variables
load_dotenv(".env")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Load YOLO model
yolo_model = YOLO("using_yolo_query/best.pt")

# Korean label map
label_kor_map = {
    "Atelectasis": "무기폐",
    "Effusion": "흉수",
    "Pneumonia": "폐렴",
    "Mass": "종괴",
    "Nodule": "결절",
    "Cardiomegaly": "심장비대",
    "Pneumothorax": "기흉",
    "Infiltrate": "침윤/경화"
}

def detect_lesions_yolo(image_path: str) -> List[Dict]:
    results = yolo_model(image_path)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": bbox
            })
    return detections

def generate_detection_summary(detections: List[Dict]) -> str:
    if not detections:
        return "감지된 병변이 없습니다."
    parts = []
    for det in detections:
        label = det["label"]
        label_kr = label_kor_map.get(label, label)
        conf = det["confidence"]
        parts.append(f"{label_kr}({label}): {conf*100:.1f}%")
    return "감지된 병변: " + ", ".join(parts)

def build_text_query_from_detections(detections: List[Dict]) -> str:
    labels = [label_kor_map.get(d["label"], d["label"]) for d in detections]
    return "유사한 흉부 X-ray에서 다음 병변이 보이는 사례를 검색: " + ", ".join(labels)

def dummy_text_embedder(text: str) -> List[float]:
    # Simulated embedding for search (for real use, replace with actual embedder)
    np.random.seed(abs(hash(text)) % (10 ** 8))
    vec = np.random.randn(1536)
    return (vec / np.linalg.norm(vec)).tolist()

def search_similar_cases(query_text: str) -> List[Dict]:
    query_vector = dummy_text_embedder(query_text)
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )
    return results["matches"]

def summarize_with_gpt(matches: List[Dict]) -> str:
    context = "\n".join([f"- {m.get('metadata', {}).get('text', '')}" for m in matches])
    prompt = f"""다음은 흉부 X-ray 병변에 대해 유사한 설명들입니다:\n{context}\n\n이 내용을 종합적으로 요약하고 해석해 주세요."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

def main(image_path: str):
    print("📸 Step 1: YOLO 병변 감지 중...")
    detections = detect_lesions_yolo(image_path)
    summary = generate_detection_summary(detections)
    print("✅ YOLO 결과:", summary)

    print("\n🔍 Step 2: Pinecone 유사 이미지 검색...")
    query_text = build_text_query_from_detections(detections)
    matches = search_similar_cases(query_text)
    for i, m in enumerate(matches):
        print(f"- [{m['id']}] {m.get('metadata', {}).get('text', '')[:100]}...")

    if OPENAI_API_KEY:
        print("\n🧠 Step 3: GPT 요약 생성 중...")
        final_summary = summarize_with_gpt(matches)
        print("\n📘 요약 결과:")
        print(final_summary)
    else:
        print("\n❗ OPENAI_API_KEY가 설정되지 않아 GPT 요약을 생략합니다.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗ 사용법: python main.py [이미지경로]")
        sys.exit(1)
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"❌ 이미지 파일이 존재하지 않습니다: {image_path}")
        sys.exit(1)
    main(image_path)
