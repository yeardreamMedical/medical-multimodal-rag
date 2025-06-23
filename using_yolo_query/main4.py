import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Dict

import torch
from dotenv import load_dotenv
from ultralytics import YOLO
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 사용자 정의 엔진 (ImageSearcher 포함)
from search.search_engine import SearchEngine

# Load environment variables
load_dotenv(".env")

# Load YOLO model
yolo_model = YOLO("using_yolo_query/best.pt")

# 한국어 맵핑
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

# 로컬 이미지 저장 경로
LOCAL_IMAGE_DIR = "C:/Users/MAIN/Workplace/yeardream/yeardreamMedical/medical-multimodal-rag/data/chestxray14/bbox_images"

def detect_lesions_yolo(image_path: str) -> List[Dict]:
    results = yolo_model(image_path)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            detections.append({"label": label, "confidence": confidence, "bbox": bbox})
    return detections

def search_similar_images_by_label(det: Dict, engine, top_k: int = 4) -> List[Dict]:
    label = det['label']
    query_bbox = det['bbox']
    print(f"   🔹 질병: {label}, bbox: {query_bbox}")
    all_candidates = []

    try:
        disease_results = engine.search_images_by_disease(label, top_k=20)
    except Exception as e:
        print(f"   ⚠️ '{label}' 질병 검색 오류: {e}")
        return []

    for result in disease_results:
        for roi_bbox_str in result.get('bbox_info', []):
            try:
                roi_bbox = list(map(int, roi_bbox_str.split(",")))
                q_cx = (query_bbox[0] + query_bbox[2]) / 2
                q_cy = (query_bbox[1] + query_bbox[3]) / 2
                r_cx = (roi_bbox[0] + roi_bbox[2]) / 2
                r_cy = (roi_bbox[1] + roi_bbox[3]) / 2
                dist = np.sqrt((q_cx - r_cx)**2 + (q_cy - r_cy)**2)
                score = -dist
                all_candidates.append((score, result))
            except:
                continue

    sorted_results = sorted(all_candidates, key=lambda x: x[0], reverse=True)
    seen = set()
    final = []
    for score, item in sorted_results:
        if item['image_id'] not in seen:
            final.append(item)
            seen.add(item['image_id'])
        if len(final) >= top_k:
            break
    print(f"✅ {label} 유사 이미지 {len(final)}개 선택")
    return final

def show_matched_images_by_labels(match_dict: Dict[str, List[Dict]]):
    for label, matches in match_dict.items():
        if not matches:
            continue

        fig, axes = plt.subplots(1, len(matches), figsize=(4 * len(matches), 4))
        if len(matches) == 1:
            axes = [axes]

        for i, match in enumerate(matches):
            image_file = match.get("image_file") or os.path.basename(match.get("image_path", ""))
            full_path = os.path.join(LOCAL_IMAGE_DIR, image_file)

            labels = match.get("labels", [])
            label_text = ", ".join(labels) if labels else match.get("primary_label", "Unknown")

            if os.path.exists(full_path):
                img = Image.open(full_path).convert("L")
                axes[i].imshow(img, cmap='gray')
                axes[i].set_title(f"{label_text}")
                axes[i].axis('off')
            else:
                print(f"❌ 이미지 없음: {full_path}")
                axes[i].set_title("이미지 없음")
                axes[i].axis('off')

        plt.suptitle(f"질병: {label_kor_map.get(label, label)}", fontsize=16)
        plt.tight_layout()
        plt.show()

def main(image_path: str):
    print("📸 YOLO 감지 중...")
    detections = detect_lesions_yolo(image_path)
    for d in detections:
        label_kr = label_kor_map.get(d['label'], d['label'])
        print(f"- {label_kr} {d['confidence']*100:.1f}%")

    print("\n🔍 Pinecone 유사도 검색 + ROI 재정렬 중...")
    engine = SearchEngine()

    label_to_matches = {}
    for det in detections:
        matches = search_similar_images_by_label(det, engine, top_k=4)
        label_to_matches[det['label']] = matches

    print("\n🖼️ 유사 이미지 시각화")
    show_matched_images_by_labels(label_to_matches)

if __name__ == "__main__":
    query_image_path = "C:/Users/MAIN/Workplace/yeardream/yeardreamMedical/medical-multimodal-rag/data/chestxray14/bbox_images/00010575_002.png"
    main(query_image_path)
