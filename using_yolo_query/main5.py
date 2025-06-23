import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict

import torch
from dotenv import load_dotenv
from ultralytics import YOLO
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from search.search_engine import SearchEngine

# Load environment variables
load_dotenv(".env")

# Load YOLO model
yolo_model = YOLO("using_yolo_query/best.pt")

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

LOCAL_IMAGE_DIR = "C:/Users/MAIN/Workplace/yeardream/yeardreamMedical/medical-multimodal-rag/data/chestxray14/bbox_images"
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"

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

def draw_bbox(image: Image.Image, detections: List[Dict], label_map=True) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(FONT_PATH, 16)
    except:
        font = ImageFont.load_default()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = det['label']
        if label_map:
            label = label_kor_map.get(label, label)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, max(0, y1 - 16)), label, fill="red", font=font)
    return image

def search_similar_images_from_yolo_detections(image_path: str, yolo_detections: List[Dict], engine, top_k: int = 4):
    print(f"\n🔍 [RAG] '{image_path}'에서 YOLO 기반 유사 이미지 검색 시작")
    if not yolo_detections:
        print("⚠️ YOLO 감지 결과 없음")
        return {}

    results_per_label = {}

    for det in yolo_detections:
        label = det['label']
        query_bbox = det['bbox']
        print(f"   🔹 질병: {label}, bbox: {query_bbox}")

        try:
            disease_results = engine.search_images_by_disease(label, top_k=20)
        except Exception as e:
            print(f"   ⚠️ '{label}' 질병 검색 오류: {e}")
            continue

        all_candidates = []
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
                    result['roi_bbox'] = roi_bbox  # Store selected bbox
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

        results_per_label[label] = final

    return results_per_label

def show_matched_images_by_label(query_image_path: str, query_detections: List[Dict], matches_per_label: Dict[str, List[Dict]]):
    for label, matches in matches_per_label.items():
        print(f"\n🖼️ {label} 유사 이미지 시각화")
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))

        img = Image.open(query_image_path).convert("RGB")
        filtered_detections = [det for det in query_detections if det['label'] == label]
        query_img = draw_bbox(img.copy(), filtered_detections)
        axes[0].imshow(query_img)
        axes[0].set_title(f"[쿼리] {label_kor_map.get(label, label)}")
        axes[0].axis('off')

        for i, match in enumerate(matches):
            if i >= 4:
                break
            image_file = match.get("image_file") or os.path.basename(match.get("image_path", ""))
            full_path = os.path.join(LOCAL_IMAGE_DIR, image_file)
            label_text = ", ".join(match.get("labels", [])) or match.get("primary_label", "Unknown")
            roi_bbox = match.get("roi_bbox")
            if os.path.exists(full_path):
                result_img = Image.open(full_path).convert("RGB")
                if roi_bbox:
                    result_img = draw_bbox(result_img.copy(), [{"label": label, "bbox": roi_bbox}], label_map=True)
                axes[i + 1].imshow(result_img)
                axes[i + 1].set_title(f"{label_text}")
                axes[i + 1].axis('off')
            else:
                print(f"❌ 이미지 없음: {full_path}")
                axes[i + 1].set_title("이미지 없음")
                axes[i + 1].axis('off')

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
    matches_per_label = search_similar_images_from_yolo_detections(image_path, detections, engine, top_k=4)

    show_matched_images_by_label(image_path, detections, matches_per_label)

if __name__ == "__main__":
    query_image_path = "C:/Users/MAIN/Workplace/yeardream/yeardreamMedical/medical-multimodal-rag/data/chestxray14/bbox_images/00010575_002.png"
    main(query_image_path)
