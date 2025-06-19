# search/search_engine.py
# 통합된 멀티모달 검색 엔진 - 모든 검색 로직을 하나의 파일에 통합
from datetime import datetime
import os
import re
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# BioViL-T 모델 import
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference

# --- 설정 및 상수 ---
load_dotenv()

class SearchConfig:
    """검색 시스템 설정 및 상수"""
    
    # API 설정
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # 모델 설정
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TEXT_EMBEDDING_MODEL = "text-embedding-3-small"
    
    # 인덱스 설정
    TEXT_INDEX_NAME = "textbook-rag"
    IMAGE_INDEX_NAME = "cxr-image-meta-v2"
    
    # 질병 정보
    DISEASE_INFO = {
        "Effusion": {"count": 51, "korean": "흉수", "exam_weight": "높음"},
        "Infiltrate": {"count": 44, "korean": "침윤/경화", "exam_weight": "높음"},
        "Atelectasis": {"count": 31, "korean": "무기폐", "exam_weight": "중간"},
        "Pneumonia": {"count": 23, "korean": "폐렴", "exam_weight": "매우높음"},
        "Mass": {"count": 22, "korean": "종괴", "exam_weight": "높음"},
        "Pneumothorax": {"count": 12, "korean": "기흉", "exam_weight": "높음"},
        "Cardiomegaly": {"count": 11, "korean": "심장비대", "exam_weight": "중간"},
        "Nodule": {"count": 3, "korean": "결절", "exam_weight": "낮음"}
    }
    
    # 쿼리 확장 템플릿
    QUERY_EXPANSION_TEMPLATES = {
        # 영어 질병명
        "pneumonia": "pneumonia diagnosis treatment bacterial viral lung infection chest disease",
        "effusion": "pleural effusion chest fluid thoracentesis drainage respiratory",
        "pleural effusion": "pleural effusion chest fluid thoracentesis drainage blunting costophrenic",
        "pneumothorax": "pneumothorax collapsed lung tension emergency chest tube",
        "atelectasis": "atelectasis lung collapse volume loss postoperative",
        "consolidation": "consolidation pneumonia lung opacity air space disease",
        "infiltrate": "infiltrate lung opacity consolidation pneumonia infection",
        "mass": "lung mass tumor lesion CT evaluation oncology",
        "cardiomegaly": "cardiomegaly enlarged heart failure cardiothoracic ratio",
        "nodule": "lung nodule pulmonary nodule CT scan evaluation",
        
        # 한국어 질병명
        "폐렴": "폐렴 진단 치료 항생제 세균성 바이러스성 폐감염 호흡기질환",
        "흉수": "흉수 늑막삼출 천자술 배액 호흡곤란 흉부 늑골횡격막각",
        "기흉": "기흉 허탈 긴장성 응급상황 흉관삽입 치료 공기누출",
        "무기폐": "무기폐 허탈 부피감소 수술후 합병증 호흡기",
        "침윤": "침윤 폐침윤 경화 감염 음영 질환 실질",
        "경화": "경화 폐경화 공기공간질환 음영 진단 폐렴",
        "종괴": "종괴 폐종괴 종양 병변 CT검사 평가 결절",
        "심장비대": "심장비대 심부전 심흉곽비 심초음파 검사",
        "결절": "결절 폐결절 종괴 병변 CT검사 악성 양성"
    }
    
    # 쿼리-질병 직접 매핑
    QUERY_DISEASE_MAPPING = {
        "pneumonia": "Pneumonia", "폐렴": "Pneumonia",
        "effusion": "Effusion", "흉수": "Effusion", "pleural effusion": "Effusion",
        "pneumothorax": "Pneumothorax", "기흉": "Pneumothorax", 
        "atelectasis": "Atelectasis", "무기폐": "Atelectasis",
        "consolidation": "Infiltrate", "침윤": "Infiltrate", "경화": "Infiltrate", "infiltrate": "Infiltrate",
        "mass": "Mass", "종괴": "Mass",
        "cardiomegaly": "Cardiomegaly", "심장비대": "Cardiomegaly",
        "nodule": "Nodule", "결절": "Nodule"
    }
    
    # 질병별 상세 매핑
    DISEASE_MAPPINGS = {
        "Pneumonia": {
            "exact_match": ["pneumonia", "폐렴"],
            "partial_match": ["bacterial pneumonia", "viral pneumonia", "lung infection"],
            "korean_match": ["폐렴", "세균성폐렴", "바이러스폐렴"],
            "exclude_if_found": ["pneumothorax"]
        },
        "Pneumothorax": {
            "exact_match": ["pneumothorax", "기흉"],
            "partial_match": ["collapsed lung", "tension pneumothorax"],
            "korean_match": ["기흉", "긴장성기흉"],
            "exclude_if_found": ["pneumonia"]
        },
        "Effusion": {
            "exact_match": ["effusion", "흉수", "pleural effusion"],
            "partial_match": ["pleural fluid", "chest fluid"],
            "korean_match": ["흉수", "늑막삼출", "가슴물"],
            "exclude_if_found": []
        },
        "Atelectasis": {
            "exact_match": ["atelectasis", "무기폐"],
            "partial_match": ["lung collapse", "partial collapse"],
            "korean_match": ["무기폐", "허탈", "폐허탈"],
            "exclude_if_found": []
        },
        "Infiltrate": {
            "exact_match": ["infiltrate", "침윤", "consolidation"],
            "partial_match": ["lung opacity", "parenchymal opacity", "경화"],
            "korean_match": ["침윤", "경화", "폐경화", "폐침윤"],
            "exclude_if_found": []
        },
        "Mass": {
            "exact_match": ["mass", "종괴"],
            "partial_match": ["tumor", "lesion", "nodular opacity"],
            "korean_match": ["종괴", "종양", "덩어리"],
            "exclude_if_found": []
        },
        "Cardiomegaly": {
            "exact_match": ["cardiomegaly", "심장비대"],
            "partial_match": ["enlarged heart", "cardiac enlargement"],
            "korean_match": ["심장비대", "심장확대"],
            "exclude_if_found": []
        },
        "Nodule": {
            "exact_match": ["nodule", "결절"],
            "partial_match": ["lung nodule", "pulmonary nodule"],
            "korean_match": ["결절", "폐결절"],
            "exclude_if_found": []
        }
    }


class QueryProcessor:
    """쿼리 확장 및 전처리"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.templates = config.QUERY_EXPANSION_TEMPLATES
        # LLM 클라이언트 추가
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.llm_client = genai.GenerativeModel('gemini-1.5-pro')
                print("✅ 쿼리 확장용 LLM 클라이언트 초기화 완료")
            else:
                self.llm_client = None
                print("⚠️ GEMINI_API_KEY 없음: 기본 템플릿 방식 사용")
        except ImportError:
            self.llm_client = None
            print("⚠️ Gemini 패키지 없음: 기본 템플릿 방식 사용")
    
    def expand_query(self, query: str) -> str:
        """LLM 기반 동적 쿼리 확장 + 기존 템플릿 fallback"""
        query_lower = query.lower().strip()
        
        # 1. 기존 템플릿 방식 시도 (빠른 처리)
        template_expanded = self._try_template_expansion(query, query_lower)
        if template_expanded != query:
            print(f"    템플릿 확장: '{query}' → '{template_expanded[:80]}...'")
            return template_expanded
        
        # 2. LLM 동적 확장 시도
        if self.llm_client:
            llm_expanded = self._expand_with_llm(query)
            if llm_expanded and llm_expanded != query:
                print(f"    LLM 동적 확장: '{query}' → '{llm_expanded[:80]}...'")
                return llm_expanded
        
        # 3. 일반 의료 키워드 추가 (최종 fallback)
        fallback_expanded = self._add_general_medical_keywords(query)
        print(f"    일반 의료 확장: '{query}' → '{fallback_expanded}'")
        return fallback_expanded
    
    def _try_template_expansion(self, query: str, query_lower: str) -> str:
        """기존 템플릿 방식 확장 시도"""
        # 직접 매칭 확인
        for term, expansion in self.templates.items():
            if term in query_lower:
                return f"{query} {expansion}"
        
        # 부분 매칭 확인
        for term, expansion in self.templates.items():
            if any(word in query_lower for word in term.split()):
                return f"{query} {expansion}"
        
        return query  # 확장 실패시 원본 반환
    
    def _expand_with_llm(self, query: str) -> str:
        """LLM을 사용한 동적 쿼리 확장 - 길이 제한 강화"""
        try:
            prompt = f"""당신은 의료 정보 검색 전문가입니다. 다음 쿼리를 의료 문헌 검색에 최적화된 형태로 확장하세요.

    원본 쿼리: "{query}"

    지침:
    1. 원본 쿼리 + 핵심 키워드 3-4개만 추가
    2. 총 길이는 원본의 2배를 넘지 말 것
    3. 동의어, 관련 증상, 진단 방법 중심
    4. 한국어와 영어를 적절히 조합
    5. 간결하고 검색에 유용한 키워드만 선택

    예시:
    - "열상" → "열상 상처 봉합 외상 laceration wound suture"
    - "폐렴" → "폐렴 pneumonia 발열 기침 감염 respiratory"

    확장된 쿼리만 출력하세요 (설명 불필요):"""

            response = self.llm_client.generate_content(prompt)
            expanded = response.text.strip()
            
            # 유연한 길이 검증 (짧은 쿼리 고려)
            if len(query) <= 5:  # 매우 짧은 쿼리 (예: "열상", "폐렴")
                max_length = 80
            elif len(query) <= 15:  # 중간 길이 쿼리 (예: "열상 환자 치료")
                max_length = 120
            else:  # 긴 쿼리
                max_length = len(query) * 2
            
            if len(expanded) > max_length:
                print(f"    ⚠️ LLM 확장 결과가 너무 김 ({len(expanded)}자 > {max_length}자), 기본 방식 사용")
                return query
            
            # 의미없는 반복이나 설명 제거
            if "예시:" in expanded or "설명:" in expanded or "지침:" in expanded:
                print(f"    ⚠️ LLM이 설명을 포함함, 기본 방식 사용")
                return query
            
            # 원본과 너무 유사하면 확장 효과 없음
            if expanded.lower().strip() == query.lower().strip():
                print(f"    ⚠️ LLM 확장 효과 없음, 기본 방식 사용")
                return query
            
            return expanded
            
        except Exception as e:
            print(f"    ⚠️ LLM 쿼리 확장 실패: {e}")
            return query
    
    def _add_general_medical_keywords(self, query: str) -> str:
        """일반 의료 키워드 추가 (최종 fallback)"""
        # 쿼리에서 의료 맥락 감지
        medical_indicators = [
            '환자', '진단', '치료', '증상', '질환', '병원', '의사',
            'patient', 'diagnosis', 'treatment', 'symptom', 'disease'
        ]
        
        if any(indicator in query.lower() for indicator in medical_indicators):
            return f"{query} 진단 치료 증상 환자 의료 clinical diagnosis treatment medical"
        else:
            return f"{query} 의료 진단 medical diagnosis clinical"


class DiseaseExtractor:
    """질병명 추출 및 직접 매칭 보너스"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.disease_mappings = config.DISEASE_MAPPINGS
        self.query_mapping = config.QUERY_DISEASE_MAPPING
        # 제외 패턴 강화 (더 구체적으로)
        self.exclusion_patterns = {
            'pneumonia': [
                r'\b(?:not|no|without|absence of|rule out|r/o)\s+pneumonia\b',
                r'\bpneumothorax\s+(?:not|rather than|instead of)\s+pneumonia\b',
                r'\b(?:pneumothorax|effusion|mass)\s+(?:not|rather than)\s+pneumonia\b'
            ],
            'pneumothorax': [
                r'\b(?:not|no|without|absence of|rule out|r/o)\s+pneumothorax\b',
                r'\bpneumonia\s+(?:not|rather than|instead of)\s+pneumothorax\b'
            ],
            'effusion': [
                r'\b(?:not|no|without|absence of|rule out|r/o)\s+(?:pleural\s+)?effusion\b',
                r'\b(?:pneumonia|pneumothorax)\s+(?:not|rather than)\s+effusion\b'
            ]
        }
    
    def extract_diseases(self, text_results: List[Dict], query: str = "") -> List[str]:
        """텍스트 결과에서 질병을 추출하고 점수 계산 - 수정된 버전"""
        
        # 기본 점수 계산 (기존 메서드 사용)
        disease_scores = self._calculate_basic_scores(text_results)
        
        # 쿼리 직접 매칭 보너스 적용
        direct_match_found = self._apply_direct_matching_bonus(disease_scores, query)
        
        # 결과 정렬 및 필터링
        predicted_diseases = self._filter_and_sort_diseases(disease_scores, direct_match_found)
        
        return predicted_diseases
    
    def _calculate_basic_scores(self, text_results: List[Dict]) -> Dict[str, float]:
        """기본 질병 점수 계산"""
        disease_scores = {disease: 0 for disease in self.disease_mappings.keys()}
        
        for chunk in text_results:
            content = chunk['content']
            similarity = chunk['similarity']
            
            for disease, mapping_info in self.disease_mappings.items():
                disease_score = 0
                
                # 제외 패턴 확인
                if self._has_exclusion_pattern(content, mapping_info["exclude_if_found"]):
                    continue
                
                # 점수 계산
                disease_score += self._calculate_exact_matches(content, mapping_info["exact_match"], similarity)
                disease_score += self._calculate_partial_matches(content, mapping_info["partial_match"], similarity)
                disease_score += self._calculate_korean_matches(content, mapping_info["korean_match"], similarity)
                
                disease_scores[disease] += disease_score
        
        return disease_scores
    
    def _has_exclusion_pattern(self, content: str, exclude_terms: List[str]) -> bool:
        """제외 패턴이 있는지 확인"""
        for exclude_term in exclude_terms:
            if self._is_exact_word_match(content, exclude_term):
                print(f"    제외 패턴 발견: '{exclude_term}' 발견")
                return True
        return False
    
    def _calculate_exact_matches(self, content: str, exact_terms: List[str], similarity: float) -> float:
        """정확한 매칭 점수 계산"""
        score = 0
        for exact_term in exact_terms:
            if self._is_exact_word_match(content, exact_term):
                count = self._count_exact_matches(content, exact_term)
                weight = 3.0
                score += similarity * weight * count
                print(f"    정확한 매칭: '{exact_term}' (가중치: {weight}, 횟수: {count})")
        return score
    
    def _calculate_partial_matches(self, content: str, partial_terms: List[str], similarity: float) -> float:
        """부분 매칭 점수 계산"""
        score = 0
        for partial_term in partial_terms:
            if partial_term.lower() in content.lower():
                count = content.lower().count(partial_term.lower())
                weight = 1.5
                score += similarity * weight * count
        return score
    
    def _calculate_korean_matches(self, content: str, korean_terms: List[str], similarity: float) -> float:
        """한국어 매칭 점수 계산"""
        score = 0
        for korean_term in korean_terms:
            if korean_term in content:
                count = content.count(korean_term)
                weight = 2.0
                score += similarity * weight * count
        return score
    
    def _apply_direct_matching_bonus(self, disease_scores: Dict[str, float], original_query: str) -> bool:
        """쿼리 직접 매칭 보너스 적용"""
        direct_match_found = False
        
        if original_query:
            query_lower = original_query.lower()
            for term, disease in self.query_mapping.items():
                if term in query_lower and disease in disease_scores:
                    original_score = disease_scores[disease]
                    disease_scores[disease] += 100.0  # 강력한 보너스
                    print(f"    🚀 쿼리 직접 매칭 보너스: {disease} {original_score:.3f} → {disease_scores[disease]:.3f}")
                    direct_match_found = True
                    break
        
        return direct_match_found
    
    def _filter_and_sort_diseases(self, disease_scores: Dict[str, float], direct_match_found: bool) -> List[str]:
        """질병 점수 정렬 및 필터링 - 텍스트 전용 플래그 추가"""
        sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 임계값 적용
        min_threshold = 0.05 if direct_match_found else 0.1
        predicted_diseases = [disease for disease, score in sorted_diseases if score > min_threshold]
        
        # 결과 출력
        print(f"   🎯 질병 예측 결과:")
        for i, (disease, score) in enumerate(sorted_diseases[:5]):
            korean = self.config.DISEASE_INFO[disease]['korean']
            status = "✅" if score > min_threshold else "❌"
            bonus_mark = "🚀" if direct_match_found and score > 50 else ""
            print(f"      {i+1}. {disease} ({korean}): {score:.3f}점 {status} {bonus_mark}")
        
        # **핵심 수정**: 모든 질병이 매칭 실패시 텍스트 전용 표시
        if not predicted_diseases:
            max_score = sorted_diseases[0][1] if sorted_diseases else 0
            if max_score < 0.1:  # 매우 낮은 점수
                print("   📝 질병 매칭 실패 → 텍스트 전용 모드 설정")
                return ["TEXT_ONLY"]  # 특별한 플래그 반환
            else:
                # 기존 fallback 로직
                predicted_diseases = sorted(self.config.DISEASE_INFO.keys(), 
                                        key=lambda x: self.config.DISEASE_INFO[x]['count'], reverse=True)
                print("   ⚠️ fallback 적용: 데이터 보유량 순")
        
        return predicted_diseases
    
    def _is_exact_word_match(self, text: str, word: str) -> bool:
        """단어 경계를 고려한 정확한 매칭"""
        pattern = r'\b' + re.escape(word.lower()) + r'\b'
        return bool(re.search(pattern, text.lower()))
    
    def _count_exact_matches(self, text: str, word: str) -> int:
        """정확한 단어 매칭 횟수 계산"""
        pattern = r'\b' + re.escape(word.lower()) + r'\b'
        return len(re.findall(pattern, text.lower()))


class ImageSearcher:
    """이미지 검색 및 임베딩 처리"""
    
    def __init__(self, config: SearchConfig, image_index, image_encoder, image_transform):
        self.config = config
        self.image_index = image_index
        self.image_encoder = image_encoder
        self.image_transform = image_transform
    
    def search_by_diseases(self, predicted_diseases: List[str], top_k: int = 3) -> List[Dict]:
        """예측된 질병들로 이미지 DB에서 정확한 라벨 매칭 검색"""
        all_images = []
        dummy_vector = [0.0] * 512  # 필터링에서는 벡터 무관
        
        print(f"🖼️ 이미지 검색: 예측된 질병별 정확한 매칭")
        
        for disease in predicted_diseases[:3]:  # 상위 3개 질병
            try:
                filter_condition = {"primary_label": {"$eq": disease}}
                available_count = self.config.DISEASE_INFO.get(disease, {}).get('count', 0)
                search_count = min(top_k, available_count)
                
                if search_count > 0:
                    results = self.image_index.query(
                        vector=dummy_vector,
                        filter=filter_condition,
                        top_k=search_count,
                        include_metadata=True
                    )
                    
                    for match in results['matches']:
                        metadata = match['metadata']
                        all_images.append({
                            'image_id': match['id'],
                            'disease': disease,
                            'labels': metadata.get('labels', []),
                            'description': metadata.get('all_descriptions', ''),
                            'primary_label': metadata.get('primary_label', ''),
                            'image_path': metadata.get('image_path', ''),
                            'bbox_info': metadata.get('bboxes', []),
                            'relevance_score': 1.0
                        })
                    
                    korean_name = self.config.DISEASE_INFO[disease]['korean']
                    print(f"   ✅ {disease} ({korean_name}): {len(results['matches'])}개 이미지 매칭")
                else:
                    print(f"   ⚠️ {disease}: 이미지 없음")
                    
            except Exception as e:
                print(f"   ❌ {disease} 이미지 검색 실패: {e}")
        
        return all_images
    
    def search_by_image(self, image_path: str, top_k: int = 3) -> Dict:
        """이미지 기반 검색"""
        # 이미지 임베딩 생성
        image_embedding = self._get_image_embedding(image_path)
        
        if not image_embedding:
            return {"error": "이미지 처리 실패", "image_path": image_path}
        
        try:
            # 이미지 DB에서 유사도 검색
            results = self.image_index.query(
                vector=image_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # 결과 처리
            matched_images = []
            for match in results['matches']:
                metadata = match['metadata']
                matched_images.append({
                    'image_id': match['id'],
                    'similarity': match['score'],
                    'disease': metadata.get('primary_label', ''),
                    'labels': metadata.get('labels', []),
                    'description': metadata.get('all_descriptions', ''),
                    'image_path': metadata.get('image_path', '')
                })
            
            # 최고 유사도 결과 기준으로 컨텍스트 생성
            if matched_images:
                best_match = matched_images[0]
                primary_disease = best_match['disease']
                korean_name = self.config.DISEASE_INFO.get(primary_disease, {}).get('korean', primary_disease)
                
                return {
                    "query": f"이미지 검색: {Path(image_path).name}",
                    "diagnosis": primary_disease,
                    "korean_diagnosis": korean_name,
                    "confidence": "high" if best_match['similarity'] > 0.95 else "medium",
                    "search_info": {
                        "strategy": "image_similarity_search",
                        "similarity_score": best_match['similarity'],
                        "matched_images": len(matched_images)
                    },
                    "images": matched_images
                }
            else:
                return {"error": "유사한 이미지를 찾을 수 없습니다", "image_path": image_path}
                
        except Exception as e:
            return {"error": f"이미지 검색 실패: {e}", "image_path": image_path}
    
    def _get_image_embedding(self, image_path: str) -> Optional[List[float]]:
        """실제 이미지 파일에 대한 임베딩 생성"""
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            print(f"경고: 이미지 파일이 존재하지 않습니다: {image_path}")
            return None
        
        try:
            # 이미지 처리
            pil_image = Image.open(image_path_obj).convert('L')
            image_tensor = self.image_transform(pil_image)
            batch_tensor = image_tensor.unsqueeze(0).to(self.config.DEVICE)
            
            with torch.no_grad():
                model_output = self.image_encoder(batch_tensor)
                embedding_tensor = model_output.img_embedding
                return embedding_tensor.cpu().detach().numpy().squeeze().tolist()
                
        except Exception as e:
            print(f"오류: 이미지 처리 실패 {image_path}: {e}")
            return None


class ContextBuilder:
    """검색 결과를 통합하여 컨텍스트 생성"""
    
    def __init__(self):
        self.max_text_length = 2000
        self.max_images = 3
    
    def create_context(
        self, 
        query: str, 
        text_results: List[Dict], 
        image_results: List[Dict], 
        predicted_diseases: List[str]
    ) -> Dict[str, Any]:
        """통합 컨텍스트 생성 - TEXT_ONLY 플래그 처리 추가"""
        
        # 텍스트 내용 통합 (길이 제한)
        text_content = self._combine_text_results(text_results)
        
        # **핵심 수정**: TEXT_ONLY 플래그 확인
        is_text_only = len(predicted_diseases) == 1 and predicted_diseases[0] == "TEXT_ONLY"
        
        if is_text_only:
            # 텍스트 전용 모드
            primary_diagnosis = "Unknown"
            korean_diagnosis = "텍스트 전용 (흉부 관련성 낮음)"
            image_info = "이미지 검색 생략 (흉부 무관 주제)"
            confidence_level = "medium"  # 텍스트 기반이므로 중간 신뢰도
            print(f"   📝 텍스트 전용 컨텍스트 생성: 흉부 무관 주제로 판단")
        else:
            # 기존 로직
            image_info = self._process_image_results(image_results)
            primary_diagnosis = predicted_diseases[0] if predicted_diseases else "Unknown"
            korean_diagnosis = self._get_korean_diagnosis(primary_diagnosis)
            confidence_level = self._calculate_confidence_level(text_results, image_results, predicted_diseases)
        
        context = {
            "query": query,
            "diagnosis": primary_diagnosis,
            "primary_diagnosis": primary_diagnosis,
            "korean_diagnosis": korean_diagnosis,
            "all_diseases": predicted_diseases,
            "text_content": text_content,
            "image_info": image_info,
            "images": [] if is_text_only else image_results,  # 텍스트 전용시 빈 리스트
            "confidence": confidence_level,
            "text_count": len(text_results),
            "image_count": 0 if is_text_only else len(image_results),
            "is_text_only_mode": is_text_only,  # 새로운 플래그 추가
            "created_at": datetime.now().isoformat()
        }
        
        print(f"   ✅ 컨텍스트 생성 완료: {korean_diagnosis}, 신뢰도: {confidence_level}")
        return context
    
    def _combine_text_results(self, text_results: List[Dict]) -> str:
        """텍스트 결과들을 결합"""
        if not text_results:
            return "관련 텍스트 정보 없음"
        
        combined_text = ""
        for result in text_results:
            content = result.get('content', '') or result.get('text', '')
            combined_text += content + " "
        
        # 길이 제한
        if len(combined_text) > self.max_text_length:
            combined_text = combined_text[:self.max_text_length] + "..."
        
        return combined_text.strip()
    
    def _process_image_results(self, image_results: List[Dict]) -> str:
        """이미지 결과들을 처리"""
        if not image_results:
            return "관련 이미지 정보 없음"
        
        processed_info = f"{len(image_results)}개 관련 이미지 발견"
        
        # 주요 이미지 정보 추가
        if image_results:
            first_image = image_results[0]
            disease = first_image.get('disease', first_image.get('primary_label', 'Unknown'))
            description = first_image.get('description', '')
            if description:
                processed_info += f" - {description[:100]}"
        
        return processed_info
    
    def _get_korean_diagnosis(self, primary_diagnosis: str) -> str:
        """영어 진단명을 한국어로 변환"""
        # 직접 DISEASE_INFO 사용
        DISEASE_INFO = {
            "Effusion": {"korean": "흉수"},
            "Infiltrate": {"korean": "침윤/경화"},
            "Atelectasis": {"korean": "무기폐"},
            "Pneumonia": {"korean": "폐렴"},
            "Mass": {"korean": "종괴"},
            "Pneumothorax": {"korean": "기흉"},
            "Cardiomegaly": {"korean": "심장비대"},
            "Nodule": {"korean": "결절"}
        }
        
        disease_info = DISEASE_INFO.get(primary_diagnosis, {})
        korean_name = disease_info.get('korean', primary_diagnosis)
        
        return f"{korean_name} ({primary_diagnosis})"
    
    def _calculate_confidence_level(self, text_results: List[Dict], image_results: List[Dict], predicted_diseases: List[str]) -> str:
        """신뢰도 레벨 계산"""
        
        confidence_score = 0
        
        # 1. 텍스트 결과 점수 (최대 40점)
        text_score = min(len(text_results) * 10, 40)
        confidence_score += text_score
        
        # 2. 이미지 결과 점수 (최대 30점)  
        image_score = min(len(image_results) * 15, 30)
        confidence_score += image_score
        
        # 3. 질병 예측 점수 (최대 30점)
        disease_score = min(len(predicted_diseases) * 10, 30)
        confidence_score += disease_score
        
        # 4. 텍스트 품질 보너스 (최대 10점)
        if text_results:
            total_text_length = sum(
                len(result.get('content', '') + result.get('text', '')) 
                for result in text_results
            )
            if total_text_length > 500:
                confidence_score += 10
        
        # 5. 이미지 품질 보너스 (최대 10점) 
        if image_results:
            high_confidence_images = 0
            for img in image_results:
                metadata = img.get('metadata', {})
                if metadata.get('image_confidence', 0) > 0.8:
                    high_confidence_images += 1
            if high_confidence_images > 0:
                confidence_score += 10
        
        # 신뢰도 레벨 결정 (총 120점 만점)
        if confidence_score >= 80:
            return "high"
        elif confidence_score >= 50:
            return "medium"  
        else:
            return "low"
        
class SearchEngine:
    """통합 멀티모달 검색 엔진"""
    
    def __init__(self):
        """검색 엔진 초기화"""
        print("🏥 멀티모달 검색 엔진 초기화 중...")
        
        # 설정 로드
        self.config = SearchConfig()
        
        # 외부 서비스 연결
        self.pc = Pinecone(api_key=self.config.PINECONE_API_KEY)
        self.openai_client = OpenAI(api_key=self.config.OPENAI_API_KEY)
        
        # 인덱스 연결
        self.text_index = self.pc.Index(self.config.TEXT_INDEX_NAME)
        self.image_index = self.pc.Index(self.config.IMAGE_INDEX_NAME)
        
        # BioViL-T 모델 초기화
        self.image_encoder = get_biovil_t_image_encoder()
        self.image_transform = create_chest_xray_transform_for_inference(resize=512, center_crop_size=480)
        self.image_encoder.to(self.config.DEVICE)
        self.image_encoder.eval()
        
        # 컴포넌트 초기화
        self.query_processor = QueryProcessor(self.config)
        self.disease_extractor = DiseaseExtractor(self.config)
        self.image_searcher = ImageSearcher(self.config, self.image_index, self.image_encoder, self.image_transform)
        self.context_builder = ContextBuilder()
        
        print("✅ 검색 엔진 초기화 완료")
    
    def search_text(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        텍스트 기반 멀티모달 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            
        Returns:
            검색 결과 딕셔너리
        """
        print(f"\n{'='*60}")
        print(f"🔍 텍스트 기반 멀티모달 검색: '{query}'")
        print(f"{'='*60}")
        
        try:
            # 1. 쿼리 확장
            expanded_query = self.query_processor.expand_query(query)
            
            # 2. 텍스트 검색
            text_results = self._search_text_knowledge(expanded_query, top_k)
            if not text_results:
                return {"error": "관련 의학 지식을 찾을 수 없습니다", "query": query}
            
            # 3. 질병 추출 (직접 매칭 보너스 포함)
            predicted_diseases = self.disease_extractor.extract_diseases(text_results, query)
            
            # 4. 이미지 검색
            image_results = self.image_searcher.search_by_diseases(predicted_diseases, top_k)
            
            # 5. 컨텍스트 생성
            context = self.context_builder.create_context(query, text_results, image_results, predicted_diseases)
            
            return context
            
        except Exception as e:
            print(f"❌ 검색 실패: {e}")
            return {"error": f"검색 중 오류 발생: {str(e)}", "query": query}
    
    def search_image(self, image_path: str, top_k: int = 5) -> Dict[str, Any]:
        """
        이미지 기반 검색
        
        Args:
            image_path: 이미지 파일 경로
            top_k: 반환할 결과 수
            
        Returns:
            검색 결과 딕셔너리
        """
        print(f"\n{'='*60}")
        print(f"🖼️ 이미지 기반 검색: {image_path}")
        print(f"{'='*60}")
        
        try:
            result = self.image_searcher.search_by_image(image_path, top_k)
            return result
            
        except Exception as e:
            print(f"❌ 이미지 검색 실패: {e}")
            return {"error": f"이미지 검색 중 오류 발생: {str(e)}", "image_path": image_path}
    
    def _search_text_knowledge(self, query: str, top_k: int = 5) -> List[Dict]:
        """텍스트 DB에서 의학 지식 검색"""
        try:
            print(f"📚 텍스트 지식 검색: '{query[:50]}...'")
            
            # OpenAI 임베딩 생성
            resp = self.openai_client.embeddings.create(
                input=[query], 
                model=self.config.TEXT_EMBEDDING_MODEL
            )
            embedding = resp.data[0].embedding
            
            # Pinecone 검색
            results = self.text_index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # 결과 처리
            text_chunks = []
            for match in results['matches']:
                metadata = match['metadata']
                content = metadata.get('text', metadata.get('content', ''))
                text_chunks.append({
                    'content': content,
                    'similarity': match['score'],
                    'source': metadata.get('source', 'unknown'),
                    'id': match['id']
                })
            
            print(f"   ✅ {len(text_chunks)}개 지식 청크 검색 완료")
            
            # 디버깅 정보
            if text_chunks:
                print(f"   🔍 검색된 내용 샘플:")
                for i, chunk in enumerate(text_chunks[:2]):
                    preview = chunk['content'][:80] + "..." if len(chunk['content']) > 80 else chunk['content']
                    print(f"      {i+1}. (유사도: {chunk['similarity']:.3f}) {preview}")
            
            return text_chunks
            
        except Exception as e:
            print(f"   ❌ 텍스트 검색 실패: {e}")
            return []
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 반환"""
        try:
            # 인덱스 통계
            text_stats = self.text_index.describe_index_stats()
            image_stats = self.image_index.describe_index_stats()
            
            return {
                "system_status": "online",
                "device": self.config.DEVICE,
                "text_index": {
                    "name": self.config.TEXT_INDEX_NAME,
                    "total_vectors": text_stats.get('total_vector_count', 0),
                    "dimension": text_stats.get('dimension', 0)
                },
                "image_index": {
                    "name": self.config.IMAGE_INDEX_NAME,
                    "total_vectors": image_stats.get('total_vector_count', 0),
                    "dimension": image_stats.get('dimension', 0)
                },
                "supported_diseases": list(self.config.DISEASE_INFO.keys()),
                "total_diseases": len(self.config.DISEASE_INFO)
            }
        except Exception as e:
            return {"system_status": "error", "error": str(e)}


# --- 테스트 및 평가 클래스 ---

class SearchTester:
    """검색 시스템 테스트 및 평가"""
    
    def __init__(self, search_engine: SearchEngine):
        self.search_engine = search_engine
    
    def test_accuracy(self) -> float:
        """질병 매칭 정확도 테스트"""
        test_cases = [
            {"query": "pneumonia", "expected": "Pneumonia"},
            {"query": "pleural effusion", "expected": "Effusion"},
            {"query": "폐렴", "expected": "Pneumonia"},
            {"query": "흉수", "expected": "Effusion"},
            {"query": "기흉", "expected": "Pneumothorax"},
            {"query": "pneumothorax", "expected": "Pneumothorax"},
            {"query": "consolidation", "expected": "Infiltrate"},
            {"query": "심장비대", "expected": "Cardiomegaly"}
        ]
        
        print("\n🎯 질병 매칭 정확도 테스트")
        print("="*50)
        
        correct = 0
        total = len(test_cases)
        
        for case in test_cases:
            try:
                result = self.search_engine.search_text(case["query"], top_k=3)
                
                if "error" not in result:
                    predicted = result["diagnosis"]
                    expected = case["expected"]
                    
                    if predicted == expected:
                        correct += 1
                        print(f"✅ '{case['query']}' → {predicted} - 정확!")
                    else:
                        print(f"❌ '{case['query']}' → {predicted} (예상: {expected})")
                else:
                    print(f"❌ '{case['query']}' → 검색 실패: {result['error']}")
                    
            except Exception as e:
                print(f"❌ '{case['query']}' → 오류: {e}")
        
        accuracy = correct / total * 100
        print(f"\n📈 정확도: {correct}/{total} ({accuracy:.1f}%)")
        
        return accuracy
    
    def test_performance(self) -> Dict[str, float]:
        """성능 테스트"""
        import time
        
        print("\n⏱️ 성능 테스트")
        print("="*30)
        
        test_queries = ["pneumonia", "pleural effusion", "폐렴"]
        times = []
        
        for query in test_queries:
            start_time = time.time()
            try:
                result = self.search_engine.search_text(query, top_k=3)
                end_time = time.time()
                
                duration = end_time - start_time
                times.append(duration)
                
                status = "✅" if "error" not in result else "❌"
                print(f"{status} '{query}': {duration:.2f}초")
                
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                times.append(duration)
                print(f"❌ '{query}': {duration:.2f}초 (오류)")
        
        avg_time = sum(times) / len(times) if times else 0
        print(f"\n📊 평균 응답 시간: {avg_time:.2f}초")
        
        return {
            "average_time": avg_time,
            "individual_times": times,
            "total_queries": len(test_queries)
        }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """포괄적 테스트 실행"""
        print("\n🧪 포괄적 테스트 실행")
        print("="*50)
        
        # 1. 정확도 테스트
        accuracy = self.test_accuracy()
        
        # 2. 성능 테스트
        performance = self.test_performance()
        
        # 3. 시스템 정보
        system_info = self.search_engine.get_system_info()
        
        # 최종 결과
        print(f"\n{'='*50}")
        print(f"🏆 종합 테스트 결과")
        print(f"{'='*50}")
        print(f"🎯 질병 매칭 정확도: {accuracy:.1f}%")
        print(f"⏱️ 평균 응답 시간: {performance['average_time']:.2f}초")
        print(f"🏥 지원 질병 수: {system_info.get('total_diseases', 0)}개")
        print(f"📊 텍스트 벡터 수: {system_info.get('text_index', {}).get('total_vectors', 0):,}개")
        print(f"🖼️ 이미지 벡터 수: {system_info.get('image_index', {}).get('total_vectors', 0):,}개")
        
        if accuracy >= 75:
            print("🎉 목표 달성! (75% 이상)")
        elif accuracy >= 50:
            print("🟡 부분 개선 (50% 이상)")
        else:
            print("🔴 추가 개선 필요")
        
        return {
            "accuracy": accuracy,
            "performance": performance,
            "system_info": system_info,
            "status": "excellent" if accuracy >= 75 else "good" if accuracy >= 50 else "needs_improvement"
        }


# --- 편의 함수들 ---

def create_search_engine() -> SearchEngine:
    """검색 엔진 인스턴스 생성"""
    return SearchEngine()

def quick_test(query: str = "폐렴 진단") -> None:
    """빠른 테스트 실행"""
    print(f"🚀 빠른 테스트: '{query}'")
    
    try:
        engine = create_search_engine()
        result = engine.search_text(query)
        
        if "error" not in result:
            print(f"✅ 결과: {result['korean_diagnosis']} ({result['diagnosis']})")
            print(f"   신뢰도: {result['confidence']}")
            print(f"   관련 이미지: {len(result.get('images', []))}개")
        else:
            print(f"❌ 실패: {result['error']}")
            
    except Exception as e:
        print(f"❌ 오류: {e}")

def run_full_evaluation() -> None:
    """전체 평가 실행"""
    try:
        engine = create_search_engine()
        tester = SearchTester(engine)
        results = tester.run_comprehensive_test()
        
        return results
    except Exception as e:
        print(f"❌ 평가 실행 실패: {e}")
        return None


# --- 메인 실행부 ---

if __name__ == "__main__":
    print("🏥 Medical Multimodal Search Engine")
    print("=" * 60)
    
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            # 빠른 테스트
            query = sys.argv[2] if len(sys.argv) > 2 else "폐렴 진단"
            quick_test(query)
            
        elif command == "eval":
            # 전체 평가
            run_full_evaluation()
            
        elif command == "info":
            # 시스템 정보
            try:
                engine = create_search_engine()
                info = engine.get_system_info()
                print("🔍 시스템 정보:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
            except Exception as e:
                print(f"❌ 시스템 정보 조회 실패: {e}")
                
        else:
            print("사용법:")
            print("  python search_engine.py test [쿼리]     # 빠른 테스트")
            print("  python search_engine.py eval           # 전체 평가")
            print("  python search_engine.py info           # 시스템 정보")
    
    else:
        # 기본 실행: 인터랙티브 모드
        try:
            engine = create_search_engine()
            print("\n🎯 인터랙티브 모드 (종료: 'quit')")
            
            while True:
                query = input("\n검색어 입력: ").strip()
                
                if query.lower() in ['quit', 'exit', '종료']:
                    print("👋 검색 엔진을 종료합니다.")
                    break
                
                if query:
                    result = engine.search_text(query)
                    
                    if "error" not in result:
                        print(f"\n📊 검색 결과:")
                        print(f"  🎯 진단: {result['korean_diagnosis']} ({result['diagnosis']})")
                        print(f"  📈 신뢰도: {result['confidence']}")
                        print(f"  🖼️ 관련 이미지: {len(result.get('images', []))}개")
                        print(f"  📝 관련 텍스트: {len(result.get('text_chunks', []))}개")
                        
                        # 주요 소견 출력
                        if 'key_finding' in result:
                            print(f"\n📋 주요 소견:")
                            for line in result['key_finding'].split('\n'):
                                if line.strip():
                                    print(f"    {line}")
                    else:
                        print(f"❌ 검색 실패: {result['error']}")
                
        except KeyboardInterrupt:
            print("\n👋 검색 엔진을 종료합니다.")
        except Exception as e:
            print(f"❌ 실행 중 오류 발생: {e}")