# generation/dynamic_question_generator.py
"""
벡터DB 기반 동적 의료 문제 생성 시스템
사용자 쿼리 → 벡터 검색 → LLM 자율 판단 → 문제 생성
"""
from pathlib import Path
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Phase IV 검색 엔진
from search.search_engine import SearchEngine

# Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

@dataclass
class SearchContext:
    query: str
    text_content: str
    image_info: str
    confidence: str
    has_images: bool
    estimated_topic: str
    primary_image_path: str = ""  # 추가
    
class DynamicQuestionGenerator:
    """벡터DB 기반 동적 문제 생성기"""
    
    def __init__(self):
        print("🤖 동적 문제 생성 시스템 초기화 중...")
        
        # 검색 엔진 연결
        try:
            self.search_engine = SearchEngine()
            print("✅ 벡터DB 검색 엔진 연결 완료")
        except Exception as e:
            print(f"❌ 검색 엔진 연결 실패: {e}")
            self.search_engine = None
        
        # Gemini 클라이언트 초기화
        self.gemini_client = self._init_gemini()
        
        print("✅ 동적 문제 생성 시스템 준비 완료")
    
    def _init_gemini(self) -> Optional[object]:
        """Gemini API 초기화"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or not GEMINI_AVAILABLE:
            print("⚠️ Gemini API 사용 불가")
            return None
        
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-pro')
            print("✅ Gemini 1.5 Pro 연결 완료")
            return model
        except Exception as e:
            print(f"❌ Gemini 초기화 실패: {e}")
            return None
    
    def generate_question_from_query(self, user_query: str, top_k: int = 8) -> Dict[str, Any]:
        """
        사용자 쿼리로부터 동적 문제 생성.
        [수정된 로직] 문제 생성 후 LLM이 이미지 필요성 판단.
        """
        
        print(f"\n[bold cyan]🔍 동적 문제 생성 시작: '{user_query}'[/bold cyan]")
        print("="*70)
        
        try:
            # 1. 벡터DB에서 관련 정보 검색
            print("1️⃣  벡터DB 검색 중...")
            search_result = self.search_engine.search_text(user_query, top_k=top_k)
            
            if "error" in search_result:
                return {"error": f"벡터 검색 실패: {search_result['error']}"}
            
            # 2. 검색 결과를 구조화된 컨텍스트로 변환
            print("2️⃣  컨텍스트 구조화 중...")
            context = self._create_search_context(search_result, user_query)
            
            # 3. LLM에게 문제 생성 요청 (이미지 선택 전에)
            print("3️⃣  LLM 문제 생성 중...")
            if not self.gemini_client:
                return {"error": "Gemini API 사용 불가"}
            
            generated_question = self._generate_question_only(context)
            
            if not generated_question or "error" in generated_question:
                error_detail = generated_question.get('raw_response', 'LLM 응답 없음')
                return {"error": f"LLM 문제 생성 실패. 상세: {error_detail}"}
            
            # 4. 생성된 문제를 기반으로 LLM이 이미지 타입 선택
            print("4️⃣  생성된 문제 분석 후, LLM 이미지 타입 선택 중...")
            image_selection = self._select_appropriate_image(generated_question, context)
            
            # 5. 선택된 타입으로 이미지 검색
                selected_images = []
            selected_type = image_selection.get("selected_image_type", "None")
                
            if selected_type != "None":
                print(f"5️⃣  선택된 타입 '{selected_type}'으로 이미지 검색 중...")
                selected_images = self._fetch_selected_images(image_selection, search_result)
                if not selected_images:
                    print(f"   ⚠️ '{selected_type}' 이미지를 찾지 못했지만, 문제는 계속 생성됩니다.")
            else:
                print("5️⃣  LLM이 이미지가 불필요하다고 판단함 (검색 생략)")

            # 6. 최종 결과 구성
            is_text_only_final = (selected_type == "None")
            
            result = {
                "generated_question": generated_question,
                "image_selection": image_selection,
                "selected_images": selected_images,
                "search_context": {
                    "original_query": user_query,
                    "estimated_topic": context.estimated_topic,
                    "has_images": len(selected_images) > 0,
                    "confidence": context.confidence,
                    "text_sources": search_result.get("text_count", 0),
                    "image_sources": len(selected_images),
                    "selected_image_type": selected_type,
                    "is_text_only_mode": is_text_only_final
                },
                "generation_metadata": {
                    "method": "dynamic_generation_with_post_image_selection",
                    "search_quality": context.confidence,
                    "llm_model": "gemini-1.5-pro",
                    "vector_db_used": True,
                    "image_selection_method": "llm_analysis_post_generation"
                },
                "created_at": datetime.now().isoformat()
            }
            
            reason = image_selection.get("reason", "선택 이유 없음")
            print(f"✅ [bold green]동적 문제 생성 완료: 최종 이미지 타입 '{selected_type}'[/bold green]")
            print(f"   💡 LLM 선택 이유: {reason}")
            
            return result
        
        except Exception as e:
            print(f"❌ 문제 생성 실패: {e}")
            return {"error": f"동적 생성 중 치명적 오류: {str(e)}"}
    
    def _generate_question_only(self, context: SearchContext) -> Optional[Dict]:
        """ 이미지 선택 없이 문제만 생성 """
        
        prompt = f"""당신은 한국 의사국가고시 문제 출제 전문가입니다.

    # 사용자 요청
    사용자가 "{context.query}"에 대한 문제를 요청했습니다.

    # 벡터DB 검색 결과
    추정 의료 주제: {context.estimated_topic}
    검색 신뢰도: {context.confidence}

    ## 관련 의학 지식
    {context.text_content[:2000]}

    # 생성 지침
    1. 한국 의사국가고시 형식의 5지선다 객관식 문제 1개 생성
    2. 실제 임상 상황을 반영한 환자 증례 포함
    3. 위의 검색된 의료 지식을 최대한 활용
    4. 한국 의료 환경과 용어에 맞게 작성

    # 출력 형식
    다음 JSON 형식으로만 응답하세요:

    {{
        "question": "문제 본문 (환자 증례 포함)",
        "options": ["보기1", "보기2", "보기3", "보기4", "보기5"],
        "answer": 정답_인덱스_숫자(0-4),
        "explanation": "정답 근거 및 해설",
        "topic_analysis": {{
            "estimated_topic": "LLM이 판단한 정확한 의료 주제",
            "difficulty_level": "초급/중급/고급",
            "clinical_relevance": "high/medium/low"
        }},
        "source_utilization": "검색된 정보를 어떻게 활용했는지 설명"
    }}

    검색된 의학 지식을 충실히 반영하여 정확하고 교육적 가치가 높은 문제를 생성하세요."""

        try:
            response = self.gemini_client.generate_content(prompt)
            
            # JSON 추출
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group())
            else:
                print("   ⚠️ LLM이 JSON 형식으로 응답하지 않음")
                return {"error": "JSON 파싱 실패", "raw_response": response.text}
                
        except Exception as e:
            print(f"   ❌ 문제 생성 실패: {e}")
            return {"error": str(e)}

    def _select_appropriate_image(self, generated_question: Dict, context: SearchContext) -> Dict:
        """생성된 문제를 분석하여 적절한 이미지 타입 선택"""
        
        question_text = generated_question.get("question", "")
        explanation = generated_question.get("explanation", "")
        topic_analysis = generated_question.get("topic_analysis", {})
        
    # 수정된 이미지 선택 프롬프트
        selection_prompt = f"""당신은 의료 영상 전문가입니다. 다음 문제를 분석하여 흉부 X-ray가 필요한지 판단하고, 가장 적절한 영상 타입을 제안해야 합니다.

# 원본 사용자 쿼리
"{context.query}"

# 생성된 문제 내용
- 문제: {question_text}
- 해설: {explanation}
- AI 추정 주제: {topic_analysis.get('estimated_topic', 'Unknown')}

# 핵심 판단 기준
**1. 분석의 초점:**
- **오직 '생성된 문제 내용'에만 집중하세요.** '원본 사용자 쿼리'가 "호흡기 문제"처럼 광범위하더라도, 생성된 문제가 명확히 '폐렴'이나 'COPD'에 대한 것이라면, 그에 맞는 이미지를 선택해야 합니다.
- 쿼리가 애매해도, 생성된 문제가 구체적이면 그 구체적인 내용을 따르세요.

**2. 반드시 `None`을 선택해야 하는 경우:**
- 생성된 문제의 주제가 아래 목록에 해당하는 경우:
    - 외상외과(열상, 골절), 정형외과(관절), 피부과, 안과, 이비인후과
    - 비흉부 내과 질환 (예: 당뇨병, 신장질환, 간질환, 내분비 질환)
- 문제의 핵심이 영상 진단이 아닌 경우 (예: 약물 용량 계산, 실험실 검사 결과 해석, 윤리 문제)

**3. 흉부 X-ray 이미지가 필요한 경우:**
- **생성된 문제에 명시적이거나 암시적인 흉부 영상 소견이 있을 때:**
    - **명시적:** "흉부 X-선에서 폐경화가 보였다", "심비대가 관찰되었다"
    - **암시적:** 환자가 "호흡곤란", "지속적인 기침", "흉통"을 호소하고, 의심되는 질환이 폐렴, 기흉, 심부전 등일 때.
- **주요 대상 질환:** 폐렴, 기흉, 흉수, 무기폐, 폐결절/종괴, 심부전, COPD, 간질성 폐질환 등

# 최종 목표
- **보수적으로 판단하지 마세요.** 생성된 문제가 흉부 질환의 임상적 시나리오를 묘사한다면, 교육적 가치를 높이기 위해 관련 이미지를 **적극적으로 제안**하는 것이 좋습니다.

# 선택 옵션 (가장 적합한 1개만 선택)
1. Pneumonia (폐렴) - 폐감염, 발열, 기침, 가래
2. Effusion (흉수) - 늑막삼출, 호흡곤란
3. Mass (종괴) - 폐종괴, 폐암, 종양
4. Nodule (결절) - 폐결절, 소결절
5. Pneumothorax (기흉) - 기흉, 흉통, 응급
6. Atelectasis (무기폐) - 폐허탈, 수술후 합병증
7. Infiltrate (침윤/경화) - 간질성 음영, 폐부종
8. Cardiomegaly (심비대) - 심장 크기 증가, 심부전
9. Normal (정상) - 특정 질환 배제를 위한 정상 소견
10. None - 이미지가 전혀 필요 없음

# 출력 형식 (JSON)
{{
    "selected_image_type": "선택한 타입 (예: Pneumonia)",
    "korean_name": "선택한 타입의 한글명 (예: 폐렴)",
    "reason": "왜 이 이미지를 선택했는지, 또는 왜 'None'을 선택했는지 문제 내용을 근거로 상세히 설명",
    "relevance_score": "문제와 이미지의 관련성 점수 (1-10)",
    "is_chest_related": "흉부 관련 문제인가? (true/false)",
    "query_match": "원본 쿼리와 최종 주제의 일치도 (high/medium/low)"
}}
"""

        try:
            response = self.gemini_client.generate_content(selection_prompt)
            
            # JSON 추출 및 파싱
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            
            if json_match:
                try:
                    parsed_json = json.loads(json_match.group())
                
                    # 유효성 검사
                    selected_type = parsed_json.get("selected_image_type", "None")
                    
                valid_types = [
                    "Pneumonia", "Effusion", "Mass", "Nodule", 
                    "Pneumothorax", "Atelectasis", "Infiltrate", "Cardiomegaly", "None"
                ]
                
                if selected_type not in valid_types:
                        print(f"   ⚠️ LLM이 유효하지 않은 이미지 타입 선택: '{selected_type}' → 'None'으로 강제 조정")
                        parsed_json["reason"] += f" (원본 선택: {selected_type})"
                        parsed_json["selected_image_type"] = "None"
                        
                    return parsed_json
                
                except json.JSONDecodeError:
                    return {"error": "LLM 응답이 유효한 JSON이 아님", "raw_response": response.text}
            else:
                return {"error": "LLM 응답에서 JSON을 찾을 수 없음", "raw_response": response.text}
                
        except Exception as e:
            print(f"   ❌ 이미지 타입 선택 실패: {e}")
            return {"error": str(e)}

    def _fetch_selected_images(self, image_selection: Dict, search_result: Dict) -> List[Dict]:
        """선택된 이미지 타입으로 벡터DB에서 이미지 검색"""
        
        selected_type = image_selection.get("selected_image_type", "None")
        if selected_type == "None" or not self.search_engine:
            return []
        
        try:
            # 검색 엔진의 이미지 검색 기능 호출 (payload_key를 'labels'로 지정)
            image_results = self.search_engine.search_images_by_disease(
                disease_name=selected_type, 
                top_k=5 
            )
            
            # 반환 형식에 맞게 데이터 가공
            formatted_results = []
            for res in image_results:
                formatted_results.append({
                    "image_path": res.get("id"),
                    "score": res.get("score"),
                    "labels": res.get("payload", {}).get("labels_kr", [])
                })
            
            print(f"   ✅ '{selected_type}' 이미지 {len(formatted_results)}개 검색 완료")
            return formatted_results
            
        except Exception as e:
            print(f"   ❌ '{selected_type}' 이미지 검색 실패: {e}")
            return []
    
    def _create_search_context(self, search_result: Dict, user_query: str) -> SearchContext:
        """검색 결과를 바탕으로 구조화된 컨텍스트 생성"""
        
        # 텍스트 컨텐츠 포맷팅
        text_chunks = [f"- {item}" for item in search_result.get("text_content", "").splitlines() if item.strip()]
        text_content_formatted = "\n\n".join(text_chunks)
        
        # 이미지 컨텐츠 포맷팅 (검색 결과에 따라)
        image_info_formatted = "이미지 정보 없음"
        primary_image_path = ""
        has_images = False
        
        # 이미지 정보가 있는 경우에만 처리
        if search_result.get("image_results") and len(search_result["image_results"]) > 0:
            image_list = []
            # 상위 5개 이미지만 사용
            for item in search_result["image_results"][:5]:
                # 경로와 레이블만 간결하게 표시
                image_list.append(f"- 경로: {item['id']}, 라벨: {item['payload'].get('labels_kr', 'N/A')}")
            
            image_info_formatted = "## 관련 이미지 데이터\n" + "\n".join(image_list)
            primary_image_path = search_result["image_results"][0]["id"]
            has_images = True

        # LLM이 추정한 주제 또는 기본값 사용
        estimated_topic = search_result.get("predicted_topic_kr", "일반 의료")
        
        return SearchContext(
            query=user_query,
            text_content=text_content_formatted,
            image_info=image_info_formatted,
            confidence=search_result.get("confidence", "unknown"),
            has_images=has_images,
            estimated_topic=estimated_topic,
            primary_image_path=primary_image_path
        )

# 편의 함수들
def create_dynamic_generator() -> DynamicQuestionGenerator:
    """동적 생성기 인스턴스 생성"""
    return DynamicQuestionGenerator()

def quick_dynamic_generate(query: str = "폐렴") -> Dict:
    """빠른 동적 문제 생성"""
    print(f"🚀 빠른 동적 생성: '{query}'")
    
    try:
        generator = create_dynamic_generator()
        result = generator.generate_question_from_query(query)
        
        if "error" not in result:
            print(f"\n✅ 동적 생성 성공!")
            question_data = result["generated_question"]
            topic_analysis = question_data.get("topic_analysis", {})
            
            print(f"📋 추정 주제: {topic_analysis.get('estimated_topic', 'Unknown')}")
            print(f"📊 난이도: {topic_analysis.get('difficulty_level', 'Unknown')}")
            print(f"🏥 임상 관련성: {topic_analysis.get('clinical_relevance', 'Unknown')}")
            print(f"🖼️ 이미지 필요: {topic_analysis.get('requires_image', False)}")
            
            return result
        else:
            print(f"❌ 생성 실패: {result['error']}")
            return result
            
    except Exception as e:
        print(f"❌ 오류: {e}")
        return {"error": str(e)}
