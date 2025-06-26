# generation/dynamic_question_generator.py
"""
벡터DB 기반 동적 의료 문제 생성 시스템
사용자 쿼리 → 벡터 검색 → LLM 자율 판단 → 문제 생성
"""
# --- 라이브러리 임포트 ---
from pathlib import Path # 파일 및 디렉토리 경로를 객체 지향적으로 다루기 위한 라이브러리입니다.
import os # 운영체제와 상호작용하기 위한 기능을 제공합니다. (예: 환경 변수 접근)
import json # JSON 데이터를 파싱하고 생성하기 위한 라이브러리입니다.
from datetime import datetime # 날짜와 시간을 다루기 위한 라이브러리입니다.
from typing import Dict, List, Any, Optional # 타입 힌팅을 위한 라이브러리. 코드의 명확성을 높여줍니다.
from dataclasses import dataclass # 데이터 클래스를 쉽게 만들기 위한 데코레이터입니다.

# Phase IV 검색 엔진
# search 폴더에 있는 search_engine.py 파일에서 SearchEngine 클래스를 가져옵니다.
# 이 클래스는 벡터 데이터베이스를 검색하는 실제 로직을 담고 있습니다.
from search.search_engine import SearchEngine

# Gemini API
# Google의 Gemini 모델을 사용하기 위한 라이브러리입니다.
# try-except 블록을 사용하여 라이브러리가 설치되어 있지 않아도 프로그램이 오류로 중단되지 않도록 합니다.
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True # 라이브러리가 성공적으로 임포트되면 True로 설정합니다.
except ImportError:
    GEMINI_AVAILABLE = False # 라이브러리가 없으면 False로 설정합니다.

# @dataclass 데코레이터를 사용하면 __init__, __repr__ 같은 특수 메서드를 자동으로 생성해줍니다.
# 이 클래스는 검색 과정에서 얻은 다양한 정보를 구조적으로 저장하는 역할을 합니다.
@dataclass
class SearchContext:
    query: str # 사용자가 입력한 원본 쿼리
    text_content: str # 벡터DB에서 검색된 관련 텍스트 정보
    image_info: str # 벡터DB에서 검색된 관련 이미지 정보
    confidence: str # 검색 결과의 신뢰도 (high, medium, low)
    has_images: bool # 관련 이미지가 있는지 여부
    estimated_topic: str # 검색 엔진이 추정한 의료 주제
    primary_image_path: str = ""  # 추가, 대표 이미지 경로 (기본값은 빈 문자열)
    
class DynamicQuestionGenerator:
    """
    벡터DB 검색 결과를 바탕으로 동적으로 의료 문제를 생성하는 핵심 클래스입니다.
    사용자 쿼리를 받아, 관련 지식을 찾고, LLM을 이용해 실제 시험과 유사한 문제를 만듭니다.
    """
    
    def __init__(self):
        """클래스가 처음 생성될 때 호출되는 초기화 메서드입니다."""
        print("🤖 동적 문제 생성 시스템 초기화 중...")
        
        # 검색 엔진 연결을 시도합니다.
        try:
            # SearchEngine 클래스의 인스턴스를 생성하여 self.search_engine에 저장합니다.
            self.search_engine = SearchEngine()
            print("✅ 벡터DB 검색 엔진 연결 완료")
        except Exception as e:
            # 연결 과정에서 오류가 발생하면 메시지를 출력하고, self.search_engine을 None으로 설정합니다.
            print(f"❌ 검색 엔진 연결 실패: {e}")
            self.search_engine = None
        
        # Gemini 클라이언트를 초기화합니다.
        self.gemini_client = self._init_gemini()
        
        print("✅ 동적 문제 생성 시스템 준비 완료")
    
    def _init_gemini(self) -> Optional[object]:
        """
        Gemini API 사용을 위한 클라이언트를 초기화하는 내부 메서드입니다.
        _로 시작하는 메서드는 클래스 내부에서만 사용하자는 약속입니다.
        """
        # 환경 변수에서 'GEMINI_API_KEY'를 가져옵니다.
        api_key = os.getenv("GEMINI_API_KEY")
        # API 키가 없거나, Gemini 라이브러리가 설치되지 않았다면 None을 반환합니다.
        if not api_key or not GEMINI_AVAILABLE:
            print("⚠️ Gemini API 사용 불가")
            return None
        
        try:
            # 가져온 API 키로 Gemini 라이브러리를 설정합니다.
            genai.configure(api_key=api_key)
            # 사용할 Gemini 모델을 지정하여 모델 객체를 생성합니다.
            model = genai.GenerativeModel('gemini-1.5-pro')
            print("✅ Gemini 1.5 Pro 연결 완료")
            return model
        except Exception as e:
            # 초기화 중 오류가 발생하면 메시지를 출력하고 None을 반환합니다.
            print(f"❌ Gemini 초기화 실패: {e}")
            return None
    
    def generate_question_from_query(self, user_query: str, top_k: int = 8) -> Dict[str, Any]:
        """
        사용자 쿼리를 입력받아 전체 동적 문제 생성 과정을 총괄하는 메인 메서드입니다.
        [수정된 로직]
        1. 쿼리로 텍스트/이미지 정보를 검색합니다.
        2. 검색된 텍스트로 먼저 문제를 생성합니다.
        3. 생성된 문제를 LLM이 분석하여 문제에 가장 적합한 이미지 유형을 판단합니다.
        4. 판단된 유형의 이미지를 다시 검색하여 최종 결과를 조합합니다.
        """
        
        print(f"\n[bold cyan]🔍 동적 문제 생성 시작: '{user_query}'[/bold cyan]")
        print("="*70)
        
        try:
            # 1. 벡터DB에서 관련 정보 검색
            print("1️⃣  벡터DB 검색 중...")
            # search_engine을 통해 텍스트를 검색합니다.
            search_result = self.search_engine.search_text(user_query, top_k=top_k)
            
            # 검색 결과에 에러가 포함되어 있으면, 에러 메시지를 담은 딕셔너리를 반환하고 함수를 종료합니다.
            if "error" in search_result:
                return {"error": f"벡터 검색 실패: {search_result['error']}"}
            
            # 2. 검색 결과를 구조화된 컨텍스트(SearchContext)로 변환
            print("2️⃣  컨텍스트 구조화 중...")
            context = self._create_search_context(search_result, user_query)
            
            # 3. LLM에게 문제 생성 요청 (이미지 선택 전에 텍스트 정보만으로)
            print("3️⃣  LLM 문제 생성 중...")
            if not self.gemini_client:
                return {"error": "Gemini API 사용 불가"}
            
            # 텍스트 컨텍스트를 기반으로 문제(질문, 보기, 정답, 해설)를 생성합니다.
            generated_question = self._generate_question_only(context)
            
            # 문제 생성에 실패했거나 LLM이 오류를 반환한 경우
            if not generated_question or "error" in generated_question:
                error_detail = generated_question.get('raw_response', 'LLM 응답 없음')
                return {"error": f"LLM 문제 생성 실패. 상세: {error_detail}"}
            
            # 4. 생성된 문제를 기반으로 LLM이 가장 적합한 이미지 타입 선택
            print("4️⃣  생성된 문제 분석 후, LLM 이미지 타입 선택 중...")
            image_selection = self._select_appropriate_image(generated_question, context)
            
            # 5. LLM이 선택한 타입으로 관련 이미지 검색
            selected_images = []
            selected_type = image_selection.get("selected_image_type", "None") # LLM의 선택 or 기본값 'None'
                
            # LLM이 'None'이 아닌 특정 이미지 타입을 선택한 경우에만 이미지 검색을 수행합니다.
            if selected_type != "None":
                print(f"5️⃣  선택된 타입 '{selected_type}'으로 이미지 검색 중...")
                selected_images = self._fetch_selected_images(image_selection, search_result)
                if not selected_images:
                    print(f"   ⚠️ '{selected_type}' 이미지를 찾지 못했지만, 문제는 계속 생성됩니다.")
            else:
                print("5️⃣  LLM이 이미지가 불필요하다고 판단함 (검색 생략)")

            # 6. 모든 결과를 종합하여 최종 결과 딕셔너리를 구성
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
                "created_at": datetime.now().isoformat() # 생성 시각 기록
            }
            
            reason = image_selection.get("reason", "선택 이유 없음")
            print(f"✅ 동적 문제 생성 완료: 최종 이미지 타입 '{selected_type}'")
            print(f"   💡 LLM 선택 이유: {reason}")
            
            return result
        
        except Exception as e:
            # 전체 과정에서 예측하지 못한 오류가 발생하면 처리합니다.
            print(f"❌ 문제 생성 실패: {e}")
            return {"error": f"동적 생성 중 치명적 오류: {str(e)}"}
    
    def _generate_question_only(self, context: SearchContext) -> Optional[Dict]:
        """
        검색된 텍스트 컨텍스트를 바탕으로, LLM을 이용해 5지선다 문제를 생성하는 내부 메서드입니다.
        """
        
        # LLM에게 전달할 프롬프트(명령서)를 작성합니다. f-string을 사용하여 동적으로 내용을 채웁니다.
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
            # Gemini 클라이언트에 프롬프트를 전달하여 콘텐츠 생성을 요청합니다.
            response = self.gemini_client.generate_content(prompt)
            
            # LLM의 응답(response.text)에서 JSON 부분만 정확히 추출하기 위해 정규표현식을 사용합니다.
            import re
            # '{'로 시작하고 '}'로 끝나는 가장 큰 문자열 조각을 찾습니다. (re.DOTALL은 줄바꿈 문자도 포함)
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            
            if json_match:
                # 찾은 JSON 문자열을 파이썬 딕셔너리로 변환하여 반환합니다.
                return json.loads(json_match.group())
            else:
                # LLM이 JSON 형식이지만 문법이 잘못된 경우, 오류를 반환합니다.
                print("   ⚠️ LLM이 JSON 형식으로 응답하지 않음")
                return {"error": "JSON 파싱 실패", "raw_response": response.text}
                
        except Exception as e:
            print(f"   ❌ 문제 생성 실패: {e}")
            return {"error": str(e)}

    def _select_appropriate_image(self, generated_question: Dict, context: SearchContext) -> Dict:
        """
        생성된 문제를 LLM이 분석하여 문제 풀이에 가장 적절한 이미지 타입을 선택하는 내부 메서드입니다.
        """
        
        # LLM에게 전달할 분석 자료들을 준비합니다.
        question_text = generated_question.get("question", "")
        explanation = generated_question.get("explanation", "")
        topic_analysis = generated_question.get("topic_analysis", {})
        
        # 이미지 타입 선택을 위한 프롬프트를 작성합니다.
        # 이 프롬프트는 LLM이 의료 영상 전문가처럼 행동하도록 지시합니다.
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
            # Gemini에 이미지 타입 선택을 요청합니다.
            response = self.gemini_client.generate_content(selection_prompt)
            
            # 응답에서 JSON 부분만 추출합니다.
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            
            if json_match:
                try:
                    # JSON 문자열을 파이썬 딕셔너리로 변환합니다.
                    parsed_json = json.loads(json_match.group())
                
                    # LLM의 응답이 유효한지 검사합니다.
                    selected_type = parsed_json.get("selected_image_type", "None")
                    
                    # 미리 정의된 유효한 이미지 타입 목록
                    valid_types = [
                        "Pneumonia", "Effusion", "Mass", "Nodule", 
                        "Pneumothorax", "Atelectasis", "Infiltrate", "Cardiomegaly", "None"
                    ]
                    
                    # LLM이 목록에 없는 이상한 값을 반환했을 경우, 'None'으로 강제 조정합니다.
                    if selected_type not in valid_types:
                            print(f"   ⚠️ LLM이 유효하지 않은 이미지 타입 선택: '{selected_type}' → 'None'으로 강제 조정")
                            parsed_json["reason"] += f" (원본 선택: {selected_type})"
                            parsed_json["selected_image_type"] = "None"
                        
                    return parsed_json
                
                except json.JSONDecodeError:
                    # JSON 형식이지만 문법이 잘못된 경우 (예: 쉼표 누락)
                    return {"error": "LLM 응답이 유효한 JSON이 아님", "raw_response": response.text}
            else:
                # 응답에서 JSON을 전혀 찾을 수 없는 경우
                return {"error": "LLM 응답에서 JSON을 찾을 수 없음", "raw_response": response.text}
                
        except Exception as e:
            print(f"   ❌ 이미지 타입 선택 실패: {e}")
            return {"error": str(e)}

    def _fetch_selected_images(self, image_selection: Dict, search_result: Dict) -> List[Dict]:
        """
        LLM이 선택한 이미지 타입을 바탕으로, 검색 엔진을 통해 실제 이미지 파일 정보를 가져옵니다.
        """
        
        selected_type = image_selection.get("selected_image_type", "None")
        # 선택된 타입이 'None'이거나 검색 엔진이 없다면 빈 리스트를 반환합니다.
        if selected_type == "None" or not self.search_engine:
            return []
        
        try:
            # 검색 엔진의 이미지 검색 기능을 호출합니다. 질병명을 전달하여 관련 이미지를 검색합니다.
            image_results = self.search_engine.search_images_by_disease(
                disease_name=selected_type, 
                top_k=5 
            )
            
            # 검색 결과를 최종 출력 형식에 맞게 가공합니다.
            formatted_results = []
            for res in image_results:
                formatted_results.append({
                    "image_path": res.get("image_path"), # 올바른 키 'image_path' 사용
                    "score": res.get("relevance_score"),      # 올바른 키 'relevance_score' 사용
                    "labels": res.get("labels", [])          # 올바른 키 'labels' 사용
                })
            
            print(f"   ✅ '{selected_type}' 이미지 {len(formatted_results)}개 검색 완료")
            return formatted_results
            
        except Exception as e:
            print(f"   ❌ '{selected_type}' 이미지 검색 실패: {e}")
            return []
    
    def _create_search_context(self, search_result: Dict, user_query: str) -> SearchContext:
        """
        검색 엔진의 초기 검색 결과를 바탕으로, LLM에게 전달할 구조화된 컨텍스트(SearchContext) 객체를 생성합니다.
        """
        
        # 텍스트 검색 결과를 LLM이 이해하기 쉬운 형태로 포맷팅합니다.
        text_chunks = [f"- {item}" for item in search_result.get("text_content", "").splitlines() if item.strip()]
        text_content_formatted = "\n\n".join(text_chunks)
        
        # 이미지 검색 결과를 포맷팅합니다.
        image_info_formatted = "이미지 정보 없음"
        primary_image_path = ""
        has_images = False
        
        # 이미지 결과가 있는 경우에만 처리합니다.
        if search_result.get("image_results") and len(search_result["image_results"]) > 0:
            image_list = []
            # 상위 5개 이미지 정보만 간결하게 요약합니다.
            for item in search_result["image_results"][:5]:
                image_list.append(f"- 경로: {item['id']}, 라벨: {item['payload'].get('labels_kr', 'N/A')}")
            
            image_info_formatted = "## 관련 이미지 데이터\n" + "\n".join(image_list)
            primary_image_path = search_result["image_results"][0]["id"] # 가장 관련도 높은 이미지를 대표 이미지로 설정
            has_images = True

        # 검색 엔진이 예측한 주제를 가져옵니다. 없으면 '일반 의료'를 기본값으로 사용합니다.
        estimated_topic = search_result.get("predicted_topic_kr", "일반 의료")
        
        # 최종적으로 SearchContext 객체를 생성하여 반환합니다.
        return SearchContext(
            query=user_query,
            text_content=text_content_formatted,
            image_info=image_info_formatted,
            confidence=search_result.get("confidence", "unknown"),
            has_images=has_images,
            estimated_topic=estimated_topic,
            primary_image_path=primary_image_path
        )

# --- 편의 함수 (Helper Functions) ---
# 이 스크립트 외부에서 DynamicQuestionGenerator를 더 쉽게 사용할 수 있도록 도와주는 함수들입니다.

def create_dynamic_generator() -> DynamicQuestionGenerator:
    """DynamicQuestionGenerator 인스턴스를 생성하여 반환하는 간단한 함수입니다."""
    return DynamicQuestionGenerator()

def quick_dynamic_generate(query: str = "폐렴") -> Dict:
    """
    쿼리 하나만으로 동적 문제 생성을 빠르게 테스트해볼 수 있는 함수입니다.
    """
    print(f"🚀 빠른 동적 생성: '{query}'")
    
    try:
        generator = create_dynamic_generator()
        result = generator.generate_question_from_query(query)
        
        # 생성 성공 시, 주요 정보를 요약하여 출력합니다.
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
            # 생성 실패 시, 에러 메시지를 출력합니다.
            print(f"❌ 생성 실패: {result['error']}")
            return result
            
    except Exception as e:
        print(f"❌ 오류: {e}")
        return {"error": str(e)}
