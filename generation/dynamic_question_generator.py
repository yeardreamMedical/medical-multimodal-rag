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
        """사용자 쿼리로부터 동적 문제 생성 + LLM 이미지 선택"""
        
        print(f"\n🔍 동적 문제 생성: '{user_query}'")
        print("="*60)
        
        try:
            # 1. 벡터DB에서 관련 정보 검색
            print("1️⃣ 벡터DB 검색 중...")
            search_result = self.search_engine.search_text(user_query, top_k=top_k)
            
            if "error" in search_result:
                return {"error": f"벡터 검색 실패: {search_result['error']}"}
            
            # 2. 검색 결과를 구조화된 컨텍스트로 변환
            print("2️⃣ 컨텍스트 구조화 중...")
            context = self._create_search_context(search_result, user_query)
            
            # 3. LLM에게 문제 생성 요청 (이미지 선택 없이)
            print("3️⃣ LLM 문제 생성 중...")
            if not self.gemini_client:
                return {"error": "Gemini API 사용 불가"}
            
            generated_question = self._generate_question_only(context)
            
            if not generated_question or "error" in generated_question:
                return {"error": "LLM 문제 생성 실패"}
            
            # 4. 텍스트 전용 모드 확인 및 조기 처리
            is_text_only_mode = search_result.get("is_text_only_mode", False)

            if is_text_only_mode:
                print("4️⃣ 텍스트 전용 모드 감지 - 이미지 검색 생략")
                # 이미지 선택 및 검색 생략
                image_selection = {
                    "selected_image_type": "None",
                    "korean_name": "텍스트 전용",
                    "reason": "질병 매칭 실패로 인한 텍스트 전용 처리 (흉부 무관 주제 추정)",
                    "relevance_score": 1,
                    "is_chest_related": False,
                    "query_match": "low"
                }
                selected_images = []
            else:
                # 기존 이미지 선택 로직
                print("4️⃣ LLM 이미지 타입 선택 중...")
                image_selection = self._select_appropriate_image(generated_question, context)
                
                print("5️⃣ 선택된 타입으로 이미지 검색 중...")
                selected_images = self._fetch_selected_images(image_selection, search_result)

            # 6. 결과 구성 (기존과 동일하되 단계 번호만 조정)
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
                    "selected_image_type": image_selection.get("selected_image_type", "None"),
                    "is_text_only_mode": is_text_only_mode  # 새 플래그 추가
                },
                "generation_metadata": {
                    "method": "dynamic_vector_search_with_early_text_only_detection",
                    "search_quality": context.confidence,
                    "llm_model": "gemini-1.5-pro",
                    "vector_db_used": True,
                    "image_selection_method": "early_detection" if is_text_only_mode else "llm_analysis"
                },
                "created_at": datetime.now().isoformat()
            }
            
            selected_type = image_selection.get("selected_image_type", "None")
            reason = image_selection.get("reason", "선택 이유 없음")
            print(f"✅ 동적 문제 생성 완료: 이미지 타입 '{selected_type}' 선택")
            print(f"   💡 선택 이유: {reason}")
            
            return result
        
        except Exception as e:
            print(f"❌ 문제 생성 실패: {e}")
            return {"error": f"동적 생성 중 오류: {str(e)}"}
    
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
        selection_prompt = f"""당신은 의료 영상 전문가입니다. 다음 문제를 분석하여 흉부 X-ray가 필요한지 판단하세요.

# 원본 사용자 쿼리
"{context.query}"

# 생성된 문제 내용
문제: {question_text}
해설: {explanation}
추정 주제: {topic_analysis.get('estimated_topic', 'Unknown')}

# 핵심 판단 기준
**반드시 None을 선택해야 하는 경우:**
- 원본 쿼리와 생성된 문제의 주제가 완전히 다른 경우
- 외상외과: 열상, 골절, 화상, 외상 등
- 정형외과: 관절, 근골격계 질환
- 내과 비흉부: 당뇨병, 신장질환, 간질환, 내분비 등
- 피부과, 안과, 이비인후과 질환
- 실험실 검사, 혈액검사, 소변검사 중심 문제
- 약물 치료, 수술 적응증 등 비영상 문제

**흉부 X-ray가 필요한 경우만:**
- 폐, 기관지, 흉막, 심장 질환
- 호흡기 감염, 폐렴, 결핵
- 흉통, 호흡곤란의 원인 진단
- 명확한 흉부 영상 소견이 언급된 경우

# 선택 옵션
1. Pneumonia (폐렴) - 폐감염, 발열, 기침, 가래
2. Effusion (흉수) - 늑막삼출, 호흡곤란
3. Mass (종괴) - 폐종괴, 폐암, 종양
4. Nodule (결절) - 폐결절, 소결절
5. Pneumothorax (기흉) - 기흉, 흉통, 응급
6. Atelectasis (무기폐) - 폐허탈, 수술후
7. Infiltrate (침윤/경화) - 폐침윤, 경화
8. Cardiomegaly (심장비대) - 심부전, 심장확대
9. **None (텍스트 전용)** - 흉부 X-ray 불필요

# 출력 형식
{{
    "selected_image_type": "선택된_타입_또는_None",
    "korean_name": "한국어명_또는_텍스트전용",
    "reason": "선택 근거 (특히 None 선택시 상세 이유)",
    "relevance_score": 점수(1-10),
    "is_chest_related": true/false,
    "query_match": "원본 쿼리와 문제 일치도 (high/medium/low)"
}}

**주의: 원본 쿼리 "{context.query}"와 생성된 문제가 다른 의료 분야라면 반드시 None을 선택하세요!**"""


        try:
            response = self.gemini_client.generate_content(selection_prompt)
            
            # JSON 추출
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            
            if json_match:
                selection_result = json.loads(json_match.group())
                
                # 유효성 검증
                valid_types = [
                    "Pneumonia", "Effusion", "Mass", "Nodule", 
                    "Pneumothorax", "Atelectasis", "Infiltrate", "Cardiomegaly", "None"
                ]
                
                selected_type = selection_result.get("selected_image_type", "None")
                if selected_type not in valid_types:
                    print(f"   ⚠️ 잘못된 이미지 타입: {selected_type}, None으로 설정")
                    selection_result["selected_image_type"] = "None"
                    selection_result["reason"] = "유효하지 않은 선택으로 인한 텍스트 전용 처리"
                
                # 흉부 무관 주제 처리 추가
                is_chest_related = selection_result.get("is_chest_related", True)
                if not is_chest_related:
                    print(f"   📝 흉부 무관 주제로 판단: 텍스트 전용 처리")
                    selection_result["selected_image_type"] = "None"
                
                return selection_result
            
            else:
                print("   ⚠️ 이미지 선택 JSON 파싱 실패")
                return {
                    "selected_image_type": "None",
                    "korean_name": "이미지 없음",
                    "reason": "LLM 응답 파싱 실패",
                    "relevance_score": 1,
                    "alternative_types": []
                }
                
        except Exception as e:
            print(f"   ❌ 이미지 선택 실패: {e}")
            return {
                "selected_image_type": "None",
                "korean_name": "이미지 없음", 
                "reason": f"선택 과정에서 오류 발생: {str(e)}",
                "relevance_score": 1,
                "alternative_types": []
            }

    def _fetch_selected_images(self, image_selection: Dict, search_result: Dict) -> List[Dict]:
        """선택된 이미지 타입으로 실제 이미지 검색"""
        
        selected_type = image_selection.get("selected_image_type", "None")
        
        if selected_type == "None":
            print("   📝 이미지 없음 (LLM 판단)")
            return []
        
        print(f"   🖼️ '{selected_type}' 타입 이미지 검색 중...")
        
        try:
            # 선택된 질병으로 이미지 검색 (기존 search_engine 활용)
            dummy_vector = [0.0] * 512
            filter_condition = {"primary_label": {"$eq": selected_type}}
            
            # 해당 질병 이미지 수 확인
            disease_info = self.search_engine.config.DISEASE_INFO.get(selected_type, {})
            available_count = disease_info.get('count', 0)
            
            if available_count == 0:
                print(f"   ⚠️ '{selected_type}' 이미지 없음")
                return []
            
            search_count = min(3, available_count)  # 최대 3개
            
            results = self.search_engine.image_index.query(
                vector=dummy_vector,
                filter=filter_condition,
                top_k=search_count,
                include_metadata=True
            )
            
            selected_images = []
            for match in results['matches']:
                metadata = match['metadata']
                selected_images.append({
                    'image_id': match['id'],
                    'disease': selected_type,
                    'labels': metadata.get('labels', []),
                    'description': metadata.get('all_descriptions', ''),
                    'primary_label': metadata.get('primary_label', ''),
                    'image_path': metadata.get('image_path', ''),
                    'bbox_info': metadata.get('bboxes', []),
                    'relevance_score': 1.0,
                    'selection_method': 'llm_choice'
                })
            
            print(f"   ✅ {len(selected_images)}개 '{selected_type}' 이미지 발견")
            return selected_images
            
        except Exception as e:
            print(f"   ❌ '{selected_type}' 이미지 검색 실패: {e}")
            return []
    
    def _create_search_context(self, search_result: Dict, user_query: str) -> SearchContext:
        """검색 결과를 구조화된 컨텍스트로 변환 (이미지 경로 포함)"""
        
        # 기본 정보 추출
        text_content = search_result.get("text_content", "")
        image_info = search_result.get("image_info", "")
        confidence = search_result.get("confidence", "low")
        
        # 이미지 결과에서 실제 경로 추출
        images = search_result.get("images", [])
        has_images = len(images) > 0
        
        # 첫 번째 이미지의 경로 추출
        primary_image_path = ""
        if images and isinstance(images, list):
            first_image = images[0]
            if isinstance(first_image, dict):
                primary_image_path = first_image.get("image_path", "")
        
        estimated_topic = search_result.get("korean_diagnosis", "미상") or search_result.get("diagnosis", "Unknown")
        
        context = SearchContext(
            query=user_query,
            text_content=text_content,
            image_info=image_info,
            confidence=confidence,
            has_images=has_images,
            estimated_topic=estimated_topic,
            primary_image_path=primary_image_path  # 추가
        )
        
        print(f"   📊 컨텍스트 생성: {context.estimated_topic} (신뢰도: {confidence})")
        print(f"   🖼️ 이미지 포함: {has_images}")
        if primary_image_path:
            print(f"   📷 이미지 경로: {Path(primary_image_path).name}")
        
        return context

    
    def _generate_with_llm_analysis(self, context: SearchContext) -> Optional[Dict]:
        """LLM이 컨텍스트를 분석하고 적절한 문제 생성"""
        
        # 동적 프롬프트 생성
        prompt = self._create_dynamic_prompt(context)
        
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
            print(f"   ❌ LLM 호출 실패: {e}")
            return {"error": str(e)}
    
    def _create_dynamic_prompt(self, context: SearchContext) -> str:
        """컨텍스트 기반 동적 프롬프트 생성"""
        
        # 이미지 유무에 따른 조건부 프롬프트
        image_instruction = ""
        if context.has_images:
            image_instruction = """
이 문제는 흉부 X-ray 이미지와 함께 제시될 예정입니다. 
문제는 반드시 이미지 해석이 필요하도록 구성하세요.
"""
        else:
            image_instruction = """
이 문제는 이미지 없이 텍스트만으로 구성됩니다.
임상 증례와 검사 결과 설명에 집중하세요.
"""
        
        # 신뢰도에 따른 조건부 프롬프트
        confidence_instruction = ""
        if context.confidence == "high":
            confidence_instruction = "검색된 의료 정보의 신뢰도가 높으므로, 구체적이고 전문적인 문제를 생성하세요."
        elif context.confidence == "medium":
            confidence_instruction = "검색된 정보를 바탕으로 하되, 일반적인 의학 지식도 활용하여 문제를 보완하세요."
        else:
            confidence_instruction = "검색 정보가 제한적이므로, 해당 주제의 기본적이고 핵심적인 내용으로 문제를 구성하세요."
        
        prompt = f"""당신은 한국 의사국가고시 문제 출제 전문가입니다.

# 사용자 요청
사용자가 "{context.query}"에 대한 문제를 요청했습니다.

# 벡터DB 검색 결과
추정 의료 주제: {context.estimated_topic}
검색 신뢰도: {context.confidence}

## 관련 의학 지식
{context.text_content[:2000]}  

## 이미지 정보
{context.image_info}

# 생성 지침

## 기본 요구사항
1. 한국 의사국가고시 형식의 5지선다 객관식 문제 1개 생성
2. 실제 임상 상황을 반영한 환자 증례 포함
3. 위의 검색된 의료 지식을 최대한 활용
4. 한국 의료 환경과 용어에 맞게 작성

## 조건부 지침
{image_instruction}

{confidence_instruction}

## 출력 형식
다음 JSON 형식으로만 응답하세요:

{{
    "question": "문제 본문 (환자 증례 포함)",
    "options": ["보기1", "보기2", "보기3", "보기4", "보기5"],
    "answer": 정답_인덱스_숫자(0-4),
    "explanation": "정답 근거 및 해설",
    "topic_analysis": {{
        "estimated_topic": "LLM이 판단한 정확한 의료 주제",
        "difficulty_level": "초급/중급/고급",
        "clinical_relevance": "high/medium/low",
        "requires_image": {str(context.has_images).lower()}
    }},
    "source_utilization": "검색된 정보를 어떻게 활용했는지 설명"
}}

검색된 의학 지식을 충실히 반영하여 정확하고 교육적 가치가 높은 문제를 생성하세요."""

        return prompt
    
    def generate_multiple_dynamic(self, queries: List[str], questions_per_query: int = 2) -> List[Dict]:
        """여러 쿼리에 대해 동적 문제 생성"""
        
        print(f"\n🔄 다중 동적 생성: {len(queries)}개 쿼리")
        print("="*50)
        
        all_results = []
        
        for i, query in enumerate(queries):
            print(f"\n📋 {i+1}/{len(queries)}: '{query}'")
            
            for j in range(questions_per_query):
                print(f"   {j+1}/{questions_per_query} 번째 문제...")
                
                result = self.generate_question_from_query(query)
                
                if "error" not in result:
                    result["sequence"] = len(all_results) + 1
                    result["batch_query"] = query
                    all_results.append(result)
                    
                    topic = result["generated_question"].get("topic_analysis", {}).get("estimated_topic", "Unknown")
                    print(f"   ✅ 생성 완료: {topic}")
                else:
                    print(f"   ❌ 생성 실패: {result['error']}")
        
        print(f"\n🎯 총 {len(all_results)}개 문제 동적 생성 완료")
        return all_results
    
    def evaluate_dynamic_quality(self, results: List[Dict]) -> Dict[str, Any]:
        """동적 생성 문제들의 품질 평가"""
        
        if not results:
            return {"error": "평가할 결과가 없음"}
        
        print(f"\n📊 동적 생성 품질 평가 ({len(results)}개 문제)")
        print("="*50)
        
        # 통계 수집
        confidence_levels = []
        has_images_count = 0
        difficulty_distribution = {"초급": 0, "중급": 0, "고급": 0}
        clinical_relevance = {"high": 0, "medium": 0, "low": 0}
        
        for result in results:
            # 검색 품질
            search_context = result.get("search_context", {})
            confidence_levels.append(search_context.get("confidence", "low"))
            
            if search_context.get("has_images", False):
                has_images_count += 1
            
            # LLM 분석 품질
            question_data = result.get("generated_question", {})
            topic_analysis = question_data.get("topic_analysis", {})
            
            difficulty = topic_analysis.get("difficulty_level", "중급")
            if difficulty in difficulty_distribution:
                difficulty_distribution[difficulty] += 1
            
            relevance = topic_analysis.get("clinical_relevance", "medium")
            if relevance in clinical_relevance:
                clinical_relevance[relevance] += 1
        
        # 평가 결과
        high_quality_count = len([c for c in confidence_levels if c == "high"])
        quality_rate = (high_quality_count / len(confidence_levels)) * 100
        
        evaluation = {
            "summary": {
                "total_questions": len(results),
                "high_confidence_rate": round(quality_rate, 1),
                "image_supported_rate": round((has_images_count / len(results)) * 100, 1),
                "average_confidence": self._calculate_avg_confidence(confidence_levels)
            },
            "quality_distribution": {
                "search_confidence": dict(zip(*zip(*[(c, confidence_levels.count(c)) for c in set(confidence_levels)]))),
                "difficulty_levels": difficulty_distribution,
                "clinical_relevance": clinical_relevance
            },
            "system_performance": {
                "vector_search_success": len([r for r in results if "error" not in r]),
                "llm_generation_success": len([r for r in results if "generated_question" in r]),
                "end_to_end_success_rate": round(len(results) / len(results) * 100, 1)
            }
        }
        
        # 결과 출력
        print(f"📈 고신뢰도 비율: {quality_rate:.1f}%")
        print(f"🖼️ 이미지 지원: {evaluation['summary']['image_supported_rate']:.1f}%")
        print(f"🎯 평균 검색 품질: {evaluation['summary']['average_confidence']}")
        
        return evaluation
    
    def _calculate_avg_confidence(self, confidence_levels: List[str]) -> str:
        """평균 신뢰도 계산"""
        scores = {"high": 3, "medium": 2, "low": 1}
        avg_score = sum(scores.get(c, 1) for c in confidence_levels) / len(confidence_levels)
        
        if avg_score >= 2.5:
            return "high"
        elif avg_score >= 1.5:
            return "medium"
        else:
            return "low"
    
    def save_dynamic_results(self, results: List[Dict], filename: str = None) -> str:
        """동적 생성 결과 저장"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dynamic_generated_{timestamp}.json"
        
        save_dir = Path("generated_questions")
        save_dir.mkdir(exist_ok=True)
        
        filepath = save_dir / filename
        
        # 메타데이터와 함께 저장
        save_data = {
            "metadata": {
                "generation_method": "dynamic_vector_search",
                "total_questions": len(results),
                "generation_date": datetime.now().isoformat(),
                "system_version": "Dynamic Generator v1.0",
                "vector_db": "Phase IV RAG System",
                "llm_model": "Gemini 1.5 Pro"
            },
            "results": results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 동적 생성 결과 저장: {filepath}")
        return str(filepath)

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

def demo_dynamic_generation() -> None:
    """동적 생성 시스템 데모"""
    print("🎯 동적 문제 생성 데모")
    print("="*50)
    
    # 다양한 의료 주제 테스트
    test_queries = [
        "폐렴 진단",           # 이미지 + 텍스트
        "결핵 치료",           # 텍스트 위주
        "기흉 응급처치",       # 이미지 + 텍스트
        "심부전 관리",         # 텍스트 위주
        "폐색전증 진단",       # 텍스트 위주
        "흉수 천자술"          # 이미지 + 텍스트
    ]
    
    try:
        generator = create_dynamic_generator()
        results = generator.generate_multiple_dynamic(test_queries, questions_per_query=1)
        
        if results:
            filepath = generator.save_dynamic_results(results, "demo_dynamic.json")
            evaluation = generator.evaluate_dynamic_quality(results)
            
            print(f"\n🎉 동적 생성 데모 완료!")
            print(f"   📁 저장: {filepath}")
            print(f"   📊 고신뢰도 비율: {evaluation['summary']['high_confidence_rate']}%")
            print(f"   🖼️ 이미지 지원: {evaluation['summary']['image_supported_rate']}%")
            
        else:
            print("❌ 문제 생성 실패")
            
    except Exception as e:
        print(f"❌ 데모 실행 실패: {e}")

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            query = sys.argv[2] if len(sys.argv) > 2 else "폐렴"
            quick_dynamic_generate(query)
            
        elif command == "demo":
            demo_dynamic_generation()
            
        else:
            print("사용법:")
            print("  python dynamic_question_generator.py test [의료주제]")
            print("  python dynamic_question_generator.py demo")
    
    else:
        print("🤖 동적 의료 문제 생성 시스템")
        print("벡터DB 검색 → LLM 자율 판단 → 적응형 문제 생성")