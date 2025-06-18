import sys
import os
import pytest
import json

# 프로젝트 루트를 Python 경로에 추가하여 모듈을 찾을 수 있도록 함
# 이 코드는 어떤 위치에서 테스트를 실행하더라도 루트 경로를 기준으로 모듈을 임포트하게 해줌
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# --- 이제 프로젝트 루트를 기준으로 모듈을 임포트할 수 있습니다 ---
from search.search_engine import SearchEngine
from context.context_builder import LLMContextBuilder
from context.prompt_engineer import PromptEngineer
# from context.llm_interface import GeminiInterface # 실제 API 호출은 테스트에서 제외하거나 mock 처리

# 테스트할 쿼리 목록
TEST_QUERIES = ["Pneumonia", "Pneumothorax", "흉수"]

@pytest.mark.parametrize("query", TEST_QUERIES)
def test_full_prompt_generation_pipeline(query):
    """
    검색부터 프롬프트 생성까지 전체 파이프라인이 오류 없이 실행되는지 테스트합니다.
    """
    print(f"\n--- 🧪 테스트 실행: 쿼리 = '{query}' ---")
    
    # 1. 검색
    print("  [1/3] 검색 엔진 실행...")
    engine = SearchEngine()
    search_result = engine.search_text(query)
    assert "error" not in search_result, f"SearchEngine에서 에러 발생: {search_result.get('error')}"
    assert "korean_diagnosis" in search_result, "검색 결과에 'korean_diagnosis' 키가 없습니다."
    print(f"  ✅ 검색 성공: {search_result['korean_diagnosis']}")

    # 2. 컨텍스트 생성
    print("  [2/3] LLMContextBuilder 실행...")
    builder = LLMContextBuilder(search_result)
    llm_context = builder.build_context_for_llm()
    primary_image_path = builder.get_primary_image_path()
    assert isinstance(llm_context, str) and len(llm_context) > 50, "LLM 컨텍스트가 올바르게 생성되지 않았습니다."
    print("  ✅ 컨텍스트 생성 성공")

    # 3. 프롬프트 엔지니어링
    print("  [3/3] PromptEngineer 실행...")
    engineer = PromptEngineer(context=llm_context, image_path=primary_image_path)
    final_prompt = engineer.create_question_generation_prompt()
    assert isinstance(final_prompt, str) and "[의료 정보 컨텍스트]" in final_prompt, "최종 프롬프트가 올바르게 생성되지 않았습니다."
    print("  ✅ 프롬프트 생성 성공")
    
    print(f"--- ✅ 테스트 통과: 쿼리 = '{query}' ---")

# 참고: 실제 Gemini API를 호출하는 테스트는 비용과 시간이 발생하므로 보통 별도로 분리하거나,
# 응답을 미리 저장해두고 테스트하는 'mocking' 기법을 사용합니다.
# 지금은 API 호출 직전까지의 파이프라인을 테스트하는 것이 목표입니다.