import sys
sys.path.append('./search')

from search.search_engine import SearchEngine
from generation.context_builder import LLMContextBuilder
from generation.prompt_engineer import PromptEngineer

def main():
    print("🚀 Phase 5: Step 2 - 전체 파이프라인 통합 테스트")
    
    # 1. 검색
    print("\n[1/4] 검색 엔진 실행...")
    engine = SearchEngine()
    search_result = engine.search_text("기흉이 의심되는 환자")
    if "error" in search_result:
        print(f"❌ 검색 실패: {search_result['error']}")
        return
    print("✅ 검색 성공!")

    # 2. 컨텍스트 생성
    print("\n[2/4] LLMContextBuilder 실행...")
    try:
        builder = LLMContextBuilder(search_result)
        llm_context = builder.build_context_for_llm()
        primary_image_path = builder.get_primary_image_path()
        print("✅ 컨텍스트 및 이미지 경로 생성 성공!")
    except ValueError as e:
        print(f"❌ 컨텍스트 생성 실패: {e}")
        return

    # 3. 프롬프트 엔지니어링
    print("\n[3/4] PromptEngineer 실행...")
    try:
        engineer = PromptEngineer(context=llm_context, image_path=primary_image_path)
        final_prompt = engineer.create_question_generation_prompt()
        print("✅ 최종 프롬프트 생성 성공!")
    except ValueError as e:
        print(f"❌ 프롬프트 생성 실패: {e}")
        return

    # 4. 최종 프롬프트 확인
    print("\n[4/4] Gemini에게 전달될 최종 프롬프트:")
    print("=" * 70)
    print(final_prompt)
    print("=" * 70)
    print("\n🎉 모든 파이프라인이 성공적으로 연결되었습니다!")

if __name__ == "__main__":
    main()