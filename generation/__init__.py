"""
Phase V & VI: LLM 연동 및 문제 생성을 위한 파이프라인

이 'generation' 패키지는 프로젝트의 핵심 두뇌 역할을 수행합니다.
검색 시스템(Phase IV)이 찾아낸 멀티모달 정보를 입력받아,
LLM(Gemini 1.5 Pro)이 이해하고 활용할 수 있는 최종 프롬프트로 가공하는
전체 파이프라인을 책임집니다.

주요 흐름:
검색 결과(dict) -> [ContextBuilder] -> 구조화된 컨텍스트(str) -> [PromptEngineer] -> 최종 프롬프트(str) -> [LLMInterface] -> LLM 응답(dict)

각 모듈의 역할:
- context_builder.py: 검색 결과를 LLM이 이해하기 쉬운 구조화된 텍스트 컨텍스트로 변환합니다. (재료 손질)
- prompt_engineer.py: 가공된 컨텍스트를 바탕으로, '문제 생성' 등 특정 작업을 지시하는 최종 프롬프트를 설계합니다. (레시피 작성)
- llm_interface.py: Gemini API와의 실제 통신을 담당하며, 프롬프트를 전송하고 응답을 받아 파싱합니다. (요리사에게 전달)
- run_question_generation.py: 위 모듈들을 통합하여 'AI 문제 생성' 기능을 실행하는 메인 스크립트입니다. (주방 총괄)
"""

# 패키지 초기화 코드가 필요하다면 여기에 작성할 수 있습니다.
# 현재는 주석 설명 외의 기능이 없으므로 비워둡니다.