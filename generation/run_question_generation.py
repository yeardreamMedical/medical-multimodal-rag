# !pip install rich
import sys
import json
import os
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Python이 모듈을 검색하는 경로 리스트의 맨 앞에 프로젝트 루트를 추가합니다.
sys.path.insert(0, PROJECT_ROOT)

# 모든 모듈 임포트
from search.search_engine import SearchEngine
from generation.context_builder import LLMContextBuilder
from generation.prompt_engineer import PromptEngineer
from generation.llm_interface import GeminiInterface

def main(query: str = "Pneumonia"):
    console = Console()
    
    with console.status("[bold green]의학 문제 생성 파이프라인 실행 중...", spinner="dots") as status:
        # 1. 검색
        status.update("[bold green]1/5: 관련 의학 정보 검색 중...")
        engine = SearchEngine()
        search_result = engine.search_text(query)
        if "error" in search_result:
            console.print(f"[bold red]❌ 검색 실패:[/bold red] {search_result['error']}")
            return
        console.log("✅ 1/5: 검색 완료")

        # 2. 컨텍스트 생성
        status.update("[bold green]2/5: LLM용 컨텍스트 생성 중...")
        builder = LLMContextBuilder(search_result)
        llm_context = builder.build_context_for_llm()
        primary_image_path = builder.get_primary_image_path()
        console.log("✅ 2/5: 컨텍스트 생성 완료")

        # 3. 프롬프트 엔지니어링
        status.update("[bold green]3/5: 최종 프롬프트 디자인 중...")
        engineer = PromptEngineer(context=llm_context, image_path=primary_image_path)
        final_prompt = engineer.create_question_generation_prompt()
        console.log("✅ 3/5: 프롬프트 생성 완료")

        # 4. Gemini API 호출
        status.update("[bold yellow]4/5: Gemini 1.5 Pro 모델 호출 중... (시간이 걸릴 수 있습니다)")
        gemini = GeminiInterface()
        generated_data = gemini.generate_question_json(final_prompt)
        if "error" in generated_data:
            console.print(f"[bold red]❌ 문제 생성 실패:[/bold red] {generated_data['error']}")
            return
        console.log("✅ 4/5: 문제 생성 완료")
        
        # 5. 결과 출력
        status.update("[bold green]5/5: 생성된 문제 포맷팅 중...")
        print_generated_question(console, generated_data, primary_image_path)
        console.log("✅ 5/5: 모든 과정 완료!")

def print_generated_question(console: Console, data: dict, image_path: str):
    """생성된 질문을 예쁘게 출력합니다."""
    
    question_panel = Panel(
        f"[bold]문제:[/bold]\n{data.get('question', '')}\n\n"
        f"[bold]제시된 이미지:[/bold] {image_path}",
        title="[bold cyan]AI 생성 의사 국가고시 문제[/bold cyan]",
        border_style="cyan"
    )
    
    options_str = ""
    answer_idx = data.get('answer', -1)
    for i, option in enumerate(data.get('options', [])):
        if i == answer_idx:
            options_str += f"  [bold green]▶ {i+1}. {option}[/bold green]\n"
        else:
            options_str += f"  - {i+1}. {option}\n"

    options_panel = Panel(
        options_str,
        title="[bold yellow]보기[/bold yellow]",
        border_style="yellow"
    )
    
    explanation_panel = Panel(
        Markdown(data.get('explanation', '')),
        title="[bold blue]해설[/bold blue]",
        border_style="blue"
    )

    console.print("\n" * 2)
    console.print(question_panel)
    console.print(options_panel)
    console.print(explanation_panel)


if __name__ == "__main__":
    # 터미널에서 `python run_question_generation.py "폐렴"` 처럼 실행 가능
    if len(sys.argv) > 1:
        user_query = sys.argv[1]
    else:
        user_query = "폐렴" # 기본 쿼리
        
    main(user_query)