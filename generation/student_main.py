# student_main.py - 학생용 대화형 문제 풀이 시스템
from pathlib import Path
import sys
import json
import os
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from typing import Optional

# 이미지 표시용 라이브러리 추가
try:
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import platform
    # 맥 OS 한글 폰트 설정
    if platform.system() == 'Darwin':  # macOS
        plt.rcParams['font.family'] = ['AppleGothic', 'Apple SD Gothic Neo', 'Helvetica']
        plt.rcParams['axes.unicode_minus'] = False
        print("✅ 맥 OS 한글 폰트 설정 완료")
    DISPLAY_AVAILABLE = True


except ImportError:
    DISPLAY_AVAILABLE = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from generation.dynamic_question_generator import DynamicQuestionGenerator

def main(query: str = "폐렴"):
    console = Console()

    console.print(f"[bold cyan]🩺 학생용 의학 문제 풀이: '{query}'[/bold cyan]")
    console.print("="*70)

    with console.status("[bold green]문제 생성 중...", spinner="dots") as status:
        try:
            status.update("[bold green]🔧 시스템 초기화 중...")
            generator = DynamicQuestionGenerator()
            console.log("✅ 생성 시스템 준비 완료")

            status.update(f"[bold yellow]🔍 '{query}' 분석 및 문제 생성 중...")
            result = generator.generate_question_from_query(query, top_k=10)

            if "error" in result:
                console.print(f"[bold red]❌ 생성 실패:[/bold red] {result['error']}")
                return

            console.log("✅ 문제 생성 완료")

            # 이미지 표시 단계 삭제 (정답 후로 이동)
            # image_displayed = display_related_image(console, result, query)
            selected_type = result.get("image_selection", {}).get("selected_image_type", "None")
            image_available = selected_type != "None" and len(result.get("selected_images", [])) > 0

            # 1. 문제와 보기 먼저 출력
            status.update("[bold green]📋 문제 포맷팅 중...")
            print_question_and_options(console, result, query, image_available)
            console.log("✅ 문제 출력 완료")
            
            # 2. 사용자에게 정답 확인 요청
            console.print("\n\n")
            Prompt.ask("[bold yellow]정답, 해설, 이미지 분석을 보려면 Enter 키를 누르세요...[/bold yellow]")

            # 3. 정답 및 해설 + 이미지 표시
            if image_available:
                display_related_image(console, result, query, show_window=True)
            print_answer_and_explanation(console, result)
            console.log("✅ 모든 과정 완료!")

        except Exception as e:
            console.print(f"[bold red]❌ 시스템 오류:[/bold red] {str(e)}")

def display_related_image(console: Console, result: dict, query: str = "", show_window: bool = True) -> bool:
    """LLM이 선택한 이미지 타입으로 이미지 표시
    show_window=False이면 표시하지 않고 단순 존재 여부만 반환"""
    
    image_selection = result.get("image_selection", {})
    selected_images = result.get("selected_images", [])
    selected_type = image_selection.get("selected_image_type", "None")
    
    if selected_type == "None" or not selected_images or not DISPLAY_AVAILABLE:
        return False

    first_image = selected_images[0]
    image_path = first_image.get("image_path", "")
    if not image_path:
        return False

    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    base_dir = project_root / "data" / "chestxray14" / "bbox_images"
    if not base_dir.exists():
        return False

    full_image_path = base_dir / image_path
    
    if not show_window:
        # 창을 띄우지 않고 존재만 확인
        return full_image_path.exists()

    try:
        img = Image.open(full_image_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(img, cmap='gray' if img.mode == 'L' else None)
        
        question_data = result.get("generated_question", {})
        topic_analysis = question_data.get("topic_analysis", {})
        estimated_topic = topic_analysis.get("estimated_topic", "Unknown")
        korean_name = image_selection.get("korean_name", selected_type)
        
        plt.title(f"참고 이미지: {korean_name}\nAI 추정 주제: {estimated_topic}", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        return True
    except Exception as e:
        console.print(f"[red]❌ 이미지 표시 실패: {e}[/red]")
        return False

def print_question_and_options(console: Console, result: dict, original_query: str, image_available: bool):
    """문제, 보기, 관련 메타정보 출력"""
    question_data = result.get("generated_question", {})
    
    question_text = question_data.get("question", "문제 생성 실패")
    options = question_data.get("options", [])
    
    topic_analysis = question_data.get("topic_analysis", {})
    estimated_topic = topic_analysis.get("estimated_topic", "주제 분석 실패")
    difficulty = topic_analysis.get("difficulty_level", "중급")

    image_status = "[green]표시됨[/green]" if image_available else "[dim]없음[/dim]"

    meta_info = (f"[bold]입력 쿼리:[/bold] {original_query} | "
                 f"[bold]AI 추정 주제:[/bold] {estimated_topic} | "
                 f"[bold]난이도:[/bold] {difficulty} | "
                 f"[bold]이미지:[/bold] {image_status}")
    
    console.print(Panel(meta_info, title="[bold cyan]문제 정보[/bold cyan]", border_style="cyan"))

    question_content = f"### 문제\n\n{question_text}\n\n"
    question_content += "--- \n"
    for i, option in enumerate(options):
        question_content += f"{i+1}. {option}\n"
        
    console.print(Panel(Markdown(question_content), 
                        title="[bold yellow]Q. 풀어보세요[/bold yellow]", 
                        border_style="yellow"))

def print_answer_and_explanation(console: Console, result: dict):
    """정답과 해설, 관련 분석 정보 출력"""
    question_data = result.get("generated_question", {})
    image_selection = result.get("image_selection", {})

    # 1. LLM 추천 이미지 정보 패널 (정답과 함께 표시)
    selected_type = image_selection.get("selected_image_type", "None")
    if selected_type != "None":
        korean_name = image_selection.get("korean_name", selected_type)
        reason = image_selection.get("reason", "선택 이유 없음")
        
        image_panel = Panel(
            f"[blue]🖼️ LLM 추천 이미지: [bold]{korean_name}[/bold] ({selected_type})\n\n[dim]이유: {reason}[/dim]", 
            title="[bold cyan]참고 이미지 분석[/bold cyan]", 
            border_style="cyan"
        )
        console.print(image_panel)

    # 2. 정답 및 해설 패널
    options = question_data.get("options", [])
    answer_idx = question_data.get("answer", -1)
    explanation = question_data.get("explanation", "해설 없음")

    answer_text = f"{answer_idx + 1}. {options[answer_idx]}" if 0 <= answer_idx < len(options) else "정답 정보 없음"

    # 정답 및 해설 패널
    answer_panel = Panel(
        f"[bold]정답:[/bold] {answer_text}\n\n---\n\n[bold]해설:[/bold]\n{explanation}",
        title="[bold green]정답 및 해설[/bold green]",
        border_style="green"
    )
    console.print(answer_panel)

    # 3. LLM 추가 분석 정보 패널 (이미지 관련 내용 제외)
    topic_analysis = question_data.get("topic_analysis", {})
    clinical_relevance = topic_analysis.get("clinical_relevance", "medium")
    
    analysis_content = (f"[bold]임상적 중요도:[/bold] {clinical_relevance}")

    analysis_panel = Panel(analysis_content, 
                           title="[bold blue]🤖 AI 추가 분석[/bold blue]",
                           border_style="blue")
                           
    console.print(analysis_panel)

if __name__ == '__main__':
    # 터미널에서 실행 시 기본 쿼리 또는 인자 사용
    # 예: python generation/student_main.py "심근경색"
    default_query = "소아 폐렴의 가장 흔한 원인균"
    query = sys.argv[1] if len(sys.argv) > 1 else default_query
    main(query) 