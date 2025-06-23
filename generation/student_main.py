# student_main.py - 학생용 대화형 문제 풀이 시스템

# --- 라이브러리 임포트 ---
# 이 파일 역시 main.py와 유사하게 필요한 라이브러리들을 가져옵니다.
from pathlib import Path
import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
# rich.prompt는 사용자로부터 입력을 받을 때 사용하는 기능입니다.
from rich.prompt import Prompt

# --- 이미지 표시 관련 라이브러리 (main.py와 동일) ---
# 이미지 표시 기능은 선택 사항이므로, 관련 라이브러리가 없어도 프로그램이 동작하도록 try-except 구문을 사용합니다.
try:
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import platform
    # 맥 OS에서 한글 폰트가 깨지지 않도록 설정합니다.
    if platform.system() == 'Darwin':
        plt.rcParams['font.family'] = ['AppleGothic', 'Apple SD Gothic Neo', 'Helvetica']
        plt.rcParams['axes.unicode_minus'] = False
        print("✅ 맥 OS 한글 폰트 설정 완료")
    DISPLAY_AVAILABLE = True
except ImportError:
    DISPLAY_AVAILABLE = False

# --- 프로젝트 경로 설정 (main.py와 동일) ---
# 다른 폴더에 있는 모듈을 불러올 수 있도록 프로젝트 루트 경로를 sys.path에 추가합니다.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# --- 핵심 모듈 임포트 ---
from generation.dynamic_question_generator import DynamicQuestionGenerator

# --- 메인 실행 함수 ---
def main(query: str = "폐렴"):
    """
    학생 사용자를 위해 문제를 단계적으로 제시하는 메인 함수입니다.
    [수정된 학습 흐름]
    1. 문제와 관련 이미지를 먼저 제시 -> 2. 사용자 입력 대기 -> 3. 정답, 해설, 문제 정보 표시
    
    Args:
        query (str): 생성할 문제의 주제. 기본값은 "폐렴"입니다.
    """
    console = Console()
    console.print(f"[bold cyan]🩺 학생용 의학 문제 풀이: '{query}'[/bold cyan]")
    console.print("="*70)

    with console.status("[bold green]문제 생성 중...", spinner="dots") as status:
        try:
            # 1. 문제 생성기 초기화 및 문제 생성 (main.py와 동일)
            status.update("[bold green]🔧 시스템 초기화 중...")
            generator = DynamicQuestionGenerator()
            console.log("✅ 생성 시스템 준비 완료")

            status.update(f"[bold yellow]🔍 '{query}' 분석 및 문제 생성 중...")
            result = generator.generate_question_from_query(query, top_k=10)

            if "error" in result:
                console.print(f"[bold red]❌ 생성 실패:[/bold red] {result['error']}")
                return

            console.log("✅ 문제 생성 완료")

            # [수정된 부분] 학습 흐름 제어
            selected_type = result.get("image_selection", {}).get("selected_image_type", "None")
            image_available = selected_type != "None" and len(result.get("selected_images", [])) > 0
            
            # 2-1. 이미지가 있다면 먼저 화면에 표시합니다.
            image_displayed = False
            if image_available:
                status.update("[bold yellow]🖼️ 참고 이미지 표시 중...")
                image_displayed = display_related_image(console, result, query, show_window=True)
            
            # 2-2. 문제와 보기만 출력합니다.
            status.update("[bold green]📋 문제 포맷팅 중...")
            print_question_and_options(console, result, image_displayed)
            console.log("✅ 문제 출력 완료")
            
            # 2-3. 사용자가 정답을 확인할 준비가 될 때까지 대기
            console.print("\n\n")
            # 프롬프트 메시지를 수정하여 이미지가 먼저 표시되었음을 암시합니다.
            Prompt.ask("[bold yellow]정답과 해설을 보려면 Enter 키를 누르세요...[/bold yellow]")

            # 2-4. 정답, 해설 및 문제의 상세 정보를 함께 출력합니다.
            print_answer_and_explanation(console, result, query)
            console.log("✅ 모든 과정 완료!")

        except Exception as e:
            console.print(f"[bold red]❌ 시스템 오류:[/bold red] {str(e)}")

def display_related_image(console: Console, result: dict, query: str = "", show_window: bool = True) -> bool:
    """
    LLM이 선택한 이미지를 화면에 표시합니다.
    이 함수는 `main.py`의 것과 거의 동일하지만, `show_window` 인자가 추가되었습니다.
    
    Args:
        show_window (bool): True이면 이미지를 화면에 창으로 띄우고, False이면 파일 존재 여부만 확인합니다.
    """
    
    # 1. 이미지 정보 추출
    image_selection = result.get("image_selection", {})
    selected_images = result.get("selected_images", [])
    selected_type = image_selection.get("selected_image_type", "None")
    
    # 2. 이미지 표시 가능 여부 확인
    if selected_type == "None" or not selected_images or not DISPLAY_AVAILABLE:
        return False

    first_image = selected_images[0]
    image_path = first_image.get("image_path", "")
    if not image_path:
        return False

    # 3. 이미지 파일 경로 구성
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    base_dir = project_root / "data" / "chestxray14" / "bbox_images"
    if not base_dir.exists():
        console.print(f"[red]오류: 이미지 기본 디렉토리({base_dir})를 찾을 수 없습니다.[/red]")
        return False

    full_image_path = base_dir / image_path
    
    # `show_window`가 False이면, 창을 띄우지 않고 파일이 실제로 존재하는지만 확인하고 반환합니다.
    if not show_window:
        return full_image_path.exists()

    # 4. 이미지 표시 (matplotlib 사용)
    try:
        img = Image.open(full_image_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(img, cmap='gray' if img.mode == 'L' else None)
        
        # 학생용이므로 제목은 조금 더 간결하게 표시합니다.
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
        console.print(f"[red]❌ 이미지 표시 중 오류 발생: {e}[/red]")
        return False

def print_question_and_options(console: Console, result: dict, image_displayed: bool):
    """
    [수정] 학생에게 문제와 보기만 먼저 보여주는 함수입니다. (문제 정보 패널 제거)
    """
    # 1. 필요한 정보 추출
    question_data = result.get("generated_question", {})
    question_text = question_data.get("question", "문제 생성 실패")
    options = question_data.get("options", [])
    
    # 2. 문제와 보기를 담은 패널 생성
    question_content = f"### 문제\n\n{question_text}"
    
    # 이미지가 별도 창으로 표시된 경우, 참고하라는 안내 메시지를 추가합니다.
    if image_displayed:
        question_content += f"\n\n[bold green] (화면에 표시된 참고 이미지를 확인하여 문제를 풀어보세요)[/bold green]"

    question_content += "\n\n--- \n" # 마크다운의 가로줄 문법
    for i, option in enumerate(options):
        question_content += f"{i+1}. {option}\n"
        
    console.print(Panel(Markdown(question_content), 
                        title="[bold yellow]Q. 다음 문제를 풀어보세요[/bold yellow]", 
                        border_style="yellow"))

def print_answer_and_explanation(console: Console, result: dict, original_query: str):
    """
    [수정] 사용자가 정답 확인을 원할 때, 문제 정보, 정답, 해설, 분석 정보를 출력합니다.
    """
    # 1. 필요한 정보 추출
    question_data = result.get("generated_question", {})
    image_selection = result.get("image_selection", {})

    # [추가] 1-A. 문제 정보 요약 패널 생성
    topic_analysis = question_data.get("topic_analysis", {})
    estimated_topic = topic_analysis.get("estimated_topic", "주제 분석 실패")
    difficulty = topic_analysis.get("difficulty_level", "중급")
    image_available = image_selection.get("selected_image_type", "None") != "None"
    image_status = "[green]있음[/green]" if image_available else "[dim]없음[/dim]"
    
    meta_info_panel = Panel(
        (f"[bold]입력 쿼리:[/bold] {original_query} | "
         f"[bold]AI 추정 주제:[/bold] {estimated_topic} | "
         f"[bold]난이도:[/bold] {difficulty} | "
         f"[bold]관련 이미지:[/bold] {image_status}"),
        title="[bold cyan]문제 정보[/bold cyan]", 
        border_style="cyan"
    )
    console.print(meta_info_panel)

    # 2. LLM이 추천한 참고 이미지 정보 패널 생성 (이미지가 있는 경우)
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

    # 3. 정답 및 해설 패널 생성
    options = question_data.get("options", [])
    answer_idx = question_data.get("answer", -1)
    explanation = question_data.get("explanation", "해설 없음")

    # 정답 인덱스를 사용하여 정답 보기 텍스트를 찾습니다.
    answer_text = f"{answer_idx + 1}. {options[answer_idx]}" if 0 <= answer_idx < len(options) else "정답 정보 없음"

    answer_panel = Panel(
        f"[bold]정답:[/bold] {answer_text}\n\n---\n\n[bold]해설:[/bold]\n{explanation}",
        title="[bold green]정답 및 해설[/bold green]",
        border_style="green"
    )
    console.print(answer_panel)

    # 4. LLM의 추가 분석 정보 패널 생성
    topic_analysis = question_data.get("topic_analysis", {})
    clinical_relevance = topic_analysis.get("clinical_relevance", "medium")
    
    analysis_content = (f"[bold]임상적 중요도:[/bold] {clinical_relevance.upper()}")

    analysis_panel = Panel(analysis_content, 
                           title="[bold blue]🤖 AI 추가 분석[/bold blue]",
                           border_style="blue")
                           
    console.print(analysis_panel)

# --- 스크립트 실행 진입점 ---
if __name__ == '__main__':
    # 이 스크립트는 터미널에서 실행될 때 인자를 받아 처리할 수 있습니다.
    # 예: python generation/student_main.py "심근경색"
    default_query = "소아 폐렴의 가장 흔한 원인균"
    # len(sys.argv) > 1 은 터미널에서 인자가 주어졌는지 확인하는 조건문입니다.
    query = sys.argv[1] if len(sys.argv) > 1 else default_query
    
    try:
        main(query)
    except (KeyboardInterrupt, EOFError):
        # 사용자가 Ctrl+C 또는 Ctrl+D로 프로그램을 강제 종료하려 할 때,
        # 에러 메시지 대신 깔끔한 종료 메시지를 보여줍니다.
        console = Console()
        console.print("\n\n[bold]👋 학습을 종료합니다.[/bold]") 