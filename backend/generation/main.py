# run_question_generation.py - 동적 시스템 + 이미지 표시

# --- 라이브러리 임포트 ---
# 파이썬의 표준 라이브러리와 외부 라이브러리들을 가져오는 부분입니다.

# pathlib: 파일 경로를 객체처럼 다루게 해주는 라이브러리입니다. 
# 운영체제(Windows, Mac, Linux)에 상관없이 경로를 쉽게 조작할 수 있어 매우 유용합니다.
from pathlib import Path
# sys: 파이썬 인터프리터와 관련된 시스템 특정 변수와 함수를 제공합니다. 
# 여기서는 다른 폴더에 있는 파이썬 모듈(파일)을 불러올 수 있도록 모듈 검색 경로를 추가하기 위해 사용됩니다.
import sys
# json: JSON(JavaScript Object Notation) 형식의 데이터를 파싱(읽기)하고 생성(쓰기)하는 데 사용됩니다.
import json
# os: 운영체제와 상호작용하기 위한 함수들을 제공합니다. 파일 시스템 경로 작업 등에 사용됩니다.
import os
# rich: 터미널에 아름다운 UI(색상, 표, 패널, 진행상황 바 등)를 쉽게 만들 수 있게 도와주는 라이브-러리입니다.
# 사용자가 프로그램을 더 편하게 사용할 수 있도록 시각적인 요소를 추가합니다.
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# --- 이미지 표시 관련 라이브러리 ---
# 이미지 표시 기능은 필수는 아니므로, 관련 라이브러리가 없어도 프로그램이 오류 없이 동작하도록 
# try-except 구문을 사용합니다. 이것을 '선택적 의존성(optional dependency)' 처리라고 합니다.
try:
    # PIL(Pillow): 파이썬 이미지 처리 라이브러리의 표준입니다. 이미지 파일을 열고, 조작하고, 저장하는 데 사용됩니다.
    from PIL import Image
    # matplotlib: 데이터 시각화를 위한 강력한 라이브러리입니다. 여기서는 이미지를 화면에 창으로 띄우는 데 사용됩니다.
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    # platform: 현재 실행 중인 운영체제(OS) 정보를 얻기 위해 사용됩니다.
    import platform 
    
    # 만약 운영체제가 'Darwin'(macOS의 공식 명칭)이라면, matplotlib에서 한글이 깨지지 않도록 폰트를 설정합니다.
    if platform.system() == 'Darwin':
        # Apple 시스템에 내장된 한글 폰트를 우선적으로 사용하도록 설정합니다.
        plt.rcParams['font.family'] = ['AppleGothic', 'Apple SD Gothic Neo', 'Helvetica']
        # 숫자 앞에 마이너스(-) 기호가 깨지는 현상을 방지합니다.
        plt.rcParams['axes.unicode_minus'] = False
        print("✅ 맥 OS 한글 폰트 설정 완료")
    
    # 이미지 관련 라이브러리가 모두 성공적으로 임포트되었음을 표시하는 플래그(깃발) 변수입니다.
    DISPLAY_AVAILABLE = True

# ImportError는 'import' 하려는 모듈을 찾지 못했을 때 발생하는 오류입니다.
except ImportError:
    # 라이브러리가 하나라도 설치되어 있지 않으면, 이미지 표시 기능을 사용할 수 없도록 플래그를 False로 설정합니다.
    DISPLAY_AVAILABLE = False

# --- 프로젝트 경로 설정 ---
# 이 파일(main.py)의 절대 경로를 기준으로 프로젝트의 최상위(루트) 폴더 경로를 계산합니다.
# os.path.dirname(__file__)는 현재 파일이 있는 디렉토리 경로를 반환합니다. ('generation' 폴더)
# '..'는 상위 폴더를 의미하므로, 'generation' 폴더의 상위 폴더인 프로젝트 루트 폴더를 가리킵니다.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 파이썬이 모듈을 찾을 때 검색하는 경로 목록(sys.path)의 맨 앞에 프로젝트 루트 폴더를 추가합니다.
# 이렇게 하면 'generation' 폴더나 'search' 폴더 등 다른 폴더에 있는 파이썬 파일(모듈)을 쉽게 `import`할 수 있습니다.
sys.path.insert(0, PROJECT_ROOT)

# --- 핵심 모듈 임포트 ---
# 위에서 설정한 경로 덕분에 'generation' 폴더 안에 있는 'dynamic_question_generator.py' 파일에서
# 'DynamicQuestionGenerator' 클래스를 성공적으로 가져올 수 있습니다.
from generation.dynamic_question_generator import DynamicQuestionGenerator

# --- 메인 실행 함수 ---
# 프로그램의 주된 흐름을 담당하는 함수입니다.
def main(query: str = "폐렴"):
    """
    사용자 쿼리를 받아 동적 문제를 생성하고 결과를 터미널에 시각적으로 출력하는 메인 함수입니다.
    
    Args:
        query (str): 사용자가 입력한 의료 관련 검색어. 함수 호출 시 값을 주지 않으면 기본값으로 "폐렴"이 사용됩니다.
    """
    # rich 라이브러리의 Console 객체를 생성하여 터미널 출력을 관리합니다.
    console = Console()
    
    # 프로그램 시작을 알리는 헤더를 출력합니다. rich의 마크업을 사용하여 글자를 굵게(bold)하고 색상(cyan)을 적용합니다.
    console.print(f"[bold cyan]🤖 동적 의학 문제 생성: '{query}'[/bold cyan]")
    console.print("벡터DB 검색 → LLM 자율 분석 → 적응형 문제 생성")
    console.print("="*70)
    
    # `with console.status(...)`는 작업이 진행 중임을 사용자에게 시각적으로 알려주는 기능입니다 (스피너 애니메이션).
    # 이 블록이 끝나면 스피너는 자동으로 사라집니다.
    with console.status("[bold green]동적 문제 생성 중...", spinner="dots") as status:
        
        # 프로그램 실행 중 발생할 수 있는 모든 예외(오류)를 잡아내어 비정상적인 종료를 막기 위한 try-except 블록입니다.
        try:
            # 1. 동적 생성기 초기화
            status.update("[bold green]🔧 시스템 초기화 중...") # 상태 메시지를 현재 진행 상황에 맞게 업데이트합니다.
            generator = DynamicQuestionGenerator() # 문제 생성기 클래스의 인스턴스(객체)를 만듭니다.
            console.log("✅ 동적 생성 시스템 준비 완료") # console.log는 현재 시간과 함께 로그 메시지를 출력하여 디버깅에 유용합니다.
            
            # 2. 문제 생성 요청
            status.update(f"[bold yellow]🔍 '{query}' 분석 및 문제 생성 중...")
            # 생성기의 핵심 메서드를 호출하여 문제를 생성합니다. top_k는 검색할 관련 정보의 개수를 의미합니다.
            result = generator.generate_question_from_query(query, top_k=10)
            
            # 3. 결과 확인 및 오류 처리
            # 생성 결과가 담긴 딕셔너리에 'error'라는 키가 있다면, 오류가 발생한 것입니다.
            if "error" in result:
                console.print(f"[bold red]❌ 생성 실패:[/bold red] {result['error']}")
                
                # 사용자에게 문제 해결을 위한 팁을 친절하게 제공합니다.
                console.print("\n[yellow]💡 문제 해결 방법:[/yellow]")
                console.print("1. 다른 의료 용어로 시도해보세요")
                console.print("2. 더 구체적인 쿼리 사용 (예: '급성 심근경색 진단' → '심전도 ST상승')")
                console.print("3. 일반적인 질병명 사용 (예: '심장마비' → '심근경색')")
                return # 오류가 발생했으므로 여기서 함수 실행을 중단합니다.
            
            console.log("✅ 동적 문제 생성 완료")
            
            # 4. 관련 이미지 표시
            status.update("[bold yellow]🖼️ 관련 이미지 검색 및 표시 중...")
            # 생성된 문제와 관련된 이미지를 찾아 화면에 창으로 띄우는 함수를 호출합니다.
            image_displayed = display_related_image(console, result, query)
            if image_displayed:
                console.log("✅ 관련 이미지 표시 완료")
            else:
                console.log("⚠️ 이미지 표시 생략 (텍스트 전용 또는 실패)")
            
            # 5. 최종 결과 출력
            status.update("[bold green]📋 결과 포맷팅 중...")
            # 생성된 모든 정보를 정리하여 rich의 Panel을 사용해 예쁘게 출력하는 함수를 호출합니다.
            print_dynamic_question(console, result, query, image_displayed)
            console.log("✅ 모든 과정 완료!")
            
        # try 블록 안에서 어떤 종류의 예외(오류)라도 발생하면 이 코드가 실행됩니다.
        except Exception as e:
            # `e`는 발생한 오류 객체입니다. `str(e)`로 오류 메시지를 볼 수 있습니다.
            console.print(f"[bold red]❌ 시스템 오류:[/bold red] {str(e)}")
            console.print("\n[yellow]디버깅 정보:[/yellow]")
            console.print(f"쿼리: {query}")
            console.print(f"오류: {str(e)}")

def display_related_image(console: Console, result: dict, query: str) -> bool:
    """
    LLM이 선택한 타입의 이미지를 찾아서 matplotlib을 사용해 화면에 표시합니다.
    
    Args:
        console (Console): rich 콘솔 객체.
        result (dict): 문제 생성 결과가 담긴 딕셔너리.
        query (str): 원본 사용자 쿼리.
        
    Returns:
        bool: 이미지 표시 성공 여부를 반환합니다 (성공: True, 실패/생략: False).
    """
    
    # 1. LLM의 이미지 선택 결과 확인
    # .get() 메서드는 딕셔너리에서 키를 찾을 때, 키가 없으면 오류를 발생시키는 대신 기본값(여기서는 빈 dict나 list)을 반환합니다.
    # 이렇게 하면 코드가 더 안전해집니다.
    image_selection = result.get("image_selection", {})
    selected_images = result.get("selected_images", [])
    selected_type = image_selection.get("selected_image_type", "None")
    
    # 2. 이미지를 표시할지 여부를 결정하는 조건문들
    # LLM이 문제에 이미지가 필요 없다고 판단한 경우
    if selected_type == "None":
        console.print("[dim]📝 텍스트 전용 문제 (LLM이 이미지 불필요로 판단)[/dim]")
        return False
    
    # LLM이 이미지가 필요하다고 판단했지만, 검색 결과 이미지가 없는 경우
    if not selected_images:
        console.print(f"[yellow]⚠️ '{selected_type}' 이미지를 찾을 수 없습니다[/yellow]")
        return False
    
    # 이미지 표시 관련 라이브러리가 설치되지 않은 경우 (맨 위 try-except에서 설정된 플래그 확인)
    if not DISPLAY_AVAILABLE:
        console.print("[yellow]⚠️ 이미지 표시 라이브러리 없음 (PIL, matplotlib 필요)[/yellow]")
        return False
    
    # 3. LLM의 이미지 선택에 대한 상세 정보 출력
    korean_name = image_selection.get("korean_name", selected_type)
    reason = image_selection.get("reason", "선택 이유 없음")
    relevance_score = image_selection.get("relevance_score", 0)
    
    console.print(f"[blue]🤖 LLM 이미지 선택: {korean_name} ({selected_type})[/blue]")
    console.print(f"[blue]💡 선택 이유: {reason}[/blue]")
    console.print(f"[blue]📊 관련성 점수: {relevance_score}/10[/blue]")
    
    # 4. 표시할 이미지의 경로 설정
    # 검색된 이미지 목록 중 가장 관련성 높은 첫 번째 이미지를 사용합니다.
    first_image = selected_images[0]
    image_path = first_image.get("image_path", "") # 이미지 파일 이름
    
    if not image_path:
        console.print("[yellow]⚠️ 이미지 경로 정보 없음[/yellow]")
        return False
    
    # 5. 이미지 파일의 전체(절대) 경로 구성
    # Path 객체를 사용하면 OS에 상관없이 파일 경로를 안전하고 쉽게 조작할 수 있습니다.
    current_dir = Path(__file__).parent  # 이 파일(main.py)이 있는 폴더 (즉, 'generation/')
    project_root = current_dir.parent  # 'generation/' 폴더의 부모 폴더 (즉, 프로젝트 루트 폴더)
    # 최종적으로 이미지들이 저장된 폴더의 경로를 만듭니다.
    base_dir = project_root / "data" / "chestxray14" / "bbox_images"
    
    # 이미지 폴더가 실제로 존재하는지 확인하고, 없다면 다른 가능한 경로를 탐색합니다.
    # 이는 프로그램을 어디서 실행하는지에 따라 상대 경로가 달라질 수 있는 문제를 해결하기 위한 안정 장치입니다.
    if not base_dir.exists():
        alternative_paths = [
            project_root / "data" / "bbox_images",
            current_dir / "data" / "chestxray14" / "bbox_images",
            current_dir / ".." / "data" / "chestxray14" / "bbox_images",
        ]
        
        found_path = False
        for alt_path in alternative_paths:
            if alt_path.resolve().exists(): # .resolve()는 심볼릭 링크 등을 모두 해석한 실제 경로를 반환합니다.
                base_dir = alt_path.resolve()
                found_path = True
                break
        
        if not found_path:
            console.print(f"[red]❌ 이미지 디렉토리를 찾을 수 없습니다[/red]")
            # 어떤 경로들을 시도했는지 알려주어 사용자가 문제를 해결(디버깅)하는 데 도움을 줍니다.
            console.print(f"[yellow]찾은 경로들:[/yellow]")
            console.print(f"  기본: {project_root / 'data' / 'chestxray14' / 'bbox_images'}")
            for i, alt_path in enumerate(alternative_paths, 1):
                console.print(f"  대안{i}: {alt_path}")
            return False

    # 이미지 파일 이름과 이미지 폴더 경로를 합쳐 전체 파일 경로를 완성합니다.
    full_image_path = base_dir / image_path
    
    # 6. Matplotlib을 사용하여 이미지 표시
    try:
        # Pillow 라이브러리의 Image.open() 함수로 이미지 파일을 엽니다.
        img = Image.open(full_image_path)
        
        # matplotlib을 사용하여 이미지를 표시할 창(figure)을 설정합니다.
        plt.figure(figsize=(12, 10))  # 가로 12인치, 세로 10인치 크기의 창을 생성합니다.
        # cmap='gray'는 이미지가 흑백(Luminance) 모드일 때 회색조 색상맵을 사용하도록 합니다. 컬러 이미지면 이 옵션은 무시됩니다.
        plt.imshow(img, cmap='gray' if img.mode == 'L' else None)
        
        # 이미지 창의 제목을 설정합니다. 문제와 관련된 다양한 정보를 포함시켜 사용자 이해를 돕습니다.
        question_data = result.get("generated_question", {})
        topic_analysis = question_data.get("topic_analysis", {})
        estimated_topic = topic_analysis.get("estimated_topic", "Unknown")
        
        plt.title(f"흉부 X-ray: {korean_name} ({selected_type})\n"
                 f"원본 쿼리: {query} | AI 추정: {estimated_topic}\n"
                 f"LLM 선택 이유: {reason[:50]}...", # 이유가 너무 길면 50자까지만 표시
                 fontsize=14, fontweight='bold', pad=20)
        # 이미지 주변의 불필요한 축(눈금, 라벨)을 숨겨 깔끔하게 만듭니다.
        plt.axis('off')
        
        # 이미지 하단에 추가 정보(파일 이름, 관련성 점수 등)를 텍스트로 표시합니다.
        plt.figtext(0.5, 0.02, # (x, y) 좌표. (0,0)은 왼쪽 아래, (1,1)은 오른쪽 위.
                   f"파일: {image_path} | 관련성: {relevance_score}/10 | LLM 자율 선택", 
                   ha='center', fontsize=11, style='italic')
        
        plt.tight_layout() # 그림의 요소들(제목 등)이 겹치지 않도록 자동으로 레이아웃을 조정합니다.
        plt.show() # 지금까지 설정한 모든 내용으로 이미지 창을 화면에 실제로 띄웁니다.
        
        console.print(f"[green]🖼️ LLM 선택 이미지 표시 완료: {image_path}[/green]")
        return True # 이미지 표시 성공
        
    except FileNotFoundError:
        console.print(f"[red]❌ 이미지 파일을 찾을 수 없습니다: {full_image_path}[/red]")
        return False
    except Exception as e:
        console.print(f"[red]❌ 이미지 표시 중 오류 발생: {e}[/red]")
        return False

def print_dynamic_question(console: Console, result: dict, original_query: str, image_displayed: bool = False):
    """
    생성된 문제, 보기, 해설 등 모든 정보를 rich 라이브러리의 Panel을 사용해 보기 좋게 출력합니다.
    """
    
    # 1. 결과 딕셔너리에서 출력에 필요한 모든 정보들을 추출합니다.
    #    .get()을 사용하여 키가 없는 경우에도 오류 없이 안전하게 값을 가져옵니다.
    question_data = result.get("generated_question", {})
    search_context = result.get("search_context", {})
    image_selection = result.get("image_selection", {})
    
    # 문제 내용
    question_text = question_data.get("question", "문제 생성에 실패했습니다.")
    options = question_data.get("options", [])
    answer_idx = question_data.get("answer", -1) # -1은 유효한 인덱스가 아님을 의미
    explanation = question_data.get("explanation", "해설 정보가 없습니다.")
    
    # LLM이 분석한 정보
    topic_analysis = question_data.get("topic_analysis", {})
    estimated_topic = topic_analysis.get("estimated_topic", "주제 분석 실패")
    difficulty = topic_analysis.get("difficulty_level", "알 수 없음")
    clinical_relevance = topic_analysis.get("clinical_relevance", "알 수 없음")
    
    # LLM이 선택한 이미지 정보
    selected_type = image_selection.get("selected_image_type", "None")
    korean_name = image_selection.get("korean_name", "선택 없음")
    selection_reason = image_selection.get("reason", "선택 이유 없음")
    relevance_score = image_selection.get("relevance_score", 0)
    
    # 검색 품질 정보
    confidence = search_context.get("confidence", "unknown")
    
    # 2. 출력할 내용들을 rich의 Panel 객체로 만듭니다. Panel은 텍스트 주위에 테두리를 그려줍니다.
    
    # 2-1. 메타 정보 패널 (전체 요약 정보)
    # 이미지 표시 상태에 따라 다른 메시지를 동적으로 보여줍니다.
    if image_displayed and selected_type != "None":
        image_status = f"[green]✅ {korean_name} 이미지가 표시되었습니다.[/green]"
    elif selected_type != "None":
        image_status = f"[yellow]🖼️ {korean_name} 이미지가 선택되었습니다.[/yellow]"
    else:
        image_status = f"[dim]📝 텍스트 전용 문제입니다 (LLM 판단).[/dim]"
    
    meta_panel = Panel(
        # f-string을 사용하여 변수와 텍스트를 조합합니다. rich 마크업으로 스타일을 적용합니다.
        f"[bold]입력 쿼리:[/bold] {original_query}\n"
        f"[bold]LLM 추정 주제:[/bold] {estimated_topic}\n"
        f"[bold]문제 난이도:[/bold] {difficulty} | [bold]임상 관련성:[/bold] {clinical_relevance}\n"
        f"[bold]LLM 이미지 선택:[/bold] {image_status}\n"
        f"[bold]선택 관련성:[/bold] {relevance_score}/10 | [bold]벡터 검색 품질:[/bold] {confidence}",
        title="[bold cyan]🤖 AI 동적 분석 및 이미지 선택 결과[/bold cyan]",
        border_style="cyan",
        expand=False # 패널 너비가 터미널에 꽉 차지 않도록 함
    )
    
    # 2-2. LLM 이미지 선택 상세 정보 패널
    if selected_type != "None":
        selection_panel = Panel(
            f"[bold]선택된 이미지 타입:[/bold] {korean_name} ({selected_type})\n"
            f"[bold]선택 이유:[/bold] {selection_reason}",
            title="[bold blue]🖼️ LLM 이미지 선택 분석[/bold blue]",
            border_style="blue",
            expand=False
        )
    else:
        selection_panel = Panel(
            f"[bold]LLM 판단:[/bold] 이 문제는 이미지가 필요하지 않습니다.\n"
            f"[bold]이유:[/bold] {selection_reason}",
            title="[bold blue]📝 텍스트 전용 문제[/bold blue]",
            border_style="blue",
            expand=False
        )
    
    # 2-3. 생성된 문제 패널
    # 이미지가 표시되었는지 여부에 따라 문제 본문 하단에 추가 안내 메시지를 넣습니다.
    image_note = ""
    if image_displayed and selected_type != "None":
        image_note = f"\n\n[bold green]📷 위에 표시된 {korean_name} 흉부 X-ray 영상을 참고하여 문제를 풀어보세요.[/bold green]"
    elif selected_type != "None":
        image_note = f"\n\n[italic yellow]📷 (참고) 이 문제는 원래 {korean_name} 이미지와 함께 제시되어야 교육적 효과가 높습니다.[/italic yellow]"
    
    question_panel = Panel(
        f"{question_text}{image_note}",
        title="[bold yellow]📋 생성된 문제[/bold yellow]",
        border_style="yellow",
        expand=False
    )
    
    # 2-4. 보기 및 정답 패널
    if options and len(options) >= 5:
        options_str = ""
        # enumerate를 사용하여 리스트의 인덱스(i)와 값(option)을 동시에 가져옵니다.
        for i, option in enumerate(options):
            # 정답인 보기는 다른 색과 아이콘으로 강조하여 표시합니다.
            if i == answer_idx:
                options_str += f"  [bold green]▶ {i+1}. {option}[/bold green] ✅\n"
            else:
                options_str += f"  {i+1}. {option}\n"
    else:
        options_str = "[red]보기를 생성하는 데 실패했습니다.[/red]"
    
    options_panel = Panel(
        options_str,
        title="[bold blue]📝 보기 및 정답[/bold blue]",
        border_style="blue",
        expand=False
    )
    
    # 2-5. 해설 및 출처 패널
    source_info = question_data.get("source_utilization", "")
    full_explanation = explanation
    if source_info:
        full_explanation += f"\n\n**정보 출처 활용:**\n{source_info}"
    
    # Markdown 객체를 사용하면 마크다운 문법(**볼드**, *이탤릭* 등)을 터미널에 예쁘게 렌더링할 수 있습니다.
    explanation_panel = Panel(
        Markdown(full_explanation),
        title="[bold magenta]💡 해설 및 근거[/bold magenta]",
        border_style="magenta",
        expand=False
    )
    
    # 2-6. 시스템 성능 정보 패널 (개발자용 정보)
    generation_meta = result.get("generation_metadata", {})
    performance_info = (
        f"생성 방법: {generation_meta.get('method', 'unknown')}\n"
        f"AI 모델: {generation_meta.get('llm_model', 'unknown')}\n"
        f"벡터DB 활용: {'Yes' if generation_meta.get('vector_db_used', True) else 'No'}\n"
        f"이미지 선택 방식: {generation_meta.get('image_selection_method', 'unknown')}"
    )
    
    performance_panel = Panel(
        f"[dim]{performance_info}[/dim]",
        title="[dim]⚙️ 시스템 정보 (개발자용)[/dim]",
        border_style="dim",
        expand=False
    )
    
    # 3. 생성된 모든 패널들을 순서대로 터미널에 출력합니다.
    console.print("")
    console.print(meta_panel)
    if selected_type != "None": # 이미지가 선택된 경우에만 상세 정보 패널 출력
        console.print(selection_panel)
    console.print(question_panel)
    console.print(options_panel)
    console.print(explanation_panel)
    console.print(performance_panel)
    
    # 4. 최종적으로 생성된 문제의 품질에 대한 요약 메시지를 출력합니다.
    if relevance_score >= 8:
        console.print("\n[green]✅ 매우 적절한 이미지가 선택되었습니다. LLM이 문제의 핵심을 잘 파악했습니다.[/green]")
    elif relevance_score >= 6:
        console.print("\n[yellow]🟡 적절한 이미지가 선택되었습니다. 문제와 이미지가 잘 매칭됩니다.[/yellow]")
    
    if confidence == "low":
        console.print("\n[yellow]⚠️ 검색된 정보의 신뢰도가 낮습니다. 더 구체적인 의료 용어를 사용하면 결과가 개선될 수 있습니다.[/yellow]")
    elif confidence == "high":
        console.print("\n[green]✅ 신뢰도 높은 의료 정보를 바탕으로 문제가 생성되었습니다.[/green]")

# --- 테스트용 함수들 ---
# 아래 함수들은 개발 과정에서 시스템의 특정 기능들이 잘 동작하는지 확인하기 위해 만들어졌습니다.
# 실제 사용자에게는 보이지 않지만, 프로그램을 유지보수하는 데 중요한 역할을 합니다.

def test_image_functionality():
    """이미지 관련 기능(라이브러리, 경로 등)이 정상적으로 동작하는지 테스트합니다."""
    # (세부 내용은 생략)
    ...

def test_problematic_queries():
    """과거에 문제가 발생했던 특정 쿼리들을 다시 테스트하여 문제가 해결되었는지 확인합니다."""
    # (세부 내용은 생략)
    ...

def quick_comparison():
    """다른 방식의 시스템과 현재 동적 시스템의 성능을 간단히 비교합니다."""
    # (세부 내용은 생략)
    ...


# --- 스크립트 실행 진입점(Entry Point) ---
# `if __name__ == "__main__":` 은 이 파이썬 파일이 '직접' 실행될 때만 내부 코드를 실행하도록 하는, 파이썬의 표준적인 약속입니다.
# 만약 다른 파일에서 이 파일을 `import`해서 클래스나 함수만 가져다 쓸 경우에는 이 아래 부분은 실행되지 않습니다.
if __name__ == "__main__":
    console = Console()
    
    # 터미널에서 `python generation/main.py [인자]` 와 같이 실행했을 때, 그 인자 값을 확인합니다.
    # sys.argv는 터미널에서 전달된 인자들의 리스트입니다. sys.argv[0]은 스크립트 파일 이름 자체입니다.
    if len(sys.argv) > 1:
        command = sys.argv[1].lower() # 첫 번째 인자(명령어)를 소문자로 변환하여 일관성 있게 처리합니다.
        
        # 사용자가 입력한 명령어에 따라 각기 다른 테스트 함수를 실행합니다.
        if command == "test":
            test_problematic_queries()
        elif command == "image-test":
            test_image_functionality()
        elif command == "compare":
            quick_comparison()
        elif command == "--help" or command == "-h":
            # 프로그램 사용법을 안내하는 도움말을 출력합니다.
            console.print("[bold cyan]동적 의학 문제 생성 시스템 사용법[/bold cyan]")
            console.print("python generation/main.py [쿼리]           # 특정 주제로 동적 문제 생성")
            console.print("python generation/main.py check-path       # 이미지 경로 등 프로젝트 설정 확인") 
            console.print("python generation/main.py test             # 문제 발생 가능성이 있는 쿼리 테스트") 
            console.print("python generation/main.py image-test       # 이미지 표시 기능 테스트")
            console.print("python generation/main.py compare          # 시스템 비교 테스트")
        
        # 새로운 'check-path' 명령어. 경로 문제 발생 시 디버깅을 돕기 위함입니다.
        elif command == "check-path":
            # 프로젝트 구조와 관련된 경로 정보들을 확인하여 출력합니다.
            current_dir = Path(__file__).parent
            project_root = current_dir.parent
            
            console.print(f"[blue]📁 프로젝트 경로 확인[/blue]")
            console.print(f"현재 스크립트 경로: {Path(__file__).resolve()}")
            console.print(f"프로젝트 루트 경로: {project_root.resolve()}")
            
            # 예상되는 이미지 폴더 경로를 확인하고 실제 존재하는지 검사합니다.
            bbox_dir = project_root / 'data' / 'chestxray14' / 'bbox_images'
            console.print(f"예상 이미지 폴더: {bbox_dir.resolve()}")
            
            if bbox_dir.exists():
                image_count = len(list(bbox_dir.glob("*.png")))
                console.print(f"[green]✅ 이미지 폴더를 찾았습니다 ({image_count}개 이미지 존재).[/green]")
            else:
                console.print(f"[red]❌ 이미지 폴더를 찾을 수 없습니다. 경로를 확인해주세요.[/red]")
        else:
            # 위에서 정의한 특정 명령어가 아니라면, 일반적인 검색 '쿼리'로 간주하고 `main` 함수를 실행합니다.
            user_query = " ".join(sys.argv[1:]) # 여러 단어로 된 쿼리도 처리하기 위해 모든 인자를 합칩니다.
            main(user_query)
    else:
        # 터미널에서 아무 인자 없이 `python generation/main.py`로만 실행한 경우,
        # 사용자에게 직접 쿼리를 입력받는 대화형 모드로 작동합니다.
        try:
            user_query = input("\n의료 주제를 입력하세요 (예: 폐렴, 기흉) | 종료하려면 Ctrl+C: ").strip()
            if not user_query: # 만약 사용자가 아무것도 입력하지 않고 엔터만 쳤다면,
                user_query = "폐렴"  # 기본값 "폐렴"을 사용합니다.
                console.print(f"[dim]입력값이 없어 기본값 '{user_query}'으로 실행합니다...[/dim]")
            main(user_query)
        except (KeyboardInterrupt, EOFError):
            # 사용자가 Ctrl+C를 누르거나 Ctrl+D를 눌러 입력을 종료하면 프로그램을 깔끔하게 종료합니다.
            console.print("\n[bold]👋 프로그램을 종료합니다.[/bold]")