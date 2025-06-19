# run_question_generation.py - 동적 시스템 + 이미지 표시
from pathlib import Path
import sys
import json
import os
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

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

# 기존 시스템 대신 동적 생성기만 사용
from generation.dynamic_question_generator import DynamicQuestionGenerator

def main(query: str = "폐렴"):
    console = Console()
    
    console.print(f"[bold cyan]🤖 동적 의학 문제 생성: '{query}'[/bold cyan]")
    console.print("벡터DB 검색 → LLM 자율 분석 → 적응형 문제 생성")
    console.print("="*70)
    
    with console.status("[bold green]동적 문제 생성 중...", spinner="dots") as status:
        
        try:
            # 동적 생성기 초기화
            status.update("[bold green]🔧 시스템 초기화 중...")
            generator = DynamicQuestionGenerator()
            console.log("✅ 동적 생성 시스템 준비 완료")
            
            # 통합 생성 과정 (벡터 검색 + LLM 분석 + 문제 생성)
            status.update(f"[bold yellow]🔍 '{query}' 분석 및 문제 생성 중...")
            result = generator.generate_question_from_query(query, top_k=10)  # 더 많은 컨텍스트
            
            if "error" in result:
                console.print(f"[bold red]❌ 생성 실패:[/bold red] {result['error']}")
                
                # 오류 시 대안 제안
                console.print("\n[yellow]💡 문제 해결 방법:[/yellow]")
                console.print("1. 다른 의료 용어로 시도해보세요")
                console.print("2. 더 구체적인 쿼리 사용 (예: '급성 심근경색 진단' → '심전도 ST상승')")
                console.print("3. 일반적인 질병명 사용 (예: '심장마비' → '심근경색')")
                return
            
            console.log("✅ 동적 문제 생성 완료")
            
            # 이미지 표시 (문제 출력 전에)
            status.update("[bold yellow]🖼️ 관련 이미지 검색 및 표시 중...")
            image_displayed = display_related_image(console, result, query)
            if image_displayed:
                console.log("✅ 관련 이미지 표시 완료")
            else:
                console.log("⚠️ 이미지 표시 생략 (텍스트 전용)")
            
            # 결과 출력
            status.update("[bold green]📋 결과 포맷팅 중...")
            print_dynamic_question(console, result, query, image_displayed)
            console.log("✅ 모든 과정 완료!")
            
        except Exception as e:
            console.print(f"[bold red]❌ 시스템 오류:[/bold red] {str(e)}")
            console.print("\n[yellow]디버깅 정보:[/yellow]")
            console.print(f"쿼리: {query}")
            console.print(f"오류: {str(e)}")

def display_related_image(console: Console, result: dict, query: str) -> bool:
    """LLM이 선택한 이미지 타입으로 이미지 표시"""
    
    # 1. LLM 이미지 선택 결과 확인
    image_selection = result.get("image_selection", {})
    selected_images = result.get("selected_images", [])
    selected_type = image_selection.get("selected_image_type", "None")
    
    if selected_type == "None":
        console.print("[dim]📝 텍스트 전용 문제 (LLM이 이미지 불필요로 판단)[/dim]")
        return False
    
    if not selected_images:
        console.print(f"[yellow]⚠️ '{selected_type}' 이미지를 찾을 수 없습니다[/yellow]")
        return False
    
    if not DISPLAY_AVAILABLE:
        console.print("[yellow]⚠️ 이미지 표시 라이브러리 없음 (PIL, matplotlib 필요)[/yellow]")
        return False
    
    # 2. LLM 선택 정보 출력
    korean_name = image_selection.get("korean_name", selected_type)
    reason = image_selection.get("reason", "선택 이유 없음")
    relevance_score = image_selection.get("relevance_score", 0)
    
    console.print(f"[blue]🤖 LLM 이미지 선택: {korean_name} ({selected_type})[/blue]")
    console.print(f"[blue]💡 선택 이유: {reason}[/blue]")
    console.print(f"[blue]📊 관련성 점수: {relevance_score}/10[/blue]")
    
    # 3. 첫 번째 이미지 표시
    first_image = selected_images[0]
    image_path = first_image.get("image_path", "")
    
    if not image_path:
        console.print("[yellow]⚠️ 이미지 경로 정보 없음[/yellow]")
        return False
    
    # 4. 절대 경로 구성 및 이미지 표시
    current_dir = Path(__file__).parent  # main.py가 있는 폴더
    project_root = current_dir.parent if current_dir.name == "generation" else current_dir
    base_dir = project_root / "data" / "chestxray14" / "bbox_images"
    
    # 경로 존재 확인 및 대안 경로 시도
    if not base_dir.exists():
        # 다른 가능한 경로들 시도
        alternative_paths = [
            project_root / "data" / "bbox_images",  # 간소화된 경로
            current_dir / "data" / "chestxray14" / "bbox_images",  # generation 폴더 기준
            current_dir / ".." / "data" / "chestxray14" / "bbox_images",  # 상위 폴더 기준
        ]
        
        for alt_path in alternative_paths:
            if alt_path.exists():
                base_dir = alt_path
                break
        else:
            console.print(f"[red]❌ 이미지 디렉토리를 찾을 수 없습니다[/red]")
            console.print(f"[yellow]찾은 경로들:[/yellow]")
            console.print(f"  기본: {project_root / 'data' / 'chestxray14' / 'bbox_images'}")
            for i, alt_path in enumerate(alternative_paths, 1):
                console.print(f"  대안{i}: {alt_path}")
            return False

    full_image_path = base_dir / image_path
    
    try:
        # 이미지 로드 및 표시
        img = Image.open(full_image_path)
        
        plt.figure(figsize=(12, 10))
        plt.imshow(img, cmap='gray' if img.mode == 'L' else None)
        
        # 제목에 LLM 선택 정보 포함
        question_data = result.get("generated_question", {})
        topic_analysis = question_data.get("topic_analysis", {})
        estimated_topic = topic_analysis.get("estimated_topic", "Unknown")
        
        plt.title(f"흉부 X-ray: {korean_name} ({selected_type})\n"
                 f"원본 쿼리: {query} | AI 추정: {estimated_topic}\n"
                 f"LLM 선택 이유: {reason[:50]}...", 
                 fontsize=14, fontweight='bold', pad=20)
        plt.axis('off')
        
        # 하단 정보
        plt.figtext(0.5, 0.02, 
                   f"파일: {image_path} | 관련성: {relevance_score}/10 | LLM 자율 선택", 
                   ha='center', fontsize=11, style='italic')
        
        plt.tight_layout()
        plt.show()
        
        console.print(f"[green]🖼️ LLM 선택 이미지 표시 완료: {image_path}[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ 이미지 표시 실패: {e}[/red]")
        return False

def print_dynamic_question(console: Console, result: dict, original_query: str, image_displayed: bool = False):
    """LLM 이미지 선택 포함 문제 출력"""
    
    question_data = result.get("generated_question", {})
    search_context = result.get("search_context", {})
    image_selection = result.get("image_selection", {})
    
    # 기본 정보 추출
    question_text = question_data.get("question", "문제 생성 실패")
    options = question_data.get("options", [])
    answer_idx = question_data.get("answer", -1)
    explanation = question_data.get("explanation", "해설 없음")
    
    # LLM 분석 정보
    topic_analysis = question_data.get("topic_analysis", {})
    estimated_topic = topic_analysis.get("estimated_topic", "주제 분석 실패")
    difficulty = topic_analysis.get("difficulty_level", "중급")
    clinical_relevance = topic_analysis.get("clinical_relevance", "medium")
    
    # LLM 이미지 선택 정보
    selected_type = image_selection.get("selected_image_type", "None")
    korean_name = image_selection.get("korean_name", "선택 없음")
    selection_reason = image_selection.get("reason", "선택 이유 없음")
    relevance_score = image_selection.get("relevance_score", 0)
    
    # 검색 품질 정보
    confidence = search_context.get("confidence", "unknown")
    text_sources = search_context.get("text_sources", 0)
    image_sources = search_context.get("image_sources", 0)
    
    # 1. 메타 정보 패널 (LLM 이미지 선택 정보 포함)
    image_status = ""
    if image_displayed and selected_type != "None":
        image_status = f"[green]✅ {korean_name} 표시됨[/green]"
    elif selected_type != "None":
        image_status = f"[yellow]🖼️ {korean_name} 선택됨[/yellow]"
    else:
        image_status = f"[dim]📝 텍스트 전용 (LLM 판단)[/dim]"
    
    meta_panel = Panel(
        f"[bold]입력 쿼리:[/bold] {original_query}\n"
        f"[bold]LLM 추정 주제:[/bold] {estimated_topic}\n"
        f"[bold]문제 난이도:[/bold] {difficulty} | [bold]임상 관련성:[/bold] {clinical_relevance}\n"
        f"[bold]LLM 이미지 선택:[/bold] {image_status}\n"
        f"[bold]선택 관련성:[/bold] {relevance_score}/10 | [bold]벡터 검색 품질:[/bold] {confidence}",
        title="[bold cyan]🤖 AI 동적 분석 및 이미지 선택 결과[/bold cyan]",
        border_style="cyan"
    )
    
    # 2. LLM 이미지 선택 상세 정보 패널
    if selected_type != "None":
        selection_panel = Panel(
            f"[bold]선택된 이미지 타입:[/bold] {korean_name} ({selected_type})\n"
            f"[bold]선택 이유:[/bold] {selection_reason}\n"
            f"[bold]관련성 점수:[/bold] {relevance_score}/10\n"
            f"[bold]대안 타입:[/bold] {', '.join(image_selection.get('alternative_types', []))}",
            title="[bold blue]🖼️ LLM 이미지 선택 분석[/bold blue]",
            border_style="blue"
        )
    else:
        selection_panel = Panel(
            f"[bold]LLM 판단:[/bold] 이 문제는 이미지가 필요하지 않음\n"
            f"[bold]이유:[/bold] {selection_reason}",
            title="[bold blue]📝 텍스트 전용 문제[/bold blue]",
            border_style="blue"
        )
    
    # 3. 생성된 문제 패널 (이미지 연동 메시지)
    image_note = ""
    if image_displayed and selected_type != "None":
        image_note = f"\n\n[bold green]📷 위에 표시된 {korean_name} 흉부 X-ray 영상을 참고하여 문제를 풀어보세요.[/bold green]"
    elif selected_type != "None":
        image_note = f"\n\n[italic yellow]📷 이 문제는 {korean_name} 이미지와 함께 제시되어야 합니다.[/italic yellow]"
    
    question_panel = Panel(
        f"{question_text}{image_note}",
        title="[bold yellow]📋 생성된 문제[/bold yellow]",
        border_style="yellow"
    )
    
    # 4. 보기 및 정답 패널 (기존과 동일)
    if options and len(options) >= 5:
        options_str = ""
        for i, option in enumerate(options):
            if i == answer_idx:
                options_str += f"  [bold green]▶ {i+1}. {option}[/bold green] ✅\n"
            else:
                options_str += f"  {i+1}. {option}\n"
    else:
        options_str = "[red]보기 생성 실패[/red]"
    
    options_panel = Panel(
        options_str,
        title="[bold blue]📝 보기 및 정답[/bold blue]",
        border_style="blue"
    )
    
    # 5. 해설 및 출처 패널 (기존과 동일)
    source_info = question_data.get("source_utilization", "")
    full_explanation = explanation
    if source_info:
        full_explanation += f"\n\n**정보 출처 활용:**\n{source_info}"
    
    explanation_panel = Panel(
        Markdown(full_explanation),
        title="[bold magenta]💡 해설 및 근거[/bold magenta]",
        border_style="magenta"
    )
    
    # 6. 시스템 성능 패널 (업데이트)
    generation_meta = result.get("generation_metadata", {})
    performance_info = (
        f"생성 방법: {generation_meta.get('method', 'dynamic_vector_search_with_llm_image_selection')}\n"
        f"AI 모델: {generation_meta.get('llm_model', 'gemini-1.5-pro')}\n"
        f"벡터DB 활용: {'Yes' if generation_meta.get('vector_db_used', True) else 'No'}\n"
        f"이미지 선택: LLM 자율 판단 ({selected_type})\n"
        f"워크플로우: 검색 → 문제생성 → 이미지선택 → 표시"
    )
    
    performance_panel = Panel(
        f"[dim]{performance_info}[/dim]",
        title="[dim]⚙️ 시스템 성능[/dim]",
        border_style="dim"
    )
    
    # 모든 패널 출력
    console.print("")
    console.print(meta_panel)
    console.print(selection_panel)
    console.print(question_panel)
    console.print(options_panel)
    console.print(explanation_panel)
    console.print(performance_panel)
    
    # 추가 성능 분석
    if relevance_score >= 8:
        console.print("\n[green]✅ 매우 적절한 이미지 선택! LLM이 높은 관련성을 보여줍니다.[/green]")
    elif relevance_score >= 6:
        console.print("\n[yellow]🟡 적절한 이미지 선택. 문제와 이미지가 잘 매칭됩니다.[/yellow]")
    elif relevance_score >= 4:
        console.print("\n[yellow]⚠️ 보통 수준의 이미지 매칭. 개선 여지가 있습니다.[/yellow]")
    
    if confidence == "low":
        console.print("\n[yellow]⚠️ 검색 품질이 낮습니다. 더 구체적인 의료 용어를 사용해보세요.[/yellow]")
    elif confidence == "high":
        console.print("\n[green]✅ 높은 품질의 의료 정보를 바탕으로 생성되었습니다.[/green]")

def test_image_functionality():
    """이미지 기능 테스트"""
    console = Console()
    
    console.print("[bold cyan]🖼️ 이미지 기능 테스트[/bold cyan]")
    console.print("="*50)
    
    # 라이브러리 확인
    if DISPLAY_AVAILABLE:
        console.print("[green]✅ 이미지 라이브러리 사용 가능[/green]")
    else:
        console.print("[red]❌ 이미지 라이브러리 없음 (pip install pillow matplotlib)[/red]")
        return
    
    # 이미지 디렉토리 확인
    # 수정:
    current_dir = Path(__file__).parent
    project_root = current_dir.parent if current_dir.name == "generation" else current_dir
    base_dir = project_root / "data" / "chestxray14" / "bbox_images"

    if base_dir.exists():
        image_count = len(list(base_dir.glob("*.png")))
        console.print(f"[green]✅ 이미지 디렉토리 발견: {image_count}개 파일[/green]")
    else:
        console.print(f"[red]❌ 이미지 디렉토리 없음: {base_dir}[/red]")
        return
    
    # 8개 질병 테스트
    test_diseases = ["폐렴", "흉수", "기흉", "무기폐"]
    
    for disease in test_diseases:
        console.print(f"\n[yellow]테스트: {disease}[/yellow]")
        try:
            generator = DynamicQuestionGenerator()
            result = generator.generate_question_from_query(disease, top_k=5)
            
            if "error" not in result:
                has_images = result.get("search_context", {}).get("has_images", False)
                console.print(f"  이미지 데이터: {'Yes' if has_images else 'No'}")
                
                if has_images:
                    # 이미지 찾기 테스트
                    image_path = find_image_for_query(console, disease, result)
                    if image_path:
                        console.print(f"  [green]✅ 이미지 파일: {Path(image_path).name}[/green]")
                    else:
                        console.print(f"  [red]❌ 이미지 파일 찾기 실패[/red]")
            else:
                console.print(f"  [red]❌ 문제 생성 실패[/red]")
                
        except Exception as e:
            console.print(f"  [red]❌ 오류: {e}[/red]")

def test_problematic_queries():
    """문제가 있던 쿼리들 테스트"""
    console = Console()
    
    console.print("[bold red]🧪 문제 쿼리 재테스트[/bold red]")
    console.print("="*50)
    
    problematic_queries = [
        "급성 심근경색",
        "심장마비",
        "관상동맥질환", 
        "협심증",
        "부정맥"
    ]
    
    for query in problematic_queries:
        console.print(f"\n[bold yellow]테스트: {query}[/bold yellow]")
        
        try:
            generator = DynamicQuestionGenerator()
            result = generator.generate_question_from_query(query, top_k=8)
            
            if "error" not in result:
                question_data = result["generated_question"]
                topic_analysis = question_data.get("topic_analysis", {})
                estimated_topic = topic_analysis.get("estimated_topic", "Unknown")
                
                console.print(f"  ✅ 성공: {estimated_topic}")
            else:
                console.print(f"  ❌ 실패: {result['error']}")
                
        except Exception as e:
            console.print(f"  ❌ 오류: {e}")

def quick_comparison():
    """기존 시스템 vs 동적 시스템 비교"""
    console = Console()
    
    console.print("[bold cyan]⚖️ 시스템 비교: 기존 vs 동적[/bold cyan]")
    console.print("="*60)
    
    test_cases = [
        ("폐렴", "기존 시스템 강점 영역"),
        ("급성 심근경색", "기존 시스템 약점 영역"),  
        ("뇌졸중", "완전히 새로운 영역"),
        ("당뇨병", "내과 일반 영역")
    ]
    
    for query, description in test_cases:
        console.print(f"\n[bold]{query}[/bold] - {description}")
        
        # 동적 시스템 테스트
        try:
            generator = DynamicQuestionGenerator()
            result = generator.generate_question_from_query(query)
            
            if "error" not in result:
                topic_analysis = result["generated_question"].get("topic_analysis", {})
                search_context = result.get("search_context", {})
                
                console.print(f"  [green]✅ 동적 시스템:[/green] {topic_analysis.get('estimated_topic', 'Unknown')}")
                console.print(f"     검색 품질: {search_context.get('confidence', 'unknown')}")
            else:
                console.print(f"  [red]❌ 동적 시스템:[/red] {result['error']}")
                
        except Exception as e:
            console.print(f"  [red]❌ 동적 시스템 오류:[/red] {e}")

if __name__ == "__main__":
    console = Console()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            test_problematic_queries()
        elif command == "image-test":
            test_image_functionality()
        elif command == "compare":
            quick_comparison()
        elif command == "--help" or command == "-h":
            console.print("[bold cyan]동적 의학 문제 생성 시스템[/bold cyan]")
            # 수정:
            console.print("python main.py [쿼리]           # 동적 문제 생성")
            console.print("python main.py check-path       # 프로젝트 경로 확인") 
            console.print("python main.py test             # 문제 쿼리 재테스트") 
            console.print("python main.py image-test       # 이미지 기능 테스트")
            console.print("python main.py compare          # 시스템 비교")
        
        # 3. 새로운 check-path 명령어 추가 (if문에 추가):
        elif command == "check-path":
            # 프로젝트 구조 확인
            current_dir = Path(__file__).parent
            project_root = current_dir.parent if current_dir.name == "generation" else current_dir
            
            console.print(f"[blue]📁 프로젝트 구조 확인[/blue]")
            console.print(f"현재 스크립트: {Path(__file__)}")
            console.print(f"현재 폴더: {current_dir}")
            console.print(f"프로젝트 루트: {project_root}")
            console.print(f"예상 이미지 경로: {project_root / 'data' / 'chestxray14' / 'bbox_images'}")
            
            # 데이터 폴더 존재 확인
            data_dir = project_root / "data"
            if data_dir.exists():
                console.print(f"[green]✅ data 폴더 존재[/green]")
                
                chestxray_dir = data_dir / "chestxray14"
                if chestxray_dir.exists():
                    console.print(f"[green]✅ chestxray14 폴더 존재[/green]")
                    
                    bbox_dir = chestxray_dir / "bbox_images"
                    if bbox_dir.exists():
                        image_count = len(list(bbox_dir.glob("*.png")))
                        console.print(f"[green]✅ bbox_images 폴더 존재 ({image_count}개 이미지)[/green]")
                    else:
                        console.print(f"[red]❌ bbox_images 폴더 없음[/red]")
                else:
                    console.print(f"[red]❌ chestxray14 폴더 없음[/red]")
            else:
                console.print(f"[red]❌ data 폴더 없음[/red]")
        else:
            # 일반 쿼리
            user_query = sys.argv[1]
            main(user_query)
    else:
        # 기본값: 대화형
        user_query = input("\n의료 주제를 입력하세요 (예: 폐렴, 기흉): ").strip()
        if not user_query:
            user_query = "폐렴"  # 기본값
        main(user_query)