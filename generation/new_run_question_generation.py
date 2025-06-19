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
    """관련 이미지 검색 및 표시"""
    
    # 1. 검색 결과에서 이미지 정보 확인
    search_context = result.get("search_context", {})
    has_images = search_context.get("has_images", False)
    
    if not has_images:
        console.print("[dim]📝 텍스트 전용 문제 (이미지 없음)[/dim]")
        return False
    
    if not DISPLAY_AVAILABLE:
        console.print("[yellow]⚠️ 이미지 표시 라이브러리 없음 (PIL, matplotlib 필요)[/yellow]")
        return False
    
    # 2. 이미지 경로 찾기
    image_path = find_image_for_query(console, query, result)
    
    if not image_path:
        console.print("[yellow]⚠️ 관련 이미지를 찾을 수 없습니다[/yellow]")
        return False
    
    # 3. 이미지 표시
    try:
        # 이미지 로드
        img = Image.open(image_path)
        
        # matplotlib으로 표시
        plt.figure(figsize=(12, 10))
        plt.imshow(img, cmap='gray' if img.mode == 'L' else None)
        
        # 제목 및 정보 추가
        question_data = result.get("generated_question", {})
        topic_analysis = question_data.get("topic_analysis", {})
        estimated_topic = topic_analysis.get("estimated_topic", "Unknown")
        
        plt.title(f"흉부 X-ray: {estimated_topic} 관련 영상\n쿼리: {query}", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        
        # 파일 정보 하단에 추가
        plt.figtext(0.5, 0.02, f"파일: {Path(image_path).name} | AI 분석: {estimated_topic}", 
                   ha='center', fontsize=12, style='italic')
        
        plt.tight_layout()
        plt.show()
        
        console.print(f"[green]🖼️ 이미지 표시 완료: {Path(image_path).name}[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ 이미지 표시 실패: {e}[/red]")
        console.print(f"[blue]📷 이미지 경로: {image_path}[/blue]")
        return False

def find_image_for_query(console: Console, query: str, result: dict) -> str:
    """쿼리와 관련된 실제 이미지 파일 찾기"""
    
    # 0. 검색 엔진에서 이미 찾은 경로가 있으면 우선 사용
    search_context = result.get("search_context", {})
    primary_image_path = search_context.get("primary_image_path", "")
    
    if primary_image_path:
        base_dir = Path("/Users/solkim/Desktop/projects/medical/project/data/chestxray14/bbox_images")
        full_path = base_dir / primary_image_path
        
        if full_path.exists():
            console.print(f"[green]✅ 검색 엔진 결과 우선 사용: {primary_image_path}[/green]")
            return str(full_path)
        else:
            console.print(f"[yellow]⚠️ 검색 결과 파일 없음: {primary_image_path}[/yellow]")
            
    # 1. 이미지 베이스 디렉토리
    base_image_dir = Path("/Users/solkim/Desktop/projects/medical/project/data/chestxray14/bbox_images")
    
    if not base_image_dir.exists():
        console.print(f"[red]❌ 이미지 디렉토리 없음: {base_image_dir}[/red]")
        return None
    
    # 2. 질병명 기반 키워드 매핑
    disease_keywords = {
        # 한국어 → 영어 키워드
        "폐렴": ["pneumonia", "infiltrate", "consolidation"],
        "흉수": ["effusion", "pleural"],
        "기흉": ["pneumothorax"],
        "무기폐": ["atelectasis"],
        "침윤": ["infiltrate", "consolidation"],
        "경화": ["consolidation", "infiltrate"],
        "종괴": ["mass", "nodule"],
        "심장비대": ["cardiomegaly"],
        "결절": ["nodule"],
        # 영어 직접 매핑
        "pneumonia": ["pneumonia", "infiltrate"],
        "effusion": ["effusion", "pleural"],
        "pneumothorax": ["pneumothorax"],
        "atelectasis": ["atelectasis"],
        "infiltrate": ["infiltrate", "consolidation"],
        "consolidation": ["consolidation", "infiltrate"],
        "mass": ["mass", "nodule"],
        "cardiomegaly": ["cardiomegaly"],
        "nodule": ["nodule"]
    }
    
    # 3. LLM 추정 주제에서도 키워드 추출
    question_data = result.get("generated_question", {})
    topic_analysis = question_data.get("topic_analysis", {})
    estimated_topic = topic_analysis.get("estimated_topic", "").lower()
    
    # 4. 검색 키워드 수집
    search_keywords = []
    query_lower = query.lower()
    
    # 쿼리에서 키워드 찾기
    for key, keywords in disease_keywords.items():
        if key.lower() in query_lower:
            search_keywords.extend(keywords)
            console.print(f"[blue]🔍 쿼리 매칭: '{key}' → {keywords}[/blue]")
            break
    
    # LLM 추정 주제에서 키워드 찾기
    if not search_keywords and estimated_topic:
        for key, keywords in disease_keywords.items():
            if key in estimated_topic:
                search_keywords.extend(keywords)
                console.print(f"[blue]🔍 LLM 주제 매칭: '{key}' → {keywords}[/blue]")
                break
    
    # 기본 키워드
    if not search_keywords:
        search_keywords = ["pneumonia"]  # 기본값
        console.print(f"[yellow]⚠️ 키워드 매칭 실패, 기본값 사용: {search_keywords}[/yellow]")
    
    # 5. 이미지 파일 검색
    console.print(f"[blue]🔍 키워드로 이미지 검색: {search_keywords}[/blue]")
    
    for keyword in search_keywords:
        # 파일명에 키워드가 포함된 이미지 찾기
        matching_files = []
        for image_file in base_image_dir.glob("*.png"):
            # 파일명 또는 메타데이터에서 키워드 확인
            if keyword.lower() in image_file.name.lower():
                matching_files.append(image_file)
        
        if matching_files:
            selected_image = matching_files[0]  # 첫 번째 매칭 파일 사용
            console.print(f"[green]✅ 매칭 이미지 발견: {selected_image.name} (키워드: {keyword})[/green]")
            return str(selected_image)
    
    # 6. 대안: 아무 이미지나 사용 (데모용)
    all_images = list(base_image_dir.glob("*.png"))
    if all_images:
        selected_image = all_images[0]
        console.print(f"[yellow]📷 대체 이미지 사용: {selected_image.name}[/yellow]")
        return str(selected_image)
    
    return None

def print_dynamic_question(console: Console, result: dict, original_query: str, image_displayed: bool = False):
    """동적 생성된 질문을 출력 (이미지 상태 포함)"""
    
    question_data = result.get("generated_question", {})
    search_context = result.get("search_context", {})
    
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
    requires_image = topic_analysis.get("requires_image", False)
    
    # 검색 품질 정보
    confidence = search_context.get("confidence", "unknown")
    has_images = search_context.get("has_images", False)
    text_sources = search_context.get("text_sources", 0)
    image_sources = search_context.get("image_sources", 0)
    
    # 1. 메타 정보 패널 (이미지 상태 추가)
    image_status = ""
    if image_displayed:
        image_status = "[green]✅ 이미지 표시됨[/green]"
    elif has_images:
        image_status = "[yellow]🖼️ 이미지 데이터 있음[/yellow]"
    else:
        image_status = "[dim]📝 텍스트 전용[/dim]"
    
    meta_panel = Panel(
        f"[bold]입력 쿼리:[/bold] {original_query}\n"
        f"[bold]LLM 추정 주제:[/bold] {estimated_topic}\n"
        f"[bold]문제 난이도:[/bold] {difficulty} | [bold]임상 관련성:[/bold] {clinical_relevance}\n"
        f"[bold]이미지 상태:[/bold] {image_status}\n"
        f"[bold]벡터 검색 품질:[/bold] {confidence} | [bold]텍스트 소스:[/bold] {text_sources}개 | [bold]이미지 소스:[/bold] {image_sources}개",
        title="[bold cyan]🤖 AI 동적 분석 결과[/bold cyan]",
        border_style="cyan"
    )
    
    # 2. 생성된 문제 패널 (이미지 연동 메시지)
    image_note = ""
    if image_displayed:
        image_note = "\n\n[bold green]📷 위에 표시된 흉부 X-ray 영상을 참고하여 문제를 풀어보세요.[/bold green]"
    elif requires_image:
        image_note = "\n\n[italic yellow]📷 이 문제는 의료 이미지와 함께 제시되어야 합니다.[/italic yellow]"
    elif has_images:
        image_note = "\n\n[italic blue]🖼️ 관련 이미지 데이터가 검색되었습니다.[/italic blue]"
    
    question_panel = Panel(
        f"{question_text}{image_note}",
        title="[bold yellow]📋 생성된 문제[/bold yellow]",
        border_style="yellow"
    )
    
    # 3. 보기 및 정답 패널
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
    
    # 4. 해설 및 출처 패널
    source_info = question_data.get("source_utilization", "")
    full_explanation = explanation
    if source_info:
        full_explanation += f"\n\n**정보 출처 활용:**\n{source_info}"
    
    explanation_panel = Panel(
        Markdown(full_explanation),
        title="[bold magenta]💡 해설 및 근거[/bold magenta]",
        border_style="magenta"
    )
    
    # 5. 시스템 성능 패널
    generation_meta = result.get("generation_metadata", {})
    performance_info = (
        f"생성 방법: {generation_meta.get('method', 'dynamic_vector_search')}\n"
        f"AI 모델: {generation_meta.get('llm_model', 'gemini-1.5-pro')}\n"
        f"벡터DB 활용: {'Yes' if generation_meta.get('vector_db_used', True) else 'No'}\n"
        f"이미지 표시: {'Yes' if image_displayed else 'No'}\n"
        f"검색 → 분석 → 생성: 통합 파이프라인"
    )
    
    performance_panel = Panel(
        f"[dim]{performance_info}[/dim]",
        title="[dim]⚙️ 시스템 성능[/dim]",
        border_style="dim"
    )
    
    # 모든 패널 출력
    console.print("")
    console.print(meta_panel)
    console.print(question_panel)
    console.print(options_panel)
    console.print(explanation_panel)
    console.print(performance_panel)
    
    # 추가 이미지 관련 팁
    if has_images and not image_displayed:
        console.print("\n[yellow]💡 이미지 표시 개선 팁:[/yellow]")
        console.print("  - 이미지 파일 경로 확인")
        console.print("  - matplotlib 백엔드 설정 확인")
        console.print("  - GUI 환경에서 실행 권장")
    
    # 추가 분석 정보
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
    base_dir = Path("/Users/solkim/Desktop/projects/medical/project/data/chestxray14/bbox_images")
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
            console.print("python new_run_question_generation.py [쿼리]       # 동적 문제 생성")
            console.print("python new_run_question_generation.py test         # 문제 쿼리 재테스트") 
            console.print("python new_run_question_generation.py image-test   # 이미지 기능 테스트")
            console.print("python new_run_question_generation.py compare      # 시스템 비교")
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