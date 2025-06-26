import os
import json
import pandas as pd
from pathlib import Path
import re
from collections import Counter

# ▶ 추가 -----------------------------
import matplotlib               # 이 줄이 반드시 pyplot import보다 먼저 와야 합니다.
matplotlib.use("Agg")           # Tk 없이 작동하는 백엔드 지정
# -----------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
from konlpy.tag import Okt
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# preprocess.py에서 전처리 함수 임포트
from simple_preprocess import preprocess_question_text

# --- 한글 폰트 설정 ---
# 시스템에 맞는 한글 폰트 사용
try:
    import platform
    if platform.system() == 'Darwin':  # macOS
        plt.rc('font', family='AppleGothic')
    elif platform.system() == 'Windows':  # Windows
        plt.rc('font', family='Malgun Gothic')
    else: # Linux
        plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지
    print("✅ 한글 폰트 설정 완료")
except Exception as e:
    print(f"⚠️ 한글 폰트 설정 실패: {e}")


# --- 데이터 로딩 ---
def load_labeled_data(base_path: Path) -> pd.DataFrame:
    """지정된 경로에서 라벨링된 JSON 데이터를 로드하여 DataFrame으로 반환"""
    data = []
    domain_map = {
        'TL_내과': '내과',
        'TL_소아청소년과': '소아청소년과',
        'TL_산부인과': '산부인과',
        'TL_응급의학과': '응급의학과',
        'TL_기타': '기타',
        'TL_마취통증의학과': '마취통증의학과',
        'TL_방사선종양학과': '방사선종양학과',
        'TL_병리과': '병리과',
        'TL_비뇨의학과': '비뇨의학과',
        'TL_신경과신경외과': '신경과/신경외과',
        'TL_안과': '안과',
        'TL_예방의학': '예방의학',
        'TL_외과': '외과',
        'TL_의료법규': '의료법규',     
        'TL_이비인후과': '이비인후과', 
        'TL_정신건강의학과': '정신건강의학과', 
        'TL_피부과': '피부과'    
    }
    
    for dir_name, domain_name in domain_map.items():
        domain_path = base_path / dir_name
        if not domain_path.is_dir():
            print(f"⚠️ 디렉토리를 찾을 수 없음: {domain_path}")
            continue
            
        for file_path in domain_path.glob('*.json'):
            # UTF-8 BOM(Byte Order Mark) 문제가 있을 수 있으므로 'utf-8-sig' 사용
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                try:
                    content = json.load(f)
                    # 'question' 필드가 있고 'q_type'이 1인 경우에만 데이터 추가
                    if 'question' in content and content['question'] and content.get('q_type') == 1:
                        data.append({
                            'domain': domain_name,
                            'text': content['question']
                        })
                except json.JSONDecodeError:
                    print(f"⚠️ JSON 파싱 오류: {file_path}")

    if not data:
        raise ValueError("데이터를 로드하지 못했습니다. 경로 또는 'q_type == 1' 조건을 확인하세요.")
        
    return pd.DataFrame(data)

# --- 텍스트 통계 계산 ---
def calculate_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    """텍스트의 기본 통계(글자, 단어, 문장 수)를 계산"""
    df['char_count'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['sentence_count'] = df['text'].apply(lambda x: len(re.split(r'[.?!]', x)))
    return df

def calculate_vocabulary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """도메인별 어휘 통계(고유어휘수, TTR) 계산"""
    okt = Okt()
    
    domain_stats = []
    for domain, group in df.groupby('domain'):
        # 도메인 전체 텍스트를 하나로 합침
        full_text = ' '.join(group['text'])
        
        # 형태소 분석기를 사용하여 명사만 추출 (토큰화)
        tokens = okt.nouns(full_text)
        
        if not tokens:
            continue

        total_tokens = len(tokens)
        unique_tokens = len(set(tokens))
        ttr = unique_tokens / total_tokens if total_tokens > 0 else 0
        
        domain_stats.append({
            'domain': domain,
            'total_tokens (nouns)': total_tokens,
            'unique_tokens (nouns)': unique_tokens,
            'TTR': ttr
        })
        
    return pd.DataFrame(domain_stats)

# --- 통계 테이블 생성 및 저장 ---
def create_descriptive_stats_tables(df: pd.DataFrame, output_path: Path):
    """도메인별 기술 통계량을 계산하여 테이블로 저장"""
    # 1. 전체 통계 테이블
    stats_overall = df.describe().T
    stats_overall.to_csv(output_path / 'overall_stats.csv')
    
    # 2. 도메인별 통계 테이블
    domain_stats = df.groupby('domain').agg({
        'char_count': ['count', 'mean', 'std', 'min', 'max', 'median'],
        'word_count': ['mean', 'std', 'min', 'max', 'median'],
        'sentence_count': ['mean', 'std', 'min', 'max', 'median']
    })
    
    # 다중 인덱스를 단일 레벨 컬럼으로 변환하기 위해 처리
    domain_stats.columns = [f'{col[0]}_{col[1]}' for col in domain_stats.columns]
    domain_stats.reset_index(inplace=True)
    domain_stats.to_csv(output_path / 'domain_stats.csv', index=False, encoding='utf-8-sig')
    
    # 3. HTML 테이블로도 저장 (웹 브라우저에서 직접 확인 가능)
    with open(output_path / 'domain_stats.html', 'w', encoding='utf-8') as f:
        f.write('<html><head><style>')
        f.write('table {border-collapse: collapse; width: 100%; margin-bottom: 20px;}')
        f.write('th, td {border: 1px solid #ddd; padding: 8px; text-align: left;}')
        f.write('th {background-color: #f2f2f2;}')
        f.write('tr:nth-child(even) {background-color: #f9f9f9;}')
        f.write('tr:hover {background-color: #f1f1f1;}')
        f.write('</style></head><body>')
        f.write('<h1>도메인별 텍스트 통계</h1>')
        f.write(domain_stats.to_html(index=False, float_format='%.2f'))
        f.write('</body></html>')
    
    return domain_stats

# --- 통계 분석 ---
def perform_statistical_analysis(df: pd.DataFrame, output_path: Path):
    """도메인 간 통계적 차이를 분석하는 ANOVA 테스트와 사후 검정 수행"""
    results = []
    features = ['char_count', 'word_count', 'sentence_count']
    
    # 도메인별 샘플이 충분히 있는지 확인
    domain_counts = df['domain'].value_counts()
    valid_domains = domain_counts[domain_counts >= 30].index.tolist()  # 최소 30개 샘플 필요
    
    if len(valid_domains) < 2:
        print("⚠️ 통계 분석을 위한 충분한 도메인(샘플 수 30 이상)이 없습니다.")
        return
    
    filtered_df = df[df['domain'].isin(valid_domains)].copy()
    
    with open(output_path / 'statistical_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("=== 의학 도메인별 텍스트 특성 통계적 분석 ===\n\n")
        
        for feature in features:
            f.write(f"\n\n== {feature} 분석 ==\n")
            
            # 1. ANOVA 분석
            formula = f"{feature} ~ domain"
            model = ols(formula, data=filtered_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            f.write("\nANOVA 분석 결과:\n")
            f.write(anova_table.to_string())
            
            # p-값 확인 및 결과 해석
            p_value = anova_table.loc['domain', 'PR(>F)']
            if p_value < 0.05:
                f.write(f"\n\n-> 결과: 도메인에 따라 {feature}에 통계적으로 유의한 차이가 있습니다. (p={p_value:.4f})\n")
                
                # 사후 검정 (Tukey's test)
                # 한글 도메인명이 있으므로 각 도메인에 숫자 ID 할당하여 처리
                domain_mapping = {domain: i for i, domain in enumerate(filtered_df['domain'].unique())}
                filtered_df['domain_id'] = filtered_df['domain'].map(domain_mapping)
                
                try:
                    from statsmodels.stats.multicomp import pairwise_tukeyhsd
                    tukey = pairwise_tukeyhsd(endog=filtered_df[feature], 
                                             groups=filtered_df['domain_id'],
                                             alpha=0.05)
                    
                    # 결과 출력
                    f.write("\n사후 검정 결과 (Tukey's HSD):\n")
                    f.write(str(tukey))
                    
                    # 유의한 쌍만 별도로 정리
                    significant_pairs = []
                    for i, row in enumerate(tukey.summary().data[1:]):
                        if row[-1] is True:  # reject는 유의한 차이 존재를 의미
                            group1_id, group2_id = row[0], row[1]
                            group1 = list(domain_mapping.keys())[list(domain_mapping.values()).index(int(group1_id))]
                            group2 = list(domain_mapping.keys())[list(domain_mapping.values()).index(int(group2_id))]
                            mean_diff = row[2]
                            significant_pairs.append((group1, group2, mean_diff))
                    
                    if significant_pairs:
                        f.write("\n\n통계적으로 유의한 차이가 있는 도메인 쌍:\n")
                        for group1, group2, diff in significant_pairs:
                            f.write(f"- {group1} vs {group2}: 평균 차이 = {diff:.2f}\n")
                    else:
                        f.write("\n사후 검정 결과 유의한 차이가 있는 쌍을 찾지 못했습니다.\n")
                        
                except Exception as e:
                    f.write(f"\n사후 검정 중 오류 발생: {str(e)}\n")
            else:
                f.write(f"\n\n-> 결과: 도메인에 따라 {feature}에 통계적으로 유의한 차이가 없습니다. (p={p_value:.4f})\n")
    
    print(f"✅ 통계 분석 결과가 '{output_path / 'statistical_analysis.txt'}'에 저장되었습니다.")

# --- 시각화 ---
def plot_distribution(df: pd.DataFrame, column: str, title: str, output_path: Path):
    """지정된 컬럼의 도메인별 분포를 시각화"""
    plt.figure(figsize=(14, 8))
    sns.histplot(data=df, x=column, hue='domain', kde=True, element="step", common_norm=False)
    plt.title(f'도메인별 {title} 분포', fontsize=16)
    plt.xlabel(title, fontsize=12)
    plt.ylabel('빈도', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path / f'{column}_distribution.png')
    plt.close()  # plt.show() 대신 plt.close() 사용하여 메모리 관리

def plot_boxplots(df: pd.DataFrame, output_path: Path):
    """도메인별 주요 지표에 대한 boxplot 생성"""
    features = ['char_count', 'word_count', 'sentence_count']
    
    for feature in features:
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='domain', y=feature, data=df)
        plt.title(f'도메인별 {feature} 분포 비교', fontsize=16)
        plt.xlabel('도메인', fontsize=12)
        plt.ylabel(feature, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_path / f'{feature}_boxplot.png')
        plt.close()

def plot_vocabulary_comparison(stats_df: pd.DataFrame, output_path: Path):
    """도메인별 어휘 통계 비교 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 고유 어휘 수 비교
    sns.barplot(ax=axes[0], data=stats_df, x='domain', y='unique_tokens (nouns)', palette='viridis')
    axes[0].set_title('도메인별 고유 어휘 수 (명사 기준)', fontsize=15)
    axes[0].set_xlabel('도메인', fontsize=12)
    axes[0].set_ylabel('고유 어휘 수', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)

    # TTR 비교
    sns.barplot(ax=axes[1], data=stats_df, x='domain', y='TTR', palette='plasma')
    axes[1].set_title('도메인별 어휘 다양성 (TTR)', fontsize=15)
    axes[1].set_xlabel('도메인', fontsize=12)
    axes[1].set_ylabel('TTR (Type-Token Ratio)', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)

    plt.suptitle('도메인별 어휘 통계 비교 분석', fontsize=20, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'vocabulary_comparison.png')
    plt.close()

# --- 메인 실행 로직 ---
def main():
    """데이터 로드, 분석, 시각화 실행"""
    # 경로 설정
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    data_path = project_root / 'data'/'AIHUB'/'qa'/'02.라벨링데이터'
    output_path = script_dir / 'analysis_results_4'  # 새 결과 폴더
    output_path.mkdir(exist_ok=True)
    
    print(f"데이터 경로: {data_path}")
    print(f"결과 저장 경로: {output_path}")

    # 1. 데이터 로드
    print("\n[1/7] 라벨링 데이터 로드 중...")
    try:
        df = load_labeled_data(data_path)
        print(f"✅ 총 {len(df)}개의 문서 로드 완료 (q_type == 1 필터 적용)")
        print("도메인별 문서 수:")
        print(df['domain'].value_counts())
    except ValueError as e:
        print(f"❌ 오류: {e}")
        return

    # 2. 텍스트 전처리
    print("\n[2/7] 텍스트 전처리 적용 중...")
    df['text'] = df['text'].apply(preprocess_question_text)
    print("✅ 전처리 완료.")

    # 3. 기본 통계 계산
    print("\n[3/7] 기본 텍스트 통계 계산 중...")
    df = calculate_text_stats(df)
    print("기본 통계 계산 완료. DataFrame 샘플:")
    print(df.head())

    # 4. 도메인별 통계 테이블 생성 (신규)
    print("\n[4/7] 도메인별 통계 테이블 생성 중...")
    domain_stats = create_descriptive_stats_tables(df, output_path)
    print(f"✅ 통계 테이블 생성 완료. 결과는 '{output_path}' 폴더에 저장됨.")

    # 5. 통계 분석 실행 (신규)
    print("\n[5/7] 통계 분석 (ANOVA) 수행 중...")
    perform_statistical_analysis(df, output_path)

    # 6. 어휘 통계 계산
    print("\n[6/7] 어휘 다양성(TTR) 통계 계산 중...")
    vocab_stats_df = calculate_vocabulary_stats(df)
    print("어휘 통계 계산 완료:")
    print(vocab_stats_df)
    vocab_stats_df.to_csv(output_path / 'vocabulary_stats.csv', index=False, encoding='utf-8-sig')
    
    # 7. 시각화
    print("\n[7/7] 통계 시각화 생성 중...")
    plot_distribution(df, 'char_count', '글자 수', output_path)
    plot_distribution(df, 'word_count', '단어 수', output_path)
    plot_distribution(df, 'sentence_count', '문장 수', output_path)
    plot_boxplots(df, output_path)  # 박스플롯 추가
    plot_vocabulary_comparison(vocab_stats_df, output_path)
    
    # 추가: 통계 요약을 텍스트 파일로 저장
    with open(output_path / 'summary_stats.txt', 'w', encoding='utf-8') as f:
        f.write("=== 의학 질문 텍스트 분석 요약 ===\n\n")
        f.write(f"총 분석 대상 문서: {len(df)}개\n")
        f.write(f"도메인 종류: {df['domain'].nunique()}개\n\n")
        
        f.write("도메인별 문서 수:\n")
        domain_counts = df['domain'].value_counts()
        for domain, count in domain_counts.items():
            f.write(f"- {domain}: {count}개\n")
        
        f.write("\n전체 통계 요약:\n")
        f.write(f"- 평균 글자 수: {df['char_count'].mean():.2f} (표준편차: {df['char_count'].std():.2f})\n")
        f.write(f"- 평균 단어 수: {df['word_count'].mean():.2f} (표준편차: {df['word_count'].std():.2f})\n")
        f.write(f"- 평균 문장 수: {df['sentence_count'].mean():.2f} (표준편차: {df['sentence_count'].std():.2f})\n")
        
    print(f"✅ 모든 분석 결과가 '{output_path}' 폴더에 저장되었습니다.")

if __name__ == '__main__':
    main() 