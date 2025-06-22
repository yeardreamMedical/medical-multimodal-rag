#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KorMedMCQA와 02.라벨링데이터 의학 데이터셋 간의 텍스트 통계 비교 스크립트

사용법:
  python compare_datasets.py \
    --mcqa_path "../data/KorMedMCQA/doctor/doctor-train.csv" \
    --label_path_08 "../data/08.전문 의학지식 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터" \
    --label_path_09 "../data/09.필수의료 의학지식 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터" \
    --out_dir "comparison_results"

- t-test를 통한 통계적 유의성 검정
- 텍스트 통계 시각화 비교
"""
from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import glob
import re
import os
from typing import Dict, List, Tuple

# Matplotlib 설정
import matplotlib
matplotlib.use("Agg")  # Tk 없이 동작하는 백엔드
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# 통계 검정
from scipy import stats

# 한글 폰트 설정 (개선된 버전)
import platform

# 디버깅용 함수: 사용 가능한 한글 폰트 목록 출력
def print_available_fonts():
    print("\n=== 사용 가능한 한글 폰트 목록 ===")
    system_fonts = sorted([f.name for f in font_manager.fontManager.ttflist if any(keyword in f.name.lower() for keyword in ['gothic', 'gulim', 'batang', 'dotum', 'malgun', 'apple', '나눔', '산돌', '고딕'])])
    for i, font in enumerate(system_fonts):
        print(f"{i+1}. {font}")
    print("=" * 35)

# Mac용 한글 폰트 설정 (방법 1-3 시도)
font_installed = False

# 방법 1: AppleGothic 폰트 사용
try:
    if platform.system() == 'Darwin':  # macOS
        font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 폰트 깨짐 방지
        print("✅ 한글 폰트 설정 성공 (AppleGothic)")
        font_installed = True
except Exception as e:
    print(f"방법 1 실패: {e}")

# 방법 2: AppleSDGothicNeo 사용
if not font_installed:
    try:
        if platform.system() == 'Darwin':  # macOS
            font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
            font_prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            print("✅ 한글 폰트 설정 성공 (AppleSDGothicNeo)")
            font_installed = True
    except Exception as e:
        print(f"방법 2 실패: {e}")

# 방법 3: 시스템별 기본 설정 방식
if not font_installed:
    try:
        if platform.system() == 'Darwin':  # macOS
            plt.rc('font', family='AppleGothic')
        elif platform.system() == 'Windows':
            plt.rc('font', family='Malgun Gothic')
        else:  # Linux
            plt.rc('font', family='NanumGothic')
        plt.rcParams['axes.unicode_minus'] = False
        print("✅ 한글 폰트 설정 성공 (일반적인 방식)")
        font_installed = True
    except Exception as e:
        print(f"방법 3 실패: {e}")

# 방법 4: 폰트 목록 출력 및 안내
if not font_installed:
    print("⚠️ 한글 폰트 설정 실패")
    print_available_fonts()
    print("코드의 font_path 변수를 위 목록의 폰트 경로로 수정해주세요.")

# 한국어 형태소 분석기
try:
    from konlpy.tag import Okt
    okt = Okt()
    print("✅ KoNLPy Okt 모듈 로드 성공")
except Exception as e:
    print(f"⚠️ KoNLPy Okt 로드 실패: {e}")
    print("명사 추출 및 어휘 다양성 분석이 비활성화됩니다.")
    okt = None

# 텍스트 전처리 함수
def preprocess_text(text: str) -> str:
    """문제 텍스트를 분석용으로 간단 전처리"""
    if not isinstance(text, str):
        return ""
    
    # HTML 태그 제거 (BeautifulSoup 의존성 제거)
    text = re.sub(r'<[^>]+>', '', text)
    
    # 보기 번호(1), 2.  같은 패턴 제거
    text = re.sub(r"\s+\d[).]\s+", " ", text)
    
    # 안내성 한국어 괄호 제거 – 그림, 사진, 도표 등
    text = re.sub(r"\((?:그림|사진|도표|표|보기|참조|단,)\s*[^)]*\)", "", text)
    
    # 제어 문자 → 공백
    text = re.sub(r"[\n\r\t]", " ", text)
    
    # 중복 공백 정리
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

# KorMedMCQA CSV 파일 로드
def load_mcqa_data(file_path: str) -> pd.DataFrame:
    """CSV 파일을 로드하여 question 열의 통계 분석"""
    try:
        # UTF-8-BOM 또는 UTF-8 인코딩 시도
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except:
        # 다른 인코딩 시도
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except:
            df = pd.read_csv(file_path, encoding='cp949')
    
    if 'question' not in df.columns:
        raise ValueError(f"CSV 파일에 'question' 열이 없습니다: {file_path}")
    
    df['text'] = df['question']
    df['source'] = 'KorMedMCQA'
    df['dataset'] = Path(file_path).stem
    
    print(f"✅ KorMedMCQA 데이터 로드 완료: {len(df)} 문항")
    return df[['text', 'source', 'dataset']]

# 라벨링 데이터 로드 (JSON)
def load_labeled_data(base_path: str, label: str) -> pd.DataFrame:
    """TL_* 폴더에서 q_type==1인 문항을 로드"""
    data = []
    base_path = Path(base_path)
    # TL_* 폴더 목록 찾기
    tl_dirs = [d for d in base_path.glob("TL_*") if d.is_dir()]
    
    for tl_dir in tl_dirs:
        domain_name = tl_dir.name
        json_files = list(tl_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8-sig") as f:
                    content = json.load(f)
                
                # q_type==1 문항만 선택
                if (isinstance(content, dict) and 
                    content.get("question") and 
                    content.get("q_type") == 1):
                    data.append({
                        'text': content["question"],
                        'source': label,
                        'dataset': domain_name
                    })
            except json.JSONDecodeError:
                print(f"⚠️ JSON 파싱 오류: {json_file}")
                continue
    
    df = pd.DataFrame(data)
    print(f"✅ {label} 라벨링 데이터 로드 완료: {len(df)} 문항")
    return df

# 텍스트 통계 계산
def calculate_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    """문자 수, 단어 수, 문장 수 계산"""
    df['clean_text'] = df['text'].apply(preprocess_text)
    df['char_count'] = df['clean_text'].str.len()
    df['word_count'] = df['clean_text'].str.split().str.len()
    # 문장 종결 문자(.) 기준으로 문장 수 계산, 없으면 1로 간주
    df['sentence_count'] = df['clean_text'].str.count(r'[.?!]') + 1
    return df

# 어휘 통계 계산
def calculate_vocab_stats(texts: list[str]) -> Dict:
    """어휘 다양성 통계 계산"""
    if okt is None:
        return {"total_nouns": 0, "unique_nouns": 0, "TTR": 0}
    
    full_text = " ".join(texts)
    nouns = okt.nouns(full_text)
    
    if not nouns:
        return {"total_nouns": 0, "unique_nouns": 0, "TTR": 0}
    
    total = len(nouns)
    unique = len(set(nouns))
    ttr = unique / total if total > 0 else 0
    
    return {
        "total_nouns": total,
        "unique_nouns": unique,
        "TTR": ttr
    }

# 통계적 유의성 검정
def perform_statistical_tests(df1: pd.DataFrame, df2: pd.DataFrame) -> Dict:
    """두 데이터셋 간의 통계적 유의성 검정 (t-test)"""
    results = {}
    
    # 각 메트릭에 대해 t-test 수행
    for metric in ['char_count', 'word_count', 'sentence_count']:
        t_stat, p_value = stats.ttest_ind(
            df1[metric].dropna(), 
            df2[metric].dropna(), 
            equal_var=False  # Welch's t-test (등분산 가정 없음)
        )
        
        results[metric] = {
            'mean_1': df1[metric].mean(),
            'mean_2': df2[metric].mean(),
            'median_1': df1[metric].median(),
            'median_2': df2[metric].median(),
            'std_1': df1[metric].std(),
            'std_2': df2[metric].std(),
            't_stat': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    return results

# 비교 시각화 생성
def create_comparison_plots(dfs: List[pd.DataFrame], out_dir: Path):
    """여러 데이터셋 간의 비교 시각화 생성"""
    # 1. 박스플롯: 문자 수
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=pd.concat(dfs), x='source', y='char_count', showfliers=False)
    plt.title('데이터셋별 문자 수 분포')
    plt.xlabel('데이터셋')
    plt.ylabel('문자 수')
    plt.tight_layout()
    plt.savefig(out_dir / 'char_count_boxplot.png')
    plt.close()
    
    # 2. 박스플롯: 단어 수
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=pd.concat(dfs), x='source', y='word_count', showfliers=False)
    plt.title('데이터셋별 단어 수 분포')
    plt.xlabel('데이터셋')
    plt.ylabel('단어 수')
    plt.tight_layout()
    plt.savefig(out_dir / 'word_count_boxplot.png')
    plt.close()
    
    # 3. 박스플롯: 문장 수
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=pd.concat(dfs), x='source', y='sentence_count', showfliers=False)
    plt.title('데이터셋별 문장 수 분포')
    plt.xlabel('데이터셋')
    plt.ylabel('문장 수')
    plt.tight_layout()
    plt.savefig(out_dir / 'sentence_count_boxplot.png')
    plt.close()
    
    # 4. 바이올린 플롯: 종합 비교
    plt.figure(figsize=(18, 12))
    
    plt.subplot(3, 1, 1)
    sns.violinplot(data=pd.concat(dfs), x='source', y='char_count', inner='quartile')
    plt.title('데이터셋별 문자 수 분포')
    
    plt.subplot(3, 1, 2)
    sns.violinplot(data=pd.concat(dfs), x='source', y='word_count', inner='quartile')
    plt.title('데이터셋별 단어 수 분포')
    
    plt.subplot(3, 1, 3)
    sns.violinplot(data=pd.concat(dfs), x='source', y='sentence_count', inner='quartile')
    plt.title('데이터셋별 문장 수 분포')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'distribution_violin_combined.png')
    plt.close()

# 어휘 다양성 비교 시각화
def create_vocab_comparison_plot(vocab_stats: Dict[str, Dict], out_dir: Path):
    """어휘 다양성 비교 시각화"""
    if okt is None:
        return
    
    plt.figure(figsize=(14, 10))
    
    # 데이터 준비
    datasets = list(vocab_stats.keys())
    metrics = ['총 명사 수', '고유 명사 수', 'TTR']
    
    # 값 정규화 (TTR과 다른 지표의 스케일 차이가 크므로)
    normalized_values = {
        '총 명사 수': [vocab_stats[ds]['total_nouns'] / 1000 for ds in datasets],  # 1000단위로 표시
        '고유 명사 수': [vocab_stats[ds]['unique_nouns'] / 1000 for ds in datasets],  # 1000단위로 표시
        'TTR': [vocab_stats[ds]['TTR'] for ds in datasets]  # TTR은 정규화 필요 없음
    }
    
    x = np.arange(len(datasets))
    width = 0.25
    
    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, metric in enumerate(metrics):
        bars = ax.bar(x + width * (i - 1), normalized_values[metric], width, label=metric)
        
        # 값 레이블 추가
        for j, bar in enumerate(bars):
            if metric == 'TTR':
                # TTR은 소수점 4자리까지
                value = vocab_stats[datasets[j]]['TTR']
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{value:.4f}", 
                        ha='center', va='bottom', fontsize=9)
            else:
                # 나머지는 정수 표시
                value = vocab_stats[datasets[j]]['total_nouns'] if metric == '총 명사 수' else vocab_stats[datasets[j]]['unique_nouns']
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{value:,}", 
                        ha='center', va='bottom', fontsize=9)
    
    ax.set_title('데이터셋별 어휘 통계 비교')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('값 (명사 수: 천 단위, TTR: 원래 값)')
    ax.set_ylim(0, max(max(normalized_values['총 명사 수']), max(normalized_values['고유 명사 수'])) * 1.2)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / 'vocab_comparison.png')
    plt.close()

# 통계 결과 텍스트 파일 생성
def save_comparison_results(
    dfs: Dict[str, pd.DataFrame], 
    vocab_stats: Dict[str, Dict], 
    statistical_tests: Dict[str, Dict], 
    out_dir: Path
):
    """비교 통계 결과를 텍스트 파일로 저장"""
    with open(out_dir / 'comparison_results.txt', 'w', encoding='utf-8') as f:
        f.write("# 데이터셋 간 통계적 비교 결과\n\n")
        
        # 1. 기본 정보
        f.write("## 데이터셋 정보\n")
        for name, df in dfs.items():
            f.write(f"- {name}: {len(df)} 문항\n")
        f.write("\n")
        
        # 2. 텍스트 통계
        f.write("## 텍스트 통계 요약\n\n")
        f.write("| 데이터셋 | 평균 문자 수 | 평균 단어 수 | 평균 문장 수 |\n")
        f.write("|---------|------------|------------|------------|\n")
        
        for name, df in dfs.items():
            f.write(f"| {name} | {df['char_count'].mean():.1f} | {df['word_count'].mean():.1f} | {df['sentence_count'].mean():.1f} |\n")
        f.write("\n")
        
        # 3. 어휘 다양성
        f.write("## 어휘 다양성\n\n")
        f.write("| 데이터셋 | 총 명사 수 | 고유 명사 수 | TTR |\n")
        f.write("|---------|-----------|------------|-----|\n")
        
        for name, stats in vocab_stats.items():
            f.write(f"| {name} | {stats['total_nouns']:,} | {stats['unique_nouns']:,} | {stats['TTR']:.4f} |\n")
        f.write("\n")
        
        # 4. 통계적 유의성 검정 결과
        f.write("## 통계적 유의성 검정 결과\n\n")
        
        datasets = list(dfs.keys())
        if len(datasets) >= 2:
            f.write(f"### {datasets[0]} vs {datasets[1]}\n\n")
            
            for metric, result in statistical_tests.items():
                if metric == 'char_count':
                    metric_name = '문자 수'
                elif metric == 'word_count':
                    metric_name = '단어 수'
                elif metric == 'sentence_count':
                    metric_name = '문장 수'
                    
                significance = "유의미함" if result['significant'] else "유의미하지 않음"
                
                f.write(f"#### {metric_name}\n")
                f.write(f"- {datasets[0]} 평균: {result['mean_1']:.2f}, 중앙값: {result['median_1']:.1f}, 표준편차: {result['std_1']:.2f}\n")
                f.write(f"- {datasets[1]} 평균: {result['mean_2']:.2f}, 중앙값: {result['median_2']:.1f}, 표준편차: {result['std_2']:.2f}\n")
                f.write(f"- t-통계량: {result['t_stat']:.3f}, p-값: {result['p_value']:.6f}\n")
                f.write(f"- 차이: **{significance}** (p < 0.05)\n\n")

def main():
    # 인자 파싱
    parser = argparse.ArgumentParser(description="KorMedMCQA와 라벨링 데이터 간의 통계적 비교")
    parser.add_argument("--mcqa_path", required=True, help="KorMedMCQA CSV 파일 경로")
    parser.add_argument("--label_path_08", required=True, help="08.전문 의학지식 라벨링 데이터 경로")
    parser.add_argument("--label_path_09", required=True, help="09.필수의료 의학지식 라벨링 데이터 경로")
    parser.add_argument("--out_dir", default="comparison_results", help="결과 저장 폴더")
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"데이터 경로:")
    print(f"- KorMedMCQA: {args.mcqa_path}")
    print(f"- 08.전문 의학지식: {args.label_path_08}")
    print(f"- 09.필수의료 의학지식: {args.label_path_09}")
    print(f"출력 폴더: {out_dir}")
    
    # 1. 데이터 로드
    df_mcqa = load_mcqa_data(args.mcqa_path)
    df_label_08 = load_labeled_data(args.label_path_08, "08.전문의학")
    df_label_09 = load_labeled_data(args.label_path_09, "09.필수의료")
    
    # 2. 텍스트 통계 계산
    df_mcqa = calculate_text_stats(df_mcqa)
    df_label_08 = calculate_text_stats(df_label_08)
    df_label_09 = calculate_text_stats(df_label_09)
    
    # 3. 어휘 다양성 분석
    vocab_mcqa = calculate_vocab_stats(df_mcqa['clean_text'].tolist())
    vocab_label_08 = calculate_vocab_stats(df_label_08['clean_text'].tolist())
    vocab_label_09 = calculate_vocab_stats(df_label_09['clean_text'].tolist())
    
    # 4. 통계적 검정 (KorMedMCQA vs 08.전문의학)
    stats_results = perform_statistical_tests(df_mcqa, df_label_08)
    
    # 5. 비교 시각화 생성
    create_comparison_plots([df_mcqa, df_label_08, df_label_09], out_dir)
    
    # 6. 어휘 다양성 시각화
    vocab_stats = {
        "KorMedMCQA": vocab_mcqa,
        "08.전문의학": vocab_label_08,
        "09.필수의료": vocab_label_09
    }
    create_vocab_comparison_plot(vocab_stats, out_dir)
    
    # 7. 결과 저장
    dfs = {
        "KorMedMCQA": df_mcqa,
        "08.전문의학": df_label_08,
        "09.필수의료": df_label_09
    }
    save_comparison_results(dfs, vocab_stats, stats_results, out_dir)
    
    print(f"✅ 분석 및 비교 완료! 결과가 {out_dir} 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main() 