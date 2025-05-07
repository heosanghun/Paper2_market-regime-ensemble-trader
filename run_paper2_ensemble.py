#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Paper2: 시장 레짐 동적제어 앙상블 기반 멀티모달 트레이딩 시스템 실행 스크립트
- 시장 상태(레짐)를 동적으로 감지하여 시간프레임별 최적 가중치 조정
- 멀티모달(이미지+뉴스) 데이터 기반 앙상블 트레이딩
"""

import os
import sys
import logging
import time
import colorama
from datetime import datetime
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 절대 경로 확인
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"현재 디렉토리: {script_dir}")
parent_dir = os.path.dirname(script_dir)
print(f"상위 디렉토리: {parent_dir}")
paper1_dir = os.path.join(parent_dir, 'paper1')
print(f"Paper1 디렉토리: {paper1_dir}")

# Python 경로에 추가
sys.path.append(script_dir)
sys.path.append(parent_dir)
sys.path.append(paper1_dir)

# Python 경로 확인
print(f"Python 경로: {sys.path}")

# 모듈 사용 가능 여부 확인
try:
    import paper1
    print("paper1 패키지 임포트 성공")
except ImportError:
    print("paper1 패키지 임포트 실패")

# paper1 디렉토리 존재 확인
if os.path.exists(paper1_dir):
    print(f"D:\\drl-candlesticks-trader-main1\\paper1 경로가 존재합니다.")
    # 모듈 경로 추가
    sys.path.append(paper1_dir)
    print(f"Paper1 모듈 경로 추가: {paper1_dir}")
    # 모든 paper1 하위 폴더 추가
    sys.path.append("paper1")
    sys.path.append(paper1_dir)
    sys.path.append(paper1_dir)
    
    # Python 경로 확인 
    print(f"Python path: {sys.path}")
else:
    print(f"D:\\drl-candlesticks-trader-main1\\paper1 경로가 존재하지 않습니다.")

# Paper2 모듈 임포트
try:
    from market_state_detector import MarketStateDetector
    from dynamic_weight_adjuster import DynamicWeightAdjuster
    from ensemble_controller import EnsembleController
    print("Paper2 모듈 임포트 성공")
except ImportError as e:
    print(f"Paper2 모듈 임포트 실패: {e}")

# paper1 디렉토리 내용 확인
if os.path.exists(paper1_dir):
    paper1_files = os.listdir(paper1_dir)
    print(f"paper1 디렉토리 파일 목록: {paper1_files}")

# Paper1 모듈 임포트
try:
    from paper1.candlestick_analyzer import CandlestickAnalyzer
    from paper1.sentiment_analyzer import SentimentAnalyzer
    from paper1.rl_trader import PPOTrader
    print("Paper1 모듈 임포트 성공")
except ImportError as e:
    print(f"Paper1 모듈 임포트 실패: {e}")
    sys.exit(1)  # 필수 모듈이 없으면 종료

# 컬러 출력 초기화
colorama.init()

# 색상 코드 정의
GREEN = colorama.Fore.GREEN
YELLOW = colorama.Fore.YELLOW
RED = colorama.Fore.RED
BLUE = colorama.Fore.BLUE
MAGENTA = colorama.Fore.MAGENTA
CYAN = colorama.Fore.CYAN
RESET = colorama.Fore.RESET
BRIGHT = colorama.Style.BRIGHT
DIM = colorama.Style.DIM

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Paper2Runner')

def print_status(message, color=GREEN, is_title=False):
    """상태 메시지 출력 함수"""
    timestamp = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
    
    if is_title:
        print("\n" + "="*80)
        print(f"{BRIGHT}{color}{timestamp} {message}{RESET}")
        print("="*80 + "\n")
    else:
        print(f"{color}{timestamp} {message}{RESET}")

def format_time(seconds):
    """시간 형식화 함수"""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{int(hours)}시간 {int(minutes)}분 {seconds:.2f}초"
    elif minutes > 0:
        return f"{int(minutes)}분 {seconds:.2f}초"
    else:
        return f"{seconds:.2f}초"

def main():
    # 시작 시간 기록
    start_time = time.time()
    print_status("시장 레짐 동적제어 앙상블 트레이딩 시뮬레이션 시작", MAGENTA, True)
    
    # GPU 확인 및 사용 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print_status(f"사용 가능 디바이스: {device}", CYAN)
    
    # 실행 경로 확인
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    print_status(f"현재 작업 디렉토리: {os.getcwd()}", BLUE)
    
    # 결과 디렉토리 생성 (paper2 폴더 내에)
    results_dir = os.path.join(current_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 실행 시간을 파일명에 포함
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_results_dir = os.path.join(results_dir, f'run_{timestamp}')
    os.makedirs(run_results_dir, exist_ok=True)
    print_status(f"결과 저장 경로: {run_results_dir}", BLUE)
    
    # 설정 정의
    config = {
        'data': {
            'chart_dir': r'D:\drl-candlesticks-trader-main1\paper1\data\chart',  # 정확한 차트 데이터 경로
            'news_file': r'D:\drl-candlesticks-trader-main1\paper1\data\news\cryptonews_2021-10-12_2023-12-19.csv',  # 정확한 뉴스 데이터 파일 경로
            'timeframes': ['1d', '4h', '1h', '30m', '15m', '5m']  # 모든 시간프레임 사용
        },
        'output': {
            'save_dir': run_results_dir  # 결과 저장 경로
        },
        # 시장 레짐 감지 설정
        'market_state': {
            'volatility_window': 20,  # 변동성 계산 윈도우
            'trend_window': 50,       # 추세 감지 윈도우
            'volume_window': 20,      # 거래량 윈도우
            'volatility_threshold': 1.5,  # 변동성 임계값
            'volume_threshold': 1.2,      # 거래량 임계값
            'trend_threshold': 0.6        # 추세 임계값
        },
        # 동적 가중치 설정
        'dynamic_weights': {
            'learning_rate': 0.2,      # 학습률
            'performance_weight': 0.3,  # 성과 기반 조정 가중치
            # 시간프레임 기본 가중치 (합이 1이 되도록 설정)
            'default_weights': {
                '5m': 0.10,   # 5분봉
                '15m': 0.15,  # 15분봉
                '1h': 0.30,   # 1시간봉
                '4h': 0.25,   # 4시간봉
                '1d': 0.20    # 일봉
            }
        },
        # 앙상블 컨트롤러 설정
        'ensemble': {
            'symbols': ['BTCUSDT'],    # 거래 심볼
            'primary_timeframe': '1d', # 주 시간프레임
            'min_timeframes': 3,       # 최소 사용 시간프레임 수
            'confidence_threshold': 0.6  # 신뢰도 임계값
        },
        # 거래 설정
        'trading': {
            'initial_capital': 10000.0,  # 초기 자본금
            'position_size': 0.1,        # 포지션 크기
            'max_trades_per_day': 5,     # 일일 최대 거래 수
            'stop_loss': 0.02,           # 손절매 비율
            'take_profit': 0.03          # 이익실현 비율
        },
        # 시그널 생성에 필요한 가중치 설정
        'pattern_weight': 0.7,          # 캔들스틱 패턴 가중치
        'news_weight': 0.3,             # 뉴스 감성 가중치
        'trade_threshold': 0.5          # 거래 신호 임계값
    }
    
    # 경로 유효성 확인
    for key, path in [('chart_dir', config['data']['chart_dir']), ('news_file', config['data']['news_file'])]:
        if os.path.exists(path):
            print_status(f"{key} 경로 확인: {path} (존재함)", GREEN)
        else:
            print_status(f"{key} 경로 확인: {path} (존재하지 않음)", RED)
    
    try:
        # 전체 진행 단계 정의
        steps = [
            '시장 상태 감지기 초기화',
            '동적 가중치 조정기 초기화',
            '앙상블 컨트롤러 초기화',
            '트레이딩 시뮬레이션 실행',
            '결과 분석 및 시각화',
            '결과 저장'
        ]
        
        # 진행 상황 표시
        overall_progress = tqdm(
            total=len(steps), 
            desc=f"{BRIGHT}{CYAN}[전체 진행률]{RESET}", 
            position=0, 
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        # 각 단계별 실행
        market_detector = None
        weight_adjuster = None
        ensemble_controller = None
        
        for i, step in enumerate(steps):
            step_start_time = time.time()
            step_desc = f"{GREEN}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {step} 중...{RESET} ({i+1}/{len(steps)})"
            print(f"\n{step_desc}")
            
            # 단계별 진행 및 실행
            if step == '시장 상태 감지기 초기화':
                print_status("시장 상태 감지기 초기화 중...", YELLOW)
                
                # 감지기 설정
                market_detector_config = config['market_state']
                
                # 시장 상태 감지기 초기화
                market_detector = MarketStateDetector(
                    config=market_detector_config,
                    results_dir=os.path.join(run_results_dir, 'market_state')
                )
                
                print_status("시장 상태 감지기 초기화 완료", GREEN)
            
            elif step == '동적 가중치 조정기 초기화':
                print_status("동적 가중치 조정기 초기화 중...", YELLOW)
                
                # 가중치 조정기 설정
                weight_adjuster_config = config['dynamic_weights']
                
                # 동적 가중치 조정기 초기화
                weight_adjuster = DynamicWeightAdjuster(
                    config=weight_adjuster_config,
                    results_dir=os.path.join(run_results_dir, 'weight_adjuster')
                )
                
                print_status("동적 가중치 조정기 초기화 완료", GREEN)
            
            elif step == '앙상블 컨트롤러 초기화':
                print_status("앙상블 컨트롤러 초기화 중...", YELLOW)
                
                # 앙상블 컨트롤러 설정
                ensemble_config = {
                    **config['ensemble'],
                    **config['trading'],
                    'data_dir': config['data']['chart_dir'],
                    'news_file': config['data']['news_file'],
                    'timeframes': config['data']['timeframes'],
                    'pattern_weight': config['pattern_weight'],
                    'news_weight': config['news_weight'],
                    'trade_threshold': config['trade_threshold']
                }
                
                # 앙상블 컨트롤러 초기화
                ensemble_controller = EnsembleController(
                    config=ensemble_config,
                    market_detector=market_detector,
                    weight_adjuster=weight_adjuster,
                    results_dir=run_results_dir
                )
                
                print_status("앙상블 컨트롤러 초기화 완료", GREEN)
            
            elif step == '트레이딩 시뮬레이션 실행':
                print_status("트레이딩 시뮬레이션 실행 중...", YELLOW)
                
                step_progress = tqdm(total=100, desc="[시뮬레이션 진행률]", position=1)
                
                # 진행률 업데이트 함수 (실제로는 EnsembleController에서 콜백 함수 사용)
                def update_progress(percent):
                    step_progress.n = int(percent)
                    step_progress.refresh()
                
                # 앙상블 거래 시뮬레이션 실행
                results = ensemble_controller.run()
                
                step_progress.close()
                
                print_status("트레이딩 시뮬레이션 실행 완료", GREEN)
            
            elif step == '결과 분석 및 시각화':
                print_status("결과 분석 및 시각화 중...", YELLOW)
                
                # 추가 시각화 생성
                # (EnsembleController에서 이미 많은 시각화를 생성하므로 여기서는 추가 작업만 수행)
                
                # 시각화: 포트폴리오 가치 변화 추적
                try:
                    portfolio_data = pd.read_csv(os.path.join(run_results_dir, 'portfolio_history.csv'))
                    
                    plt.figure(figsize=(12, 6))
                    plt.plot(portfolio_data['timestamp'], portfolio_data['portfolio_value'], linewidth=2)
                    plt.title('Portfolio Value Over Time', fontsize=16)
                    plt.xlabel('Date')
                    plt.ylabel('Portfolio Value ($)')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(run_results_dir, 'portfolio_value_chart.png'), dpi=300)
                    plt.close()
                except Exception as e:
                    print_status(f"포트폴리오 차트 생성 오류: {str(e)}", RED)
                
                print_status("결과 분석 및 시각화 완료", GREEN)
            
            elif step == '결과 저장':
                print_status("결과 저장 중...", YELLOW)
                
                # 성능 메트릭 저장
                if ensemble_controller:
                    ensemble_controller.save_performance_metrics()
                    ensemble_controller.save_ensemble_weights()
                
                # 가중치 조정 히스토리 저장
                if weight_adjuster:
                    weight_history = weight_adjuster.get_weight_history()
                    
                    # JSON으로 저장
                    with open(os.path.join(run_results_dir, 'weight_history.json'), 'w') as f:
                        import json
                        # datetime은 직렬화할 수 없으므로 문자열로 변환
                        weight_history_serializable = {
                            'timestamp': weight_history['timestamp'],
                            'market_state': weight_history['market_state'],
                            'weights': [{k: float(v) for k, v in w.items()} for w in weight_history['weights']]
                        }
                        json.dump(weight_history_serializable, f, indent=4)
                
                print_status("결과 저장 완료", GREEN)
            
            # 진행률 업데이트
            overall_progress.update(1)
            
            # 단계별 소요 시간 출력
            step_elapsed = time.time() - step_start_time
            print_status(f"{step} 완료: {format_time(step_elapsed)}", DIM + GREEN)
        
        overall_progress.close()
        
        # 전체 시뮬레이션 완료 및 총 소요 시간 출력
        total_elapsed = time.time() - start_time
        print_status(f"시장 레짐 동적제어 앙상블 트레이딩 시뮬레이션 완료! 총 소요 시간: {format_time(total_elapsed)}", MAGENTA, True)
        
        # 주요 성능 지표 요약 출력
        try:
            with open(os.path.join(run_results_dir, 'performance_metrics.json'), 'r') as f:
                import json
                metrics = json.load(f)
                
                print("\n" + "-"*50)
                print(f"{BRIGHT}{CYAN}성능 지표 요약{RESET}")
                print("-"*50)
                print(f"총 수익률: {BRIGHT}{GREEN if metrics.get('total_return_pct', 0) >= 0 else RED}{metrics.get('total_return_pct', 0):.2f}%{RESET}")
                print(f"샤프 비율: {metrics.get('sharpe_ratio', 0):.2f}")
                print(f"최대 낙폭: {RED}{metrics.get('max_drawdown_pct', 0):.2f}%{RESET}")
                print(f"승률: {metrics.get('win_rate', 0):.2f}%")
                print(f"총 거래 수: {metrics.get('total_trades', 0)}")
                print("-"*50)
        except Exception as e:
            print_status(f"성능 지표 요약 출력 오류: {str(e)}", RED)
    
    except Exception as e:
        import traceback
        print_status(f"시뮬레이션 실행 중 오류 발생: {str(e)}", RED)
        traceback.print_exc()
    
    finally:
        # 컬러 설정 리셋
        print(RESET)

if __name__ == "__main__":
    main() 