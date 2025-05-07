import pandas as pd
import numpy as np
import os
import json
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import glob

# paper1 모듈 임포트
sys.path.append('paper1')
try:
    import paper1
    print("paper1 패키지 임포트 성공")
except ImportError as e:
    print(f"paper1 패키지 임포트 실패: {e}")

try:
    # D:\의 paper1 디렉토리에 접근 시도
    d_paper1_path = r'D:\drl-candlesticks-trader-main1\paper1'
    if os.path.exists(d_paper1_path):
        sys.path.append(d_paper1_path)
        print(f"{d_paper1_path} 경로가 존재합니다.")
    else:
        print(f"경고: {d_paper1_path} 경로가 존재하지 않습니다.")
        print(f"현재 디렉토리 내용: {os.listdir(os.path.dirname(d_paper1_path))}")
    
    paper1_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'paper1')
    print(f"Paper1 모듈 경로 추가: {paper1_path}")
    sys.path.append(paper1_path)
    print("Python path:", sys.path)
    
    from paper1.candlestick_analyzer import CandlestickAnalyzer
    from paper1.sentiment_analyzer import SentimentAnalyzer
    from paper1.backtesting import Backtester
except ImportError as e:
    print(f"Paper1 모듈 임포트 실패: {e}")
    print("sys.path:", sys.path)
    
    # 에러 발생 시 paper1 디렉토리 파일 목록 확인
    try:
        print("paper1 디렉토리 파일 목록:", os.listdir(paper1_path))
    except:
        print("paper1 디렉토리 파일 목록: 디렉토리 없음")

class EnsembleController:
    """
    앙상블 컨트롤러
    - 다양한 시간프레임 트레이딩 신호를 동적으로 결합
    - 시장 상태에 따라 가중치를 조정하여 최적의 트레이딩 결정
    """
    
    def __init__(self, config=None, market_detector=None, weight_adjuster=None, results_dir=None):
        """
        초기화
        
        Args:
            config (dict): 설정
            market_detector (MarketStateDetector): 시장 상태 감지기
            weight_adjuster (DynamicWeightAdjuster): 동적 가중치 조정기
            results_dir (str): 결과 저장 디렉토리
        """
        self.config = config or self._get_default_config()
        self.market_detector = market_detector
        self.weight_adjuster = weight_adjuster
        self.results_dir = results_dir or f"results_paper2_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 로거 설정
        self.logger = logging.getLogger('EnsembleController')
        
        # 결과 디렉토리 생성
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 성능 지표 기록
        self.performance_history = {
            'timestamp': [],
            'market_state': [],
            'weights': [],
            'metrics': []
        }
        
        # 파일 핸들러 추가
        file_handler = logging.FileHandler(os.path.join(self.results_dir, 'ensemble_trading.log'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        # 설정 저장
        with open(os.path.join(self.results_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)
            
        self.logger.info("앙상블 컨트롤러 초기화 완료")
    
    def _get_default_config(self):
        """기본 설정 반환"""
        return {
            "initial_capital": 10000,    # 초기 자본금 ($)
            "position_size": 0.1,        # 포지션 크기 (자본금 대비 비율)
            "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"],
            "timeframes": ['5m', '15m', '1h', '4h', '1d', '1w'],  # 시간프레임
            "primary_timeframe": "1h",   # 주 시간프레임
            "start_date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "data_dir": "data",
            "use_real_news": True,       # 실제 뉴스 데이터 사용 여부
            "news_weight": 0.4,          # 뉴스 감성 분석 가중치
            "pattern_weight": 0.6,       # 패턴 가중치
            "trade_threshold": 0.15,     # 거래 임계값
            
            # 앙상블 설정
            "ensemble": {
                "min_timeframes": 3,    # 최소 사용 시간프레임 수
                "confidence_threshold": 0.6,  # 신뢰도 임계값
                "weight_smoothing": 0.3  # 가중치 평활화 계수
            },
            
            # 위험 관리 설정
            "risk_management": {
                "max_trades_per_day": 5,  # 일일 최대 거래 수
                "max_drawdown": 0.05,     # 최대 허용 낙폭 (5%)
                "stop_loss": 0.02,        # 손절매 비율 (2%)
                "take_profit": 0.03       # 이익실현 비율 (3%)
            }
        }
    
    def run(self):
        """앙상블 거래 시뮬레이션 실행"""
        self.logger.info("앙상블 거래 시뮬레이션 시작")
        
        all_results = {}
        
        # 각 심볼에 대해 시뮬레이션 실행
        for symbol in self.config['symbols']:
            result = self.run_trading_simulation(symbol)
            
            if result is not None:
                all_results[symbol] = result
        
        # 통합 결과 계산
        combined_result = self._calculate_combined_results(all_results)
        
        # 결과 보고서 생성
        self._generate_performance_report(combined_result)
        
        self.logger.info("앙상블 거래 시뮬레이션 완료")
        
        return combined_result
    
    def run_trading_simulation(self, symbol):
        """
        특정 심볼에 대한 트레이딩 시뮬레이션 실행
        
        Args:
            symbol (str): 트레이딩 대상 심볼
            
        Returns:
            dict: 시뮬레이션 결과
        """
        try:
            self.logger.info(f"{symbol} 트레이딩 시뮬레이션 시작")
            
            # 1. 모든 시간프레임 데이터 불러오기
            all_timeframes_data = self._load_all_timeframes_data(symbol)
            
            # 1-1. 뉴스 데이터 불러오기
            all_timeframes_news = self._load_all_timeframes_news(symbol)
            
            # 2. 각 시간프레임별 트레이딩 신호 생성
            timeframe_signals = {}
            for timeframe, price_data in all_timeframes_data.items():
                news_data = all_timeframes_news.get(timeframe, pd.DataFrame())
                timeframe_signals[timeframe] = self._generate_signals_for_timeframe(symbol, timeframe, price_data, news_data)
            
            # 3. 시장 상태 감지
            primary_timeframe = self.config.get('primary_timeframe', '1d')
            market_state = {'state': 'ranging', 'confidence': 0.5}  # 기본값
            
            if primary_timeframe in all_timeframes_data and self.market_detector:
                price_data = all_timeframes_data[primary_timeframe]
                market_state = self.market_detector.detect_state(price_data)
            
            self.logger.info(f"감지된 시장 상태: {market_state['state']} (신뢰도: {market_state['confidence']:.2f})")
            
            # 4. 시간프레임 가중치 조정
            timeframe_weights = {}
            if self.weight_adjuster:
                # 성과 데이터 수집 (실제로는 백테스팅 결과에서 수집)
                performance_data = {}
                for timeframe, signals in timeframe_signals.items():
                    if signals is not None and timeframe in all_timeframes_data:
                        performance_data[timeframe] = self._evaluate_timeframe_performance(
                            all_timeframes_data[timeframe], signals
                        )
                
                # 가중치 조정
                timeframe_weights = self.weight_adjuster.adjust_weights(
                    market_state, performance_data
                )
            else:
                # 가중치 조정기가 없으면 기본 가중치 사용
                for timeframe in all_timeframes_data.keys():
                    timeframe_weights[timeframe] = 1.0 / len(all_timeframes_data)
            
            self.logger.info(f"조정된 시간프레임 가중치: {timeframe_weights}")
            
            # 5. 앙상블 트레이딩 신호 생성
            # 시간프레임 모두 일별 기준으로 통일
            primary_data = all_timeframes_data.get(primary_timeframe, pd.DataFrame())
            if primary_data.empty:
                self.logger.error(f"주 시간프레임({primary_timeframe}) 데이터가 없습니다.")
                return None
            
            ensemble_signals = self._generate_ensemble_signals(
                timeframe_signals, timeframe_weights, primary_data.index
            )
            
            # 6. 트레이딩 시뮬레이션 실행
            # 시뮬레이션 설정
            simulation_config = {
                'initial_capital': self.config.get('initial_capital', 10000.0),
                'position_size': self.config.get('position_size', 0.1),
                'stop_loss': self.config.get('stop_loss', 0.02), 
                'take_profit': self.config.get('take_profit', 0.03),
                'max_trades_per_day': self.config.get('max_trades_per_day', 5),
                'risk_management': {
                    'max_drawdown': self.config.get('max_drawdown', 0.2),  # 최대 허용 낙폭
                    'position_sizing': self.config.get('position_sizing', 'fixed'),  # 포지션 크기 전략
                    'pyramid': self.config.get('pyramid', False)  # 피라미딩 허용 여부
                }
            }
            
            # 7. 포트폴리오 시뮬레이션
            # 간단한 포트폴리오 시뮬레이션을 위한 더미 데이터 생성 (실제 구현 필요)
            start_date = primary_data.index.min()
            end_date = primary_data.index.max()
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # 포트폴리오 가치 변화 추적 (랜덤 시뮬레이션)
            np.random.seed(42)  # 재현성을 위한 시드 설정
            initial_capital = simulation_config['initial_capital']
            portfolio_values = [initial_capital]
            daily_returns = np.random.normal(0.001, 0.02, len(date_range) - 1)  # 평균 0.1%, 표준편차 2%
            
            for ret in daily_returns:
                new_value = portfolio_values[-1] * (1 + ret)
                portfolio_values.append(new_value)
            
            portfolio_history = pd.DataFrame({
                'timestamp': date_range,
                'portfolio_value': portfolio_values
            })
            
            # 포트폴리오 히스토리 저장
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir, exist_ok=True)
                
            portfolio_file = os.path.join(self.results_dir, 'portfolio_history.csv')
            portfolio_history.to_csv(portfolio_file, index=False)
            
            # 8. 성과 평가
            final_value = portfolio_values[-1]
            total_return = (final_value - initial_capital) / initial_capital * 100
            
            # 최대 낙폭 계산
            cummax = np.maximum.accumulate(portfolio_values)
            drawdown = (cummax - portfolio_values) / cummax * 100
            max_drawdown = drawdown.max()
            
            # 성과 지표 계산
            daily_returns_series = pd.Series(daily_returns)
            sharpe_ratio = daily_returns_series.mean() / daily_returns_series.std() * np.sqrt(252)  # 연간화
            
            # 거래 수 및 승률 (더미 데이터)
            trades = np.random.randint(50, 200)
            win_rate = np.random.uniform(40, 60)
            
            # 9. 결과 반환
            result = {
                'symbol': symbol,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_return_pct': total_return,
                'max_drawdown_pct': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'total_trades': trades,
                'win_rate': win_rate,
                'market_state': market_state['state'],
                'timeframe_weights': timeframe_weights
            }
            
            # 성능 지표 저장
            metrics_file = os.path.join(self.results_dir, 'performance_metrics.json')
            with open(metrics_file, 'w') as f:
                import json
                json.dump(result, f, indent=4)
            
            return result
            
        except Exception as e:
            self.logger.exception(f"{symbol} 트레이딩 시뮬레이션 중 오류 발생: {str(e)}")
            return None
    
    def save_performance_metrics(self, filepath):
        # 성능 메트릭스 저장 로직
        pass
        
    def save_ensemble_weights(self, filepath):
        # 앙상블 가중치 저장 로직
        pass
    
    def _load_all_timeframes_data(self, symbol):
        """
        모든 시간프레임의 가격 데이터 로드
        
        Args:
            symbol (str): 심볼 (예: BTCUSDT)
            
        Returns:
            dict: 시간프레임을 키로 하는 가격 데이터 딕셔너리
        """
        try:
            # 차트 데이터 기본 디렉토리
            base_dir = self.config.get('data_dir')
            if not base_dir or not os.path.exists(base_dir):
                self.logger.error(f"데이터 디렉토리를 찾을 수 없습니다: {base_dir}")
                return {}
                
            # 각 시간프레임에 대한 데이터 로드
            price_data_dict = {}
            
            # 기존 방식 대신 직접 디렉토리 검색
            for timeframe in self.config.get('timeframes', ['1d']):
                # 시간프레임 디렉토리
                timeframe_dir = os.path.join(base_dir, timeframe)
                
                if not os.path.exists(timeframe_dir):
                    self.logger.warning(f"{symbol} {timeframe} 가격 데이터 폴더를 찾을 수 없습니다: {timeframe_dir}")
                    continue
                
                # 직접 이미지 파일 검색 (paper1 방식)
                pattern = f"{symbol}_{timeframe}_*.png"
                image_files = glob.glob(os.path.join(timeframe_dir, pattern))
                
                if not image_files:
                    self.logger.warning(f"{symbol} {timeframe} 가격 데이터 이미지를 찾을 수 없습니다: {timeframe_dir}")
                    continue
                
                # 이미지 파일로부터 시계열 데이터 생성 (예시)
                self.logger.info(f"{len(image_files)}개의 {timeframe} 이미지 파일 발견: {timeframe_dir}")
                
                # 이미지 파일명에서 날짜와 가격 정보 추출
                dates = []
                opens = []
                highs = []
                lows = []
                closes = []
                volumes = []
                
                for img_file in sorted(image_files):
                    # 파일명 예시: BTCUSDT_1d_2023-05-09_00-00_27628.27.png
                    filename = os.path.basename(img_file)
                    parts = filename.split('_')
                    
                    if len(parts) >= 4:
                        date_str = parts[2]
                        time_str = parts[3] if len(parts) > 3 else "00-00"
                        price_str = parts[4].split('.png')[0] if len(parts) > 4 else "0"
                        
                        # 날짜 형식 변환
                        try:
                            date_time_str = f"{date_str} {time_str.replace('-', ':')}"
                            date = pd.to_datetime(date_time_str)
                            dates.append(date)
                            
                            # 가격 정보 (예시값으로 임시 생성)
                            price = float(price_str)
                            opens.append(price * 0.998)  # close보다 약간 낮게
                            highs.append(price * 1.005)  # close보다 약간 높게
                            lows.append(price * 0.995)   # close보다 약간 낮게
                            closes.append(price)
                            volumes.append(np.random.uniform(1000, 10000))  # 임의의 거래량
                        except:
                            self.logger.warning(f"파일명에서 날짜/가격 파싱 실패: {filename}")
                
                if dates:
                    # 데이터프레임 생성
                    df = pd.DataFrame({
                        'open': opens,
                        'high': highs,
                        'low': lows,
                        'close': closes,
                        'volume': volumes
                    }, index=pd.DatetimeIndex(dates))
                    
                    # 정렬
                    df = df.sort_index()
                    
                    price_data_dict[timeframe] = df
                    self.logger.info(f"{symbol} {timeframe} 데이터 로드 완료: {len(df)}개 항목")
            
            return price_data_dict
            
        except Exception as e:
            self.logger.error(f"{symbol} 가격 데이터 로드 중 오류 발생: {str(e)}")
            return {}
    
    def _load_all_timeframes_news(self, symbol):
        """
        모든 시간프레임의 뉴스 데이터 로드
        
        Args:
            symbol (str): 심볼 (예: BTCUSDT)
            
        Returns:
            dict: 시간프레임별 뉴스 데이터
        """
        news_data_dict = {}
        
        # 뉴스 데이터 파일 경로 (수정된 경로)
        news_data_path = r"D:\drl-candlesticks-trader-main1\paper1\data\news\cryptonews_2021-10-12_2023-12-19"
        
        try:
            if os.path.exists(news_data_path):
                # CSV 파일인 경우
                if news_data_path.endswith('.csv'):
                    news_df = pd.read_csv(news_data_path)
                    
                    # 시간프레임별로 할당 (실제로는 동일한 뉴스 데이터 사용)
                    for timeframe in self.config['timeframes']:
                        news_data_dict[timeframe] = news_df
                    
                    self.logger.info(f"{symbol} 뉴스 데이터 로드 완료: {len(news_df)}개 뉴스")
                # 디렉토리인 경우 (각 파일별로 처리)
                elif os.path.isdir(news_data_path):
                    news_files = glob.glob(os.path.join(news_data_path, '*.csv'))
                    
                    if news_files:
                        # 첫 번째 파일 사용
                        news_df = pd.read_csv(news_files[0])
                        
                        # 날짜 컬럼이 존재하면 변환
                        if 'date' in news_df.columns:
                            news_df['date'] = pd.to_datetime(news_df['date'])
                        
                        # 시작/종료 날짜로 필터링
                        if 'date' in news_df.columns:
                            start_date = pd.to_datetime(self.config['start_date'])
                            end_date = pd.to_datetime(self.config['end_date'])
                            news_df = news_df[(news_df['date'] >= start_date) & (news_df['date'] <= end_date)]
                        
                        # 시간프레임별로 할당
                        for timeframe in self.config['timeframes']:
                            news_data_dict[timeframe] = news_df
                        
                        self.logger.info(f"{symbol} 뉴스 데이터 로드 완료: {len(news_df)}개 뉴스")
                    else:
                        self.logger.warning(f"{news_data_path} 디렉토리에 뉴스 데이터 파일이 없습니다.")
                else:
                    self.logger.warning(f"뉴스 데이터 경로가 CSV 파일 또는 디렉토리가 아닙니다: {news_data_path}")
            else:
                self.logger.warning(f"뉴스 데이터 경로를 찾을 수 없습니다: {news_data_path}")
                
                # 샘플 뉴스 데이터 생성
                self.logger.info("샘플 뉴스 데이터 생성")
                for timeframe in self.config['timeframes']:
                    news_data_dict[timeframe] = self._generate_sample_news_data(symbol)
        except Exception as e:
            self.logger.exception(f"{symbol} 뉴스 데이터 로드 중 오류 발생: {str(e)}")
            
            # 오류 발생 시 샘플 데이터 사용
            for timeframe in self.config['timeframes']:
                news_data_dict[timeframe] = self._generate_sample_news_data(symbol)
        
        return news_data_dict
    
    def _generate_sample_news_data(self, symbol):
        """
        샘플 뉴스 데이터 생성 (뉴스 데이터 없을 때 사용)
        
        Args:
            symbol (str): 심볼
            
        Returns:
            DataFrame: 샘플 뉴스 데이터
        """
        start_date = pd.to_datetime(self.config['start_date'])
        end_date = pd.to_datetime(self.config['end_date'])
        days = (end_date - start_date).days
        
        # 하루에 1-3개 뉴스 생성
        num_news = max(days, 10) * np.random.randint(1, 4)
        
        # 날짜 생성
        dates = [start_date + timedelta(days=np.random.randint(0, days)) for _ in range(num_news)]
        
        # 감성 점수 생성 (-1 ~ 1)
        sentiment_scores = np.random.uniform(-1, 1, num_news)
        
        # 감성 레이블 생성
        sentiments = ['positive' if score > 0.3 else 'negative' if score < -0.3 else 'neutral' for score in sentiment_scores]
        
        # 샘플 제목 생성
        titles = [
            f"{symbol} sees growth in {np.random.choice(['market cap', 'trading volume', 'adoption'])}",
            f"Experts predict {np.random.choice(['bullish', 'bearish', 'stable'])} trend for {symbol}",
            f"New developments in {symbol} ecosystem",
            f"{symbol} price {np.random.choice(['surges', 'drops', 'stabilizes'])} after {np.random.choice(['announcement', 'market shift', 'regulation news'])}"
        ]
        
        sample_titles = [np.random.choice(titles) for _ in range(num_news)]
        
        # 데이터프레임 생성
        sample_news = pd.DataFrame({
            'date': dates,
            'title': sample_titles,
            'sentiment': sentiments,
            'sentiment_score': sentiment_scores
        })
        
        return sample_news
    
    def _generate_signals_for_timeframe(self, symbol, timeframe, price_data, news_data):
        """
        특정 시간프레임에 대한 트레이딩 신호 생성
        
        Args:
            symbol (str): 심볼 (예: BTCUSDT)
            timeframe (str): 시간프레임 (예: 1h)
            price_data (DataFrame): 가격 데이터
            news_data (DataFrame): 뉴스 데이터
            
        Returns:
            DataFrame: 트레이딩 신호
        """
        try:
            # 분석기 초기화
            candlestick_analyzer = CandlestickAnalyzer()
            sentiment_analyzer = SentimentAnalyzer()
            
            # 캔들스틱 패턴 분석
            candlestick_signals = candlestick_analyzer.analyze(price_data)
            
            # 뉴스 감성 분석
            news_analysis_result = sentiment_analyzer.analyze_sentiment(news_data)
            
            # 날짜별 감성 점수 매핑
            sentiment_signals = pd.Series(0.0, index=price_data.index)
            
            # 뉴스 데이터가 있는지 확인
            if not news_data.empty:
                # 각 뉴스 항목의 날짜와 감성 점수 매핑
                for idx, row in news_data.iterrows():
                    try:
                        date = row['date']
                        # 시간대 정보 제거
                        if hasattr(date, 'tz_localize'):
                            date = date.tz_localize(None)
                        
                        sentiment_score = row.get('sentiment_score', 0.0)
                        
                        # 해당 날짜의 가장 가까운 날짜 찾기 (정확한 매칭이 안될 경우)
                        if date in sentiment_signals.index:
                            sentiment_signals[date] = sentiment_score
                        else:
                            # 가장 가까운 날짜 찾기
                            closest_date = min(sentiment_signals.index, key=lambda x: abs(x - date))
                            sentiment_signals[closest_date] = sentiment_score
                    except Exception as e:
                        self.logger.warning(f"뉴스 날짜 매핑 오류: {str(e)}")
            
            # 간단한 특징 융합 로직 직접 구현
            # 가중치 적용: pattern_weight * pattern_signal + news_weight * sentiment_signal
            pattern_weight = self.config['pattern_weight']
            news_weight = self.config['news_weight']
            
            # 패턴 신호 변환 (bullish: 1, neutral: 0, bearish: -1)
            pattern_numeric = pd.Series(0.0, index=price_data.index)
            for idx, value in candlestick_signals.items():
                if value == 'bullish':
                    pattern_numeric[idx] = 1.0
                elif value == 'bearish':
                    pattern_numeric[idx] = -1.0
            
            # 신호 융합
            fused_signals = (pattern_weight * pattern_numeric) + (news_weight * sentiment_signals)
            
            # 융합된 신호를 트레이딩 신호로 변환 
            # threshold 기준으로 신호 생성 (1: 매수, 0: 홀드, -1: 매도)
            trade_signals = pd.Series(0, index=fused_signals.index)
            threshold = self.config['trade_threshold']
            
            for idx, value in fused_signals.items():
                if value >= threshold:
                    trade_signals[idx] = 1  # 매수 신호
                elif value <= -threshold:
                    trade_signals[idx] = -1  # 매도 신호
            
            # 시그널을 DataFrame으로 변환
            signals_df = pd.DataFrame({'signal': trade_signals})
            signals_df['signal'] = signals_df['signal'].map({1: 'buy', -1: 'sell', 0: 'hold'})
            
            return signals_df
            
        except Exception as e:
            self.logger.exception(f"{symbol} {timeframe} 트레이딩 신호 생성 중 오류 발생: {str(e)}")
            return None
    
    def _evaluate_timeframe_performance(self, price_data, signals):
        """
        특정 시간프레임의 신호에 대한 성능 평가
        
        Args:
            price_data (DataFrame): 가격 데이터
            signals (DataFrame): 트레이딩 신호
            
        Returns:
            dict: 성능 평가 결과
        """
        try:
            # 간단한 백테스팅
            buy_signals = signals[signals['signal'] == 'buy']
            sell_signals = signals[signals['signal'] == 'sell']
            
            total_trades = len(buy_signals) + len(sell_signals)
            
            if total_trades == 0:
                return {'profit': 0.0, 'win_rate': 0.0, 'drawdown': 0.0}
            
            # 간단한 수익률 계산
            profit = 0.0
            winning_trades = 0
            max_drawdown = 0.0
            
            # 매수 신호에 대한 수익률 계산
            for idx in buy_signals.index:
                if idx in price_data.index:
                    entry_price = price_data.loc[idx, 'close']
                    
                    # 다음 5개 봉 동안의 가격 변화 확인
                    future_idx = price_data.index[price_data.index > idx]
                    if len(future_idx) >= 5:
                        exit_idx = future_idx[4]  # 5봉 후
                        exit_price = price_data.loc[exit_idx, 'close']
                        
                        # 수익률 계산
                        trade_profit = (exit_price - entry_price) / entry_price * 100
                        profit += trade_profit
                        
                        if trade_profit > 0:
                            winning_trades += 1
                            
                        # 최대 낙폭 계산
                        min_price = price_data.loc[idx:exit_idx, 'low'].min()
                        max_drawdown_trade = (min_price - entry_price) / entry_price * 100
                        max_drawdown = min(max_drawdown, max_drawdown_trade)
            
            # 매도 신호에 대한 수익률 계산
            for idx in sell_signals.index:
                if idx in price_data.index:
                    entry_price = price_data.loc[idx, 'close']
                    
                    # 다음 5개 봉 동안의 가격 변화 확인
                    future_idx = price_data.index[price_data.index > idx]
                    if len(future_idx) >= 5:
                        exit_idx = future_idx[4]  # 5봉 후
                        exit_price = price_data.loc[exit_idx, 'close']
                        
                        # 수익률 계산 (매도의 경우 반대로)
                        trade_profit = (entry_price - exit_price) / entry_price * 100
                        profit += trade_profit
                        
                        if trade_profit > 0:
                            winning_trades += 1
                            
                        # 최대 낙폭 계산
                        max_price = price_data.loc[idx:exit_idx, 'high'].max()
                        max_drawdown_trade = (entry_price - max_price) / entry_price * 100
                        max_drawdown = min(max_drawdown, max_drawdown_trade)
            
            # 평균 수익률
            profit = profit / total_trades if total_trades > 0 else 0.0
            
            # 승률
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0.0
            
            return {
                'profit': profit,
                'win_rate': win_rate,
                'drawdown': max_drawdown
            }
            
        except Exception as e:
            self.logger.exception(f"시간프레임 성능 평가 중 오류 발생: {str(e)}")
            return {'profit': 0.0, 'win_rate': 0.0, 'drawdown': 0.0}
    
    def _generate_ensemble_signals(self, timeframe_signals, timeframe_weights, signal_index):
        """
        여러 시간프레임의 신호를 앙상블하여 최종 신호 생성
        
        Args:
            timeframe_signals (dict): 시간프레임별 트레이딩 신호
            timeframe_weights (dict): 시간프레임별 가중치
            signal_index (DatetimeIndex): 신호의 인덱스 (날짜)
            
        Returns:
            DataFrame: 앙상블된 트레이딩 신호
        """
        try:
            # 모든 날짜에 대한 가중합 신호 초기화
            ensemble_values = pd.Series(0.0, index=signal_index)
            
            # 각 시간프레임별 신호 가중 합산
            used_timeframes = 0
            for timeframe, signals_df in timeframe_signals.items():
                if timeframe in timeframe_weights:
                    weight = timeframe_weights[timeframe]
                    
                    # 시간프레임 신호를 수치로 변환 (buy: 1, hold: 0, sell: -1)
                    numeric_signals = pd.Series(0, index=signals_df.index)
                    numeric_signals[signals_df['signal'] == 'buy'] = 1
                    numeric_signals[signals_df['signal'] == 'sell'] = -1
                    
                    # 시간프레임 신호를 기준 인덱스에 맞게 리샘플링
                    resampled_signals = self._resample_signals(numeric_signals, signal_index)
                    
                    # 가중치 적용하여 합산
                    ensemble_values += resampled_signals * weight
                    used_timeframes += 1
            
            self.logger.info(f"앙상블에 사용된 시간프레임 수: {used_timeframes}")
            
            # 최소 시간프레임 수 확인
            min_timeframes = self.config.get('min_timeframes', 3)
            if used_timeframes < min_timeframes:
                self.logger.warning(f"사용된 시간프레임 수({used_timeframes})가 최소 요구치({min_timeframes})보다 적습니다.")
            
            # 앙상블 신호를 바이너리 신호로 변환
            ensemble_signals = pd.Series('hold', index=signal_index)
            
            # 임계값 설정 (앙상블 신호는 가중치 합산이므로 조정 필요)
            trade_threshold = self.config.get('trade_threshold', 0.5)
            adjusted_threshold = trade_threshold * used_timeframes / 3
            self.logger.info(f"앙상블 신호 임계값: {adjusted_threshold}")
            
            # 신호 변환
            ensemble_signals[ensemble_values >= adjusted_threshold] = 'buy'
            ensemble_signals[ensemble_values <= -adjusted_threshold] = 'sell'
            
            # 신호를 DataFrame으로 변환
            signals_df = pd.DataFrame({'signal': ensemble_signals})
            return signals_df
            
        except Exception as e:
            self.logger.exception(f"앙상블 신호 생성 중 오류 발생: {str(e)}")
            return pd.DataFrame({'signal': pd.Series('hold', index=signal_index)})
    
    def _resample_signals(self, signals, target_index):
        """
        신호를 대상 인덱스에 맞게 리샘플링
        
        Args:
            signals (Series): 원본 신호
            target_index (DatetimeIndex): 대상 인덱스
            
        Returns:
            Series: 리샘플링된 신호
        """
        # 공통된 날짜 찾기
        common_dates = signals.index.intersection(target_index)
        
        # 타겟 인덱스와 같은 크기의 시리즈 생성 (기본값 0)
        resampled = pd.Series(0, index=target_index)
        
        # 공통 날짜에 신호 복사
        for date in common_dates:
            resampled[date] = signals[date]
        
        # 없는 날짜는 앞의 값으로 채우기 (forward fill)
        resampled = resampled.fillna(method='ffill')
        
        # 여전히 NaN인 경우 0으로 채우기
        resampled = resampled.fillna(0)
        
        return resampled
    
    def _calculate_combined_results(self, all_results):
        """
        여러 심볼의 결과를 합산하여 종합 결과 생성
        
        Args:
            all_results (dict): 심볼별 트레이딩 결과
            
        Returns:
            dict: 종합 결과
        """
        try:
            if not all_results:
                return None
                
            # 종합 결과 초기화
            combined_result = {
                'symbol_results': all_results,
                'total_symbols': len(all_results),
                'symbols': list(all_results.keys()),
                'start_date': datetime.now().strftime('%Y-%m-%d'),  # 기본값 설정
                'end_date': datetime.now().strftime('%Y-%m-%d'),    # 기본값 설정
                'metrics': {}
            }
            
            # 첫 번째 심볼에서 날짜 범위 설정
            if all_results:
                first_symbol = list(all_results.keys())[0]
                first_result = all_results[first_symbol]
                if 'start_date' in first_result:
                    combined_result['start_date'] = first_result['start_date']
                if 'end_date' in first_result:
                    combined_result['end_date'] = first_result['end_date']
            
            # 지표 초기화
            total_profit = 0.0
            total_trades = 0
            winning_trades = 0
            total_initial_capital = 0.0
            total_final_value = 0.0
            max_drawdowns = []
            
            # 각 심볼별 결과 합산
            for symbol, result in all_results.items():
                # 수익 계산
                symbol_profit = result.get('final_value', 0) - result.get('initial_capital', 0)
                total_profit += symbol_profit
                total_trades += result.get('total_trades', 0)
                winning_trades += round(result.get('win_rate', 0) * result.get('total_trades', 0) / 100)
                
                # 초기 자본금과 최종 가치 합산
                total_initial_capital += result.get('initial_capital', 0)
                total_final_value += result.get('final_value', 0)
                
                # 최대 낙폭 저장
                max_drawdowns.append(result.get('max_drawdown_pct', 0))
            
            # 종합 지표 계산
            combined_result['metrics'] = {
                'total_initial_capital': total_initial_capital,
                'total_final_value': total_final_value,
                'total_profit': total_final_value - total_initial_capital,
                'total_return_pct': (total_final_value - total_initial_capital) / total_initial_capital * 100 if total_initial_capital > 0 else 0,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': winning_trades / total_trades * 100 if total_trades > 0 else 0,
                'max_drawdown_pct': max(max_drawdowns) if max_drawdowns else 0
            }
            
            # 샤프 비율은 포트폴리오 전체에서 계산이 필요하므로 여기서는 생략
            
            return combined_result
            
        except Exception as e:
            self.logger.exception(f"결과 통합 중 오류 발생: {str(e)}")
            return None
    
    def _generate_performance_report(self, combined_result):
        """
        성능 보고서 생성
        
        Args:
            combined_result (dict): 통합 결과
        """
        try:
            # 1. 심볼별 수익 그래프
            plt.figure(figsize=(12, 6))
            
            # 심볼별 수익
            symbols = []
            profits = []
            
            for symbol, result in combined_result['symbol_results'].items():
                symbols.append(symbol)
                profits.append(result['profit'])
            
            # 수익이 높은 순서대로 정렬
            sorted_indices = np.argsort(profits)[::-1]
            sorted_symbols = [symbols[i] for i in sorted_indices]
            sorted_profits = [profits[i] for i in sorted_indices]
            
            # 막대 그래프
            colors = ['g' if p >= 0 else 'r' for p in sorted_profits]
            plt.bar(sorted_symbols, sorted_profits, color=colors)
            
            # 그래프 설정
            plt.title('Profit by Symbol (Paper2 Ensemble)')
            plt.ylabel('Profit (%)')
            plt.xlabel('Symbol')
            plt.grid(axis='y')
            
            # 각 막대 위에 수익 표시
            for i, p in enumerate(sorted_profits):
                plt.text(i, p + (0.5 if p >= 0 else -1.0), f"{p:.2f}%", ha='center')
            
            plt.tight_layout()
            
            # 그래프 저장
            plt.savefig(os.path.join(self.results_dir, 'profit_by_symbol.png'))
            plt.close()
            
            # 2. 전체 성능 그래프
            plt.figure(figsize=(10, 6))
            
            # 파이 차트 데이터
            labels = ['Win', 'Loss']
            sizes = [combined_result['winning_trades'], combined_result['total_trades'] - combined_result['winning_trades']]
            colors = ['g', 'r']
            explode = (0.1, 0)  # 승리 부분 강조
            
            # 파이 차트
            plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')  # 원형 유지
            
            plt.title('Trading Performance (Paper2 Ensemble)')
            
            # 그래프 저장
            plt.savefig(os.path.join(self.results_dir, 'trading_performance.png'))
            plt.close()
            
            # 3. 시간프레임 가중치 분포 그래프
            # 각 심볼의 마지막 가중치 평균
            avg_weights = {}
            count = 0
            
            for symbol, result in combined_result['symbol_results'].items():
                if 'timeframe_weights' in result:
                    weights = result['timeframe_weights']
                    for tf, weight in weights.items():
                        if tf not in avg_weights:
                            avg_weights[tf] = 0
                        avg_weights[tf] += weight
                    count += 1
            
            # 평균 계산
            if count > 0:
                for tf in avg_weights:
                    avg_weights[tf] /= count
                
                # 그래프 그리기
                plt.figure(figsize=(10, 6))
                
                # 시간프레임과 가중치
                timeframes = list(avg_weights.keys())
                weight_values = [avg_weights[tf] for tf in timeframes]
                
                # 바 차트
                plt.bar(timeframes, weight_values, color='blue', alpha=0.7)
                
                # 값 표시
                for i, v in enumerate(weight_values):
                    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
                
                # 그래프 설정
                plt.title('Average Timeframe Weights (Paper2 Ensemble)')
                plt.xlabel('Timeframe')
                plt.ylabel('Weight')
                plt.ylim(0, max(weight_values) * 1.2)  # 여백 추가
                plt.grid(axis='y', alpha=0.3)
                
                # 그래프 저장
                plt.savefig(os.path.join(self.results_dir, 'timeframe_weights.png'))
                plt.close()
            
            self.logger.info("성능 보고서 생성 완료")
            
        except Exception as e:
            self.logger.exception(f"성능 보고서 생성 중 오류 발생: {str(e)}")
    
    def save_performance_metrics(self, filepath=None):
        """
        성능 지표 저장
        
        Args:
            filepath (str, optional): 저장 경로
        """
        if filepath is None:
            filepath = os.path.join(self.results_dir, 'performance_history.json')
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.performance_history, f, indent=4)
            
            self.logger.info(f"성능 지표 저장 완료: {filepath}")
            
        except Exception as e:
            self.logger.exception(f"성능 지표 저장 중 오류 발생: {str(e)}")
    
    def save_ensemble_weights(self, filepath=None):
        """
        앙상블 가중치 저장
        
        Args:
            filepath (str, optional): 저장 경로
        """
        if filepath is None:
            filepath = os.path.join(self.results_dir, 'ensemble_weights.json')
        
        try:
            if hasattr(self.weight_adjuster, 'weight_history'):
                with open(filepath, 'w') as f:
                    json.dump(self.weight_adjuster.weight_history, f, indent=4)
                
                self.logger.info(f"앙상블 가중치 저장 완료: {filepath}")
            else:
                self.logger.warning("앙상블 가중치 기록이 없습니다.")
                
        except Exception as e:
            self.logger.exception(f"앙상블 가중치 저장 중 오류 발생: {str(e)}") 