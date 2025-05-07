import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from datetime import datetime

# TA-Lib 대신 대체 구현 사용
try:
    import talib
    TALIB_INSTALLED = True
except ImportError:
    TALIB_INSTALLED = False
    from talib_alternative import ATR, ADX, MACD

class MarketStateDetector:
    """
    시장 상태 감지 클래스
    - 가격 변동성, 추세, 거래량을 분석하여 시장 상태를 감지
    - 상태: 'trending', 'ranging', 'volatile'
    """
    
    def __init__(self, config=None, results_dir=None):
        self.config = config or self._get_default_config()
        self.results_dir = results_dir or "results_market_state"
        self.logger = logging.getLogger('MarketStateDetector')
        
        # 결과 디렉토리 생성
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _get_default_config(self):
        """기본 설정 반환"""
        return {
            'volatility_window': 20,         # 변동성 계산 윈도우
            'volatility_threshold': 2.0,     # 변동성 임계값 (표준편차 배수)
            'trend_window': 50,              # 추세 감지 윈도우
            'trend_threshold': 0.03,         # 추세 임계값 (3%)
            'volume_window': 20,             # 거래량 윈도우
            'volume_threshold': 1.5          # 거래량 임계값 (평균 대비)
        }
    
    def detect_state(self, price_data, visualize=False):
        """
        시장 상태 감지
        
        Args:
            price_data (DataFrame): 가격 데이터 (OHLCV)
            visualize (bool): 시각화 여부
            
        Returns:
            dict: 감지된 시장 상태 정보
        """
        if price_data is None or len(price_data) < self.config['trend_window']:
            self.logger.warning(f"데이터가 충분하지 않습니다. 최소 {self.config['trend_window']}개 이상 필요합니다.")
            return {'state': 'unknown', 'confidence': 0.0}
        
        try:
            # 데이터 복사
            data = price_data.copy()
            
            # 1. 변동성 계산 (ATR 기반)
            if TALIB_INSTALLED:
                data['atr'] = talib.ATR(
                    data['high'].values, 
                    data['low'].values, 
                    data['close'].values, 
                    timeperiod=self.config['volatility_window']
                )
            else:
                data['atr'] = ATR(
                    data['high'].values, 
                    data['low'].values, 
                    data['close'].values, 
                    timeperiod=self.config['volatility_window']
                )
            
            # ATR의 현재 값과 과거 평균 대비 비율
            current_atr = data['atr'].iloc[-1]
            avg_atr = data['atr'].iloc[-self.config['volatility_window']:].mean()
            volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
            
            # 2. 추세 강도 계산 (ADX 기반)
            if TALIB_INSTALLED:
                data['adx'] = talib.ADX(
                    data['high'].values, 
                    data['low'].values, 
                    data['close'].values, 
                    timeperiod=self.config['trend_window']
                )
            else:
                data['adx'] = ADX(
                    data['high'].values, 
                    data['low'].values, 
                    data['close'].values, 
                    timeperiod=self.config['trend_window']
                )
            
            current_adx = data['adx'].iloc[-1]
            
            # 3. 추세 방향 (MACD 기반)
            if TALIB_INSTALLED:
                macd, signal, hist = talib.MACD(
                    data['close'].values,
                    fastperiod=12,
                    slowperiod=26,
                    signalperiod=9
                )
            else:
                macd, signal, hist = MACD(
                    data['close'].values,
                    fastperiod=12,
                    slowperiod=26,
                    signalperiod=9
                )
            
            data['macd'] = macd
            data['macd_signal'] = signal
            data['macd_hist'] = hist
            
            # 현재 MACD와 시그널 라인
            current_macd = data['macd'].iloc[-1]
            current_signal = data['macd_signal'].iloc[-1]
            
            # 추세 방향 (-1: 하락, 0: 중립, 1: 상승)
            if current_macd > current_signal:
                trend_direction = 1
            elif current_macd < current_signal:
                trend_direction = -1
            else:
                trend_direction = 0
            
            # 4. 거래량 분석
            # 거래량 이동평균
            data['volume_ma'] = data['volume'].rolling(window=self.config['volume_window']).mean()
            
            # 현재 거래량과 평균 거래량 비율
            current_volume = data['volume'].iloc[-1]
            avg_volume = data['volume_ma'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 5. 시장 상태 판단
            # 기본값은 'ranging' (횡보)
            market_state = 'ranging'
            confidence = 0.5
            
            # 높은 변동성 + 높은 거래량 = 'volatile' (고변동성)
            if volatility_ratio > self.config['volatility_threshold'] and volume_ratio > self.config['volume_threshold']:
                market_state = 'volatile'
                confidence = min(0.5 + volatility_ratio/10 + volume_ratio/10, 0.95)
            
            # 강한 추세 + 방향성 있는 MACD = 'trending' (추세)
            elif current_adx > 25 and abs(trend_direction) > 0:
                market_state = 'trending'
                confidence = min(0.5 + current_adx/100, 0.95)
                
                # 추세 방향 (상승/하락) 추가
                if trend_direction > 0:
                    market_state = 'uptrend'
                else:
                    market_state = 'downtrend'
            
            # 결과
            result = {
                'state': market_state,
                'confidence': confidence,
                'volatility': {
                    'atr': current_atr,
                    'ratio': volatility_ratio
                },
                'trend': {
                    'adx': current_adx,
                    'direction': trend_direction
                },
                'volume': {
                    'current': current_volume,
                    'ratio': volume_ratio
                }
            }
            
            # 시각화 (필요한 경우)
            if visualize and self.results_dir:
                self._visualize_market_state(data, result)
            
            return result
        
        except Exception as e:
            self.logger.exception(f"시장 상태 감지 중 오류 발생: {str(e)}")
            return {'state': 'unknown', 'confidence': 0.0}
    
    def _visualize_market_state(self, data, state_result):
        """
        시장 상태 시각화
        
        Args:
            data (DataFrame): 가격 데이터 (지표 포함)
            state_result (dict): 감지된 시장 상태 정보
        """
        try:
            # 플롯 생성
            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})
            
            # 1. 가격 차트
            axes[0].set_title(f"Market State: {state_result['state'].upper()} (Confidence: {state_result['confidence']:.2f})")
            axes[0].plot(data.index, data['close'], label='Close Price')
            
            # 배경색 설정 (시장 상태에 따라)
            market_state = state_result['state']
            if market_state == 'uptrend':
                axes[0].set_facecolor('rgba(0, 255, 0, 0.1)')  # 연한 녹색
            elif market_state == 'downtrend':
                axes[0].set_facecolor('rgba(255, 0, 0, 0.1)')  # 연한 빨간색
            elif market_state == 'volatile':
                axes[0].set_facecolor('rgba(255, 255, 0, 0.1)')  # 연한 노란색
            
            # 2. ATR (변동성)
            axes[1].set_title(f"ATR (Volatility) - Ratio: {state_result['volatility']['ratio']:.2f}")
            axes[1].plot(data.index, data['atr'], color='purple', label='ATR')
            axes[1].axhline(y=data['atr'].mean(), color='gray', linestyle='--', label='Avg ATR')
            
            # 3. ADX (추세 강도)
            axes[2].set_title(f"ADX (Trend Strength) - Value: {state_result['trend']['adx']:.2f}")
            axes[2].plot(data.index, data['adx'], color='blue', label='ADX')
            axes[2].axhline(y=25, color='red', linestyle='--', label='Threshold (25)')
            
            # 4. Volume
            axes[3].set_title(f"Volume - Ratio: {state_result['volume']['ratio']:.2f}")
            axes[3].bar(data.index, data['volume'], color='green', alpha=0.5)
            axes[3].plot(data.index, data['volume_ma'], color='red', label='Volume MA')
            
            # 레이블 및 그리드 설정
            for ax in axes:
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
            
            # 날짜 포맷 설정
            fig.autofmt_xdate()
            
            # 그래프 저장
            plt.tight_layout()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(self.results_dir, f'market_state_{timestamp}.png'))
            plt.close()
        
        except Exception as e:
            self.logger.exception(f"시장 상태 시각화 중 오류 발생: {str(e)}") 