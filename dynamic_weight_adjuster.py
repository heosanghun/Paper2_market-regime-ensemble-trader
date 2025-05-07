import pandas as pd
import numpy as np
import json
import os
import logging
import matplotlib.pyplot as plt
from datetime import datetime

class DynamicWeightAdjuster:
    """
    동적 가중치 조정기
    - 시장 상태에 따라 다양한 시간프레임 신호의 가중치를 동적으로 조정
    - 추세, 횡보, 변동성이 높은 시장 상태에 최적화된 가중치 설정
    """
    
    def __init__(self, config=None, results_dir=None):
        self.config = config or self._get_default_config()
        self.results_dir = results_dir or "results_weight_adjuster"
        self.logger = logging.getLogger('DynamicWeightAdjuster')
        
        # 결과 디렉토리 생성
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 가중치 기록
        self.weight_history = {
            'timestamp': [],
            'market_state': [],
            'weights': []
        }
    
    def _get_default_config(self):
        """기본 설정 반환"""
        return {
            # 시간프레임 기본 가중치 (합이 1이 되도록 설정)
            'default_weights': {
                '5m': 0.10,   # 5분봉
                '15m': 0.15,  # 15분봉
                '1h': 0.30,   # 1시간봉
                '4h': 0.25,   # 4시간봉
                '1d': 0.15,   # 일봉
                '1w': 0.05    # 주봉
            },
            # 추세 시장 가중치
            'trending_weights': {
                '5m': 0.05,   # 5분봉 (낮은 가중치)
                '15m': 0.10,  # 15분봉 
                '1h': 0.20,   # 1시간봉
                '4h': 0.30,   # 4시간봉 (높은 가중치)
                '1d': 0.25,   # 일봉 (높은 가중치)
                '1w': 0.10    # 주봉
            },
            # 횡보 시장 가중치
            'ranging_weights': {
                '5m': 0.20,   # 5분봉 (높은 가중치)
                '15m': 0.25,  # 15분봉 (높은 가중치)
                '1h': 0.30,   # 1시간봉 (높은 가중치)
                '4h': 0.15,   # 4시간봉
                '1d': 0.07,   # 일봉 (낮은 가중치)
                '1w': 0.03    # 주봉 (낮은 가중치)
            },
            # 고변동성 시장 가중치
            'volatile_weights': {
                '5m': 0.05,   # 5분봉 (낮은 가중치 - 노이즈가 많음)
                '15m': 0.10,  # 15분봉
                '1h': 0.25,   # 1시간봉
                '4h': 0.30,   # 4시간봉 (높은 가중치)
                '1d': 0.20,   # 일봉
                '1w': 0.10    # 주봉
            },
            # 상승 추세 시장 가중치
            'uptrend_weights': {
                '5m': 0.08,   # 5분봉
                '15m': 0.12,  # 15분봉
                '1h': 0.25,   # 1시간봉
                '4h': 0.30,   # 4시간봉 (높은 가중치)
                '1d': 0.20,   # 일봉
                '1w': 0.05    # 주봉
            },
            # 하락 추세 시장 가중치
            'downtrend_weights': {
                '5m': 0.05,   # 5분봉 (낮은 가중치)
                '15m': 0.10,  # 15분봉
                '1h': 0.20,   # 1시간봉
                '4h': 0.35,   # 4시간봉 (높은 가중치)
                '1d': 0.20,   # 일봉
                '1w': 0.10    # 주봉
            },
            # 학습률 (가중치 조정 정도)
            'learning_rate': 0.2,
            
            # 성과 기반 가중치 조정 설정
            'performance_weight': 0.3,  # 성과 기반 조정 가중치
            'window_size': 10,          # 성과 평가 윈도우
            
            # 평가 메트릭
            'metrics': {
                'profit': 0.6,          # 수익률 가중치
                'win_rate': 0.2,        # 승률 가중치
                'drawdown': 0.2,        # 낙폭 가중치
            }
        }
    
    def adjust_weights(self, market_state, performance_data=None):
        """
        시장 상태에 따라 가중치 조정
        
        Args:
            market_state (dict): 시장 상태 정보
                {'state': 'trending'|'ranging'|'volatile'|'uptrend'|'downtrend', 
                 'confidence': 0.0-1.0}
            performance_data (dict, optional): 시간프레임별 성과 데이터
                {'5m': {'profit': 0.0, 'win_rate': 0.0, 'drawdown': 0.0}, ...}
                
        Returns:
            dict: 조정된 가중치
        """
        state = market_state.get('state', 'ranging')
        confidence = market_state.get('confidence', 0.5)
        
        # 기본 가중치 복사
        adjusted_weights = self.config['default_weights'].copy()
        
        try:
            # 1. 시장 상태에 따른 가중치 선택
            # 필요한 가중치 설정이 config에 없으면 default config에서 가져옴
            default_config = self._get_default_config()
            
            if state == 'uptrend':
                target_weights = self.config.get('uptrend_weights', default_config.get('uptrend_weights', adjusted_weights))
            elif state == 'downtrend':
                target_weights = self.config.get('downtrend_weights', default_config.get('downtrend_weights', adjusted_weights))
            elif state == 'volatile':
                target_weights = self.config.get('volatile_weights', default_config.get('volatile_weights', adjusted_weights))
            elif state == 'trending':
                target_weights = self.config.get('trending_weights', default_config.get('trending_weights', adjusted_weights))
            else:  # ranging 또는 unknown
                target_weights = self.config.get('ranging_weights', default_config.get('ranging_weights', adjusted_weights))
            
            # 2. 현재 가중치와 목표 가중치 사이의 보간
            # 신뢰도에 따라 학습률 조정
            effective_lr = self.config.get('learning_rate', default_config.get('learning_rate', 0.2)) * confidence
            
            # 각 시간프레임에 대해 가중치 보간
            for timeframe in adjusted_weights.keys():
                current_weight = adjusted_weights[timeframe]
                target_weight = target_weights[timeframe]
                
                # 선형 보간: current + lr * (target - current)
                adjusted_weights[timeframe] = current_weight + effective_lr * (target_weight - current_weight)
            
            # 3. 성과 기반 가중치 조정 (optional)
            if performance_data is not None:
                self._adjust_by_performance(adjusted_weights, performance_data)
            
            # 4. 가중치 정규화 (합이 1이 되도록)
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                for timeframe in adjusted_weights:
                    adjusted_weights[timeframe] /= total_weight
            
            # 5. 가중치 기록
            self.weight_history['timestamp'].append(datetime.now().isoformat())
            self.weight_history['market_state'].append(state)
            self.weight_history['weights'].append(adjusted_weights.copy())
            
            # 6. 가중치 시각화 및 저장
            if len(self.weight_history['weights']) % 10 == 0:  # 10번째마다 시각화
                self._visualize_weights(adjusted_weights, state)
                self._save_weight_history()
            
            return adjusted_weights
            
        except Exception as e:
            self.logger.exception(f"가중치 조정 중 오류 발생: {str(e)}")
            # 오류 발생 시 기본 가중치 반환
            return self.config['default_weights']
    
    def _adjust_by_performance(self, adjusted_weights, performance_data):
        """
        성과 데이터 기반 가중치 미세 조정
        
        Args:
            adjusted_weights (dict): 조정할 가중치
            performance_data (dict): 시간프레임별 성과 데이터
        """
        if not performance_data:
            return
        
        # 성과 점수 계산 (수익률, 승률, 낙폭 기반)
        performance_scores = {}
        
        # 기본 메트릭 설정
        default_config = self._get_default_config()
        metrics = self.config.get('metrics', default_config.get('metrics', {
            'profit': 0.6,
            'win_rate': 0.2,
            'drawdown': 0.2
        }))
        
        for timeframe, perf in performance_data.items():
            # 각 메트릭에 대한 점수 계산
            profit_score = perf.get('profit', 0) * metrics.get('profit', 0.6)
            win_rate_score = perf.get('win_rate', 0) * metrics.get('win_rate', 0.2)
            # 낙폭은 낮을수록 좋음 (역수 사용)
            drawdown = max(0.01, abs(perf.get('drawdown', 0)))  # 0 방지
            drawdown_score = (1 / drawdown) * metrics.get('drawdown', 0.2)
            
            # 종합 점수
            performance_scores[timeframe] = profit_score + win_rate_score + drawdown_score
        
        # 점수 정규화
        total_score = sum(performance_scores.values())
        if total_score > 0:
            for timeframe in performance_scores:
                performance_scores[timeframe] /= total_score
        
        # 가중치 미세 조정
        for timeframe in adjusted_weights:
            if timeframe in performance_scores:
                # 성과 기반 조정 (성과 좋은 시간프레임에 더 높은 가중치)
                perf_adjustment = self.config['performance_weight'] * (performance_scores[timeframe] - adjusted_weights[timeframe])
                adjusted_weights[timeframe] += perf_adjustment
    
    def _visualize_weights(self, weights, market_state):
        """
        가중치 시각화
        
        Args:
            weights (dict): 시간프레임별 가중치
            market_state (str): 시장 상태
        """
        try:
            plt.figure(figsize=(10, 6))
            
            # 시간프레임과 가중치
            timeframes = list(weights.keys())
            weight_values = list(weights.values())
            
            # 바 차트 색상 (시장 상태에 따라)
            if market_state == 'uptrend':
                color = 'green'
            elif market_state == 'downtrend':
                color = 'red'
            elif market_state == 'volatile':
                color = 'orange'
            elif market_state == 'trending':
                color = 'blue'
            else:  # ranging 또는 unknown
                color = 'gray'
            
            # 바 차트 그리기
            plt.bar(timeframes, weight_values, color=color, alpha=0.7)
            
            # 값 표시
            for i, v in enumerate(weight_values):
                plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
            
            # 그래프 설정
            plt.title(f"Timeframe Weights for {market_state.upper()} Market")
            plt.xlabel("Timeframe")
            plt.ylabel("Weight")
            plt.ylim(0, max(weight_values) * 1.2)  # 여백 추가
            plt.grid(axis='y', alpha=0.3)
            
            # 그래프 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(self.results_dir, f'weights_{market_state}_{timestamp}.png'))
            plt.close()
            
        except Exception as e:
            self.logger.exception(f"가중치 시각화 중 오류 발생: {str(e)}")
    
    def _save_weight_history(self):
        """가중치 기록 저장"""
        try:
            # 최대 100개까지만 저장
            if len(self.weight_history['timestamp']) > 100:
                # 가장 오래된 항목 삭제
                self.weight_history['timestamp'] = self.weight_history['timestamp'][-100:]
                self.weight_history['market_state'] = self.weight_history['market_state'][-100:]
                self.weight_history['weights'] = self.weight_history['weights'][-100:]
            
            # JSON으로 저장
            with open(os.path.join(self.results_dir, 'weight_history.json'), 'w') as f:
                json.dump(self.weight_history, f, indent=4)
                
        except Exception as e:
            self.logger.exception(f"가중치 기록 저장 중 오류 발생: {str(e)}")
    
    def get_weight_history(self):
        """가중치 기록 반환"""
        return self.weight_history 