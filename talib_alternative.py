"""
TA-Lib 함수의 간단한 대체 구현체
ATR, ADX, MACD 등의 기술 지표 계산
"""

import numpy as np
import pandas as pd

def ATR(high, low, close, timeperiod=14):
    """
    ATR(Average True Range) 계산
    
    Args:
        high (array): 고가 배열
        low (array): 저가 배열
        close (array): 종가 배열
        timeperiod (int): 기간
        
    Returns:
        array: ATR 값
    """
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    
    # True Range 계산
    tr1 = high - low  # 당일 고가 - 당일 저가
    tr2 = np.abs(high - np.roll(close, 1))  # 당일 고가 - 전일 종가
    tr3 = np.abs(low - np.roll(close, 1))  # 당일 저가 - 전일 종가
    
    # 첫번째 값은 NaN으로 처리
    tr2[0] = 0
    tr3[0] = 0
    
    # True Range는 위 세 값 중 최대값
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # ATR 계산 (이동평균)
    atr = np.zeros_like(tr)
    atr[0:timeperiod] = np.nan  # 처음 timeperiod 개는 NaN
    
    # 첫번째 ATR은 True Range의 평균
    if len(tr) >= timeperiod:
        atr[timeperiod-1] = np.mean(tr[0:timeperiod])
        
        # 나머지 ATR은 이전 ATR의 가중치 적용
        for i in range(timeperiod, len(tr)):
            atr[i] = (atr[i-1] * (timeperiod-1) + tr[i]) / timeperiod
    
    return atr

def ADX(high, low, close, timeperiod=14):
    """
    ADX(Average Directional Index) 계산
    
    Args:
        high (array): 고가 배열
        low (array): 저가 배열
        close (array): 종가 배열
        timeperiod (int): 기간
        
    Returns:
        array: ADX 값
    """
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    
    # True Range 계산 (ATR과 동일)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    
    # 첫번째 값은 0으로 처리
    tr2[0] = 0
    tr3[0] = 0
    
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # +DM과 -DM 계산
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    
    # 첫번째 값은 0으로 처리
    up_move[0] = 0
    down_move[0] = 0
    
    # +DM: 상승폭이 하락폭보다 크고, 상승폭이 0보다 클 때
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    
    # -DM: 하락폭이 상승폭보다 크고, 하락폭이 0보다 클 때
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # +DI와 -DI 계산
    plus_di = np.zeros_like(plus_dm)
    minus_di = np.zeros_like(minus_dm)
    
    # 처음 timeperiod 개는 NaN
    plus_di[0:timeperiod] = np.nan
    minus_di[0:timeperiod] = np.nan
    
    # 첫번째 +DI, -DI 계산
    if len(plus_dm) >= timeperiod:
        plus_di[timeperiod-1] = 100 * np.sum(plus_dm[0:timeperiod]) / np.sum(tr[0:timeperiod])
        minus_di[timeperiod-1] = 100 * np.sum(minus_dm[0:timeperiod]) / np.sum(tr[0:timeperiod])
        
        # 나머지 +DI, -DI 계산
        for i in range(timeperiod, len(plus_dm)):
            plus_di[i] = 100 * ((plus_di[i-1] * (timeperiod-1) / 100) + plus_dm[i]) / ((tr[i-1] * (timeperiod-1) / 100) + tr[i])
            minus_di[i] = 100 * ((minus_di[i-1] * (timeperiod-1) / 100) + minus_dm[i]) / ((tr[i-1] * (timeperiod-1) / 100) + tr[i])
    
    # DX 계산: |+DI - -DI| / (+DI + -DI) * 100
    dx = np.zeros_like(plus_di)
    for i in range(timeperiod-1, len(dx)):
        if plus_di[i] + minus_di[i] > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        else:
            dx[i] = 0
    
    # ADX 계산: DX의 이동평균
    adx = np.zeros_like(dx)
    adx[0:timeperiod*2-1] = np.nan  # 처음 2*timeperiod-1 개는 NaN
    
    # 첫번째 ADX 계산
    if len(dx) >= timeperiod*2-1:
        adx[timeperiod*2-2] = np.mean(dx[timeperiod-1:timeperiod*2-1])
        
        # 나머지 ADX 계산
        for i in range(timeperiod*2-1, len(dx)):
            adx[i] = (adx[i-1] * (timeperiod-1) + dx[i]) / timeperiod
    
    return adx

def MACD(close, fastperiod=12, slowperiod=26, signalperiod=9):
    """
    MACD(Moving Average Convergence Divergence) 계산
    
    Args:
        close (array): 종가 배열
        fastperiod (int): 빠른 이동평균 기간
        slowperiod (int): 느린 이동평균 기간
        signalperiod (int): 신호선 기간
        
    Returns:
        tuple: MACD, Signal, Histogram
    """
    close = np.array(close)
    
    # EMA 계산 함수
    def ema(data, period):
        alpha = 2 / (period + 1)
        ema_values = np.zeros_like(data)
        ema_values[0:period-1] = np.nan  # 처음 period-1 개는 NaN
        
        # 첫번째 EMA는 SMA로 계산
        if len(data) >= period:
            ema_values[period-1] = np.mean(data[0:period])
            
            # 나머지 EMA 계산
            for i in range(period, len(data)):
                ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i-1]
        
        return ema_values
    
    # 빠른 EMA와 느린 EMA 계산
    fast_ema = ema(close, fastperiod)
    slow_ema = ema(close, slowperiod)
    
    # MACD 계산
    macd = fast_ema - slow_ema
    
    # Signal 계산
    signal = ema(macd, signalperiod)
    
    # Histogram 계산
    hist = macd - signal
    
    return macd, signal, hist 