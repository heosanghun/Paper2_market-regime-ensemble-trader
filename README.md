# Paper 2: Market Regime Dynamic Control Ensemble Trading System

## Introduction
This project implements an advanced trading system that combines market regime detection (bull/bear/sideways) with dynamic timeframe weighting and ensemble strategies. It builds upon the multimodal approach (image + news) from Paper 1 to create an adaptive system that responds to changing market conditions.

## Directory Structure
```
paper2/
├── __init__.py                          # Package initialization file
├── ensemble_controller.py               # Ensemble controller implementation
├── dynamic_weight_adjuster.py           # Dynamic weight adjustment for timeframes
├── market_state_detector.py             # Market regime detection module
├── run_paper2_ensemble.py               # Main execution script 
├── run_comparative_analysis.py          # Comparative analysis with Paper 1
├── comparative_visualizer.py            # Visualization tools for comparison
├── visualization_utils.py               # Visualization utilities
├── talib_alternative.py                 # Technical indicators without TA-Lib dependency
├── data/                                # Data directory
│   ├── charts/                          # Chart images (if needed)
│   └── example_data.csv                 # Example data file
├── docs/                                # Documentation
└── results/                             # Results directory
    └── run_YYYYMMDD_HHMMSS/             # Individual run results
        ├── portfolio_history.csv        # Portfolio value history
        ├── performance_metrics.json     # Performance metrics
        ├── strategy_weights.json        # Strategy weights over time
        └── market_regime_history.csv    # Detected market regimes
```

## Core Components

### Market State Detector
Analyzes price patterns, volatility, and volume to determine the current market state (bullish, bearish, sideways, or volatile).

### Dynamic Weight Adjuster
Adjusts weights for different timeframes and strategies based on their recent performance and the current market regime.

### Ensemble Controller
Combines signals from multiple trading strategies, with weights dynamically adjusted based on market conditions.

## Features
- **Market Regime Detection**: Automatic identification of market states
- **Dynamic Timeframe Weighting**: Adjusts importance of different timeframes
- **Adaptive Strategy Selection**: Selects optimal strategies for current market conditions
- **Performance-Based Adjustment**: Learns from recent trading performance
- **Comparative Analysis**: Tools to compare with Paper 1 results

## Data Download

프로젝트 실행에 필요한 시장 데이터는 다음 Google Drive 링크에서 다운로드할 수 있습니다:
- **Google Drive**: [https://drive.google.com/drive/folders/1vHxKgrkjguXfgmIOUWqbbSdD1XXDnbSK?usp=sharing](https://drive.google.com/drive/folders/1vHxKgrkjguXfgmIOUWqbbSdD1XXDnbSK?usp=sharing)
- 캔들 차트 이미지(224X224) 데이터 용량: 8.19GB/369,456장 | 2021-10-12 ~ 2023-12-19
- 암호화폐 뉴스 기사(감성분석) 데이터 용량: 12.6MB/31,038개 |  2021-10-12 ~ 2023-12-19
다운로드한 파일들을 `data/` 디렉토리에 위치시키세요. 데이터셋에는 다음이 포함됩니다:
- 다양한 시장 상태(강세/약세/횡보)의 가격 데이터
- 시장 상태 분석용 참조 데이터
- 앙상블 테스트를 위한 전처리된 데이터
- 성능 비교를 위한 벤치마크 데이터

## Installation
1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
Run the main ensemble simulation:
```
python run_paper2_ensemble.py
```

Run comparative analysis with Paper 1:
```
python run_comparative_analysis.py
```

## Notes
- Detailed documentation available in the docs/ folder
- Data and results are excluded from Git using .gitignore
- This system requires Paper 1 components to be available in the parent directory
