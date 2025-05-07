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