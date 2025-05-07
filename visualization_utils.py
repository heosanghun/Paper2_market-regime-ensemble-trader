import os
import json
import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend to avoid Tk errors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns

def compare_performance_metrics(paper1_metrics, paper2_metrics, output_path):
    """
    Create a chart comparing performance metrics of Paper 1 and Paper 2
    
    Args:
        paper1_metrics (dict): Performance metrics for Paper 1
        paper2_metrics (dict): Performance metrics for Paper 2
        output_path (str): Path to save the result
    """
    # Extract key performance metrics
    metrics = ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'win_rate', 'profit_factor']
    
    # Create the graph
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 15))
    
    # Comparison chart for each metric
    for i, metric in enumerate(metrics):
        # Validation
        paper1_value = paper1_metrics.get(metric, 0)
        paper2_value = paper2_metrics.get(metric, 0)
        
        # Bar chart
        bars = axes[i].bar(['Paper 1', 'Paper 2'], [paper1_value, paper2_value], 
                        color=['#3498db', '#e74c3c'])
        
        # Display values
        for bar in bars:
            height = bar.get_height()
            axes[i].annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=12)
        
        # Titles and labels
        metric_labels = {
            'total_return_pct': 'Total Return (%)',
            'sharpe_ratio': 'Sharpe Ratio',
            'max_drawdown_pct': 'Max Drawdown (%)',
            'win_rate': 'Win Rate (%)',
            'profit_factor': 'Profit Factor'
        }
        
        axes[i].set_title(metric_labels.get(metric, metric), fontsize=14)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'papers_comparison.png'), dpi=300)
    plt.close()

def compare_equity_curves(paper1_equity, paper2_equity, output_path):
    """
    Create a chart comparing equity curves of Paper 1 and Paper 2
    
    Args:
        paper1_equity (DataFrame): Equity curve for Paper 1
        paper2_equity (DataFrame): Equity curve for Paper 2
        output_path (str): Path to save the result
    """
    plt.figure(figsize=(15, 8))
    
    # Date formatting
    plt.plot(paper1_equity.index, paper1_equity['portfolio_value'], 
             label='Paper 1', linewidth=2, color='#3498db')
    plt.plot(paper2_equity.index, paper2_equity['portfolio_value'], 
             label='Paper 2', linewidth=2, color='#e74c3c')
    
    # Initial capital horizontal line
    initial_capital = min(
        paper1_equity['portfolio_value'].iloc[0],
        paper2_equity['portfolio_value'].iloc[0]
    )
    plt.axhline(y=initial_capital, color='gray', linestyle='--', label='Initial Capital')
    
    plt.title('Portfolio Value Comparison', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Date format settings
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'equity_curve_comparison.png'), dpi=300)
    plt.close()

def create_comparative_analysis_report(paper1_results_dir, paper2_results_dir, output_dir):
    """
    Generate a comprehensive comparison report for Paper 1 and Paper 2
    
    Args:
        paper1_results_dir (str): Results directory for Paper 1
        paper2_results_dir (str): Results directory for Paper 2
        output_dir (str): Directory to save the results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load performance metrics
    try:
        with open(os.path.join(paper1_results_dir, 'performance_metrics.json'), 'r') as f:
            paper1_metrics = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Could not find or load performance metrics file for Paper 1")
        paper1_metrics = {}
    
    try:
        with open(os.path.join(paper2_results_dir, 'performance_metrics.json'), 'r') as f:
            paper2_metrics = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Could not find or load performance metrics file for Paper 2")
        paper2_metrics = {}
    
    # Load portfolio data
    try:
        paper1_equity = pd.read_csv(os.path.join(paper1_results_dir, 'portfolio_history.csv'))
        paper1_equity['timestamp'] = pd.to_datetime(paper1_equity['timestamp'])
        paper1_equity.set_index('timestamp', inplace=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Could not find or load portfolio history file for Paper 1")
        paper1_equity = pd.DataFrame({
            'timestamp': pd.date_range(start='2021-01-01', periods=10, freq='D'),
            'portfolio_value': np.linspace(10000, 10000, 10)
        }).set_index('timestamp')
    
    try:
        paper2_equity = pd.read_csv(os.path.join(paper2_results_dir, 'portfolio_history.csv'))
        paper2_equity['timestamp'] = pd.to_datetime(paper2_equity['timestamp'])
        paper2_equity.set_index('timestamp', inplace=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Could not find or load portfolio history file for Paper 2")
        paper2_equity = pd.DataFrame({
            'timestamp': pd.date_range(start='2021-01-01', periods=10, freq='D'),
            'portfolio_value': np.linspace(10000, 10000, 10)
        }).set_index('timestamp')
    
    # Create performance metrics comparison chart
    compare_performance_metrics(paper1_metrics, paper2_metrics, output_dir)
    
    # Create equity curve comparison chart
    compare_equity_curves(paper1_equity, paper2_equity, output_dir)
    
    # Monthly returns comparison chart
    try:
        # Calculate monthly returns
        def calculate_monthly_returns(equity_data):
            monthly_returns = equity_data['portfolio_value'].resample('M').last().pct_change() * 100
            monthly_returns = monthly_returns.fillna(0)
            return monthly_returns
        
        paper1_monthly = calculate_monthly_returns(paper1_equity)
        paper2_monthly = calculate_monthly_returns(paper2_equity)
        
        # Monthly returns chart
        plt.figure(figsize=(15, 8))
        
        # Common date range
        common_dates = paper1_monthly.index.intersection(paper2_monthly.index)
        
        # Create bar chart
        bar_width = 0.35
        indices = range(len(common_dates))
        
        plt.bar([i - bar_width/2 for i in indices], 
                paper1_monthly.loc[common_dates].values, 
                width=bar_width, label='Paper 1', color='#3498db', alpha=0.7)
        
        plt.bar([i + bar_width/2 for i in indices], 
                paper2_monthly.loc[common_dates].values, 
                width=bar_width, label='Paper 2', color='#e74c3c', alpha=0.7)
        
        # Monthly labels
        plt.xticks(indices, [d.strftime('%Y-%m') for d in common_dates], rotation=45)
        
        plt.title('Monthly Returns Comparison', fontsize=16)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Monthly Return (%)', fontsize=12)
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'monthly_returns_comparison.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating monthly returns comparison chart: {str(e)}")
    
    # Trade distribution comparison chart (win/loss trades)
    try:
        # Load trade data
        paper1_trades = pd.read_csv(os.path.join(paper1_results_dir, 'trades.csv'))
        paper2_trades = pd.read_csv(os.path.join(paper2_results_dir, 'trades.csv'))
        
        # Classify win/loss trades
        paper1_trades['profit'] = paper1_trades['exit_price'] - paper1_trades['entry_price']
        paper1_trades['is_win'] = paper1_trades['profit'] > 0
        
        paper2_trades['profit'] = paper2_trades['exit_price'] - paper2_trades['entry_price']
        paper2_trades['is_win'] = paper2_trades['profit'] > 0
        
        # Create pie charts
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Paper 1 pie chart
        paper1_win_count = paper1_trades['is_win'].sum()
        paper1_loss_count = len(paper1_trades) - paper1_win_count
        axes[0].pie([paper1_win_count, paper1_loss_count], 
                    labels=['Win', 'Loss'], 
                    colors=['#2ecc71', '#e74c3c'],
                    autopct='%1.1f%%',
                    startangle=90,
                    explode=(0.05, 0))
        axes[0].set_title('Paper 1 Trade Results', fontsize=14)
        
        # Paper 2 pie chart
        paper2_win_count = paper2_trades['is_win'].sum()
        paper2_loss_count = len(paper2_trades) - paper2_win_count
        axes[1].pie([paper2_win_count, paper2_loss_count], 
                    labels=['Win', 'Loss'], 
                    colors=['#2ecc71', '#e74c3c'],
                    autopct='%1.1f%%',
                    startangle=90,
                    explode=(0.05, 0))
        axes[1].set_title('Paper 2 Trade Results', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trade_results_comparison.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating trade distribution comparison chart: {str(e)}")
    
    # Report completion message
    print(f"Comparative analysis report generated in {output_dir}") 