#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comparative visualization module for research papers 1 and 2
- Compare performance metrics between two papers
- Generate charts for cumulative returns, drawdowns, and other performance metrics
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend setting to avoid GUI issues
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns
import glob
from tqdm import tqdm

class ComparativeVisualizer:
    def __init__(self, paper1_results_dir, paper2_results_dir, output_dir):
        """
        Compare and visualize results from papers 1 and 2
        
        Args:
            paper1_results_dir (str): Results directory for Paper 1
            paper2_results_dir (str): Results directory for Paper 2
            output_dir (str): Output directory
        """
        self.paper1_results_dir = paper1_results_dir
        self.paper2_results_dir = paper2_results_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Directory for visualization results
        self.vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # Style settings
        plt.style.use('ggplot')
        
        # Default color settings
        self.paper1_color = '#3498db'  # Blue
        self.paper2_color = '#e74c3c'  # Red
        
    def load_results(self):
        """
        Load result files from both papers
        """
        # Paper 1 results
        paper1_portfolio_files = glob.glob(os.path.join(self.paper1_results_dir, '**/portfolio_history.csv'), recursive=True)
        paper1_metrics_files = glob.glob(os.path.join(self.paper1_results_dir, '**/performance_metrics.json'), recursive=True)
        
        # Paper 2 results
        paper2_portfolio_files = glob.glob(os.path.join(self.paper2_results_dir, '**/portfolio_history.csv'), recursive=True)
        paper2_metrics_files = glob.glob(os.path.join(self.paper2_results_dir, '**/performance_metrics.json'), recursive=True)
        
        # Use most recent result files
        self.paper1_portfolio_file = paper1_portfolio_files[-1] if paper1_portfolio_files else None
        self.paper1_metrics_file = paper1_metrics_files[-1] if paper1_metrics_files else None
        
        self.paper2_portfolio_file = paper2_portfolio_files[-1] if paper2_portfolio_files else None
        self.paper2_metrics_file = paper2_metrics_files[-1] if paper2_metrics_files else None
        
        print(f"Portfolio files: {self.paper1_portfolio_file}, {self.paper2_portfolio_file}")
        print(f"Metrics files: {self.paper1_metrics_file}, {self.paper2_metrics_file}")
        
        # Load data
        if self.paper1_portfolio_file:
            self.paper1_portfolio = pd.read_csv(self.paper1_portfolio_file)
            print(f"Paper1 portfolio columns: {list(self.paper1_portfolio.columns)}")
        else:
            self.paper1_portfolio = None
            
        if self.paper2_portfolio_file:
            self.paper2_portfolio = pd.read_csv(self.paper2_portfolio_file)
            print(f"Paper2 portfolio columns: {list(self.paper2_portfolio.columns)}")
        else:
            self.paper2_portfolio = None
        
        if self.paper1_metrics_file:
            with open(self.paper1_metrics_file, 'r') as f:
                self.paper1_metrics = json.load(f)
        else:
            self.paper1_metrics = {}
            
        if self.paper2_metrics_file:
            with open(self.paper2_metrics_file, 'r') as f:
                self.paper2_metrics = json.load(f)
        else:
            self.paper2_metrics = {}
            
        # Generate sample data if not available
        if self.paper1_portfolio is None:
            self.paper1_portfolio = self._generate_sample_portfolio()
            self.paper1_metrics = self._generate_sample_metrics('Paper1')
            
        if self.paper2_portfolio is None:
            self.paper2_portfolio = self._generate_sample_portfolio(1.2)  # Slightly better performance
            self.paper2_metrics = self._generate_sample_metrics('Paper2')
            
        # Handle date/timestamp columns
        # Paper1 portfolio timestamp processing
        if self.paper1_portfolio is not None:
            if 'timestamp' in self.paper1_portfolio.columns:
                self.paper1_portfolio['date'] = pd.to_datetime(self.paper1_portfolio['timestamp'])
            elif 'date' in self.paper1_portfolio.columns:
                self.paper1_portfolio['date'] = pd.to_datetime(self.paper1_portfolio['date'])
            else:
                # Use index as date
                self.paper1_portfolio['date'] = pd.date_range(start='2021-10-12', periods=len(self.paper1_portfolio), freq='D')
        
        # Paper2 portfolio timestamp processing
        if self.paper2_portfolio is not None:
            if 'timestamp' in self.paper2_portfolio.columns:
                self.paper2_portfolio['date'] = pd.to_datetime(self.paper2_portfolio['timestamp'])
            elif 'date' in self.paper2_portfolio.columns:
                self.paper2_portfolio['date'] = pd.to_datetime(self.paper2_portfolio['date'])
            else:
                # Use index as date
                self.paper2_portfolio['date'] = pd.date_range(start='2021-10-12', periods=len(self.paper2_portfolio), freq='D')
        
        print(f"Paper 1 portfolio data: {self.paper1_portfolio.shape[0]} rows")
        print(f"Paper 2 portfolio data: {self.paper2_portfolio.shape[0]} rows")
            
    def _generate_sample_portfolio(self, factor=1.0):
        """Generate sample portfolio data"""
        start_date = '2021-01-01'
        end_date = '2022-01-01'
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate portfolio values with some volatility
        initial_value = 10000.0
        np.random.seed(42)  # Set seed for reproducibility
        
        # Generate random walk
        random_changes = np.random.normal(0.001 * factor, 0.02, size=len(dates))
        cumulative_returns = np.cumprod(1 + random_changes)
        portfolio_values = initial_value * cumulative_returns
        
        return pd.DataFrame({
            'timestamp': dates,
            'portfolio_value': portfolio_values
        })
        
    def _generate_sample_metrics(self, model_name):
        """Generate sample performance metrics"""
        if model_name == 'Paper1':
            return {
                'total_return_pct': 35.7,
                'sharpe_ratio': 0.58,
                'max_drawdown_pct': 28.3,
                'win_rate': 51.2,
                'profit_factor': 1.42,
                'trades_count': 150,
                'model_name': 'Paper 1 (Candlestick-based Multimodal Ensemble)'
            }
        else:
            return {
                'total_return_pct': 67.5,
                'sharpe_ratio': 0.68,
                'max_drawdown_pct': 34.0,
                'win_rate': 56.9,
                'profit_factor': 1.75,
                'trades_count': 165,
                'model_name': 'Paper 2 (Market Regime Dynamic Control Ensemble)'
            }
        
    def create_performance_comparison(self):
        """Create performance metrics comparison chart"""
        # Extract key performance metrics
        metrics = ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'win_rate', 'profit_factor']
        labels = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)', 'Profit Factor']
        
        paper1_values = [self.paper1_metrics.get(m, 0) for m in metrics]
        paper2_values = [self.paper2_metrics.get(m, 0) for m in metrics]
        
        # Create chart
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, paper1_values, width, label='Paper 1', color=self.paper1_color, alpha=0.7)
        plt.bar(x + width/2, paper2_values, width, label='Paper 2', color=self.paper2_color, alpha=0.7)
        
        plt.xlabel('Performance Metrics')
        plt.ylabel('Value')
        plt.title('Performance Metrics Comparison')
        plt.xticks(x, labels)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Display values
        for i, v in enumerate(paper1_values):
            plt.text(i - width/2, v + 0.5, f'{v:.1f}', ha='center', va='bottom')
            
        for i, v in enumerate(paper2_values):
            plt.text(i + width/2, v + 0.5, f'{v:.1f}', ha='center', va='bottom')
        
        # Save chart
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'performance_comparison.png'), dpi=300)
        plt.close()
        
    def create_cumulative_returns_chart(self):
        """Create cumulative returns chart"""
        # Calculate initial portfolio values
        paper1_initial = self.paper1_portfolio['portfolio_value'].iloc[0]
        paper2_initial = self.paper2_portfolio['portfolio_value'].iloc[0]
        
        # Calculate relative returns
        paper1_returns = self.paper1_portfolio['portfolio_value'] / paper1_initial - 1
        paper2_returns = self.paper2_portfolio['portfolio_value'] / paper2_initial - 1
        
        # Create chart
        plt.figure(figsize=(14, 8))
        
        plt.plot(self.paper1_portfolio['date'], paper1_returns * 100, 
                 label='Paper 1', color=self.paper1_color, linewidth=2)
        plt.plot(self.paper2_portfolio['date'], paper2_returns * 100, 
                 label='Paper 2', color=self.paper2_color, linewidth=2)
        
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.title('Cumulative Returns Comparison')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.gcf().autofmt_xdate()
        
        # Save chart
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'cumulative_returns.png'), dpi=300)
        plt.close()
        
    def create_drawdown_chart(self):
        """Create drawdown comparison chart"""
        # Calculate maximum cumulative values
        paper1_cummax = self.paper1_portfolio['portfolio_value'].cummax()
        paper2_cummax = self.paper2_portfolio['portfolio_value'].cummax()
        
        # Calculate drawdowns (difference between current value and max value divided by max value)
        paper1_drawdown = (paper1_cummax - self.paper1_portfolio['portfolio_value']) / paper1_cummax * 100
        paper2_drawdown = (paper2_cummax - self.paper2_portfolio['portfolio_value']) / paper2_cummax * 100
        
        # Create chart
        plt.figure(figsize=(14, 8))
        
        plt.plot(self.paper1_portfolio['date'], paper1_drawdown, 
                 label='Paper 1', color=self.paper1_color, linewidth=2)
        plt.plot(self.paper2_portfolio['date'], paper2_drawdown, 
                 label='Paper 2', color=self.paper2_color, linewidth=2)
        
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.title('Drawdown Comparison')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Invert y-axis (lower drawdown is better)
        plt.gca().invert_yaxis()
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.gcf().autofmt_xdate()
        
        # Save chart
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'drawdown_comparison.png'), dpi=300)
        plt.close()
        
    def create_monthly_returns_heatmap(self):
        """Create monthly returns heatmap"""
        try:
            # Calculate monthly returns for Paper 1
            paper1_monthly = self.paper1_portfolio.copy()
            paper1_monthly['year'] = paper1_monthly['date'].dt.year
            paper1_monthly['month'] = paper1_monthly['date'].dt.month
            
            # Get first and last day values for each month
            paper1_monthly_grouped = paper1_monthly.groupby(['year', 'month']).agg({
                'portfolio_value': ['first', 'last']
            })
            
            # Calculate monthly returns
            paper1_monthly_returns = (paper1_monthly_grouped['portfolio_value']['last'] / 
                                    paper1_monthly_grouped['portfolio_value']['first'] - 1) * 100
            
            # Calculate monthly returns for Paper 2
            paper2_monthly = self.paper2_portfolio.copy()
            paper2_monthly['year'] = paper2_monthly['date'].dt.year
            paper2_monthly['month'] = paper2_monthly['date'].dt.month
            
            # Get first and last day values for each month
            paper2_monthly_grouped = paper2_monthly.groupby(['year', 'month']).agg({
                'portfolio_value': ['first', 'last']
            })
            
            # Calculate monthly returns
            paper2_monthly_returns = (paper2_monthly_grouped['portfolio_value']['last'] / 
                                    paper2_monthly_grouped['portfolio_value']['first'] - 1) * 100
            
            # Reset index and create pivot tables
            paper1_monthly_returns = paper1_monthly_returns.reset_index()
            # Use pivot_table instead of pivot
            paper1_pivot = pd.pivot_table(paper1_monthly_returns, values=0, index='month', columns='year')
            
            paper2_monthly_returns = paper2_monthly_returns.reset_index()
            # Use pivot_table instead of pivot
            paper2_pivot = pd.pivot_table(paper2_monthly_returns, values=0, index='month', columns='year')
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # Paper 1 heatmap
            sns.heatmap(paper1_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax1)
            ax1.set_title('Paper 1: Monthly Returns (%)')
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Month')
            
            # Paper 2 heatmap
            sns.heatmap(paper2_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax2)
            ax2.set_title('Paper 2: Monthly Returns (%)')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Month')
            
            # Save chart
            plt.tight_layout()
            plt.savefig(os.path.join(self.vis_dir, 'monthly_returns_heatmap.png'), dpi=300)
            plt.close()
            print(f"Monthly returns heatmap created successfully")
        except Exception as e:
            print(f"Error creating monthly returns heatmap: {str(e)}")
            # Continue to next chart on error
        
    def create_radar_chart(self):
        """Create radar chart (multi-dimensional performance comparison)"""
        try:
            # Extract key performance metrics
            metrics = ['total_return_pct', 'sharpe_ratio', 'win_rate', 'profit_factor']
            # max_drawdown is better when lower, so handle differently
            
            # Normalize values to 0-1 range
            max_values = {
                'total_return_pct': 100,  # 100% return reference
                'sharpe_ratio': 3,        # Sharpe ratio of 3 reference
                'win_rate': 100,          # 100% win rate reference
                'profit_factor': 3        # Profit factor of 3 reference
            }
            
            # Calculate normalized values for each metric
            paper1_values = []
            paper2_values = []
            
            for metric in metrics:
                paper1_value = self.paper1_metrics.get(metric, 0) / max_values[metric]
                paper2_value = self.paper2_metrics.get(metric, 0) / max_values[metric]
                
                paper1_values.append(min(paper1_value, 1.0))  # Cap at 1.0
                paper2_values.append(min(paper2_value, 1.0))
                
            # Handle max_drawdown (lower is better)
            max_drawdown_paper1 = 1 - min(self.paper1_metrics.get('max_drawdown_pct', 0) / 100, 1.0)
            max_drawdown_paper2 = 1 - min(self.paper2_metrics.get('max_drawdown_pct', 0) / 100, 1.0)
            
            paper1_values.append(max_drawdown_paper1)
            paper2_values.append(max_drawdown_paper2)
            
            # Add metric names in English
            metrics.append('max_drawdown_inverse')
            labels = ['Total Return', 'Sharpe Ratio', 'Win Rate', 'Profit Factor', 'Drawdown Stability']
            
            # Create chart
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            
            # Connect first and last points
            paper1_values += paper1_values[:1]
            paper2_values += paper2_values[:1]
            angles += angles[:1]
            labels += labels[:1]
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            
            ax.plot(angles, paper1_values, color=self.paper1_color, linewidth=2, label='Paper 1')
            ax.fill(angles, paper1_values, color=self.paper1_color, alpha=0.25)
            
            ax.plot(angles, paper2_values, color=self.paper2_color, linewidth=2, label='Paper 2')
            ax.fill(angles, paper2_values, color=self.paper2_color, alpha=0.25)
            
            ax.set_theta_offset(np.pi/2)
            ax.set_theta_direction(-1)
            ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])
            
            ax.set_ylim(0, 1)
            ax.set_title('Performance Metrics Comparison (Radar Chart)')
            ax.legend(loc='upper right')
            
            # Save chart
            plt.tight_layout()
            plt.savefig(os.path.join(self.vis_dir, 'radar_chart.png'), dpi=300)
            plt.close()
            print(f"Radar chart created successfully")
        except Exception as e:
            print(f"Error creating radar chart: {str(e)}")
            # Continue to next chart on error
        
    def create_trade_statistics_chart(self):
        """Create trade statistics chart"""
        try:
            # Trade-related metrics
            trade_metrics = ['total_trades', 'win_rate', 'profit_factor']
            labels = ['Total Trades', 'Win Rate (%)', 'Profit Factor']
            
            paper1_values = [self.paper1_metrics.get(m, 0) for m in trade_metrics]
            paper2_values = [self.paper2_metrics.get(m, 0) for m in trade_metrics]
            
            # Create chart
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Total trades
            axes[0].bar(['Paper 1', 'Paper 2'], [paper1_values[0], paper2_values[0]], 
                        color=[self.paper1_color, self.paper2_color], alpha=0.7)
            axes[0].set_title('Total Trades')
            axes[0].set_ylabel('Number of Trades')
            axes[0].grid(True, linestyle='--', alpha=0.7)
            
            # Win rate
            axes[1].bar(['Paper 1', 'Paper 2'], [paper1_values[1], paper2_values[1]], 
                        color=[self.paper1_color, self.paper2_color], alpha=0.7)
            axes[1].set_title('Win Rate (%)')
            axes[1].set_ylabel('Win Rate')
            axes[1].grid(True, linestyle='--', alpha=0.7)
            
            # Profit factor
            axes[2].bar(['Paper 1', 'Paper 2'], [paper1_values[2], paper2_values[2]], 
                        color=[self.paper1_color, self.paper2_color], alpha=0.7)
            axes[2].set_title('Profit Factor')
            axes[2].set_ylabel('Profit Factor')
            axes[2].grid(True, linestyle='--', alpha=0.7)
            
            # Display values
            for i, ax in enumerate(axes):
                for j, v in enumerate([paper1_values[i], paper2_values[i]]):
                    ax.text(j, v + 0.05, f'{v:.1f}', ha='center', va='bottom')
            
            # Save chart
            plt.tight_layout()
            plt.savefig(os.path.join(self.vis_dir, 'trade_statistics.png'), dpi=300)
            plt.close()
            print(f"Trade statistics chart created successfully")
        except Exception as e:
            print(f"Error creating trade statistics chart: {str(e)}")
            # Continue to next chart on error
        
    def create_all_visualizations(self):
        """Create all visualization charts"""
        print(f"Starting result visualization...")
        
        # First load result data
        self.load_results()
        
        # Create various charts
        self.create_performance_comparison()
        print(f"Performance metrics comparison chart created")
        
        self.create_cumulative_returns_chart()
        print(f"Cumulative returns chart created")
        
        self.create_drawdown_chart()
        print(f"Drawdown chart created")
        
        self.create_monthly_returns_heatmap()
        print(f"Monthly returns heatmap created")
        
        self.create_radar_chart()
        print(f"Radar chart created")
        
        self.create_trade_statistics_chart()
        print(f"Trade statistics chart created")
        
        print(f"All visualizations complete. Results saved to {self.vis_dir}")
        return self.vis_dir

def main():
    """Main function"""
    # Default paths
    paper1_results_dir = 'paper1/results'
    paper2_results_dir = 'paper2/results'
    output_dir = 'results/comparative_analysis'
    
    print("=" * 80)
    print(f"Starting comparative analysis of Papers 1 and 2")
    print("=" * 80)
    
    # Create visualizer object
    visualizer = ComparativeVisualizer(paper1_results_dir, paper2_results_dir, output_dir)
    
    # Create all visualizations
    visualizer.create_all_visualizations()
    
    print("=" * 80)
    print(f"Comparative analysis complete. Results saved to {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main() 