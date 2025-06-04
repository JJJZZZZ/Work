"""
Visualization Module for Propensity Matching Tool
Handles plot generation for before/after matching analysis
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class MatchingVisualizer:
    """Handles visualization generation for matching analysis"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 4), style: str = 'whitegrid'):
        """
        Initialize the visualizer
        
        Args:
            figsize: Default figure size for plots
            style: Seaborn style to use
        """
        self.figsize = figsize
        plt.style.use('default')
        sns.set_style(style)
        sns.set_palette("husl")
        
    def plot_numeric_comparison(self, 
                               df_before: pd.DataFrame, 
                               df_after: pd.DataFrame,
                               column: str,
                               treatment_flag: str = 'treatment_flag') -> str:
        """
        Create before/after comparison plot for a numeric column
        
        Args:
            df_before: DataFrame before matching
            df_after: DataFrame after matching
            column: Column name to plot
            treatment_flag: Name of treatment flag column
            
        Returns:
            Base64 encoded plot image
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Before matching
        treatment_before = df_before[df_before[treatment_flag] == 1][column].dropna()
        control_before = df_before[df_before[treatment_flag] == 0][column].dropna()
        
        ax1.hist(treatment_before, alpha=0.7, label='Treatment', bins=30, color='#FF6B6B')
        ax1.hist(control_before, alpha=0.7, label='Control', bins=30, color='#4ECDC4')
        ax1.set_title(f'{column} - Before Matching', fontsize=12, fontweight='bold')
        ax1.set_xlabel(column)
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # After matching
        treatment_after = df_after[df_after[treatment_flag] == 1][column].dropna()
        control_after = df_after[df_after[treatment_flag] == 0][column].dropna()
        
        ax2.hist(treatment_after, alpha=0.7, label='Treatment', bins=30, color='#FF6B6B')
        ax2.hist(control_after, alpha=0.7, label='Control', bins=30, color='#4ECDC4')
        ax2.set_title(f'{column} - After Matching', fontsize=12, fontweight='bold')
        ax2.set_xlabel(column)
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data
    
    def plot_categorical_comparison(self, 
                                   df_before: pd.DataFrame, 
                                   df_after: pd.DataFrame,
                                   column: str,
                                   treatment_flag: str = 'treatment_flag',
                                   max_categories: int = 10) -> str:
        """
        Create before/after comparison plot for a categorical column
        
        Args:
            df_before: DataFrame before matching
            df_after: DataFrame after matching
            column: Column name to plot
            treatment_flag: Name of treatment flag column
            max_categories: Maximum number of categories to show
            
        Returns:
            Base64 encoded plot image
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Get top categories
        top_categories = df_before[column].value_counts().head(max_categories).index.tolist()
        
        # Filter data to top categories
        df_before_filtered = df_before[df_before[column].isin(top_categories)]
        df_after_filtered = df_after[df_after[column].isin(top_categories)]
        
        # Before matching - Treatment
        treatment_before = df_before_filtered[df_before_filtered[treatment_flag] == 1][column].value_counts()
        treatment_before = treatment_before.reindex(top_categories, fill_value=0)
        ax1.bar(range(len(treatment_before)), treatment_before.values, color='#FF6B6B', alpha=0.7)
        ax1.set_title(f'{column} - Treatment (Before)', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(top_categories)))
        ax1.set_xticklabels(top_categories, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Before matching - Control
        control_before = df_before_filtered[df_before_filtered[treatment_flag] == 0][column].value_counts()
        control_before = control_before.reindex(top_categories, fill_value=0)
        ax2.bar(range(len(control_before)), control_before.values, color='#4ECDC4', alpha=0.7)
        ax2.set_title(f'{column} - Control (Before)', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(top_categories)))
        ax2.set_xticklabels(top_categories, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # After matching - Treatment
        treatment_after = df_after_filtered[df_after_filtered[treatment_flag] == 1][column].value_counts()
        treatment_after = treatment_after.reindex(top_categories, fill_value=0)
        ax3.bar(range(len(treatment_after)), treatment_after.values, color='#FF6B6B', alpha=0.7)
        ax3.set_title(f'{column} - Treatment (After)', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(len(top_categories)))
        ax3.set_xticklabels(top_categories, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # After matching - Control
        control_after = df_after_filtered[df_after_filtered[treatment_flag] == 0][column].value_counts()
        control_after = control_after.reindex(top_categories, fill_value=0)
        ax4.bar(range(len(control_after)), control_after.values, color='#4ECDC4', alpha=0.7)
        ax4.set_title(f'{column} - Control (After)', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(len(top_categories)))
        ax4.set_xticklabels(top_categories, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data
    
    def plot_balance_summary(self, 
                           df_before: pd.DataFrame, 
                           df_after: pd.DataFrame,
                           numeric_columns: List[str],
                           treatment_flag: str = 'treatment_flag') -> str:
        """
        Create a summary plot showing standardized mean differences
        
        Args:
            df_before: DataFrame before matching
            df_after: DataFrame after matching
            numeric_columns: List of numeric columns to analyze
            treatment_flag: Name of treatment flag column
            
        Returns:
            Base64 encoded plot image
        """
        smd_before = []
        smd_after = []
        columns = []
        
        for col in numeric_columns:
            if col in df_before.columns and col in df_after.columns:
                # Calculate standardized mean differences
                # Before matching
                treat_before = df_before[df_before[treatment_flag] == 1][col].dropna()
                control_before = df_before[df_before[treatment_flag] == 0][col].dropna()
                
                if len(treat_before) > 0 and len(control_before) > 0:
                    pooled_std_before = np.sqrt((treat_before.var() + control_before.var()) / 2)
                    smd_before_val = (treat_before.mean() - control_before.mean()) / pooled_std_before if pooled_std_before > 0 else 0
                    
                    # After matching
                    treat_after = df_after[df_after[treatment_flag] == 1][col].dropna()
                    control_after = df_after[df_after[treatment_flag] == 0][col].dropna()
                    
                    if len(treat_after) > 0 and len(control_after) > 0:
                        pooled_std_after = np.sqrt((treat_after.var() + control_after.var()) / 2)
                        smd_after_val = (treat_after.mean() - control_after.mean()) / pooled_std_after if pooled_std_after > 0 else 0
                        
                        smd_before.append(abs(smd_before_val))
                        smd_after.append(abs(smd_after_val))
                        columns.append(col)
        
        if not columns:
            # Return empty plot if no data
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No numeric data available for balance assessment', 
                   ha='center', va='center', transform=ax.transAxes)
            plt.tight_layout()
        else:
            # Create the plot
            x = np.arange(len(columns))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(max(self.figsize[0], len(columns) * 0.8), self.figsize[1]))
            
            bars1 = ax.bar(x - width/2, smd_before, width, label='Before Matching', color='#FF6B6B', alpha=0.7)
            bars2 = ax.bar(x + width/2, smd_after, width, label='After Matching', color='#4ECDC4', alpha=0.7)
            
            ax.set_xlabel('Variables')
            ax.set_ylabel('Absolute Standardized Mean Difference')
            ax.set_title('Balance Assessment: Before vs After Matching', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(columns, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add horizontal line at 0.1 (common threshold for good balance)
            ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Good Balance Threshold (0.1)')
            
            plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data
    
    def generate_all_plots(self, 
                          df_before: pd.DataFrame, 
                          df_after: pd.DataFrame,
                          numeric_columns: List[str] = None,
                          categorical_columns: List[str] = None,
                          treatment_flag: str = 'treatment_flag',
                          max_plots: int = 5) -> List[Dict[str, str]]:
        """
        Generate all comparison plots
        
        Args:
            df_before: DataFrame before matching
            df_after: DataFrame after matching
            numeric_columns: List of numeric columns to plot
            categorical_columns: List of categorical columns to plot
            treatment_flag: Name of treatment flag column
            max_plots: Maximum number of plots to generate
            
        Returns:
            List of dictionaries with plot information
        """
        plots = []
        
        # Auto-detect columns if not provided
        if numeric_columns is None:
            numeric_columns = df_before.select_dtypes(include=["number"]).columns.tolist()
            if treatment_flag in numeric_columns:
                numeric_columns.remove(treatment_flag)
        
        if categorical_columns is None:
            categorical_columns = df_before.select_dtypes(include=['object']).columns.tolist()
        
        # Generate numeric plots
        for i, col in enumerate(numeric_columns[:max_plots]):
            try:
                plot_data = self.plot_numeric_comparison(df_before, df_after, col, treatment_flag)
                plots.append({
                    'column': col,
                    'type': 'numeric',
                    'plot': plot_data
                })
            except Exception as e:
                print(f"Error generating plot for {col}: {str(e)}")
        
        # Generate balance summary
        try:
            balance_plot = self.plot_balance_summary(df_before, df_after, numeric_columns, treatment_flag)
            plots.append({
                'column': 'Balance Summary',
                'type': 'balance',
                'plot': balance_plot
            })
        except Exception as e:
            print(f"Error generating balance plot: {str(e)}")
        
        return plots
