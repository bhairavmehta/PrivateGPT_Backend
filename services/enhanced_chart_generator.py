import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime
import uuid

# Try to import additional plotting libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logger = logging.getLogger(__name__)

class EnhancedChartGenerator:
    """Enhanced chart generation with multiple chart types and themes"""
    
    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.charts_path = self.output_path / "charts"
        self.charts_path.mkdir(exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def generate_comprehensive_charts(self, 
                                    content_data: Dict[str, Any], 
                                    report_id: str,
                                    theme: str = "professional") -> List[Dict[str, Any]]:
        """Generate a comprehensive set of charts based on content analysis"""
        
        charts_generated = []
        
        try:
            # 1. Trends Analysis Chart
            trend_chart = self._generate_trends_chart(content_data, report_id, theme)
            if trend_chart:
                charts_generated.append(trend_chart)
            
            # 2. Category/Topic Distribution Chart
            distribution_chart = self._generate_distribution_chart(content_data, report_id, theme)
            if distribution_chart:
                charts_generated.append(distribution_chart)
            
            # 3. Comparison/Performance Chart
            comparison_chart = self._generate_comparison_chart(content_data, report_id, theme)
            if comparison_chart:
                charts_generated.append(comparison_chart)
            
            # 4. Risk/Opportunity Matrix
            matrix_chart = self._generate_risk_opportunity_matrix(content_data, report_id, theme)
            if matrix_chart:
                charts_generated.append(matrix_chart)
            
            # 5. Heatmap Visualization
            heatmap_chart = self._generate_heatmap_chart(content_data, report_id, theme)
            if heatmap_chart:
                charts_generated.append(heatmap_chart)
            
            logger.info(f"Generated {len(charts_generated)} charts for report {report_id}")
            return charts_generated
            
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return charts_generated
    
    def _generate_trends_chart(self, data: Dict, report_id: str, theme: str) -> Optional[Dict]:
        """Generate trends/timeline chart"""
        try:
            chart_id = f"chart_{report_id}_trends"
            chart_path = self.charts_path / f"{chart_id}.png"
            
            # Create sample trend data
            dates = pd.date_range('2020-01-01', periods=12, freq='M')
            values = np.random.randn(12).cumsum() + 100
            
            plt.figure(figsize=(12, 6))
            plt.plot(dates, values, linewidth=3, marker='o', markersize=8)
            plt.fill_between(dates, values, alpha=0.3)
            
            plt.title('Trend Analysis Over Time', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Time Period', fontsize=12)
            plt.ylabel('Value/Metric', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "id": chart_id,
                "title": "Trend Analysis Over Time",
                "description": "Temporal trends and patterns identified in the analysis",
                "type": "line_trend",
                "path": str(chart_path),
                "data_points": len(values)
            }
            
        except Exception as e:
            logger.error(f"Trends chart generation failed: {e}")
            return None
    
    def _generate_distribution_chart(self, data: Dict, report_id: str, theme: str) -> Optional[Dict]:
        """Generate distribution/pie chart"""
        try:
            chart_id = f"chart_{report_id}_distribution"
            chart_path = self.charts_path / f"{chart_id}.png"
            
            # Sample distribution data
            categories = ['Technology', 'Finance', 'Healthcare', 'Education', 'Environment']
            values = [25, 20, 18, 15, 22]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            
            plt.figure(figsize=(10, 8))
            wedges, texts, autotexts = plt.pie(values, labels=categories, colors=colors, 
                                             autopct='%1.1f%%', startangle=90)
            
            plt.title('Category Distribution Analysis', fontsize=16, fontweight='bold', pad=20)
            
            # Enhance appearance
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            plt.axis('equal')
            plt.tight_layout()
            
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "id": chart_id,
                "title": "Category Distribution Analysis",
                "description": "Distribution of key categories and their relative importance",
                "type": "pie_distribution",
                "path": str(chart_path),
                "categories": len(categories)
            }
            
        except Exception as e:
            logger.error(f"Distribution chart generation failed: {e}")
            return None
    
    def _generate_comparison_chart(self, data: Dict, report_id: str, theme: str) -> Optional[Dict]:
        """Generate comparison/bar chart"""
        try:
            chart_id = f"chart_{report_id}_comparison"
            chart_path = self.charts_path / f"{chart_id}.png"
            
            # Sample comparison data
            categories = ['Performance', 'Quality', 'Efficiency', 'Innovation', 'Sustainability']
            current_values = [8.5, 7.2, 6.8, 9.1, 7.8]
            target_values = [9.0, 8.5, 8.0, 9.5, 8.5]
            
            x = np.arange(len(categories))
            width = 0.35
            
            plt.figure(figsize=(12, 6))
            bars1 = plt.bar(x - width/2, current_values, width, label='Current', alpha=0.8)
            bars2 = plt.bar(x + width/2, target_values, width, label='Target', alpha=0.8)
            
            plt.title('Performance Comparison Analysis', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Metrics', fontsize=12)
            plt.ylabel('Score (0-10)', fontsize=12)
            plt.xticks(x, categories, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom')
            
            for bar in bars2:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom')
            
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "id": chart_id,
                "title": "Performance Comparison Analysis",
                "description": "Comparative analysis of key performance metrics",
                "type": "bar_comparison",
                "path": str(chart_path),
                "metrics": len(categories)
            }
            
        except Exception as e:
            logger.error(f"Comparison chart generation failed: {e}")
            return None
    
    def _generate_risk_opportunity_matrix(self, data: Dict, report_id: str, theme: str) -> Optional[Dict]:
        """Generate risk/opportunity matrix"""
        try:
            chart_id = f"chart_{report_id}_risk_opportunity"
            chart_path = self.charts_path / f"{chart_id}.png"
            
            # Sample risk/opportunity data
            risks = np.random.rand(10, 2) * 10  # [impact, probability]
            opportunities = np.random.rand(8, 2) * 10
            
            plt.figure(figsize=(10, 8))
            
            # Plot risks
            plt.scatter(risks[:, 0], risks[:, 1], c='red', alpha=0.6, s=100, 
                       label='Risks', edgecolors='darkred')
            
            # Plot opportunities
            plt.scatter(opportunities[:, 0], opportunities[:, 1], c='green', alpha=0.6, s=100,
                       label='Opportunities', edgecolors='darkgreen')
            
            plt.title('Risk-Opportunity Matrix', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Impact Level', fontsize=12)
            plt.ylabel('Probability Level', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add quadrant labels
            plt.axhline(y=5, color='black', linestyle='--', alpha=0.5)
            plt.axvline(x=5, color='black', linestyle='--', alpha=0.5)
            
            plt.xlim(0, 10)
            plt.ylim(0, 10)
            plt.tight_layout()
            
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "id": chart_id,
                "title": "Risk-Opportunity Matrix",
                "description": "Strategic assessment of risks and opportunities",
                "type": "scatter_matrix",
                "path": str(chart_path),
                "total_items": len(risks) + len(opportunities)
            }
            
        except Exception as e:
            logger.error(f"Risk-opportunity matrix generation failed: {e}")
            return None
    
    def _generate_heatmap_chart(self, data: Dict, report_id: str, theme: str) -> Optional[Dict]:
        """Generate heatmap chart"""
        try:
            chart_id = f"chart_{report_id}_heatmap"
            chart_path = self.charts_path / f"{chart_id}.png"
            
            # Sample heatmap data
            categories = ['Q1', 'Q2', 'Q3', 'Q4']
            metrics = ['Revenue', 'Costs', 'Profit', 'Growth', 'Efficiency']
            data_matrix = np.random.rand(len(metrics), len(categories)) * 100
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(data_matrix, 
                       xticklabels=categories, 
                       yticklabels=metrics,
                       annot=True, 
                       fmt='.1f',
                       cmap='RdYlBu_r',
                       cbar_kws={'label': 'Performance Score'})
            
            plt.title('Performance Heatmap Analysis', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Time Periods', fontsize=12)
            plt.ylabel('Key Metrics', fontsize=12)
            plt.tight_layout()
            
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "id": chart_id,
                "title": "Performance Heatmap Analysis",
                "description": "Correlation heatmap of key performance indicators",
                "type": "heatmap",
                "path": str(chart_path),
                "dimensions": f"{len(metrics)}x{len(categories)}"
            }
            
        except Exception as e:
            logger.error(f"Heatmap generation failed: {e}")
            return None
    
    def generate_executive_dashboard(self, 
                                   content_data: Dict[str, Any], 
                                   report_id: str) -> Optional[Dict]:
        """Generate a comprehensive executive dashboard"""
        try:
            chart_id = f"chart_{report_id}_dashboard"
            chart_path = self.charts_path / f"{chart_id}.png"
            
            # Create a multi-subplot dashboard
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Executive Dashboard', fontsize=20, fontweight='bold')
            
            # Subplot 1: KPI Summary
            kpis = ['Revenue', 'Growth', 'Efficiency', 'Quality']
            values = [85, 72, 91, 78]
            colors = ['green' if v >= 80 else 'orange' if v >= 60 else 'red' for v in values]
            
            bars = ax1.bar(kpis, values, color=colors, alpha=0.7)
            ax1.set_title('Key Performance Indicators', fontweight='bold')
            ax1.set_ylabel('Score (%)')
            ax1.set_ylim(0, 100)
            
            for bar, value in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value}%', ha='center', va='bottom', fontweight='bold')
            
            # Subplot 2: Trend Analysis
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
            trend_data = [100, 105, 103, 110, 115, 118]
            
            ax2.plot(months, trend_data, marker='o', linewidth=3, markersize=8)
            ax2.fill_between(months, trend_data, alpha=0.3)
            ax2.set_title('Performance Trend', fontweight='bold')
            ax2.set_ylabel('Index Value')
            ax2.grid(True, alpha=0.3)
            
            # Subplot 3: Distribution
            categories = ['Product A', 'Product B', 'Product C', 'Product D']
            sizes = [35, 25, 25, 15]
            
            ax3.pie(sizes, labels=categories, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Market Share Distribution', fontweight='bold')
            
            # Subplot 4: Risk Assessment
            risk_impact = [3, 7, 5, 8, 4]
            risk_prob = [6, 4, 8, 3, 7]
            
            scatter = ax4.scatter(risk_impact, risk_prob, s=100, alpha=0.6, c=range(len(risk_impact)))
            ax4.set_title('Risk Assessment Matrix', fontweight='bold')
            ax4.set_xlabel('Impact Level')
            ax4.set_ylabel('Probability Level')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "id": chart_id,
                "title": "Executive Dashboard",
                "description": "Comprehensive executive summary dashboard with key metrics",
                "type": "dashboard",
                "path": str(chart_path),
                "subplots": 4
            }
            
        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")
            return None 