"""
Comparison Dashboard Generator
Creates HTML dashboard for visualizing A/B test results
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from src.ab_testing.metrics import MetricsCollector, VariantMetrics


class DashboardGenerator:
    """
    Generates HTML dashboard for A/B test comparison
    """

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.metrics_collector = MetricsCollector(experiment_name)

    def _format_timestamp(self, timestamp: datetime) -> str:
        """Format timestamp for display"""
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")

    def generate_dashboard(self, output_path: Optional[str] = None) -> str:
        """
        Generate complete dashboard HTML
        
        Args:
            output_path: Path to save HTML file
            
        Returns:
            Path to generated HTML file
        """
        if output_path is None:
            experiment_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "experiments",
                self.experiment_name
            )
            output_path = os.path.join(
                experiment_dir,
                f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )

        # Collect all metrics
        all_metrics = self.metrics_collector.collect_all_metrics()
        summary_stats = self.metrics_collector.get_summary_stats()

        # Generate HTML
        html = self._generate_html(all_metrics, summary_stats)

        with open(output_path, 'w') as f:
            f.write(html)

        return output_path

    def _generate_html(
        self,
        all_metrics: Dict[str, VariantMetrics],
        summary_stats: Dict[str, Any]
    ) -> str:
        """Generate complete HTML dashboard"""

        # Prepare data for charts
        variant_labels = list(all_metrics.keys())
        prediction_counts = [m.total_predictions for m in all_metrics.values()]
        avg_confidences = [m.avg_confidence for m in all_metrics.values()]
        avg_times = [m.avg_inference_time_ms for m in all_metrics.values()]

        # Sentiment data
        sentiment_data = {}
        for vid, metrics in all_metrics.items():
            sentiment_data[vid] = metrics.sentiment_distribution or {"Positive": 0, "Negative": 0, "Neutral": 0}

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>A/B Test Dashboard - {self.experiment_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .dashboard {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .header h1 {{
            color: #2d3748;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            color: #718096;
            font-size: 14px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .stat-card .label {{
            color: #718096;
            font-size: 14px;
            margin-bottom: 8px;
        }}
        
        .stat-card .value {{
            color: #2d3748;
            font-size: 32px;
            font-weight: bold;
        }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .chart-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .chart-card h3 {{
            color: #2d3748;
            margin-bottom: 20px;
            font-size: 18px;
        }}
        
        .variant-table {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow-x: auto;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        th {{
            color: #4a5568;
            font-weight: 600;
            font-size: 14px;
        }}
        
        td {{
            color: #2d3748;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}
        
        .badge-positive {{
            background: #c6f6d5;
            color: #22543d;
        }}
        
        .badge-negative {{
            background: #fed7d7;
            color: #742a2a;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>A/B Test Dashboard</h1>
            <div class="subtitle">Experiment: {self.experiment_name} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="label">Total Variants</div>
                <div class="value">{summary_stats.get('total_variants', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="label">Total Predictions</div>
                <div class="value">{summary_stats.get('total_predictions', 0):,}</div>
            </div>
            <div class="stat-card">
                <div class="label">Best Avg Confidence</div>
                <div class="value">{max(avg_confidences) if avg_confidences else 0:.2%}</div>
            </div>
            <div class="stat-card">
                <div class="label">Fastest Avg Time</div>
                <div class="value">{min(avg_times) if avg_times else 0:.1f}ms</div>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-card">
                <h3>Prediction Volume by Variant</h3>
                <canvas id="volumeChart"></canvas>
            </div>
            <div class="chart-card">
                <h3>Average Confidence by Variant</h3>
                <canvas id="confidenceChart"></canvas>
            </div>
            <div class="chart-card">
                <h3>Average Inference Time (ms)</h3>
                <canvas id="timeChart"></canvas>
            </div>
            <div class="chart-card">
                <h3>Sentiment Distribution</h3>
                <canvas id="sentimentChart"></canvas>
            </div>
        </div>
        
        <div class="variant-table">
            <h3 style="margin-bottom: 20px;">Variant Comparison</h3>
            <table>
                <thead>
                    <tr>
                        <th>Variant ID</th>
                        <th>Predictions</th>
                        <th>Traffic %</th>
                        <th>Avg Confidence</th>
                        <th>Avg Time (ms)</th>
                        <th>Positive %</th>
                        <th>Negative %</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_table_rows(all_metrics, summary_stats)}
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        // Chart.js configuration
        const chartConfig = {{
            responsive: true,
            maintainAspectRatio: true,
            plugins: {{
                legend: {{
                    display: true,
                    position: 'bottom'
                }}
            }}
        }};
        
        // Volume Chart
        new Chart(document.getElementById('volumeChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(variant_labels)},
                datasets: [{{
                    label: 'Predictions',
                    data: {json.dumps(prediction_counts)},
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 1
                }}]
            }},
            options: chartConfig
        }});
        
        // Confidence Chart
        new Chart(document.getElementById('confidenceChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(variant_labels)},
                datasets: [{{
                    label: 'Avg Confidence',
                    data: {json.dumps(avg_confidences)},
                    backgroundColor: 'rgba(52, 211, 153, 0.8)',
                    borderColor: 'rgba(52, 211, 153, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                ...chartConfig,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1
                    }}
                }}
            }}
        }});
        
        // Time Chart
        new Chart(document.getElementById('timeChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(variant_labels)},
                datasets: [{{
                    label: 'Avg Inference Time (ms)',
                    data: {json.dumps(avg_times)},
                    backgroundColor: 'rgba(251, 191, 36, 0.8)',
                    borderColor: 'rgba(251, 191, 36, 1)',
                    borderWidth: 1
                }}]
            }},
            options: chartConfig
        }});
        
        // Sentiment Chart
        new Chart(document.getElementById('sentimentChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(variant_labels)},
                datasets: [
                    {{
                        label: 'Positive',
                        data: {json.dumps([sentiment_data[v].get('Positive', 0) for v in variant_labels])},
                        backgroundColor: 'rgba(52, 211, 153, 0.8)'
                    }},
                    {{
                        label: 'Negative',
                        data: {json.dumps([sentiment_data[v].get('Negative', 0) for v in variant_labels])},
                        backgroundColor: 'rgba(248, 113, 113, 0.8)'
                    }}
                ]
            }},
            options: {{
                ...chartConfig,
                scales: {{
                    x: {{ stacked: true }},
                    y: {{ stacked: true }}
                }}
            }}
        }});
    </script>
</body>
</html>
        """

        return html

    def _generate_table_rows(
        self,
        all_metrics: Dict[str, VariantMetrics],
        summary_stats: Dict[str, Any]
    ) -> str:
        """Generate table rows for variant comparison"""
        rows = []

        for variant_id, metrics in all_metrics.items():
            variant_stats = summary_stats["variants"].get(variant_id, {})

            dist = metrics.sentiment_distribution or {"Positive": 0, "Negative": 0, "Neutral": 0}
            total_sentiment = sum(dist.values())
            pos_pct = (dist.get("Positive", 0) / total_sentiment * 100) if total_sentiment > 0 else 0
            neg_pct = (dist.get("Negative", 0) / total_sentiment * 100) if total_sentiment > 0 else 0

            row = f"""
                <tr>
                    <td><strong>{variant_id}</strong></td>
                    <td>{metrics.total_predictions:,}</td>
                    <td>{variant_stats.get('percentage', 0):.1f}%</td>
                    <td>{metrics.avg_confidence:.2%}</td>
                    <td>{metrics.avg_inference_time_ms:.2f}</td>
                    <td>{pos_pct:.1f}%</td>
                    <td>{neg_pct:.1f}%</td>
                </tr>
            """
            rows.append(row)

        return '\n'.join(rows)
