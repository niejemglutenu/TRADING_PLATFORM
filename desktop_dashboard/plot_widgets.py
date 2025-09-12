# desktop_dashboard/plot_widgets.py
import logging
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt6.QtCore import Qt, QDateTime, QRect
from PyQt6 import QtCore

from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPixmap
from PyQt6.QtCharts import (
    QChart, QChartView, QLineSeries, QScatterSeries, QValueAxis, 
    QDateTimeAxis
)
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

# --- Column name constants ---
COL_DATE = 'PredictionDate'
COL_ACTUAL = 'ActualReturn'
COL_PREDICTED = 'PredictedReturn'
COL_PORTFOLIO_DATE = 'Date'
COL_PORTFOLIO_VALUE = 'PortfolioValue'
COL_FORECAST_HORIZON = 'ForecastHorizon'
COL_FORECAST_ORIGIN = 'ForecastOriginDate'

class ResultsPlotWidget(QWidget):
    """A self-contained widget for displaying professional, academic-style plots."""
    
    # --- UNIFIED ACADEMIC (Light) THEME COLORS ---
    BACKGROUND_COLOR = QColor("#FFFFFF")
    TEXT_COLOR = QColor("#000000")
    GRID_LINES_COLOR = QColor("#E0E0E0")
    
    EQUITY_COLOR = QColor("#000000")      # Black for Equity Curve
    ACTUAL_COLOR = QColor("#007BFF")      # Blue for Actual Returns
    PREDICTED_COLOR = QColor("#D14036")   # Red for Predicted Returns
    SCATTER_COLOR = QColor("#007BFF")
    RESIDUALS_COLOR = QColor("#000000")   # Black points for residuals
    REGRESSION_COLOR = QColor("#D14036")  # Red trend line
    ZERO_LINE_COLOR = QColor("#555555")    # A dark gray for the zero line
    
    def __init__(self, initial_title: str, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.chart_view = QChartView()
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.layout.addWidget(self.chart_view)
        self.current_treemap_widget = None
        self.clear_plot(initial_title)

    def _setup_new_chart(self, title: str) -> QChart:
        """Creates and configures a new chart object with the light theme."""
        chart = QChart()
        chart.setTitle(title)
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        chart.setBackgroundBrush(QBrush(self.BACKGROUND_COLOR))
        title_font = QFont("Segoe UI", 12, QFont.Weight.Bold)
        chart.setTitleFont(title_font)
        chart.setTitleBrush(QBrush(self.TEXT_COLOR))
        chart.legend().setLabelColor(self.TEXT_COLOR)
        chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.chart_view.setChart(chart)
        return chart

    def _setup_themed_axes(self, chart: QChart, x_title: str, y_title: str, is_datetime: bool = False):
        """Helper to create and style axes for the light theme."""
        if is_datetime: axis_x = QDateTimeAxis(); axis_x.setFormat("yyyy-MM-dd")
        else: axis_x = QValueAxis()
        
        for axis, title in [(axis_x, x_title), (axis_y := QValueAxis(), y_title)]:
            axis.setTitleText(title)
            axis.setLabelsColor(self.TEXT_COLOR)
            axis.setTitleBrush(QBrush(self.TEXT_COLOR))
            axis.setGridLineVisible(True)
            axis.setGridLinePen(QPen(self.GRID_LINES_COLOR))

        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        return axis_x, axis_y

    def _aggregate_multi_day_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates multi-day predictions into single-day predictions for plotting.
        
        For each prediction origin date, we:
        1. Take the mean of all predictions for that origin date
        2. Compare it to the actual return on the next business day after the origin date
        
        This handles cases where prediction_horizon > 1 (e.g., 5-day predictions).
        """
        if df.empty:
            return df
            
        # Check if we have multi-day predictions
        if COL_FORECAST_HORIZON not in df.columns:
            # Single-day predictions, return as is
            return df
            
        # Group by origin date and calculate mean prediction
        aggregated_df = df.groupby([COL_FORECAST_ORIGIN, 'Ticker']).agg({
            COL_PREDICTED: 'mean',
            COL_ACTUAL: 'first'  # Take the first actual value (next day after origin)
        }).reset_index()
        
        # Rename the origin date to prediction date for consistency
        aggregated_df[COL_DATE] = aggregated_df[COL_FORECAST_ORIGIN]
        
        # Add horizon info for reference
        horizon = df[COL_FORECAST_HORIZON].max()
        aggregated_df['AggregatedHorizon'] = horizon
        
        
        return aggregated_df[['Ticker', COL_DATE, COL_PREDICTED, COL_ACTUAL, 'AggregatedHorizon']]

    def clear_plot(self, message: str = "Wybierz dane do wyświetlenia"):
        """Clears the plot and shows a placeholder title."""
        # Remove any existing treemap widget
        if hasattr(self, 'current_treemap_widget') and self.current_treemap_widget:
            self.layout.removeWidget(self.current_treemap_widget)
            self.current_treemap_widget.deleteLater()
            self.current_treemap_widget = None
        
        # Show the chart view
        self.chart_view.show()
        self._setup_new_chart(message)

    def save_to_file(self, file_path: str):
        chart = self.chart_view.chart()
        scene = self.chart_view.scene()
        rect = scene.sceneRect().toRect()
        
        image = QPixmap(rect.size())
        image.fill(Qt.GlobalColor.transparent)

        painter = QPainter(image)
        scene.render(painter, target=QtCore.QRectF(image.rect()), source=scene.sceneRect())
        painter.end()

        if not image.save(file_path, "PNG"):
            logger.error(f"Failed to save plot to {file_path}.")
        else:
            logger.info(f"Successfully saved plot to {file_path}")

    def plot_equity_curve(self, df: pd.DataFrame):
        chart = self._setup_new_chart("Krzywa Kapitału Portfela")
        if df.empty or COL_PORTFOLIO_VALUE not in df.columns:
            self.clear_plot("Brak danych krzywej kapitału")
            return

        plot_df = df.sort_values(by=COL_PORTFOLIO_DATE)
        axis_x, axis_y = self._setup_themed_axes(chart, "Data", "Wartość Portfela ($)", is_datetime=True)
        chart.legend().setVisible(False)

        series = QLineSeries()
        series.setPen(QPen(self.EQUITY_COLOR, 2))
        for _, row in plot_df.iterrows():
            # Always convert to pd.Timestamp
            dt = pd.to_datetime(row[COL_PORTFOLIO_DATE])
            series.append(QDateTime(dt).toMSecsSinceEpoch(), float(row[COL_PORTFOLIO_VALUE]))

        chart.addSeries(series)
        series.attachAxis(axis_x)
        series.attachAxis(axis_y)

        # Y axis padding
        min_y, max_y = plot_df[COL_PORTFOLIO_VALUE].min(), plot_df[COL_PORTFOLIO_VALUE].max()
        padding = (max_y - min_y) * 0.05 if max_y > min_y else abs(min_y) * 0.05 or 1000
        axis_y.setRange(min_y - padding, max_y + padding)

        # X axis padding (add 2% of the range as margin)
        min_x = pd.to_datetime(plot_df[COL_PORTFOLIO_DATE].min())
        max_x = pd.to_datetime(plot_df[COL_PORTFOLIO_DATE].max())
        if isinstance(min_x, pd.Timestamp) and isinstance(max_x, pd.Timestamp):
            date_range = max_x - min_x
            margin = pd.Timedelta(days=max(1, int(date_range.days * 0.02)))
            axis_x.setRange(QDateTime(min_x - margin), QDateTime(max_x + margin))
        else:
            axis_x.setRange(QDateTime(min_x), QDateTime(max_x))

        axis_x.setGridLineVisible(True)
        axis_y.setGridLineVisible(True)


    def plot_time_series(self, df: pd.DataFrame, ticker: str):
        chart = self._setup_new_chart(f"Prognoza vs. Rzeczywistość dla {ticker}")
        if df.empty: return

        # Handle multi-day predictions
        plot_df = self._aggregate_multi_day_predictions(df)
        if plot_df.empty: return

        axis_x, axis_y = self._setup_themed_axes(chart, "Data", "Wartość Zwrotu", is_datetime=True)
        chart.legend().setVisible(True)

        actual_series = QLineSeries(); actual_series.setName("Zwroty Rzeczywiste"); actual_series.setPen(QPen(self.ACTUAL_COLOR, 2))
        pred_series = QLineSeries(); pred_series.setName("Zwroty Przewidywane"); pred_series.setPen(QPen(self.PREDICTED_COLOR, 2, style=Qt.PenStyle.DashLine))
        
        for _, row in plot_df.iterrows():
            x_val = QDateTime(pd.to_datetime(row[COL_DATE])).toMSecsSinceEpoch()
            actual_series.append(x_val, float(row[COL_ACTUAL]))
            pred_series.append(x_val, float(row[COL_PREDICTED]))
            
        chart.addSeries(actual_series); chart.addSeries(pred_series)
        actual_series.attachAxis(axis_x); actual_series.attachAxis(axis_y)
        pred_series.attachAxis(axis_x); pred_series.attachAxis(axis_y)
        
        all_values = pd.concat([plot_df[COL_ACTUAL], plot_df[COL_PREDICTED]]); min_y, max_y = all_values.min(), all_values.max()
        padding = (max_y - min_y) * 0.1 if max_y > min_y else abs(min_y) * 0.1 or 0.01
        axis_y.setRange(min_y - padding, max_y + padding)
        
    def plot_scatter(self, df: pd.DataFrame, ticker: str):
        chart = self._setup_new_chart(f"Wykres Rozrzutu (Przewidywane vs Rzeczywiste) dla {ticker}")
        if df.empty: return
        
        # Handle multi-day predictions
        plot_df = self._aggregate_multi_day_predictions(df)
        if plot_df.empty: return
        
        axis_x, axis_y = self._setup_themed_axes(chart, "Zwroty Rzeczywiste", "Zwroty Przewidywane")
        
        scatter_series = QScatterSeries(); scatter_series.setName("Obserwacje")
        scatter_series.setMarkerSize(7.0); scatter_series.setColor(self.SCATTER_COLOR); scatter_series.setOpacity(0.6)
        for _, row in plot_df.iterrows():
            scatter_series.append(float(row[COL_ACTUAL]), float(row[COL_PREDICTED]))
        chart.addSeries(scatter_series); scatter_series.attachAxis(axis_x); scatter_series.attachAxis(axis_y)

        all_vals = pd.concat([plot_df[COL_ACTUAL], plot_df[COL_PREDICTED]]); min_val, max_val = all_vals.min(), all_vals.max()
        padding = (max_val - min_val) * 0.1 if max_val > min_val else abs(min_val) * 0.1 or 0.01
        axis_min, axis_max = min_val - padding, max_val + padding

        perfect_line = QLineSeries(); perfect_line.setName("Idealna Prognoza")
        perfect_line.setPen(QPen(self.TEXT_COLOR, 2, Qt.PenStyle.DashLine))
        perfect_line.append(axis_min, axis_min); perfect_line.append(axis_max, axis_max)
        chart.addSeries(perfect_line); perfect_line.attachAxis(axis_x); perfect_line.attachAxis(axis_y)
        
        axis_x.setRange(axis_min, axis_max); axis_y.setRange(axis_min, axis_max)
        chart.legend().setVisible(True)

    def plot_residuals(self, df: pd.DataFrame, ticker: str):
        chart = self._setup_new_chart(f"Analiza Reszt vs. Prognoza dla {ticker}")
        if df.empty: return
            
        # Handle multi-day predictions
        plot_df = self._aggregate_multi_day_predictions(df)
        if plot_df.empty: return
        
        plot_df['residual'] = plot_df[COL_ACTUAL] - plot_df[COL_PREDICTED]
        axis_x, axis_y = self._setup_themed_axes(chart, "Wartości Przewidywane", "Reszty (Błąd)")
        
        residual_series = QScatterSeries(); residual_series.setName("Reszty")
        residual_series.setMarkerSize(7.0); residual_series.setColor(self.RESIDUALS_COLOR); residual_series.setOpacity(0.6)
        for _, row in plot_df.iterrows():
            residual_series.append(float(row[COL_PREDICTED]), float(row['residual']))
        chart.addSeries(residual_series); residual_series.attachAxis(axis_x); residual_series.attachAxis(axis_y)
        
        min_x, max_x = plot_df[COL_PREDICTED].min(), plot_df[COL_PREDICTED].max()

        if len(plot_df) > 1:
            X_fit = plot_df[[COL_PREDICTED]].values; y_fit = plot_df[['residual']].values
            regression = LinearRegression().fit(X_fit, y_fit)
            fit_line = QLineSeries(); fit_line.setName("Linia Trendu Reszt")
            fit_line.setPen(QPen(self.REGRESSION_COLOR, 2))
            fit_line.append(min_x, regression.predict([[min_x]])[0][0])
            fit_line.append(max_x, regression.predict([[max_x]])[0][0])
            chart.addSeries(fit_line); fit_line.attachAxis(axis_x); fit_line.attachAxis(axis_y)
        
        zero_line = QLineSeries(); zero_line.setName("Błąd = 0")
        zero_line.setPen(QPen(self.ZERO_LINE_COLOR, 2, Qt.PenStyle.DashLine))
        zero_line.append(min_x, 0.0); zero_line.append(max_x, 0.0)
        chart.addSeries(zero_line); zero_line.attachAxis(axis_x); zero_line.attachAxis(axis_y)
        
        chart.legend().setVisible(True)
        max_abs_res = plot_df['residual'].abs().max()
        x_pad = (max_x - min_x) * 0.05 if max_x > min_x else abs(max_x)*0.1 or 0.01
        y_pad = max_abs_res * 0.1 or 0.01
        axis_x.setRange(min_x - x_pad, max_x + x_pad)
        axis_y.setRange(-max_abs_res - y_pad, max_abs_res + y_pad)

    def plot_residuals_vs_time(self, df: pd.DataFrame, ticker: str):
        chart = self._setup_new_chart(f"Reszty w Czasie dla {ticker}")
        if df.empty: self.clear_plot(f"Brak danych dla {ticker}"); return

        # Handle multi-day predictions
        plot_df = self._aggregate_multi_day_predictions(df)
        if plot_df.empty: return
        
        plot_df = plot_df.sort_values(by=COL_DATE).copy()
        plot_df['residual'] = plot_df[COL_ACTUAL] - plot_df[COL_PREDICTED]
        
        axis_x, axis_y = self._setup_themed_axes(chart, "Data", "Reszty (Błąd)", is_datetime=True)
        chart.legend().setVisible(False)
        
        residual_series = QLineSeries(); residual_series.setPen(QPen(self.RESIDUALS_COLOR, 2))
        for _, row in plot_df.iterrows():
            residual_series.append(QDateTime(pd.to_datetime(row[COL_DATE])).toMSecsSinceEpoch(), float(row['residual']))
        
        chart.addSeries(residual_series); residual_series.attachAxis(axis_x); residual_series.attachAxis(axis_y)
        
        max_abs_residual = plot_df['residual'].abs().max()
        padding = max_abs_residual * 0.1 or 0.01
        axis_y.setRange(-max_abs_residual - padding, max_abs_residual + padding)

    def plot_final_holdings(self, holdings_dict: dict):
        """
        Creates a treemap visualization of portfolio holdings using a custom widget.
        """
        try:
            if not holdings_dict or not any(v.get('weight_pct', 0) > 0 for v in holdings_dict.values()):
                self.clear_plot("No holdings data available")
                return

            # Remove any existing treemap widget
            if hasattr(self, 'current_treemap_widget') and self.current_treemap_widget:
                self.layout.removeWidget(self.current_treemap_widget)
                self.current_treemap_widget.deleteLater()
                self.current_treemap_widget = None

            # Hide the chart view
            self.chart_view.hide()

            # Create rectangles for the treemap
            rectangles = TreemapWidget.create_treemap_layout_from_holdings(holdings_dict)
            
            # Check if we got valid rectangles
            if not rectangles:
                self.clear_plot("Could not create holdings visualization")
                return
            
            # Create and add the treemap widget
            self.current_treemap_widget = TreemapWidget(rectangles, self.BACKGROUND_COLOR, self.TEXT_COLOR)
            self.layout.addWidget(self.current_treemap_widget)
            
        except Exception as e:
            logger.error(f"Error creating holdings plot: {e}")
            self.clear_plot("Error creating holdings visualization")


class TreemapWidget(QWidget):
    """Custom widget for drawing treemap rectangles."""
    
    def __init__(self, rectangles, bg_color, text_color, parent=None):
        super().__init__(parent)
        self.rectangles = rectangles
        self.bg_color = bg_color
        self.text_color = text_color
        
        # Color palette
        self.palette = [
            QColor("#6c757d"), QColor("#495057"), QColor("#adb5bd"), QColor("#9a8c98"),
            QColor("#7b9acc"), QColor("#a2a2a2"), QColor("#52796f"), QColor("#84a98c"),
            QColor("#cad2c5"), QColor("#c9ada7"),
        ]
        
        # Set widget properties for better expansion
        self.setMinimumSize(300, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"background-color: {self.bg_color.name()};")
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get widget dimensions
        widget_width = self.width()
        widget_height = self.height()
        
        if not self.rectangles or widget_width <= 0 or widget_height <= 0:
            return
        
        # Calculate scale factors to fit rectangles in widget
        max_x = max(rect['x'] + rect['width'] for rect in self.rectangles)
        max_y = max(rect['y'] + rect['height'] for rect in self.rectangles)
        
        # Use the full widget space with small margins
        margin = 30
        available_width = widget_width - 2 * margin
        available_height = widget_height - 2 * margin
        
        scale_x = available_width / max_x
        scale_y = available_height / max_y
        
        # Use the smaller scale to maintain aspect ratio
        scale = min(scale_x, scale_y)
        
        # Center the treemap in the widget
        scaled_width = max_x * scale
        scaled_height = max_y * scale
        offset_x = margin + (available_width - scaled_width) / 2
        offset_y = margin + (available_height - scaled_height) / 2
        
        # Draw each rectangle
        for i, rect in enumerate(self.rectangles):
            # Calculate scaled position and size
            x = offset_x + rect['x'] * scale
            y = offset_y + rect['y'] * scale
            width = rect['width'] * scale
            height = rect['height'] * scale
            
            # Set color with better contrast
            color = self.palette[i % len(self.palette)]
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor("#FFFFFF"), 2))
            
            # Draw rectangle with rounded corners
            painter.drawRoundedRect(int(x), int(y), int(width), int(height), 5, 5)
            
            # Draw text if rectangle is large enough
            if width > 60 and height > 40:
                painter.setPen(QPen(self.text_color))
                font = QFont("Segoe UI", 11, QFont.Weight.Bold)
                painter.setFont(font)
                
                # Draw ticker name
                text_rect = QRect(int(x + 8), int(y + 8), int(width - 16), int(height - 16))
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, rect['label'])
                
                # Draw percentage
                percentage_text = f"{rect['value']:.1f}%"
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight, percentage_text)
            elif width > 40 and height > 25:
                # For medium rectangles, show ticker and percentage
                painter.setPen(QPen(self.text_color))
                font = QFont("Segoe UI", 9)
                painter.setFont(font)
                
                text_rect = QRect(int(x + 5), int(y + 5), int(width - 10), int(height - 10))
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, f"{rect['label']}\n{rect['value']:.1f}%")
            elif width > 25 and height > 15:
                # For smaller rectangles, just show the ticker
                painter.setPen(QPen(self.text_color))
                font = QFont("Segoe UI", 8)
                painter.setFont(font)
                
                text_rect = QRect(int(x + 3), int(y + 3), int(width - 6), int(height - 6))
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, rect['label'])

    @staticmethod
    def create_treemap_layout_from_holdings(holdings_dict):
        """
        Static method to create treemap layout from holdings dictionary.
        Uses a simplified but effective layout algorithm.
        """
        if not holdings_dict or not any(v.get('weight_pct', 0) > 0 for v in holdings_dict.values()):
            return []
        
        # Convert holdings_dict to the format expected by the treemap algorithm
        data_for_treemap = []
        for ticker, data in holdings_dict.items():
            weight = data.get('weight_pct', 0)
            if weight > 0:
                data_for_treemap.append({'label': ticker.upper(), 'value': weight})
        
        if not data_for_treemap:
            return []
        
        # Sort by weight (descending)
        data_for_treemap.sort(key=lambda x: x['value'], reverse=True)
        
        # Calculate total weight
        total_weight = sum(item['value'] for item in data_for_treemap)
        
        # Safety check: ensure total_weight is not zero
        if total_weight <= 0:
            logger.warning("Total weight is zero or negative, cannot create treemap layout")
            return []
        
        # Use a simple grid-based layout for better visual appeal
        return TreemapWidget._create_grid_layout(data_for_treemap, total_weight)
    
    @staticmethod
    def _create_grid_layout(data, total_weight):
        """
        Creates a weight-based layout that's more visually appealing and properly sized.
        """
        if not data:
            return []
        
        # Create a 100x100 coordinate system
        container_width = 100
        container_height = 100
        
        # Sort by weight for better visual hierarchy
        data = sorted(data, key=lambda x: x['value'], reverse=True)
        
        # Calculate total area
        total_area = container_width * container_height
        
        rectangles = []
        current_x = 0
        current_y = 0
        row_height = 0
        max_width = container_width
        
        for item in data:
            # Calculate area based on weight
            weight_ratio = item['value'] / total_weight
            area = total_area * weight_ratio
            
            # Calculate dimensions to maintain good aspect ratio
            if current_x == 0:
                # Start new row
                row_height = min(container_height - current_y, max(20, area / max_width))
                # Ensure row_height is never zero
                if row_height <= 0:
                    row_height = 20  # Default minimum height
                width = min(max_width, area / row_height)
                height = row_height
            else:
                # Continue current row
                # Ensure row_height is never zero
                if row_height <= 0:
                    row_height = 20  # Default minimum height
                width = min(max_width - current_x, area / row_height)
                height = row_height
            
            # Ensure minimum size
            width = max(width, 10)
            height = max(height, 10)
            
            # Check if we need to start a new row
            if current_x + width > max_width:
                current_x = 0
                current_y += row_height
                row_height = min(container_height - current_y, max(20, area / max_width))
                # Ensure row_height is never zero
                if row_height <= 0:
                    row_height = 20  # Default minimum height
                width = min(max_width, area / row_height)
                height = row_height
                width = max(width, 10)
                height = max(height, 10)
            
            # Check if we're out of space
            if current_y + height > container_height:
                break
            
            rectangles.append({
                'label': item['label'],
                'value': item['value'],
                'x': current_x,
                'y': current_y,
                'width': width,
                'height': height
            })
            
            current_x += width
        
        return rectangles
    
    def clear_plot(self):
        """Clear the treemap by setting empty rectangles."""
        self.rectangles = []
        self.update()  # Trigger repaint

    def resizeEvent(self, event):
        """Handle widget resize to ensure proper redrawing."""
        super().resizeEvent(event)
        self.update()  # Trigger repaint when widget is resized