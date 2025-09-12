import logging
from typing import Dict, Any
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QLabel
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtCore import Qt

logger = logging.getLogger(__name__)

class MetricsDisplayWidget(QWidget):
    """A widget to display nested dictionaries of metrics in a QTableWidget."""
    def __init__(self, initial_title: str = "Metryki", parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Add a title label for consistency
        self.title_label = QLabel(f"<b>{initial_title}</b>")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.title_label.font(); font.setPointSize(12)
        self.title_label.setFont(font)
        self.layout.addWidget(self.title_label)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Metryka", "Wartość"]) # Polish translation
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.layout.addWidget(self.table)

    def display_metrics(self, metrics_data: Dict[str, Any]):
        """Clears the table and populates it with new metrics data."""
        self.table.setRowCount(0) # Clear existing rows
        if not metrics_data or not isinstance(metrics_data, dict):
            return
        
        self._populate_table_recursive(metrics_data)
        self.table.resizeColumnsToContents()

    def _populate_table_recursive(self, data: Dict, level: int = 0):
        """Recursively populates the table, indenting sub-metrics."""
        header_font = QFont()
        header_font.setBold(True)
        header_bg_color = QColor("#e9e9e9")

        for key, value in data.items():
            row_position = self.table.rowCount()
            self.table.insertRow(row_position)
            
            if isinstance(value, dict):
                # This is a section header (e.g., "Predictive Performance")
                header_item = QTableWidgetItem(f"{key.replace('_', ' ').title()}")
                header_item.setFont(header_font)
                header_item.setBackground(header_bg_color)
                self.table.setItem(row_position, 0, header_item)
                # Make the header span both columns
                self.table.setSpan(row_position, 0, 1, 2)
                # Recursively populate with sub-items
                self._populate_table_recursive(value, level + 1)
            else:
                # This is a regular key-value pair
                key_item = QTableWidgetItem(f"{'    ' * level}{key.replace('_', ' ').title()}")
                self.table.setItem(row_position, 0, key_item)
                
                # Format the value nicely for display
                if isinstance(value, float):
                    # Use scientific notation for very small numbers, otherwise standard float
                    val_str = f"{value:.4f}" if abs(value) > 1e-5 else f"{value:.2e}"
                elif value is None:
                    val_str = "N/A"
                else:
                    val_str = str(value)
                
                value_item = QTableWidgetItem(val_str)
                self.table.setItem(row_position, 1, value_item)