# trading_platform/desktop_dashboard/main_gui.py
import sys
import os
import json
import logging
import pandas as pd
import time 
from datetime import date, datetime, timedelta
from typing import Optional, Dict, Any, List 
import numpy as np 
import subprocess
from pathlib import Path
import stat
from PIL import Image, ImageTk
from metrics_widgets import MetricsDisplayWidget
from plot_widgets import ResultsPlotWidget, TreemapWidget
import re
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QLabel, QTableWidget, QTableWidgetItem,
    QTextEdit, QProgressBar, QScrollArea, QSplitter, QGroupBox,
    QGridLayout, QLayout, QLineEdit, QMessageBox, QDateEdit, QSizePolicy,QCheckBox, QFileDialog, QStackedWidget
)
from PyQt6.QtGui import QPixmap, QFont, QColor, QPen, QBrush
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from data_loader import RAW_PREDICTIONS_DIR
from desktop_dashboard.data_loader import get_list_of_backtest_runs, load_run_data
from desktop_dashboard.data_loader import get_list_of_backtest_runs
from data_loader import get_raw_predictions_filepath
from app.common.constants import get_all_portfolio_strategies, get_feature_engineering_strategies, get_predictive_strategies

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("desktop_dashboard.main_gui")
STATUS_UPDATE_PREFIX = "GUI_STATUS_UPDATE::"

DATA_DIR = Path(PROJECT_ROOT) / "data"
PLOTS_DIR = DATA_DIR / "plots"
STATUS_UPDATE_PREFIX = "GUI_STATUS_UPDATE::"

def convert_host_path_to_container(host_path: Path, project_root: Path) -> str:
    relative_path = host_path.relative_to(project_root)
    container_path = Path("/opt/app") / relative_path
    return container_path.as_posix()

def ensure_directory_permissions(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        return True
    except Exception as e:
        logger.error(f"Failed to set permissions for directory {directory}: {e}")
        return False

for dir_path in [DATA_DIR, PLOTS_DIR]:
    if ensure_directory_permissions(dir_path):
        logger.info(f"Ensured directory exists with proper permissions: {dir_path}")
    else:
        logger.warning(f"Could not ensure proper permissions for directory: {dir_path}")

def _convert_container_path_to_host(container_path: str) -> Path:
    if not container_path:
        return Path()
    if container_path.startswith('/opt/app/'):
        container_path = container_path[9:]
    return Path(PROJECT_ROOT) / container_path

class BacktestRunnerThread(QThread):
    log_message = pyqtSignal(str)
    process_finished = pyqtSignal(int)

    def __init__(self, command_list: List[str], project_root_for_subprocess: str):
        super().__init__()
        self.command_list = command_list
        self.project_root_for_subprocess = project_root_for_subprocess
        self.process: Optional[subprocess.Popen] = None
        logger.info(f"BacktestRunnerThread initialized. CWD for subprocess: {self.project_root_for_subprocess}")

    def run(self):
        try:
            self.log_message.emit(f"GUI_LOG: Starting backtest via Docker Compose...")
            self.log_message.emit(f"GUI_LOG: Executing: {' '.join(self.command_list)}")
            self.process = subprocess.Popen(
                self.command_list,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding='utf-8', errors='replace', bufsize=1, universal_newlines=True,
                cwd=self.project_root_for_subprocess,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            pid_msg = f"GUI_LOG: Subprocess started (PID: {self.process.pid if self.process else 'N/A'}). Monitoring output..."
            self.log_message.emit(pid_msg)
            logger.info(pid_msg)
            if self.process and self.process.stdout:
                for line in iter(self.process.stdout.readline, ''):
                    if self.isInterruptionRequested():
                        self.log_message.emit("GUI_LOG: Backtest process interruption requested.")
                        break
                    self.log_message.emit(line.strip())
            exit_code = -1
            if self.process:
                if not self.isInterruptionRequested() and self.process.stdout is not None:
                    self.process.stdout.close()
                    self.process.wait()
                    exit_code = self.process.returncode
                    self.log_message.emit(f"GUI_LOG: Backtest subprocess finished with exit code: {exit_code}")
                else:
                    self.log_message.emit("GUI_LOG: Backtest subprocess was interrupted by GUI signal.")
                    exit_code = -2
            else:
                self.log_message.emit("GUI_ERROR: Subprocess Popen object is None, likely failed to start.")
                exit_code = -102
            self.process_finished.emit(exit_code)
        except FileNotFoundError:
            self.log_message.emit(f"GUI_ERROR: Command '{self.command_list[0]}' not found. Is Docker Compose in PATH?")
            self.process_finished.emit(-100)
        except Exception as e:
            self.log_message.emit(f"GUI_ERROR: Exception running backtest subprocess: {e}")
            logger.error(f"Exception in BacktestRunnerThread: {e}", exc_info=True)
            self.process_finished.emit(-101)

    def stop_process(self):
        self.requestInterruption()
        if self.process and self.process.poll() is None:
            self.log_message.emit("GUI_LOG: Attempting to stop backtest process tree...")
            try:
                if os.name == 'nt':
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                        check=True,
                        capture_output=True
                    )
                    self.log_message.emit("GUI_LOG: taskkill command sent.")
                else:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                    self.log_message.emit("GUI_LOG: SIGTERM sent to process group.")
                self.process.wait(timeout=5)
                self.log_message.emit("GUI_LOG: Subprocess terminated successfully.")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, PermissionError, ProcessLookupError) as e:
                self.log_message.emit(f"GUI_WARN: Graceful terminate failed ({e}), attempting final kill.")
                try:
                    self.process.kill()
                    self.process.wait()
                    self.log_message.emit("GUI_LOG: Subprocess killed.")
                except Exception as kill_e:
                    self.log_message.emit(f"GUI_ERROR: Final kill attempt also failed: {kill_e}")

class BacktestDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading System - Backtest Dashboard")
        self.showMaximized
        self.current_run_id: Optional[str] = None
        self.run_data: Optional[Dict] = None
        self.merged_eval_df: Optional[pd.DataFrame] = None
        self.equity_curve_df: Optional[pd.DataFrame] = None
        self.backtest_thread: Optional[BacktestRunnerThread] = None
        self.active_run_id_from_status: Optional[str] = None
        self.active_run_gui_start_time_monotonic: Optional[float] = None
        self.eta_countdown_timer = QTimer(self)
        self.eta_seconds_remaining = 0
        self._init_controls()
        self._init_ui_layout()
        self._connect_signals()
        QTimer.singleShot(0, self.initial_load)

    def _init_controls(self):
        plot_min_height = 450

        self.new_run_base_config_input = QLineEdit("app_config.yaml")
        self.new_run_profile_config_input = QLineEdit()
        self.new_run_scope_selector = QComboBox()
        self.new_run_scope_selector.addItems(["all_stocks_model", "single_stock_model"])
        self.feature_engineering_strategy_selector = QComboBox()
        self.feature_strategy_label = QLabel("Feature Strategia:")
        self.new_run_tickers_input = QLineEdit("AAPL GOOGL TSLA NVDA AMZN SPY")
        today = date.today()
        self.new_run_training_start_input = QLineEdit((today - timedelta(days=365*2)).strftime('%d-%m-%Y'))
        self.new_run_bt_start_input = QLineEdit((today - timedelta(days=180)).strftime('%d-%m-%Y'))
        self.new_run_bt_end_input = QLineEdit(today.strftime('%d-%m-%Y'))
        self.new_run_horizon_input = QLineEdit("2")
        self.rebalance_days_input = QLineEdit("2")
        self.portfolio_strategy_selector = QComboBox()
        self.top_k_input = QLineEdit("25")
        self.allow_shorting_checkbox = QCheckBox("Krótka sprzedaż")
        self.fully_invested_checkbox = QCheckBox("Gotówka")
        self.fully_invested_checkbox.setChecked(True)
        self.max_position_size_input = QLineEdit("0.10")
        self.max_position_size_input.setPlaceholderText(" 0.10 = 10%")
        self.min_position_size_input = QLineEdit("0.01")
        self.min_position_size_input.setPlaceholderText(" 0.01 = 1%")
        self.load_model_selector = QComboBox()
        self.load_model_selector.setPlaceholderText("Nowy model")

        self.load_predictions_selector = QComboBox(); self.load_predictions_selector.setPlaceholderText("Zestaw z prognozą")

        self.model = QComboBox()
        self.model.addItems(["LSTM_Shuffle", "LSTM_NoShuffle"])
        self.save_model_as_input = QLineEdit()
        self.save_model_as_input.setPlaceholderText("Custom Tag")
        self.new_run_force_retrain_checkbox = QCheckBox("Initial trening"); self.new_run_force_retrain_checkbox.setChecked(True)
        self.new_run_force_retrain_steps_checkbox = QCheckBox("Retrain Every Step")
        self.retrain_frequency_input = QLineEdit(); self.retrain_frequency_input.setPlaceholderText("Odstęp pomiędzy treningami: 5")
        self.epochs_input = QLineEdit("50")
        self.batch_size_input = QLineEdit("64")
        self.run_backtest_button = QPushButton("Nowy Backtest")
        self.stop_backtest_button = QPushButton("Stop Backtest"); self.stop_backtest_button.setEnabled(False)
        self.stop_backtest_button.setStyleSheet("background-color: #ff6b6b; color: white;")
        self.refresh_button = QPushButton("Odśwież")
        self.save_results_button = QPushButton("Zapisz Raport"); self.save_results_button.setEnabled(False)
        self.status_label = QLabel("Status: Idle.")
        self.progress_bar = QProgressBar()
        self.eta_label = QLabel("ETA: N/A")
        self.elapsed_time_label = QLabel("Upłynęło: N/A")
        self.log_view_area = QTextEdit(); self.log_view_area.setReadOnly(True)
        self.run_selector = QComboBox()
        self.ticker_selector = QComboBox()
        self.metrics_display = MetricsDisplayWidget("Metryki")
        
        self.equity_plot = ResultsPlotWidget(initial_title= "Krzywa Kapitału Portfela", parent= self)
        self.equity_plot.setMinimumHeight(plot_min_height)

        self.holdings_plot = ResultsPlotWidget(initial_title="Portfolio Holdings", parent=self)
        self.holdings_plot.setMinimumHeight(plot_min_height)

        self.timeseries_plot = ResultsPlotWidget(initial_title="Prognoza vs. Rzeczywistość",parent=self)
        self.timeseries_plot.setMinimumHeight(plot_min_height)

        self.scatter_plot = ResultsPlotWidget(initial_title="Analiza Reszt vs. Prognoza", parent= self)
        self.scatter_plot.setMinimumHeight(plot_min_height)

        self.residuals_plot = ResultsPlotWidget(initial_title="Analiza Reszt", parent= self)
        self.residuals_plot.setMinimumHeight(plot_min_height)

        self.residuals_time = ResultsPlotWidget(initial_title="Analiza Reszt w Czasie", parent=self)
        self.residuals_time.setMinimumHeight(plot_min_height)

        self.lookback_period_input = QLineEdit("252")
        self.lookback_period_input.setPlaceholderText("252 = 1 rok")

        self.entry_threshold_input = QLineEdit("0.005")
        self.entry_threshold_input.setPlaceholderText("0.005 = 0.5%")
        self.market_ticker_input = QLineEdit("SPY")
        self.market_ticker_input.setPlaceholderText("SPY")
        
        self.stop_loss_pct_input = QLineEdit("0.05")
        self.stop_loss_pct_input.setPlaceholderText("e.g., 0.05 for 5%")
        self.take_profit_pct_input = QLineEdit("0.15")
        self.take_profit_pct_input.setPlaceholderText("e.g., 0.15 for 15%")
        self.trailing_stop_pct_input = QLineEdit("0.03")
        self.trailing_stop_pct_input.setPlaceholderText("e.g., 0.03 for 3%")
        self.enable_stop_loss_checkbox = QCheckBox("Stop-Loss/Take-Profit")
        self.enable_stop_loss_checkbox.setChecked(True)
        self.use_trailing_stop_checkbox = QCheckBox("Trailing Stop")
        self.use_trailing_stop_checkbox.setChecked(True)
        



    def _init_ui_layout(self):
        
        main_container = QWidget()
        main_layout = QVBoxLayout(main_container)
        scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(main_container)
        self.setCentralWidget(scroll_area)
        
        top_bottom_splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(top_bottom_splitter)

        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_bottom_splitter.addWidget(top_widget)
        
        new_run_gb = QGroupBox("Nowy Backtest")
        new_run_grid = QGridLayout(new_run_gb)
        new_run_grid.addWidget(QLabel("<b> Configuracja </b>"), 0, 0, 1, 4)
        new_run_grid.addWidget(QLabel("Base Cfg:"), 1, 0); new_run_grid.addWidget(self.new_run_base_config_input, 1, 1)
        new_run_grid.addWidget(QLabel("Profile Cfg:"), 1, 2); new_run_grid.addWidget(self.new_run_profile_config_input, 1, 3)
        new_run_grid.addWidget(QLabel("Model Scope:"), 2, 0); new_run_grid.addWidget(self.new_run_scope_selector, 2, 1)
        new_run_grid.addWidget(self.feature_strategy_label, 2, 2); new_run_grid.addWidget(self.feature_engineering_strategy_selector, 2, 3)
        new_run_grid.addWidget(QLabel("<b>Data</b>"), 3, 0, 1, 4)
        new_run_grid.addWidget(QLabel("Tickers:"), 4, 0); new_run_grid.addWidget(self.new_run_tickers_input, 4, 1, 1, 3)
        new_run_grid.addWidget(QLabel("Train Start:"), 5, 0); new_run_grid.addWidget(self.new_run_training_start_input, 5, 1)
        new_run_grid.addWidget(QLabel("Backtest Start:"), 5, 2); new_run_grid.addWidget(self.new_run_bt_start_input, 5, 3)
        new_run_grid.addWidget(QLabel("Backtest End:"), 6, 0); new_run_grid.addWidget(self.new_run_bt_end_input, 6, 1)
        new_run_grid.addWidget(QLabel("Prediction Horizon:"), 6, 2); new_run_grid.addWidget(self.new_run_horizon_input, 6, 3)
        new_run_grid.addWidget(QLabel("<b>Portfolio & Model Settings</b>"), 7, 0, 1, 4)
        new_run_grid.addWidget(QLabel("Portfolio Strategy:"), 8, 0); new_run_grid.addWidget(self.portfolio_strategy_selector, 8, 1)
        new_run_grid.addWidget(QLabel("Top K:"), 8, 2); new_run_grid.addWidget(self.top_k_input, 8, 3)
        new_run_grid.addWidget(self.allow_shorting_checkbox, 8, 4)
        new_run_grid.addWidget(self.fully_invested_checkbox, 9, 4)
        
        new_run_grid.addWidget(QLabel("Max Position Size:"), 9, 0); new_run_grid.addWidget(self.max_position_size_input, 9, 1)
        new_run_grid.addWidget(QLabel("Min Position Size:"), 10, 0); new_run_grid.addWidget(self.min_position_size_input, 10, 1)
        new_run_grid.addWidget(QLabel("Rebalance (Days):"), 10, 2); new_run_grid.addWidget(self.rebalance_days_input, 10, 3)
        new_run_grid.addWidget(QLabel("Lookback Period:"), 11, 0); new_run_grid.addWidget(self.lookback_period_input, 11, 1)
        new_run_grid.addWidget(QLabel("Entry Threshold:"), 11, 2); new_run_grid.addWidget(self.entry_threshold_input, 11, 3)
        new_run_grid.addWidget(QLabel("Market Ticker:"), 12, 2); new_run_grid.addWidget(self.market_ticker_input, 12, 3)
        
        new_run_grid.addWidget(QLabel("<b>Risk Management</b>"), 13, 0, 1, 4)
        new_run_grid.addWidget(QLabel("Enable Stop Loss:"), 14, 0); new_run_grid.addWidget(self.enable_stop_loss_checkbox, 14, 1)
        new_run_grid.addWidget(QLabel("Stop Loss %:"), 14, 2); new_run_grid.addWidget(self.stop_loss_pct_input, 14, 3)
        new_run_grid.addWidget(QLabel("Take Profit %:"), 15, 0); new_run_grid.addWidget(self.take_profit_pct_input, 15, 1)
        new_run_grid.addWidget(QLabel("Trailing Stop %:"), 15, 2); new_run_grid.addWidget(self.trailing_stop_pct_input, 15, 3)
        new_run_grid.addWidget(QLabel("Use Trailing Stop:"), 16, 0); new_run_grid.addWidget(self.use_trailing_stop_checkbox, 16, 1)
        
        new_run_grid.addWidget(QLabel("Model Type:"), 19, 2); new_run_grid.addWidget(self.model, 19, 3)
        new_run_grid.addWidget(QLabel("<b>Model Training</b>"), 20, 0, 1, 4)
        new_run_grid.addWidget(QLabel("Load Pre-Trained Model:"), 21, 0); new_run_grid.addWidget(self.load_model_selector, 21, 1, 1, 2)
        new_run_grid.addWidget(QLabel("Load Old Predictions:"), 20, 2); new_run_grid.addWidget(self.load_predictions_selector, 20, 3)

        new_run_grid.addWidget(self.refresh_button, 21, 3)
        new_run_grid.addWidget(QLabel("Epochs:"), 22, 0); new_run_grid.addWidget(self.epochs_input, 22, 1)
        new_run_grid.addWidget(QLabel("Batch Size:"), 22, 2); new_run_grid.addWidget(self.batch_size_input, 22, 3)
        new_run_grid.addWidget(self.new_run_force_retrain_checkbox, 23, 0); new_run_grid.addWidget(self.new_run_force_retrain_steps_checkbox, 23, 1)
        new_run_grid.addWidget(QLabel("Retrain Freq (Days):"), 23, 2); new_run_grid.addWidget(self.retrain_frequency_input, 23, 3)
        new_run_grid.addWidget(QLabel("Custom Name Tag:"), 24, 0); new_run_grid.addWidget(self.save_model_as_input, 24, 1, 1, 3)
        new_run_grid.addWidget(self.run_backtest_button, 25, 0, 1, 2); new_run_grid.addWidget(self.stop_backtest_button, 25, 2, 1, 2)
        top_layout.addWidget(new_run_gb)
        
        progress_gb = QGroupBox("Live Backtest Monitor"); progress_grid = QGridLayout(progress_gb)

        self.progress_bar.setRange(0,100); self.progress_bar.setValue(0); self.progress_bar.setTextVisible(True); self.progress_bar.setFormat("0%")
        self.log_view_area.setReadOnly(True); self.log_view_area.setMinimumHeight(250); self.log_view_area.setFont(QFont("Courier", 8))
        progress_grid.addWidget(QLabel("Status:"),0,0); progress_grid.addWidget(self.status_label,0,1,1,3)
        progress_grid.addWidget(QLabel("Progress:"),1,0); progress_grid.addWidget(self.progress_bar,1,1)
        progress_grid.addWidget(QLabel("ETA:"),1,2); progress_grid.addWidget(self.eta_label,1,3)
        progress_grid.addWidget(QLabel("Elapsed:"),2,0); progress_grid.addWidget(self.elapsed_time_label,2,1)
        progress_grid.addWidget(QLabel("Process Output Log:"),3,0); progress_grid.addWidget(self.log_view_area,4,0,1,4)
        progress_gb.setMinimumHeight(300)
        top_layout.addWidget(progress_gb)

        bottom_widget = QWidget(); bottom_layout = QVBoxLayout(bottom_widget)
        top_bottom_splitter.addWidget(bottom_widget)
        
        controls_group = QGroupBox("Completed Results Viewer"); controls_layout = QHBoxLayout(controls_group)
        controls_layout.addWidget(QLabel("Select Backtest Run:")); controls_layout.addWidget(self.run_selector, 1)
        controls_layout.addWidget(QLabel("View Ticker Details:")); controls_layout.addWidget(self.ticker_selector)
        controls_layout.addStretch(); controls_layout.addWidget(self.save_results_button)
        bottom_layout.addWidget(controls_group)
        
        results_splitter = QSplitter(Qt.Orientation.Horizontal); bottom_layout.addWidget(results_splitter)
        results_splitter.addWidget(self.metrics_display)
        
        self.plot_stack = QStackedWidget(); results_splitter.addWidget(self.plot_stack)
        
        overall_page = QWidget(); overall_layout = QVBoxLayout(overall_page)
        overall_layout.addWidget(self.equity_plot); overall_layout.addWidget(self.holdings_plot)
        self.plot_stack.addWidget(overall_page)
        


        
        self.ticker_plot_page = QSplitter(Qt.Orientation.Vertical)
        self.ticker_plot_page .addWidget(self.timeseries_plot);  self.ticker_plot_page .addWidget(self.scatter_plot);  self.ticker_plot_page .addWidget(self.residuals_plot);  self.ticker_plot_page .addWidget(self.residuals_time)
        self.plot_stack.addWidget( self.ticker_plot_page )

        self.ticker_plot_page.setSizes([400, 400, 400])
        self.plot_stack.addWidget(self.ticker_plot_page)



        top_bottom_splitter.setSizes([600, 800])
        results_splitter.setSizes([450, 1150])
     


    def _connect_signals(self):
        self.run_backtest_button.clicked.connect(self.on_start_new_backtest_clicked)
        self.stop_backtest_button.clicked.connect(self.on_stop_backtest_clicked)
        self.refresh_button.clicked.connect(self.on_refresh_button_clicked)
        self.save_results_button.clicked.connect(self.on_save_results_clicked)
        self.run_selector.currentTextChanged.connect(self.on_run_selected)
        self.ticker_selector.currentTextChanged.connect(self.on_ticker_selected)
        self.load_model_selector.currentTextChanged.connect(self._on_load_option_changed)
        self.load_predictions_selector.currentTextChanged.connect(self._on_load_option_changed)
        self.eta_countdown_timer.timeout.connect(self.update_eta_countdown)
        
        self.portfolio_strategy_selector.currentTextChanged.connect(self._on_strategy_changed)

        self.new_run_force_retrain_steps_checkbox.toggled.connect(
            lambda checked: self.retrain_frequency_input.setDisabled(checked)
        )
        self.retrain_frequency_input.textChanged.connect(
            lambda text: self.new_run_force_retrain_steps_checkbox.setDisabled(bool(text.strip()))
        )

 
    def on_run_selected(self, run_id: str):
        if not run_id or "--" in run_id:
            self.clear_all_displays()
            return
        if run_id == self.current_run_id:
            return
        
        self.load_and_display_run(run_id)


    def on_ticker_selected(self, ticker: str):
        if not self.run_data:
            return

        has_predictions = self.merged_eval_df is not None and not self.merged_eval_df.empty
        is_overall_view = not has_predictions or "--" in ticker

        self.update_metrics_display(ticker)

        if is_overall_view:
            self.plot_stack.setCurrentIndex(0)
            self.equity_plot.plot_equity_curve(self.equity_curve_df)
            self.update_holdings_plot(self.run_data.get("portfolio_performance", {}).get("FinalHoldings", {}))
            self.timeseries_plot.clear_plot(); self.scatter_plot.clear_plot(); self.residuals_plot.clear_plot()
        else:
            self.plot_stack.setCurrentIndex(1)
            ticker_data = self.merged_eval_df[self.merged_eval_df['Ticker'] == ticker]
            self.timeseries_plot.plot_time_series(ticker_data, ticker)
            self.scatter_plot.plot_scatter(ticker_data, ticker)
            self.residuals_plot.plot_residuals(ticker_data, ticker)
            self.equity_plot.clear_plot(); self.holdings_plot.clear_plot()

    def on_refresh_button_clicked(self):
        logger.info("--- User triggered full refresh ---")
        self.initial_load()


    def populate_run_selector(self, select_run_id: Optional[str] = None):
        logger.info(f"Refreshing list of runs. Will try to select: {select_run_id}")
        current_text = self.run_selector.currentText()
        self.run_selector.blockSignals(True)
        self.run_selector.clear()
        available_runs = get_list_of_backtest_runs()
        
        if not available_runs:
            self.run_selector.addItem("-- No Runs Found --"); self.run_selector.setEnabled(False)
            self.run_selector.blockSignals(False)
            self.clear_all_displays(); return

        self.run_selector.setEnabled(True); self.run_selector.addItems(available_runs)
        target_selection = select_run_id or current_text
        if target_selection in available_runs: self.run_selector.setCurrentText(target_selection)
        else: self.run_selector.setCurrentIndex(0)
        
        logger.info(f"Run selector populated. Selected: {self.run_selector.currentText()}")
        self.run_selector.blockSignals(False)
        
        self.on_run_selected(self.run_selector.currentText())

    def populate_model_selector(self):
        logger.info("Refreshing list of available pre-trained models...")
        
        self.load_model_selector.blockSignals(True)
        
        current_selection = self.load_model_selector.currentText()
        self.load_model_selector.clear()
        self.load_model_selector.addItem("")
        
        try:
            models_dir = PROJECT_ROOT / "data" / "models"
            if models_dir.exists():
                model_ids = sorted([d.name for d in models_dir.iterdir() if d.is_dir()])
                self.load_model_selector.addItems(model_ids)
                logger.info(f"Found {len(model_ids)} existing models.")
                
                if current_selection in model_ids:
                    self.load_model_selector.setCurrentText(current_selection)
        except Exception as e:
            logger.error(f"Could not list models in {models_dir}: {e}")
        finally:
            self.load_model_selector.blockSignals(False)
            self.on_load_model_changed(self.load_model_selector.currentText())

    def load_and_display_run(self, run_id: str):
        logger.info(f"Loading data for run: {run_id}")
        self.current_run_id = run_id
        self.run_data = load_run_data(run_id)
        
        if not self.run_data:
            self.clear_all_displays()
            QMessageBox.critical(self, "Load Error", f"Could not load data for run:\n{run_id}")
            return
            
        self.merged_eval_df = self.run_data.get('merged_eval_df', pd.DataFrame())
        self.equity_curve_df = self.run_data.get('equity_curve_df', pd.DataFrame())
        
        self.save_results_button.setEnabled(True)

        self.populate_ticker_selector()
    
    
    def populate_ticker_selector(self):
        self.ticker_selector.blockSignals(True)
        self.ticker_selector.clear()

        has_predictions = self.merged_eval_df is not None and not self.merged_eval_df.empty
        if has_predictions:
            self.ticker_selector.setEnabled(True)
            self.ticker_selector.addItem("-- View Overall Run --")
            tickers = sorted(self.merged_eval_df['Ticker'].unique())
            self.ticker_selector.addItems(tickers)
        else:
            self.ticker_selector.addItem("-- Overall Run Only --")
            self.ticker_selector.setEnabled(False)
        
        self.ticker_selector.blockSignals(False)
        
        self.on_ticker_selected(self.ticker_selector.currentText())


    def update_metrics_display(self, ticker: str):
        metrics_to_show = self.get_metrics_for_display(ticker)
        self.metrics_display.display_metrics(metrics_to_show)
    
    def update_displays_for_run(self):
        if not self.run_data: self.clear_all_displays(); return
        
        current_ticker = self.ticker_selector.currentText()
        has_predictions = self.merged_eval_df is not None and not self.merged_eval_df.empty
        is_overall_view = not has_predictions or "--" in current_ticker

        self.update_metrics_display(current_ticker)
        self.populate_ticker_selector()

        if is_overall_view:
            self.plot_stack.setCurrentIndex(0)
            self.equity_plot.plot_equity_curve(self.equity_curve_df)
            self.update_holdings_plot(self.run_data.get("portfolio_performance", {}).get("FinalHoldings", {}))
        else:
            self.plot_stack.setCurrentIndex(1)
            ticker_data = self.merged_eval_df[self.merged_eval_df['Ticker'] == current_ticker]
            self.timeseries_plot.plot_time_series(ticker_data, current_ticker)
            self.scatter_plot.plot_scatter(ticker_data, current_ticker)
            self.residuals_plot.plot_residuals(ticker_data, current_ticker)
            self.residuals_time_plot.plot_residuals_vs_time(ticker_data, current_ticker)

    def get_metrics_for_display(self, ticker: str) -> Dict:
        if not self.run_data: return {}
        
        is_overall_view = ("--" in ticker or not ticker)
        
        metrics_to_display = {
            "Wyniki Portfela": self.run_data.get("portfolio_performance", {})
        }
        
        if "predictive_performance" in self.run_data and self.run_data["predictive_performance"]:
            metrics_to_display["Ogólne Wyniki Predykcji"] = self.run_data.get("predictive_performance", {})
            if not is_overall_view:
                per_ticker_metrics = self.run_data.get('per_ticker_predictive_performance', {})
                metrics_to_display[f"Wyniki Predykcji dla {ticker}"] = per_ticker_metrics.get(ticker, {})
                
        return metrics_to_display

    def initial_load(self):
        logger.info("Performing initial data load...")
        self._populate_strategy_selectors()
        
        default_strategy = self.portfolio_strategy_selector.currentText()
        if default_strategy:
            self._on_strategy_changed(default_strategy)
        
        self.populate_model_selector()
        self.populate_prediction_run_selector()
        self.populate_run_selector()
        self.clear_all_displays()

        
    def _on_load_option_changed(self, text: str):
        model_to_load = self.load_model_selector.currentText().strip()
        preds_to_load = self.load_predictions_selector.currentText().strip()

        if preds_to_load:
            self.set_training_controls_enabled(False)
            self.set_date_controls_enabled(True)
            self.load_model_selector.setDisabled(True)
        elif model_to_load:
            self.set_training_controls_enabled(False)
            self.set_date_controls_enabled(True)
            self.load_predictions_selector.setDisabled(True)
        else:
            self.set_training_controls_enabled(True)
            self.set_date_controls_enabled(True)
            self.load_model_selector.setDisabled(False)
            self.load_predictions_selector.setDisabled(False)

    def set_training_controls_enabled(self, is_enabled: bool):
        self.new_run_force_retrain_checkbox.setEnabled(is_enabled)
        self.new_run_force_retrain_steps_checkbox.setEnabled(is_enabled)
        self.retrain_frequency_input.setEnabled(is_enabled)
        self.save_model_as_input.setEnabled(is_enabled)
        self.feature_engineering_strategy_selector.setEnabled(is_enabled)
        self.model.setEnabled(is_enabled)
        self.epochs_input.setEnabled(is_enabled)
        self.batch_size_input.setEnabled(is_enabled)
    
    
    
    def set_date_controls_enabled(self, is_enabled: bool):
        self.new_run_training_start_input.setEnabled(is_enabled)
        self.new_run_bt_start_input.setEnabled(is_enabled)
        self.new_run_bt_end_input.setEnabled(is_enabled)


    def auto_set_dates_from_prediction_file(self, run_id: str):
        try:
            preds_filepath = get_raw_predictions_filepath(run_id)
            if preds_filepath.exists():
                df = pd.read_csv(preds_filepath)
                df['PredictionDate'] = pd.to_datetime(df['PredictionDate'])
                min_date, max_date = df['PredictionDate'].min(), df['PredictionDate'].max()
                self.new_run_bt_start_input.setText(min_date.strftime('%Y-%m-%d'))
                self.new_run_bt_end_input.setText(max_date.strftime('%Y-%m-%d'))
                logger.info(f"Backtest dates automatically set from '{run_id}'")
        except Exception as e:
            logger.error(f"Error reading prediction file to set dates: {e}")


    def on_load_model_changed(self, text: str):
        is_loading = bool(text.strip())
        self.new_run_force_retrain_checkbox.setDisabled(is_loading)
        self.new_run_force_retrain_steps_checkbox.setDisabled(is_loading)
        self.retrain_frequency_input.setDisabled(is_loading)
        self.save_model_as_input.setDisabled(is_loading)





    def populate_prediction_run_selector(self):
        self.load_predictions_selector.blockSignals(True)
        self.load_predictions_selector.clear()
        self.load_predictions_selector.addItem("")
        
        try:
            if not RAW_PREDICTIONS_DIR.exists():
                logger.warning(f"Raw predictions directory not found at: {RAW_PREDICTIONS_DIR}")
                self.load_predictions_selector.blockSignals(False)
                return

            runs_with_preds = [
                file.stem.replace('predictions_', '') 
                for file in RAW_PREDICTIONS_DIR.glob("predictions_*.csv")
            ]
            
            self.load_predictions_selector.addItems(sorted(runs_with_preds, reverse=True))
            logger.info(f"Found {len(runs_with_preds)} runs with saved predictions.")

        except Exception as e:
            logger.error(f"Could not list prediction files: {e}", exc_info=True)
        finally:
            self.load_predictions_selector.blockSignals(False)



    def _populate_strategy_selectors(self):
        feature_strategies = get_feature_engineering_strategies()
        self.feature_engineering_strategy_selector.addItems(feature_strategies)
        if "ReturnsRelativeStrengthStrategy" in feature_strategies:
            self.feature_engineering_strategy_selector.setCurrentText("ReturnsRelativeStrengthStrategy")

        portfolio_strategies = get_all_portfolio_strategies()
        self.portfolio_strategy_selector.addItems(portfolio_strategies)
        if "MarkowitzHistoric" in portfolio_strategies:
            self.portfolio_strategy_selector.setCurrentText("MarkowitzHistoric")


    def _reset_live_monitor_ui(self, status_message="Status: Idle."):
        self.status_label.setText(status_message)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0%")
        self.eta_label.setText("ETA: N/A")
        self.elapsed_time_label.setText("Elapsed: N/A")
        
 
        logger.debug("Live monitor UI elements reset.")


    def get_next_run_increment(self) -> int:
        try:
            metrics_dir = PROJECT_ROOT / "data" / "metrics"
            if not metrics_dir.exists(): return 1
            max_num = 0
            for file in metrics_dir.glob("*.json"):
                match = re.search(r'^run(\d+)_', file.name)
                if match:
                    num = int(match.group(1))
                    if num > max_num: max_num = num
            return max_num + 1
        except Exception:
            return 1


    def on_start_new_backtest_clicked(self):
        try:
            tickers_list = self.new_run_tickers_input.text().strip().split()
            if not tickers_list or not tickers_list[0]:
                QMessageBox.warning(self, "Input Error", "Please provide at least one ticker.")
                return

            portfolio_strategy = self.portfolio_strategy_selector.currentText()
            feature_strategy = self.feature_engineering_strategy_selector.currentText()
            model_name = self.model.currentText()
            custom_run_name = self.save_model_as_input.text().strip()
            
            model_to_load = self.load_model_selector.currentText().strip()
            predictions_to_load = self.load_predictions_selector.currentText().strip()

            run_increment = self.get_next_run_increment()
            run_id_parts = [f"run{run_increment:03d}"]

            if custom_run_name:
                safe_custom_name = re.sub(r'[\s\W]+', '_', custom_run_name).strip('_')
                run_id_parts.append(safe_custom_name)

            run_id_parts.append(portfolio_strategy)
            
            PREDICTIVE_STRATEGIES = get_predictive_strategies()
            if portfolio_strategy in PREDICTIVE_STRATEGIES:
                if predictions_to_load:
                    run_id_parts.append("REUSED_PREDS")
                else:
                    run_id_parts.append(feature_strategy)
                    run_id_parts.append(model_name)
            
            run_id_parts.append(datetime.now().strftime('%Y%m%d-%H%M%S'))
            gui_generated_run_id = "_".join(run_id_parts)

            python_command_parts = ["python", "-m", "app.cli", "--run-id", gui_generated_run_id]
            
            python_command_parts.extend(["--base-config", convert_host_path_to_container(PROJECT_ROOT / "configs" / self.new_run_base_config_input.text(), PROJECT_ROOT)])
            if self.new_run_profile_config_input.text():
                python_command_parts.extend(["--profile-config", convert_host_path_to_container(PROJECT_ROOT / "configs" / self.new_run_profile_config_input.text(), PROJECT_ROOT)])

            python_command_parts.extend(["--mode", "backtest"])
            python_command_parts.extend(["--tickers"] + tickers_list)
            python_command_parts.extend(["--training-start-date", self.new_run_training_start_input.text()])
            python_command_parts.extend(["--backtest-start-date", self.new_run_bt_start_input.text()])
            python_command_parts.extend(["--backtest-end-date", self.new_run_bt_end_input.text()])
            python_command_parts.extend(["--portfolio-strategy", portfolio_strategy])
            python_command_parts.extend(["--top-k", self.top_k_input.text()])
            python_command_parts.extend(["--rebalance-days", self.rebalance_days_input.text()])
            
            if self.allow_shorting_checkbox.isChecked():
                python_command_parts.append("--allow-shorting")
            
            if self.fully_invested_checkbox.isChecked():
                python_command_parts.append("--fully-invested")
                

            python_command_parts.extend(["--max-position-size", self.max_position_size_input.text()])
            python_command_parts.extend(["--min-position-size", self.min_position_size_input.text()])
            
            python_command_parts.extend(["--lookback-period", self.lookback_period_input.text()])
            python_command_parts.extend(["--entry-threshold", self.entry_threshold_input.text()])
            python_command_parts.extend(["--market-ticker", self.market_ticker_input.text()])
            
            if self.enable_stop_loss_checkbox.isChecked():
                python_command_parts.append("--enable-stop-loss-take-profit")
                
            if self.use_trailing_stop_checkbox.isChecked():
                python_command_parts.append("--use-trailing-stop")
            python_command_parts.extend(["--stop-loss-pct", self.stop_loss_pct_input.text()])
            python_command_parts.extend(["--take-profit-pct", self.take_profit_pct_input.text()])
            python_command_parts.extend(["--trailing-stop-pct", self.trailing_stop_pct_input.text()])
            

            

            
            if portfolio_strategy in PREDICTIVE_STRATEGIES:
                python_command_parts.extend(["--feature-strategy", feature_strategy])
                python_command_parts.extend(["--model-scope", self.new_run_scope_selector.currentText()])
                python_command_parts.extend(["--prediction-horizon", self.new_run_horizon_input.text()])

                if predictions_to_load:
                    python_command_parts.extend(["--load-predictions-from-run", predictions_to_load])
                elif model_to_load:
                    python_command_parts.extend(["--load-model-id", model_to_load])
                    python_command_parts.extend(["--model", model_name])
                else:
                    python_command_parts.extend(["--model", model_name])
                    python_command_parts.extend(["--epochs", self.epochs_input.text()])
                    python_command_parts.extend(["--batch-size", self.batch_size_input.text()])
                    if self.new_run_force_retrain_checkbox.isChecked():
                        python_command_parts.append("--force-retrain")
                    if self.new_run_force_retrain_steps_checkbox.isChecked():
                        python_command_parts.append("--force-retrain-steps")
                    retrain_freq_str = self.retrain_frequency_input.text().strip()
                    if retrain_freq_str.isdigit() and int(retrain_freq_str) > 0:
                        python_command_parts.extend(["--retrain-frequency", retrain_freq_str])
                    if custom_run_name:
                        python_command_parts.extend(["--save-model-as", custom_run_name])

            command = ["docker-compose", "run", "--rm", "backtest-runner"] + python_command_parts
            
            self.active_run_id_from_status = gui_generated_run_id
            logger.info(f"GUI generated Run ID: {gui_generated_run_id}")
            logger.info(f"Starting DOCKER backtest with command: {' '.join(command)}")

            self.backtest_thread = BacktestRunnerThread(command, str(PROJECT_ROOT))
            self.backtest_thread.log_message.connect(self.process_subprocess_log_message)
            self.backtest_thread.process_finished.connect(self.on_backtest_process_finished)
            
            self.active_run_gui_start_time_monotonic = time.monotonic()
            self.backtest_thread.start()

            self.run_backtest_button.setEnabled(False)
            self.stop_backtest_button.setEnabled(True)

        except Exception as e:
            logger.error(f"Error starting backtest: {e}", exc_info=True)
            QMessageBox.critical(self, "Backtest Start Error", f"An error occurred while preparing the backtest: {e}")

    def on_stop_backtest_clicked(self):
        if self.backtest_thread and self.backtest_thread.isRunning():
            logger.info("GUI: Stop button clicked. Attempting to stop backtest thread.")
            self.backtest_thread.stop_process()
            self.stop_backtest_button.setText("Stopping...")
        else:
            logger.info("GUI: Stop button clicked, but no backtest thread is active.")
    

    def on_backtest_process_finished(self, exit_code: int):
        logger.info(f"GUI: Backtest subprocess finished (Code: {exit_code}).")
        
        run_id_that_just_finished = self.active_run_id_from_status
        
        self.active_run_id_from_status = None
        self._reset_live_monitor_ui() 
        self.run_backtest_button.setEnabled(True)
        self.stop_backtest_button.setEnabled(False)
        self.stop_backtest_button.setText("Stop Current Backtest")
        
        if exit_code == 0:
            QTimer.singleShot(500, self.on_refresh_button_clicked)
            
            if run_id_that_just_finished:
                 QTimer.singleShot(1000, lambda: self.populate_run_selector(select_run_id=run_id_that_just_finished))
        else:
            QMessageBox.critical(self, "Backtest Error", f"Backtest failed with exit code: {exit_code}. Please check the logs for details.")
    

    def update_eta_countdown(self):
        if self.eta_seconds_remaining > 0:
            self.eta_seconds_remaining -= 1
            eta_countdown_str = str(timedelta(seconds=self.eta_seconds_remaining))
            self.eta_label.setText(f"ETA: {eta_countdown_str}")
        else:
            self.eta_countdown_timer.stop()
            self.eta_label.setText("ETA: Finishing...")

    def on_save_results_clicked(self):
        if not self.current_run_id or not self.run_data:
            QMessageBox.warning(self, "Brak Danych", "Proszę wybrać przebieg testu do zapisania.")
            return

        report_dir = PLOTS_DIR / self.current_run_id
        try:
            report_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving essential report to directory: {report_dir}")

            self.equity_plot.save_to_file(str(report_dir / "plot_equity_curve.png"))
            self.holdings_plot.save_to_file(str(report_dir / "plot_final_holdings.png"))

            if self.equity_curve_df is not None and not self.equity_curve_df.empty:
                equity_curve_csv_path = report_dir / "equity_curve.csv"
                self.equity_curve_df.to_csv(equity_curve_csv_path, index=False)
                logger.info(f"Equity curve data saved to {equity_curve_csv_path}")

            metrics_file_path = report_dir / "metrics_summary.txt"
            with open(metrics_file_path, 'w', encoding='utf-8') as f:
                f.write(f"Podsumowanie wyników dla przebiegu: {self.current_run_id}\n")
                f.write("="*80 + "\n\n")
                
                metrics_to_save = self.get_metrics_for_display("-- View Overall Run --")
                if "per_ticker_predictive_performance" in self.run_data:
                    metrics_to_save["Wyniki Predykcji (per Ticker)"] = self.run_data["per_ticker_predictive_performance"]
                
                for title, metrics_dict in metrics_to_save.items():
                    f.write(f"--- {title} ---\n")
                    if not metrics_dict:
                        f.write("  Brak danych.\n")
                    else:
                        for key, value in metrics_dict.items():
                            if isinstance(value, dict):
                                f.write(f"  {key}:\n")
                                for sub_key, sub_val in value.items():
                                    f.write(f"    - {sub_key+':':<25} {sub_val}\n")
                            elif isinstance(value, (int, float)):
                                f.write(f"  {key+':':<30} {value: >10.4f}\n")
                            else:
                                f.write(f"  {key+':':<30} {value}\n")
                    f.write("\n")
            logger.info(f"Metrics summary saved to {metrics_file_path}")

            if self.merged_eval_df is not None and not self.merged_eval_df.empty:
                predictions_csv_path = report_dir / "predictions_and_actuals.csv"
                self.merged_eval_df.to_csv(predictions_csv_path, index=False)
                logger.info(f"Prediction data saved to {predictions_csv_path}")
            
            QMessageBox.information(self, "Zapisano Pomyślnie", f"Raport został zapisany w folderze:\n{report_dir}")

        except Exception as e:
            logger.error(f"Failed to save report: {e}", exc_info=True)
            QMessageBox.critical(self, "Błąd Zapisu", f"Wystąpił błąd podczas zapisywania raportu: {e}")
        
    def clear_all_displays(self):
        self.current_run_id = None
        self.run_data = None
        self.merged_eval_df = pd.DataFrame()
        self.equity_curve_df = pd.DataFrame()
        
        self.metrics_display.display_metrics({})
        self.equity_plot.clear_plot("Wybierz przebieg testu")
        self.holdings_plot.clear_plot("No holdings data")
        self.timeseries_plot.clear_plot()
        self.scatter_plot.clear_plot()
        self.residuals_plot.clear_plot()
        self.residuals_time.clear_plot()
        
        self.ticker_selector.blockSignals(True)
        self.ticker_selector.clear(); self.ticker_selector.addItem("-- Brak Danych --")
        self.ticker_selector.setEnabled(False)
        self.ticker_selector.blockSignals(False)
        
        self.save_results_button.setEnabled(False)
   

    def update_holdings_plot(self, holdings_data):
        if not holdings_data:
            self.holdings_plot.clear_plot("No holdings data available")
            return
        
        self.holdings_plot.plot_final_holdings(holdings_data)

    def process_subprocess_log_message(self, message: str):
            
        scrollbar = self.log_view_area.verticalScrollBar()
        
        is_at_bottom = scrollbar.value() >= (scrollbar.maximum() - 10)

        self.log_view_area.append(message)

        if is_at_bottom:
            scrollbar.setValue(scrollbar.maximum())

        if message.startswith(STATUS_UPDATE_PREFIX):
            try:
                status_json_str = message[len(STATUS_UPDATE_PREFIX):]
                status_data = json.loads(status_json_str)
                self.update_gui_from_status_data(status_data)
            except Exception as e: 
                logger.error(f"GUI: Error processing status: {e} - Data: {message}", exc_info=True)
        
    def update_gui_from_status_data(self, status_data: Dict[str, Any]):
        if run_id := status_data.get('run_id_for_gui'): self.active_run_id_from_status = run_id

        status_msg = status_data.get('status_message', 'N/A')
        current_iter = status_data.get("current_iteration", 0)
        total_iter = status_data.get("total_iterations_approx", 0)
        self.status_label.setText(f"Status: {status_msg}")
        self.progress_bar.setValue(status_data.get("progress_percent", 0))
        self.progress_bar.setFormat(f"{status_data.get('progress_percent', 0)}% ({current_iter}/{total_iter})")
        if self.active_run_gui_start_time_monotonic is not None:
            elapsed_seconds = time.monotonic() - self.active_run_gui_start_time_monotonic
            self.elapsed_time_label.setText(f"Elapsed: {str(timedelta(seconds=int(elapsed_seconds)))}")
        else:
            self.elapsed_time_label.setText("Elapsed: N/A")

        if not self.eta_countdown_timer.isActive():
            
            time_first = status_data.get("time_for_first_iter")
            time_second = status_data.get("time_for_second_iter")
            
            if time_first is not None and time_second is not None:
                
                retrain_freq_str = self.retrain_frequency_input.text().strip()
                retrain_freq = int(retrain_freq_str) if retrain_freq_str.isdigit() and int(retrain_freq_str) > 0 else 0
                is_loading_model = bool(self.load_model_selector.currentText().strip())
                is_retrain_every_step = self.new_run_force_retrain_steps_checkbox.isChecked()

                est_train_time = time_first
                est_predict_time = time_second
                
                if is_loading_model:
                    est_train_time = est_predict_time

                total_iterations = status_data.get("total_iterations_approx", 1)
                current_iteration = status_data.get("current_iteration", 2)
                remaining_steps = total_iterations - current_iteration

                num_future_trains = 0
                if not is_loading_model and remaining_steps > 0:
                    if is_retrain_every_step:
                        num_future_trains = remaining_steps
                    elif retrain_freq > 0:
                        days_since_last = status_data.get("days_since_last_train", 1)
                        steps_to_next_train = retrain_freq - days_since_last
                        if steps_to_next_train <= 0: steps_to_next_train = retrain_freq

                        if remaining_steps >= steps_to_next_train:
                            remaining_after_next = remaining_steps - steps_to_next_train
                            num_future_trains = 1 + (remaining_after_next // retrain_freq)
                
                num_future_predicts = remaining_steps - num_future_trains

                eta_seconds = int((num_future_trains * est_train_time) + (num_future_predicts * est_predict_time))

                self.eta_seconds_remaining = eta_seconds
                self.eta_label.setText(f"ETA: {str(timedelta(seconds=eta_seconds))}")
                self.eta_countdown_timer.stop()
                self.eta_countdown_timer.start(1000)
                logger.info(f"Smart ETA calculated: {eta_seconds}s ({num_future_trains} train, {num_future_predicts} predict steps remaining).")

            elif self.active_run_gui_start_time_monotonic is not None:
                self.eta_label.setText("ETA: Calculating...")

    def _on_strategy_changed(self, strategy_name: str):
        if not strategy_name:
            return
            
        logger.info(f"Strategy changed to: {strategy_name}")
        
        strategy_params = {
            'MarkowitzHistoric': {
                'essential': ['lookback_period', 'top_k', 'rebalance_days', 'allow_shorting', 
                            'fully_invested', 'max_position_size', 'min_position_size', 'entry_threshold'],
                'risk_management': ['enable_stop_loss', 'stop_loss_pct', 'take_profit_pct', 
                                  'trailing_stop_pct', 'use_trailing_stop']
            },
            'MarkowitzHistoricEfficientReturn': {
                'essential': ['lookback_period', 'top_k', 'rebalance_days', 'allow_shorting',
                            'fully_invested', 'max_position_size', 'min_position_size', 'entry_threshold'],
                'risk_management': ['enable_stop_loss', 'stop_loss_pct', 'take_profit_pct',
                                  'trailing_stop_pct', 'use_trailing_stop']            },
            'MinSemiVarianceHistoric': {
                'essential': ['lookback_period', 'top_k', 'rebalance_days', 'allow_shorting',
                            'fully_invested', 'max_position_size', 'min_position_size', 'entry_threshold'],
                'risk_management': ['enable_stop_loss', 'stop_loss_pct', 'take_profit_pct',
                                  'trailing_stop_pct', 'use_trailing_stop']
            },
            'MeanCVaRHistoric': {
                'essential': ['lookback_period', 'top_k', 'rebalance_days', 'allow_shorting',
                            'fully_invested', 'max_position_size', 'min_position_size', 'entry_threshold'],
                'risk_management': ['enable_stop_loss', 'stop_loss_pct', 'take_profit_pct',
                                  'trailing_stop_pct', 'use_trailing_stop']
            },
            
            'MarkowitzPredicted': {
                'essential': ['lookback_period', 'top_k', 'rebalance_days', 'allow_shorting',
                            'fully_invested', 'max_position_size', 'min_position_size', 'entry_threshold'],
                'risk_management': ['enable_stop_loss', 'stop_loss_pct', 'take_profit_pct',
                                  'trailing_stop_pct', 'use_trailing_stop']
            },
            'EnhancedMarkowitzPredicted': {
                'essential': ['lookback_period', 'top_k', 'rebalance_days', 'allow_shorting',
                            'fully_invested', 'max_position_size', 'min_position_size', 'entry_threshold'],
                'risk_management': ['enable_stop_loss', 'stop_loss_pct', 'take_profit_pct',
                                  'trailing_stop_pct', 'use_trailing_stop'],
                'advanced': ['use_shrinkage']
            },
            'TopKPredicted': {
                'essential': ['lookback_period', 'top_k', 'rebalance_days', 'allow_shorting',
                            'fully_invested', 'max_position_size', 'min_position_size', 'entry_threshold'],
                'risk_management': ['enable_stop_loss', 'stop_loss_pct', 'take_profit_pct',
                                  'trailing_stop_pct', 'use_trailing_stop'],
                'advanced': ['use_shrinkage']
            },
            'MinSemiVariancePredicted': {
                'essential': ['lookback_period', 'top_k', 'rebalance_days', 'allow_shorting',
                            'fully_invested', 'max_position_size', 'min_position_size','entry_threshold' ],
                'risk_management': ['enable_stop_loss', 'stop_loss_pct', 'take_profit_pct',
                                  'trailing_stop_pct', 'use_trailing_stop'],
                'advanced': ['use_shrinkage']
            },
            'MinCVaRPredicted': {
                'essential': ['lookback_period', 'top_k', 'rebalance_days', 'allow_shorting',
                            'fully_invested', 'max_position_size', 'min_position_size', 'entry_threshold'],
                'risk_management': ['enable_stop_loss', 'stop_loss_pct', 'take_profit_pct',
                                  'trailing_stop_pct', 'use_trailing_stop'],
                'advanced': ['use_shrinkage']
            },
            'PredictiveMomentumFilter': {
                'essential': ['lookback_period', 'top_k', 'rebalance_days', 'allow_shorting',
                            'fully_invested', 'max_position_size', 'min_position_size', 'entry_threshold'],
                'risk_management': ['enable_stop_loss', 'stop_loss_pct', 'take_profit_pct',
                                'trailing_stop_pct', 'use_trailing_stop'],
                'advanced': ['use_shrinkage']
            }

            
        }
        
        strategy_config = strategy_params.get(strategy_name, {})
        
        param_widgets = {
            'lookback_period': self.lookback_period_input,
            'top_k': self.top_k_input,
            'rebalance_days': self.rebalance_days_input,
            'allow_shorting': self.allow_shorting_checkbox,
            'fully_invested': self.fully_invested_checkbox,
            'max_position_size': self.max_position_size_input,
            'min_position_size': self.min_position_size_input,
            'entry_threshold': self.entry_threshold_input,
            'enable_stop_loss': self.enable_stop_loss_checkbox,
            'stop_loss_pct': self.stop_loss_pct_input,
            'take_profit_pct': self.take_profit_pct_input,
            'trailing_stop_pct': self.trailing_stop_pct_input,
            'use_trailing_stop': self.use_trailing_stop_checkbox,
        }
        
        for param_name, widget in param_widgets.items():
            if param_name in strategy_config.get('essential', []):
                widget.setVisible(True)
                widget.setEnabled(True)
                widget.setStyleSheet("")
            elif param_name in strategy_config.get('risk_management', []):
                widget.setVisible(True)
                widget.setEnabled(True)
                widget.setStyleSheet("")
            elif param_name in strategy_config.get('advanced', []):
                widget.setVisible(True)
                widget.setEnabled(True)
                widget.setStyleSheet("")
            else:
                widget.setVisible(False)
                widget.setEnabled(False)
                widget.setStyleSheet("color: gray; background-color: #f0f0f0;")
        
        PREDICTIVE_STRATEGIES = get_predictive_strategies()
        if strategy_name in PREDICTIVE_STRATEGIES:
            self.feature_engineering_strategy_selector.setEnabled(True)
            self.feature_engineering_strategy_selector.setStyleSheet("")
            if hasattr(self, 'feature_strategy_label'):
                self.feature_strategy_label.setText("Feature Strategy (Required for Predictive)")
                self.feature_strategy_label.setStyleSheet("color: #2E7D32; font-weight: bold;")
        else:
            self.feature_engineering_strategy_selector.setEnabled(False)
            self.feature_engineering_strategy_selector.setStyleSheet("color: gray; background-color: #f0f0f0;")
            self.feature_engineering_strategy_selector.setCurrentText("")
            if hasattr(self, 'feature_strategy_label'):
                self.feature_strategy_label.setText("Feature Strategy (Not needed for Historic)")
                self.feature_strategy_label.setStyleSheet("color: gray;")
        
        logger.info(f"Updated parameter visibility for strategy: {strategy_name}")






if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = BacktestDashboard()
    main_window.show()
    sys.exit(app.exec())









