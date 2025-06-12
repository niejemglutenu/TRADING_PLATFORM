# trading_platform/desktop_dashboard/main_gui.py
import sys
import os
import json
import logging
import pandas as pd
import time 
import datetime
from typing import Optional, Dict, Any, List 
import numpy as np 
import subprocess

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QLabel, QTableWidget, QTableWidgetItem,
    QTextEdit, QProgressBar, QScrollArea, QSplitter, QGroupBox,
    QGridLayout, QLayout, QLineEdit, QMessageBox
)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal # QTimer only for post-run refresh

# Assuming data_loader.py is in the same directory and defines PROJECT_ROOT
from data_loader import get_list_of_backtest_runs, load_run_data, PROJECT_ROOT
# STATUS_DIR and LOGS_DIR are not directly used by main_gui anymore if not polling files

import matplotlib.pyplot as plt

# --- Setup Logger for GUI ---
if not logging.getLogger("desktop_dashboard").handlers:
    gui_logger_instance = logging.getLogger("desktop_dashboard")
    gui_logger_instance.setLevel(logging.INFO) # INFO for GUI, DEBUG for subprocess is fine
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s:%(filename)s:%(lineno)d] - %(message)s')
    stream_handler.setFormatter(formatter)
    gui_logger_instance.addHandler(stream_handler)
    gui_logger_instance.propagate = False
logger = logging.getLogger("desktop_dashboard.main_gui")

STATUS_UPDATE_PREFIX = "GUI_STATUS_UPDATE::" # Must match run_system.py


# trading_platform/desktop_dashboard/main_gui.py
import sys
import os
import json
import logging
import pandas as pd
import time 
import datetime
from typing import Optional, Dict, Any, List
import numpy as np
import subprocess

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QLabel, QTableWidget, QTableWidgetItem,
    QTextEdit, QProgressBar, QScrollArea, QSplitter, QGroupBox,
    QGridLayout, QLayout, QLineEdit, QMessageBox, QHeaderView
)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal

# Assuming data_loader.py is in the same directory and defines PROJECT_ROOT
from data_loader import get_list_of_backtest_runs, load_run_data, PROJECT_ROOT

import matplotlib.pyplot as plt # For _create_scatter_plot_image

# Setup basic logging for the GUI
if not logging.getLogger("desktop_dashboard").handlers:
    gui_logger_instance = logging.getLogger("desktop_dashboard")
    gui_logger_instance.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s:%(filename)s:%(lineno)d] - %(message)s')
    stream_handler.setFormatter(formatter)
    gui_logger_instance.addHandler(stream_handler)
    gui_logger_instance.propagate = False
logger = logging.getLogger("desktop_dashboard.main_gui")

STATUS_UPDATE_PREFIX = "GUI_STATUS_UPDATE::" # Must match run_system.py

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
            self.log_message.emit(f"GUI_LOG: Starting backtest via Docker Compose...") # GUI internal log
            self.log_message.emit(f"GUI_LOG: Executing: {' '.join(self.command_list)}")
            
            self.process = subprocess.Popen(
                self.command_list,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding='utf-8', errors='replace', bufsize=1, universal_newlines=True,
                cwd=self.project_root_for_subprocess,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            pid_msg = f"GUI_LOG: Subprocess started (PID: {self.process.pid if self.process else 'N/A'}). Monitoring output..."
            self.log_message.emit(pid_msg); logger.info(pid_msg)

            if self.process and self.process.stdout:
                for line in iter(self.process.stdout.readline, ''):
                    if self.isInterruptionRequested():
                        self.log_message.emit("GUI_LOG: Backtest process interruption requested.")
                        break
                    self.log_message.emit(line.strip()) # Emit each line from subprocess
            
            exit_code = -1 # Default if issues before process.wait()
            if self.process:
                if not self.isInterruptionRequested() and self.process.stdout is not None:
                    # Ensure process has a chance to finish writing before getting returncode
                    self.process.stdout.close() # Close stdout to help Popen.wait() if process writes a lot
                    self.process.wait() 
                    exit_code = self.process.returncode
                    self.log_message.emit(f"GUI_LOG: Backtest subprocess finished with exit code: {exit_code}")
                else:
                    self.log_message.emit("GUI_LOG: Backtest subprocess was interrupted by GUI signal.")
                    exit_code = -2 # Custom code for GUI interruption
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
        # finally: # stdout is closed in the try block now

    def stop_process(self): # Unchanged, looks good
        self.requestInterruption()
        if self.process and self.process.poll() is None:
            self.log_message.emit("GUI_LOG: Attempting to stop backtest subprocess..."); self.process.terminate()
            try:
                self.process.wait(timeout=3); self.log_message.emit("GUI_LOG: Subprocess terminated.")
            except subprocess.TimeoutExpired:
                self.log_message.emit("GUI_LOG: Subprocess kill timeout."); self.process.kill(); self.process.wait()
                self.log_message.emit("GUI_LOG: Subprocess killed.")


class BacktestDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading System - Backtest Dashboard"); self.setGeometry(50, 50, 1600, 1000)
        self.current_run_data: Optional[Dict[str, Any]] = None
        self.backtest_thread: Optional[BacktestRunnerThread] = None
        
        self.active_run_gui_start_time_monotonic: Optional[float] = None # When GUI clicked "Start"
        self.active_run_process_start_time_monotonic: Optional[float] = None # From status: overall_start_time_unix
        self.active_run_total_iterations: Optional[int] = None
        self.active_run_id_from_status: Optional[str] = None

        # UI Elements (initialized here)
        self.run_selector = QComboBox(); self.refresh_runs_button = QPushButton("Refresh Runs List")
        self.status_label = QLabel(); self.progress_bar = QProgressBar(); self.eta_label = QLabel(); self.elapsed_time_label = QLabel()
        self.log_view_area = QTextEdit()
        self.new_run_base_config_input = QLineEdit("app_config.yaml"); self.new_run_profile_config_input = QLineEdit()
        self.new_run_scope_selector = QComboBox(); self.new_run_scope_selector.addItems(["single_stock_model", "all_stocks_model"])
        self.new_run_strategy_selector = QComboBox()
        self.new_run_tickers_input = QLineEdit("AAPL"); self.new_run_training_start_input = QLineEdit(datetime.date.today().replace(year=datetime.date.today().year-1).strftime('%Y-%m-%d'))
        self.new_run_bt_start_input = QLineEdit((datetime.date.today() - datetime.timedelta(days=14)).strftime('%Y-%m-%d')) # Adjusted default
        self.new_run_bt_end_input = QLineEdit((datetime.date.today() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')) # Adjusted default
        self.new_run_horizon_input = QLineEdit("2")
        self.new_run_force_retrain_checkbox = QPushButton("Force Retrain Initial"); self.new_run_force_retrain_checkbox.setCheckable(True); self.new_run_force_retrain_checkbox.setChecked(True)
        self.new_run_force_retrain_steps_checkbox = QPushButton("Force Retrain Steps"); self.new_run_force_retrain_steps_checkbox.setCheckable(True)
        self.run_backtest_button = QPushButton("Start New Dockerized Backtest")
        self.metrics_layout = QVBoxLayout(); self.plot_ticker_selector = QComboBox()
        self.plots_display_layout = QVBoxLayout(); self.plots_scroll_area = QScrollArea()

        self._init_ui_layout()
        self._connect_signals()
        self._populate_new_run_strategy_selector()
        self.populate_run_selector() 
        self._reset_live_monitor_ui()

    def _init_ui_layout(self): # This method arranges pre-initialized widgets
        # ... (Your _init_ui_layout seems fine, ensure all self.widgets are added to layouts) ...
        # Example for Progress GroupBox, ensure all widgets are 'self.'
        main_widget = QWidget(); self.setCentralWidget(main_widget); main_layout = QVBoxLayout(main_widget)
        select_run_gb = QGroupBox("View Completed Backtest Results"); select_run_lay = QHBoxLayout(select_run_gb)
        select_run_lay.addWidget(QLabel("Select Run:")); select_run_lay.addWidget(self.run_selector, 1); select_run_lay.addWidget(self.refresh_runs_button)
        main_layout.addWidget(select_run_gb)
        new_run_gb = QGroupBox("Execute New Backtest (via Docker Compose)"); new_run_grid = QGridLayout(new_run_gb)
        new_run_grid.addWidget(QLabel("Base Cfg:"),0,0); new_run_grid.addWidget(self.new_run_base_config_input,0,1)
        new_run_grid.addWidget(QLabel("Profile Cfg (opt):"),0,2); new_run_grid.addWidget(self.new_run_profile_config_input,0,3)
        new_run_grid.addWidget(QLabel("Scope:"),1,0); new_run_grid.addWidget(self.new_run_scope_selector,1,1)
        new_run_grid.addWidget(QLabel("Strategy:"),1,2); new_run_grid.addWidget(self.new_run_strategy_selector,1,3)
        new_run_grid.addWidget(QLabel("Tickers (CSV):"),2,0); new_run_grid.addWidget(self.new_run_tickers_input,2,1)
        new_run_grid.addWidget(QLabel("Train Start Date:"),2,2); new_run_grid.addWidget(self.new_run_training_start_input,2,3)
        new_run_grid.addWidget(QLabel("BT Start:"),3,0); new_run_grid.addWidget(self.new_run_bt_start_input,3,1)
        new_run_grid.addWidget(QLabel("BT End:"),3,2); new_run_grid.addWidget(self.new_run_bt_end_input,3,3)
        new_run_grid.addWidget(QLabel("Horizon (d):"),4,0); new_run_grid.addWidget(self.new_run_horizon_input,4,1)
        new_run_grid.addWidget(self.new_run_force_retrain_checkbox, 4,2)
        new_run_grid.addWidget(self.new_run_force_retrain_steps_checkbox, 4,3)
        new_run_grid.addWidget(self.run_backtest_button,5,0,1,4)
        main_layout.addWidget(new_run_gb)
        progress_gb = QGroupBox("Live Backtest Monitor"); progress_grid = QGridLayout(progress_gb)
        self.progress_bar.setRange(0,100); self.progress_bar.setValue(0); self.progress_bar.setTextVisible(True); self.progress_bar.setFormat("0%")
        self.log_view_area.setReadOnly(True); self.log_view_area.setMinimumHeight(200); self.log_view_area.setFont(QFont("Courier", 8))
        progress_grid.addWidget(QLabel("Status:"),0,0); progress_grid.addWidget(self.status_label,0,1,1,3)
        progress_grid.addWidget(QLabel("Progress:"),1,0); progress_grid.addWidget(self.progress_bar,1,1)
        progress_grid.addWidget(QLabel("ETA:"),1,2); progress_grid.addWidget(self.eta_label,1,3)
        progress_grid.addWidget(QLabel("Elapsed:"),2,0); progress_grid.addWidget(self.elapsed_time_label,2,1)
        progress_grid.addWidget(QLabel("Process Output Log:"),3,0); progress_grid.addWidget(self.log_view_area,4,0,1,4)
        main_layout.addWidget(progress_gb)
        results_splitter = QSplitter(Qt.Orientation.Horizontal)
        metrics_scroll = QScrollArea(); metrics_scroll.setWidgetResizable(True)
        metrics_content = QWidget(); metrics_content.setLayout(self.metrics_layout)
        metrics_scroll.setWidget(metrics_content); results_splitter.addWidget(metrics_scroll)
        plots_main_w = QWidget(); plots_main_l = QVBoxLayout(plots_main_w)
        plot_ticker_ctrl_w = QWidget(); plot_ticker_sel_l = QHBoxLayout(plot_ticker_ctrl_w)
        plot_ticker_sel_l.addWidget(QLabel("View Plots for Ticker:")); plot_ticker_sel_l.addWidget(self.plot_ticker_selector,1)
        plots_main_l.addWidget(plot_ticker_ctrl_w)
        self.plots_scroll_area.setWidgetResizable(True)
        plots_display_content_w = QWidget(); plots_display_content_w.setLayout(self.plots_display_layout)
        self.plots_scroll_area.setWidget(plots_display_content_w); plots_main_l.addWidget(self.plots_scroll_area,1)
        results_splitter.addWidget(plots_main_w)
        results_splitter.setSizes([600,900]); main_layout.addWidget(results_splitter,1)


    def _connect_signals(self): # Same as before
        self.run_selector.currentTextChanged.connect(self.on_run_selected)
        self.refresh_runs_button.clicked.connect(self.populate_run_selector)
        self.plot_ticker_selector.currentTextChanged.connect(self.on_plot_ticker_selected)
        self.run_backtest_button.clicked.connect(self.on_start_new_backtest_clicked)

    def _populate_new_run_strategy_selector(self): # Same as before
        strategies = ["PastReturnsStrategy", "ReturnsVariationStrategy", "ReturnsVarCorrStrategy"]
        self.new_run_strategy_selector.addItems(strategies)
        if strategies: self.new_run_strategy_selector.setCurrentText("ReturnsVariationStrategy")

    def _reset_live_monitor_ui(self, status_message="Status: Idle."):
        logger.debug(f"Resetting live monitor UI with message: {status_message}")
        self.status_label.setText(status_message)
        self.progress_bar.setValue(0); self.progress_bar.setFormat("0%"); self.progress_bar.setRange(0,100)
        self.eta_label.setText("ETA: N/A"); self.elapsed_time_label.setText("Elapsed: N/A")
        
        # These are CRITICAL to reset
        self.active_run_overall_start_time_monotonic = None
        self.active_run_total_iterations = None
        self.active_run_id_from_status = None

    def on_start_new_backtest_clicked(self):
        if self.backtest_thread and self.backtest_thread.isRunning():
            QMessageBox.warning(self, "Backtest Running", "A backtest process is already running.")
            return

        base_cfg = self.new_run_base_config_input.text().strip()
        profile_cfg = self.new_run_profile_config_input.text().strip()
        model_scope = self.new_run_scope_selector.currentText()
        tickers_str = self.new_run_tickers_input.text().strip()
        feat_strat_key = self.new_run_strategy_selector.currentText()
        train_pool_start = self.new_run_training_start_input.text().strip()
        bt_start = self.new_run_bt_start_input.text().strip()
        bt_end = self.new_run_bt_end_input.text().strip()
        horizon_str = self.new_run_horizon_input.text().strip()
        force_initial = self.new_run_force_retrain_checkbox.isChecked()
        force_steps = self.new_run_force_retrain_steps_checkbox.isChecked()

        if not all([base_cfg, tickers_str, feat_strat_key, train_pool_start, bt_start, bt_end, horizon_str]):
            QMessageBox.critical(self, "Input Error", "All fields for new backtest (except Profile Config) must be filled."); return
        try: pred_horizon = int(horizon_str); assert pred_horizon > 0
        except: QMessageBox.critical(self, "Input Error", "Prediction horizon must be a positive integer."); return

        command = ["docker-compose", "run", "--rm", "backtest-runner", "python", "-m", "app.cli"]
        command.extend(["--base-config", base_cfg])
        if profile_cfg: command.extend(["--profile-config", profile_cfg])
        command.extend(["--mode", "backtest"]); command.extend(["--model-scope", model_scope])
        command.extend(["--tickers-to-predict", tickers_str]); command.extend(["--feature-strategy-key", feat_strat_key])
        command.extend(["--training-pool-start-date", train_pool_start]); command.extend(["--backtest-start-date", bt_start])
        command.extend(["--backtest-end-date", bt_end]); command.extend(["--prediction-horizon-days", str(pred_horizon)])
        if force_initial: command.append("--force-retrain-models")
        if force_steps: command.append("--force-retrain-each-step")
        command.append("--log-level"); command.append("DEBUG"); command.append("--output-artifacts-json")

        self.log_view_area.clear()
        self.process_subprocess_log_message("GUI_LOG: ---- Initiating New Dockerized Backtest ----")
        self._reset_live_monitor_ui("Status: Starting Docker container...")
        self.active_run_overall_start_time_monotonic = time.monotonic() # GUI's knowledge of when it launched the process
        
        self.backtest_thread = BacktestRunnerThread(command, PROJECT_ROOT) # PROJECT_ROOT from data_loader
        self.backtest_thread.log_message.connect(self.process_subprocess_log_message)
        self.backtest_thread.process_finished.connect(self.on_backtest_process_finished)
        self.backtest_thread.start()
        self.run_backtest_button.setEnabled(False)
        logger.info("New backtest process thread started by GUI.")

    def process_subprocess_log_message(self, message: str):
        self.log_view_area.append(message)
        if self.log_view_area.verticalScrollBar is True:
            self.log_view_area.verticalScrollBar().setValue(self.log_view_area.verticalScrollBar().maximum())
        if message.startswith(STATUS_UPDATE_PREFIX):
            try:
                status_json_str = message[len(STATUS_UPDATE_PREFIX):]
                status_data = json.loads(status_json_str)
                self.update_gui_from_status_data(status_data)
            except Exception as e: logger.error(f"GUI: Error processing status update: {e} - Data: {message}", exc_info=True)
        
    def update_gui_from_status_data(self, status_data: Dict[str, Any]):
        if not all([self.status_label, self.progress_bar, self.eta_label, self.elapsed_time_label]): return

        status_run_id = status_data.get("run_id_for_gui")
        current_iter = status_data.get("current_iteration", 0)
        total_iter_from_status = status_data.get("total_iterations_approx", 0)
        status_msg = status_data.get('status_message', 'N/A')
        time_per_iter_from_status_ema = status_data.get("time_per_iteration_sec")
        is_final_status = status_data.get("is_final_run_status", False)
        # This is the monotonic start time *reported by run_system.py*
        process_start_monotonic_from_status = status_data.get("overall_start_time_unix")

        # Latch onto total iterations for the run ID reported by the status
        if status_run_id and (self.active_run_id_from_status != status_run_id or self.active_run_total_iterations is None):
            self.active_run_id_from_status = status_run_id
            self.active_run_total_iterations = total_iter_from_status
            # If the GUI didn't start this specific run (e.g., GUI restarted),
            # use the start time from the status for elapsed calculation.
            if self.active_run_overall_start_time_monotonic is None and process_start_monotonic_from_status is not None:
                 self.active_run_overall_start_time_monotonic = process_start_monotonic_from_status
            logger.info(f"GUI: Monitoring run '{status_run_id}'. Total It: {self.active_run_total_iterations}, Ref Start: {self.active_run_overall_start_time_monotonic}")


        overall_start_from_status_monotonic = status_data.get("overall_start_time_unix")

        if (self.active_run_id_from_status is None and status_run_id) or \
           (status_run_id and self.active_run_id_from_status != status_run_id):
            
            self.active_run_id_from_status = status_run_id
            self.active_run_total_iterations = status_data.get("total_iterations_approx")
            # USE the start time reported by run_system.py as the authoritative start
            self.active_run_overall_start_time_monotonic = overall_start_from_status_monotonic
            logger.info(f"GUI: Monitoring run '{self.active_run_id_from_status}'. "
                        f"Process Start Ref: {self.active_run_overall_start_time_monotonic}, "
                        f"Total It: {self.active_run_total_iterations}")

        self.status_label.setText(f"Status: {status_msg}")
        
        display_total_iter = self.active_run_total_iterations or 0
        
        progress_val = 0
        if display_total_iter > 0:
            progress_val = min(100, int((current_iter / display_total_iter) * 100))
        if is_final_status:
            progress_val = 100
            if display_total_iter > 0: current_iter = display_total_iter

        self.progress_bar.setRange(0, display_total_iter if display_total_iter > 0 else 100)
        self.progress_bar.setValue(current_iter if display_total_iter > 0 and current_iter <= display_total_iter else progress_val)
        self.progress_bar.setFormat(f"{progress_val}% ({current_iter}/{display_total_iter if display_total_iter > 0 else '?'})")
        logger.debug(f"GUI STATUS UPDATE RECEIVED: {status_data}")
        logger.debug(f"BEFORE LATCH: self.active_run_id_from_status={self.active_run_id_from_status}, self.active_run_overall_start_time_monotonic={self.active_run_overall_start_time_monotonic}")
# ... the latching logic ...
        logger.debug(f"AFTER LATCH: self.active_run_id_from_status={self.active_run_id_from_status}, self.active_run_overall_start_time_monotonic={self.active_run_overall_start_time_monotonic}")
        if self.active_run_overall_start_time_monotonic is not None:
            elapsed_seconds = time.monotonic() - self.active_run_overall_start_time_monotonic
            self.elapsed_time_label.setText(f"Elapsed: {str(datetime.timedelta(seconds=int(elapsed_seconds)))}")

            if is_final_status:
                self.eta_label.setText("ETA: Completed" if "error" not in status_msg.lower() else "ETA: Error")
            elif display_total_iter > 0 and current_iter > 0 : # Must have total iterations and some progress
                effective_time_per_iter = 0.0
                if time_per_iter_from_status_ema and time_per_iter_from_status_ema > 0.01:
                    effective_time_per_iter = time_per_iter_from_status_ema
                elif elapsed_seconds > 0.1 and current_iter > 0: # Fallback only if EMA is bad and we have elapsed time
                    effective_time_per_iter = elapsed_seconds / current_iter
                
                if effective_time_per_iter > 0.001:
                    remaining_iters = display_total_iter - current_iter
                    if remaining_iters >= 0:
                        eta_seconds = remaining_iters * effective_time_per_iter
                        self.eta_label.setText(f"ETA: {str(datetime.timedelta(seconds=int(eta_seconds)))}")
                    else: self.eta_label.setText("ETA: Finishing...")
                else: self.eta_label.setText("ETA: Calculating (short iter time)...")
            elif not is_final_status: self.eta_label.setText("ETA: Calculating (iter 0)...")
        elif is_final_status: # Final, but GUI didn't have a start time for it
            self.elapsed_time_label.setText("Elapsed: N/A (run completed)"); self.eta_label.setText("ETA: Completed")
        else: # No GUI-tracked start and not final
            self.elapsed_time_label.setText("Elapsed: N/A"); self.eta_label.setText("ETA: N/A")


    def on_backtest_process_finished(self, exit_code: int):
        logger.info(f"GUI: Backtest subprocess finished (Code: {exit_code}).")
        if self.backtest_thread: self.backtest_thread = None
        self.run_backtest_button.setEnabled(True)
        
        # Update UI to a final state based on the last known status or exit code
        last_status_msg = self.status_label.text()
        if exit_code == 0:
            if "completed" not in last_status_msg.lower() and "error" not in last_status_msg.lower():
                 self._reset_live_monitor_ui("Status: Completed. Refreshing results...")
                 self.progress_bar.setValue(self.progress_bar.maximum()); self.progress_bar.setFormat("100% (Done)")
                 self.eta_label.setText("ETA: Completed")
            logger.info("Backtest successful. Triggering runs list refresh.")
            QTimer.singleShot(1500, self.populate_run_selector)
        else:
            error_ui_msg = f"Status: ERROR (Code: {exit_code}). Check Output Log."
            if "error" not in last_status_msg.lower(): # Avoid overwriting specific error from status update
                self._reset_live_monitor_ui(error_ui_msg)
                self.progress_bar.setValue(self.progress_bar.maximum()); self.progress_bar.setFormat("Error")
                self.eta_label.setText("ETA: Error")
            logger.error(error_ui_msg.replace("Status: ", ""))
            QMessageBox.critical(self, "Backtest Execution Error", error_ui_msg.replace("Status: ", ""))
        
        # Important: Reset these regardless of success/failure for the next GUI-initiated run
        self.active_run_overall_start_time_monotonic = None
        self.active_run_total_iterations = None
        self.active_run_id_from_status = None
    
    
    
    
    def _clear_layout(self, layout: Optional[QLayout]): # Your previous robust version
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                if item is not None: 
                    widget = item.widget()
                    if widget: widget.setParent(None); widget.deleteLater()
                    else:
                        sub_layout = item.layout()
                        if sub_layout: self._clear_layout(sub_layout)
    
    def _create_metrics_table(self, df: pd.DataFrame) -> QTableWidget: # Your previous robust version
        if df.empty: tbl=QTableWidget(0,0); tbl.setVisible(False); return tbl
        tbl=QTableWidget(df.shape[0], df.shape[1]); str_cols=[str(c) for c in df.columns]; tbl.setColumnCount(len(str_cols)); tbl.setHorizontalHeaderLabels(str_cols)
        for r_idx in range(df.shape[0]):
            for c_idx, col_name in enumerate(str_cols): # Iterate over str_cols for df.iloc
                val = df.iloc[r_idx, c_idx] 
                item_str=f"{val:.4f}" if isinstance(val,(float,np.floating)) and not pd.isna(val) and val!=int(val) else (str(int(val)) if isinstance(val,(float,np.floating)) and not pd.isna(val) and val==int(val) else ("N/A" if pd.isna(val) else str(val)))
                item=QTableWidgetItem(item_str)
                if isinstance(val,(float,np.floating,int,np.integer)) and not pd.isna(val): item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter|Qt.AlignmentFlag.AlignRight)
                else: item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter|Qt.AlignmentFlag.AlignLeft)
                tbl.setItem(r_idx,c_idx,item)
        tbl.resizeColumnsToContents(); tbl.setAlternatingRowColors(True); tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tbl.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows); tbl.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        horizontal_header = tbl.horizontalHeader()
        header_h = horizontal_header.height() if horizontal_header is not None else 25
        content_h = sum(tbl.rowHeight(i) for i in range(tbl.rowCount()))
        total_h = header_h + content_h + 10 
        min_h_sensible = header_h + (25 if tbl.rowCount() == 0 else 50) 
        final_h = max(min_h_sensible, total_h)
        tbl.setMinimumHeight(final_h); tbl.setMaximumHeight(final_h + 20)
        tbl.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff); tbl.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return tbl

    
    def display_loaded_run_data(self):
        self.clear_results_display()

        if not self.current_run_data:
            if self.metrics_layout: self.metrics_layout.addWidget(QLabel(f"Failed to load data for run: {self.run_selector.currentText()}"))
            return

        run_id_disp = self.current_run_data.get('run_id', 'N/A')
        if self.metrics_layout: self.metrics_layout.addWidget(QLabel(f"<b>Displaying Results for Run: {run_id_disp}</b>"))
        
        overall_metrics = self.current_run_data.get('overall_horizon_metrics', {})
        if overall_metrics and self.metrics_layout:
            self.metrics_layout.addWidget(QLabel("<b>Overall Metrics per Horizon:</b>"))
            overall_data_for_table = []
            # Convert string keys from JSON (horizons) to int for sorting
            try:
                sorted_horizons = sorted([int(h) for h in overall_metrics.keys()])
            except ValueError:
                logger.error(f"Could not convert all horizon keys in overall_metrics to int: {overall_metrics.keys()}")
                sorted_horizons = sorted(overall_metrics.keys()) # Fallback to string sort

            for horizon_key in sorted_horizons: # Iterate using sorted keys
                metrics = overall_metrics.get(str(horizon_key), {}) # Access with original string key from JSON
                
                row: Dict[str, Any] = {"Horizon": int(horizon_key)} # Ensure horizon is int
                
                for k, v_metric in metrics.items():
                    # Let DataFrame handle initial type inference, preserve NaN for numerics
                    if pd.isna(v_metric):
                        row[k] = np.nan 
                    elif isinstance(v_metric, (float, np.floating, int, np.integer)):
                        row[k] = v_metric # Keep as number
                    else:
                        # Attempt to convert to number if possible, otherwise string
                        try:
                            num_v = float(v_metric) # Try float first
                            if num_v == int(num_v): # Check if it's a whole number
                                row[k] = int(num_v)
                            else:
                                row[k] = num_v
                        except (ValueError, TypeError):
                            row[k] = str(v_metric) # Fallback to string
                overall_data_for_table.append(row)
            
            if overall_data_for_table:
                df_overall = pd.DataFrame(overall_data_for_table)
                # Ensure 'Horizon' is first column for display
                if 'Horizon' in df_overall.columns:
                    cols = ['Horizon'] + [col for col in df_overall.columns if col != 'Horizon']
                    df_overall = df_overall[cols]
                
                table = self._create_metrics_table(df_overall)
                self.metrics_layout.addWidget(table)
            else:
                self.metrics_layout.addWidget(QLabel("No overall metrics data to display."))
        elif self.metrics_layout:
            self.metrics_layout.addWidget(QLabel("No overall metrics data structure found for this run."))
        
        if self.metrics_layout: self.metrics_layout.addSpacing(10) # Add some spacing

        # --- Per-Ticker Metrics Display (similar logic) ---
        per_ticker_metrics = self.current_run_data.get('per_ticker_horizon_metrics', {})
        predictions_df = self.current_run_data.get('predictions_df') # Used to get ticker list

        if per_ticker_metrics and predictions_df is not None and not predictions_df.empty and self.metrics_layout:
            # For simplicity, let's make a dropdown for ticker selection for detailed metrics
            # Or just show the first one as before
            example_ticker = predictions_df['Ticker'].unique()[0]
            if example_ticker in per_ticker_metrics:
                self.metrics_layout.addWidget(QLabel(f"<b>Per-Ticker Metrics for {example_ticker} (per Horizon):</b>"))
                ticker_metric_table_data = []
                ticker_horizons_data = per_ticker_metrics[example_ticker]
                try:
                    sorted_ticker_horizons = sorted([int(h) for h in ticker_horizons_data.keys()])
                except ValueError:
                    sorted_ticker_horizons = sorted(ticker_horizons_data.keys())

                for horizon_key in sorted_ticker_horizons:
                    metrics = ticker_horizons_data.get(str(horizon_key), {})
                    row_ticker: Dict[str, Any] = {"Horizon": int(horizon_key)}
                    for k, v_metric in metrics.items():
                        if pd.isna(v_metric): row_ticker[k] = np.nan
                        elif isinstance(v_metric, (float, np.floating, int, np.integer)): row_ticker[k] = v_metric
                        else:
                            try:
                                num_v = float(v_metric); row_ticker[k] = int(num_v) if num_v == int(num_v) else num_v
                            except (ValueError, TypeError): row_ticker[k] = str(v_metric)
                    ticker_metric_table_data.append(row_ticker)
                
                if ticker_metric_table_data:
                    df_ticker_metrics = pd.DataFrame(ticker_metric_table_data)
                    if 'Horizon' in df_ticker_metrics.columns: # Ensure Horizon is first
                        cols_tick = ['Horizon'] + [col for col in df_ticker_metrics.columns if col != 'Horizon']
                        df_ticker_metrics = df_ticker_metrics[cols_tick]
                    table_ticker = self._create_metrics_table(df_ticker_metrics)
                    self.metrics_layout.addWidget(table_ticker)
        elif self.metrics_layout:
            self.metrics_layout.addWidget(QLabel("No per-ticker metrics data found or no predictions to determine tickers."))

        if self.metrics_layout: self.metrics_layout.addSpacing(10)

        predictions_df = self.current_run_data.get('predictions_df')
        if predictions_df is not None and not predictions_df.empty and self.plot_ticker_selector:
            unique_tickers = sorted(predictions_df['Ticker'].unique())
            self.plot_ticker_selector.blockSignals(True)
            current_plot_ticker = self.plot_ticker_selector.currentText()
            self.plot_ticker_selector.clear()
            self.plot_ticker_selector.addItems(unique_tickers)
            if current_plot_ticker in unique_tickers:
                self.plot_ticker_selector.setCurrentText(current_plot_ticker)
            elif unique_tickers:
                self.plot_ticker_selector.setCurrentIndex(0)
            self.plot_ticker_selector.blockSignals(False)
            
            # Manually trigger if text actually changed or if it's the first population
            if self.plot_ticker_selector.currentText() and \
               (self.plot_ticker_selector.property("last_selected_text") != self.plot_ticker_selector.currentText() or \
                not self.plots_display_layout.count()): # Check if plots area is empty
                self.on_plot_ticker_selected(self.plot_ticker_selector.currentText())
            self.plot_ticker_selector.setProperty("last_selected_text", self.plot_ticker_selector.currentText())

        elif self.plots_display_layout:
             self.plots_display_layout.addWidget(QLabel("No prediction data available to generate plots."))
        
        if self.metrics_layout: self.metrics_layout.addStretch(1)

    def load_selected_run_data(self):
        selected_run_id = self.run_selector.currentText()
        if not selected_run_id:
            self.clear_results_display()
            if self.metrics_layout: self.metrics_layout.addWidget(QLabel("No run selected."))
            return
        logger.info(f"Loading data for completed run: {selected_run_id}")
        self.current_run_data = load_run_data(selected_run_id)
        self.display_loaded_run_data()

    def clear_results_display(self): # Your previous robust version
        self._clear_layout(self.metrics_layout)
        self._clear_layout(self.plots_display_layout)
        if self.plot_ticker_selector:
            self.plot_ticker_selector.blockSignals(True); self.plot_ticker_selector.clear(); self.plot_ticker_selector.blockSignals(False)

    def on_run_selected(self, run_id: str):
            if not run_id: self.clear_results_display(); return
            if self.current_run_data and self.current_run_data.get('run_id') == run_id: return
            self.load_selected_run_data()


    def _create_scatter_plot_image(self, df: pd.DataFrame, ticker: str, run_id_suffix: str) -> Optional[str]: # Your previous robust version
        if df.empty or 'ActualValue' not in df.columns or 'PredictedReturn' not in df.columns: return None
        try:
            fig, ax = plt.subplots(figsize=(7,7)); ax.scatter(df['ActualValue'], df['PredictedReturn'], alpha=0.6, s=25, ec='k', lw=0.3, label="Preds")
            all_v = pd.concat([df['ActualValue'], df['PredictedReturn']]).dropna();
            if all_v.empty: min_v,max_v = -0.05,0.05
            else: min_v,max_v = all_v.min(),all_v.max()
            pad = (max_v-min_v)*0.1 if (max_v-min_v)>1e-6 else 0.01; lims=[min_v-pad,max_v+pad]
            ax.plot(lims,lims,'r--',alpha=0.7,zorder=0,label="y=x"); ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
            ax.set_title(f"Scatter: {ticker} ({run_id_suffix[:30]})"); ax.grid(True,ls=':',alpha=0.5);ax.axhline(0,c='k',lw=0.5,ls='--');ax.axvline(0,c='k',lw=0.5,ls='--')
            ax.legend(); fig.tight_layout()
            gui_plots_dir = os.path.join(PROJECT_ROOT, "data", "gui_temp_scatter_plots") 
            os.makedirs(gui_plots_dir, exist_ok=True)
            safe_id = run_id_suffix.replace('/','_').replace('\\','_')[:50]
            f_name = f"scatter_{ticker.replace('.','_').replace('^','')}_{safe_id}.png"; p_path = os.path.join(gui_plots_dir, f_name)
            fig.savefig(p_path); plt.close(fig); logger.info(f"Scatter plot saved: {p_path}"); return p_path
        except Exception as e: logger.error(f"Scatter plot error for {ticker}: {e}",exc_info=True); return None



    def on_plot_ticker_selected(self, ticker: str): # Your previous robust version
        if not ticker or not self.current_run_data or not self.plots_display_layout: return
        logger.info(f"Displaying plots for: {ticker}")
        self._clear_layout(self.plots_display_layout)
        preds_df = self.current_run_data.get('predictions_df'); plot_map = self.current_run_data.get('plot_files', {})
        if preds_df is not None and not preds_df.empty:
            ticker_df = preds_df[preds_df['Ticker'] == ticker]
            if not ticker_df.empty:
                vp = self.plots_scroll_area.viewport(); max_w = vp.width() - 25 if vp else 600; max_w = max(max_w,100)
                if ticker in plot_map and os.path.exists(plot_map[ticker]): # plot_map value is absolute path
                    self.plots_display_layout.addWidget(QLabel(f"<b>Eval Plot: {ticker}</b>")); lbl=QLabel(); pxm=QPixmap(plot_map[ticker])
                    if pxm.width()>max_w: pxm=pxm.scaledToWidth(max_w, Qt.TransformationMode.SmoothTransformation)
                    lbl.setPixmap(pxm); self.plots_display_layout.addWidget(lbl)
                else: self.plots_display_layout.addWidget(QLabel(f"Line plot for {ticker} not found: '{plot_map.get(ticker)}'"))
                if 'ActualValue' in ticker_df.columns and 'PredictedReturn' in ticker_df.columns:
                    scatter_df = ticker_df[['ActualValue', 'PredictedReturn']].dropna()
                    if not scatter_df.empty:
                        run_id = self.current_run_data.get('run_id', 'curr'); s_path = self._create_scatter_plot_image(scatter_df, ticker, run_id)
                        if s_path and os.path.exists(s_path):
                            self.plots_display_layout.addWidget(QLabel(f"<b>Scatter Plot: {ticker}</b>")); lbl_s=QLabel();pxm_s=QPixmap(s_path)
                            if pxm_s.width()>max_w: pxm_s=pxm_s.scaledToWidth(max_w, Qt.TransformationMode.SmoothTransformation)
                            lbl_s.setPixmap(pxm_s); self.plots_display_layout.addWidget(lbl_s)
                        else: self.plots_display_layout.addWidget(QLabel(f"Scatter plot for {ticker} missing."))
            else: self.plots_display_layout.addWidget(QLabel(f"No data for ticker {ticker}."))
        if self.plots_display_layout: self.plots_display_layout.addStretch(1)




# In class BacktestDashboard(QMainWindow):

    def populate_run_selector(self):
        logger.debug("Populating run selector for completed runs...")
        self.run_selector.blockSignals(True) # Block signals during modification
        
        previous_selection = self.run_selector.currentText() # Store what was selected
        self.run_selector.clear()
        
        runs = get_list_of_backtest_runs() # Get fresh list of runs
        
        newly_selected_run_id_after_populate: Optional[str] = None

        if runs:
            self.run_selector.addItems(runs)
            # Try to re-select the previously selected run if it still exists
            if previous_selection and previous_selection in runs:
                self.run_selector.setCurrentText(previous_selection)
                newly_selected_run_id_after_populate = previous_selection
            else: # Otherwise, select the first (newest) run
                self.run_selector.setCurrentIndex(0)
                newly_selected_run_id_after_populate = self.run_selector.currentText()
            logger.info(f"Run selector populated. Current selection: {newly_selected_run_id_after_populate}")
        else:
            self.clear_results_display() 
            # Ensure metrics_layout exists before adding a widget
            if self.metrics_layout:
                self._clear_layout(self.metrics_layout) # Clear previous "No runs" message if any
                self.metrics_layout.addWidget(QLabel("No completed backtest runs found."))
            logger.info("No completed backtest runs found to populate selector.")
            
        self.run_selector.blockSignals(False)
        
        # Explicitly trigger data loading for the current selection
        # This ensures that even if setCurrentText didn't trigger currentTextChanged
        # (e.g., if the text was the same), the data is loaded or re-evaluated.
        if newly_selected_run_id_after_populate:
            # Check if we need to force a reload or if the data is already current
            if not self.current_run_data or self.current_run_data.get('run_id') != newly_selected_run_id_after_populate:
                self.on_run_selected(newly_selected_run_id_after_populate) # This will call load and display
            else:
                logger.debug(f"Data for run '{newly_selected_run_id_after_populate}' seems to be already loaded. No explicit reload.")
        elif not runs: # If selector is empty after populating
            self.clear_results_display()
            if self.metrics_layout: # Check again for safety
                self._clear_layout(self.metrics_layout)
                self.metrics_layout.addWidget(QLabel("No completed backtest runs found."))


    def closeEvent(self, event): # Same as before
        logger.info("Close event for dashboard. Stopping active processes.")
        if self.backtest_thread and self.backtest_thread.isRunning():
            self.backtest_thread.stop_process()
            self.backtest_thread.quit()
            if not self.backtest_thread.wait(3000):
                logger.warning("Backtest thread did not finish cleanly during GUI close.")
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = BacktestDashboard()
    main_window.show()
    sys.exit(app.exec())









