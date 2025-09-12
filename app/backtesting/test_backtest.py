import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import backtrader as bt
from app.backtesting.backtest import PredictionThresholdStrategy
from app.backtesting.performance import evaluate_predictions

class TestBacktest(unittest.TestCase):
    def setUp(self):
        # Create sample data
        self.start_date = datetime(2024, 1, 1)
        self.end_date = datetime(2024, 1, 10)
        self.dates = pd.date_range(self.start_date, self.end_date, freq='B')
        
        # Create sample predictions
        self.predictions = pd.DataFrame({
            'PredictionDate': self.dates,
            'Ticker': ['AAPL'] * len(self.dates),
            'PredictedReturn': np.random.normal(0.001, 0.02, len(self.dates)),
            'ForecastHorizon': [1] * len(self.dates)
        })
        # Normalize prediction dates
        self.predictions['PredictionDateNormalized'] = pd.to_datetime(self.predictions['PredictionDate']).dt.normalize()
        
        # Create sample price data
        self.prices = pd.DataFrame({
            'date': self.dates,
            'open': np.random.normal(100, 1, len(self.dates)),
            'high': np.random.normal(101, 1, len(self.dates)),
            'low': np.random.normal(99, 1, len(self.dates)),
            'close': np.random.normal(100, 1, len(self.dates)),
            'volume': np.random.normal(1000000, 100000, len(self.dates))
        })
        self.prices['close'] = self.prices['close'].cumsum()  # Make it more realistic
        
    def test_backtest_strategy(self):
        # Create Cerebro instance
        cerebro = bt.Cerebro()
        
        # Add data
        data = bt.feeds.PandasData(
            dataname=self.prices,
            datetime='date',
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            name='AAPL'
        )
        cerebro.adddata(data)
        
        # Add strategy
        cerebro.addstrategy(
            PredictionThresholdStrategy,
            predictions_df=self.predictions,
            target_horizon=1,
            buy_threshold=0.001,
            sell_threshold=-0.001,
            position_size_pct=0.1
        )
        
        # Set initial cash
        cerebro.broker.setcash(100000.0)
        
        # Run backtest
        initial_value = cerebro.broker.getvalue()
        results = cerebro.run()
        final_value = cerebro.broker.getvalue()
        
        # Basic assertions
        self.assertIsNotNone(results)
        self.assertGreater(final_value, 0)
        
        # Test plot data generation
        plot_data = evaluate_predictions(
            predictions_df=self.predictions,
            feature_engineer=None,  # We'll mock this in a real test
            data_manager=None,      # We'll mock this in a real test
            ohlcv_table_name='test_table',
            target_column_name='close',
            plots_output_dir='./test_plots',
            output_suffix_for_plots='_test'
        )
        
        self.assertIsNotNone(plot_data)
        self.assertIsInstance(plot_data, tuple)
        self.assertEqual(len(plot_data), 4)  # Should return 4 items

if __name__ == '__main__':
    unittest.main() 