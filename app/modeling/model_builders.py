from typing import Tuple
from keras import backend
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, LayerNormalization, Dropout, Dense, LeakyReLU


def create_lstm_model(input_shape_param: Tuple[int, int]):
    model = Sequential()
    model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=0.2), input_shape=input_shape_param))
    model.add(LayerNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(100, return_sequences=False, recurrent_dropout=0.2))
    model.add(LayerNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(50))
    model.add(LeakyReLU(negative_slope=0.01))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='huber') # Specify optimizer as string
    # logging.info("LSTM model structure created and compiled.")
    # model.summary(print_fn=logging.debug) # Use debug for summary
    return model
