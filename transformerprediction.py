import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
from tensorflow import keras
from keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from google.colab import drive

drive.mount('/content/drive')
path = '/content/drive/MyDrive/ColabNotebooks/AmznStock.csv'

#tiempo para correr el script
start_time = time.time()

#importar data
df = pd.read_csv(path, usecols = ['Date','Close'])
df.set_index('Date', inplace = True)
df['Close'] = df['Close'].astype(float)

#poner la data en sets de entrenamiento
train_size = int(len(df)* 0.8) #80% de la data para el entrenamiento
train_data = df.iloc[:train_size].values
test_data = df.iloc[train_size:].values
#mostrar la cantidad de filas en entrenamiento y testeo
print('numero de filas en entrenamiento:',len(train_data))
print('numero de filas en testeo:',len(test_data))

def transformer_encoder(inputs,head_size,num_heads,ff_dim,dropout = 0):
  x = layers.LayerNormalization(epsilon=1e-6)(inputs)
  x = layers.MultiHeadAttention(key_dim = head_size, num_heads = num_heads, dropout = dropout)(x, x)
  x = layers.Dropout(dropout)(x)
  res = x + inputs
  x = layers.LayerNormalization(epsilon=1e-6)(res)
  x = layers.Conv1D(filters = ff_dim, kernel_size = 1, activation = "relu")(x)
  x = layers.Dropout(dropout)(x)
  x = layers.Conv1D(filters = inputs.shape[-1], kernel_size = 1)(x)
  return x + res

def create_model(inputshape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout = 0, mlp_dropout = 0):
  inputs = keras.Input(shape = inputshape)
  x = inputs
  for _ in range(num_transformer_blocks):
    x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
  x = layers.GlobalAveragePooling1D(data_format = "channels_first")(x)
  for dim in mlp_units:
    x = layers.Dense(dim, activation = "relu")(x)
    x = layers.Dropout(mlp_dropout)(x)
  outputs = layers.Dense(1)(x)
  return keras.Model(inputs, outputs)

time_steps = 1
X_train = []
y_train = []

for i in range(len(train_data)- time_steps):
  window = train_data[i:(i + time_steps)]
  after_window = train_data[i + time_steps]
  X_train.append(window)
  y_train.append(after_window)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = []
y_test = []

for i in range(len(test_data)- time_steps):
  window = test_data[i:(i + time_steps)]
  after_window = test_data[i + time_steps]
  X_test.append(window)
  y_test.append(after_window)

X_test = np.array(X_test)
y_test = np.array(y_test)

input_shape = X_train.shape[1:]
model = create_model(input_shape, head_size = 4, num_heads = 4, ff_dim = 64, num_transformer_blocks = 1, mlp_units = [128], mlp_dropout = 0.4, dropout = 0.25)
model.compile(loss = "mean_squared_error", optimizer=keras.optimizers.Adam(learning_rate = 1e-4))
history = model.fit(X_train, y_train, validation_split = 0.2, epochs = 100, batch_size=64)
model.evaluate(X_test, y_test, verbose=1)
pred = model.predict(X_test)

fig = go.Figure()
fig.add_trace(go.Scatter(x = df.index[train_size + time_steps:], y=y_test.flatten(), mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x = df.index[train_size + time_steps:], y=pred.flatten(), mode='lines', name='Predicción'))
fig.update_layout(title='Actual vs Predicha del precio de stock de Amazon (Transformers)', xaxis_title='Fecha', yaxis_title='Precio ($)')
fig.update_layout(title_x=0.5, title_font_size=24, xaxis_title_font_size=18, yaxis_title_font_size=18)
fig.update_xaxes(tickformat='%Y/%m/%d')
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x = df.index[:train_size], y=train_data.flatten(), mode='lines', name = 'Data de entrenamiento'))
fig.add_trace(go.Scatter(x = df.index[train_size + time_steps:], y=y_test.flatten(), mode = 'lines', name = 'Actual'))
fig.add_trace(go.Scatter(x = df.index[train_size + time_steps:], y=pred.flatten(), mode='lines', name='Predicción'))
fig.update_layout(title='Entrenamiento, Actual vs Predicha del precio de stock de Amazon(Transformens)', xaxis_title='Fecha', yaxis_title='Precio ($)')
fig.update_layout(title_x=0.5, title_font_size=24, xaxis_title_font_size=18, yaxis_title_font_size=18)
fig.update_xaxes(tickformat='%Y/%m/%d')
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x = history.epoch, y=history.history['loss'], name='Perdida de entranamiento'))
fig.add_trace(go.Scatter(x = history.epoch, y=history.history['val_loss'], name='Perdida de validacion'))
fig.update_layout(title='Perdida del modelo Amazon(Transformers)', xaxis_title='Epoca', yaxis_title='Perdida')
fig.update_layout(title_x=0.5, title_font_size=24, xaxis_title_font_size=18, yaxis_title_font_size = 18)
fig.show()

y_pred = model.predict(X_test)
test_dates = df.iloc[train_size + time_steps:].index
result_df = pd.DataFrame({'Date': test_dates, 'Actual': y_test.reshape(-1),'Predicción': y_pred.reshape(-1)})
result_df.to_csv('resuLtTRANSFORMERS.csv', index=False)

result_df = pd.read_csv('resuLtTRANSFORMERS.csv')
result_df['Actual'] = result_df['Actual'] / df['Close'].max()
result_df['Predicción'] = result_df['Predicción'] / df['Close'].max()

mse = mean_squared_error(result_df['Actual'], result_df['Predicción'])
print('Mean squared error:', mse)

rmse = np.sqrt(mse)
print('Root mean squared error:', rmse)

mae = mean_absolute_error(result_df['Actual'], result_df['Predicción'])
print('Mean absolute error:', mae)

result_df['Difference'] = (result_df['Actual'] - result_df['Predicción']) / result_df['Actual'] * 100
print('Precisión del modelo: {} %'.format(100 - result_df['Difference'].mean()))


print( "-----%s segundos-----"% (time.time() - start_time))

