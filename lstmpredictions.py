from sre_constants import error
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import time
from google.colab import drive

drive.mount('/content/drive')
path = '/content/drive/MyDrive/ColabNotebooks/AmznStock.csv'

#tiempo para correr el script
start_time = time.time()

#importar data
df = pd.read_csv(path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace = True)

#normalizar usando el maximo y minimo escalar
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns = df.columns, index = df.index)
#guardar la data normalizada
df = df_scaled

#poner la data en sets de entrenamiento
train_size = int(len(df)* 0.8) #80% de la data para el entrenamiento
train_data = df.iloc[:train_size].values
test_data = df.iloc[train_size:].values
#mostrar la cantidad de filas en entrenamiento y testeo
print('numero de filas en entrenamiento:',len(train_data))
print('numero de filas en testeo:',len(test_data))
#definir el tiempo de los pasos para la lstm
time_steps = 1
num_features = 6

X_train = []
y_train = []

for i in range(time_steps, train_size):
  X_train.append(train_data[i - time_steps:i, :])
  y_train.append(train_data[i, 4])

X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], time_steps, num_features))

X_test = []
y_test = []

for i in range(time_steps, len(test_data)):
  X_test.append(test_data[i - time_steps:i, :])
  y_test.append(test_data[i, 4])

X_test = np.array(X_test)
y_test = np.array(y_test)

X_test = np.reshape(X_test, (X_test.shape[0], time_steps, num_features))

# LSTM

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.LSTM(units=64, input_shape=(time_steps, num_features), activation='tanh', recurrent_activation='sigmoid'))
model.add(tf.keras.layers.Dense(1))

# Compilar lstm
model.compile(optimizer='adam', loss='mean_squared_error')

# ajustar el modelo a la data de entrenamiento
history = model.fit(X_train, y_train, epochs=150, batch_size=64, validation_data=(X_test, y_test))

# Predicciones
predictions = model.predict(X_test)

# cargar resultados en CSV para ser posteriormente utilizados

df_orig = pd.read_csv(path, parse_dates=['Date'], index_col='Date')

y_test_unscaled = scaler.inverse_transform(np.hstack((test_data[time_steps:, :4], y_test.reshape(-1, 1),
test_data[time_steps:, 5:])))[:, 4]

predictions_unscaled = scaler.inverse_transform(np.hstack((test_data[time_steps:, :4], predictions,
test_data[time_steps:, 5:])))[:, 4]


fig = go.Figure()
fig.add_trace(go.Scatter(x=df_orig.index[train_size+time_steps:], y=y_test_unscaled, name='Actual'))
fig.add_trace(go.Scatter(x=df_orig.index[train_size+time_steps:], y=predictions_unscaled, name='Predicción'))
fig.update_layout(title='Actual vs Predicción del precio de stock de Amazon(LSTM)', xaxis_title='Fecha', yaxis_title='Precio ($)')
fig.update_layout(title_x=0.5, title_font_size=24, xaxis_title_font_size=18, yaxis_title_font_size=18)
fig.update_xaxes(tickformat='%Y/%m/%d"') 
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_orig.index[:train_size], y=df_orig['Close'][:train_size], name='Data de entrenamiento'))
fig.add_trace(go.Scatter(x=df_orig.index[train_size+time_steps:], y=y_test_unscaled, name='Actual', line=dict(color='green')))
fig.add_trace(go.Scatter(x=df_orig.index[train_size+time_steps:], y=predictions_unscaled, name='Predicción', line=dict(color='red')))
fig.update_layout(title='Data de entrenamiento, Actual y predicha del precio de stock de Amazon(LSTM)', xaxis_title='Fecha', yaxis_title='Precio ($)')
fig.update_layout(title_x=0.5, title_font_size=24, xaxis_title_font_size=18, yaxis_title_font_size=18)
fig.update_xaxes(tickformat='%Y/%m/%d') 
fig.show()


fig = go.Figure()
fig.add_trace(go.Scatter(x=history.epoch, y=history.history['loss'], name='Perdida en entrenamiento'))
fig.add_trace(go.Scatter(x=history.epoch, y=history.history['val_loss'], name='Perdida de validación'))
fig.update_layout(title='Perdida del modelo Amazon(LSTM)', xaxis_title='Epoca', yaxis_title='Perdida')
fig.update_layout(title_x=0.5, title_font_size=24, xaxis_title_font_size=18, yaxis_title_font_size=18)
fig.show()

accuracy = (100 - np.mean((y_test_unscaled - predictions_unscaled) / predictions_unscaled) * 100)
print('Precisión del modelo: {} %'.format (accuracy))

y_test = (y_test - np.min(y_test)) / (np.max(y_test) - np.min(y_test))
predictions = (predictions - np.min(predictions)) / (np.max(predictions) - np.min(predictions))

mse = mean_squared_error(y_test, predictions)
print('Mean squared error: {}'.format(mse))

rmse = math.sqrt(mse)
print('Root mean squared error: {}'.format(rmse))

mae = mean_absolute_error(y_test, predictions)
print('Mean absolute error: {}'.format(mae))

print("Tiempo de compilación: %s segundos " % (time.time() - start_time))

model.summary ()
