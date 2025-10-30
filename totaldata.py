import pandas as pd
import plotly.graph_objects as go
from google.colab import drive

drive.mount('/content/drive')
path = '/content/drive/MyDrive/ColabNotebooks/AmazonStock.csv'
# carga del precio de SEP 500 en un dataframe de pandas
df = pd.read_csv(path)

# data a datetime
df['Date'] = pd.to_datetime(df['Date'])

# convertir los datos para modelos graficos
x = df['Date']

y = df['Close']

# muestra historica de datos utilizando plotly
fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', name='Price', line=dict(color='green')))
fig.update_layout(title='Data historica precio stock Amazon', xaxis_title='Date', yaxis_title='Precio cierre en USD ($)')
fig.update_layout(title_x=0.5, title_font_size=24, xaxis_title_font_size=18, yaxis_title_font_size=18)
fig.update_xaxes(tickformat='%d/%m/%Y') 

fig.show()