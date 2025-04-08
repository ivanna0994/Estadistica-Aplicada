import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")

# Cargar el dataset
df = pd.read_csv("C:/Users/10Pearls/Documents/Proyecto final/datos/jena_climate_2009_2016.csv")

# Parsear fecha y establecer como índice
df['Date Time'] = pd.to_datetime(df['Date Time'], dayfirst=True)
df.set_index('Date Time', inplace=True)


# Seleccionar la variable de temperatura
df_temp = df[['T (degC)']].copy()

# Resamplear a diario
df_daily = df_temp.resample('D').mean()

# Visualización inicial
df_daily.plot(title="Temperatura diaria promedio en Jena")
plt.show()

# Verificar estacionariedad con ADF
adf_result = adfuller(df_daily.dropna())
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")

# Diferenciación si no es estacionaria
df_diff = df_daily.diff().dropna()

# ACF y PACF para identificar órdenes
plot_acf(df_diff)
plt.title("ACF")
plt.show()

plot_pacf(df_diff)
plt.title("PACF")
plt.show()

# Definir y entrenar el modelo SARIMA (puedes ajustar los parámetros)
model = SARIMAX(df_daily, 
                order=(1,1,1), 
                seasonal_order=(1,1,1,7),  # Estacionalidad semanal
                enforce_stationarity=False, 
                enforce_invertibility=False)
results = model.fit()

# Imprimir resumen del modelo
print(results.summary())

# Pronóstico
forecast = results.get_forecast(steps=30)
forecast_ci = forecast.conf_int()

# Gráfico del pronóstico
df_daily.plot(label='Observado', figsize=(12, 6))
forecast.predicted_mean.plot(label='Pronóstico')
plt.fill_between(forecast_ci.index, 
                 forecast_ci.iloc[:, 0], 
                 forecast_ci.iloc[:, 1], 
                 color='pink', alpha=0.3)
plt.title("Pronóstico de Temperatura - SARIMA")
plt.legend()
plt.show()

import pyperclip
from io import StringIO
import sys

# Capturar salida
output = StringIO()
sys.stdout = output

# Aquí va tu print o modelo.summary()
print(results.summary())

# Restaurar consola y copiar
sys.stdout = sys.__stdout__
pyperclip.copy(output.getvalue())