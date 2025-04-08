import pandas as pd
import numpy as np
import pyperclip
import matplotlib.pyplot as plt
import time

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# === CARGAR DATOS ===
ruta = "C:/Users/10Pearls/Documents/Proyecto final/datos/jena_climate_2009_2016.csv"
df = pd.read_csv(ruta)
df['Date Time'] = pd.to_datetime(df['Date Time'], dayfirst=True)
df.set_index('Date Time', inplace=True)

# === RESAMPLEAR A SEMANAL ===
df_weekly = df['T (degC)'].resample('W').mean()

# === SEPARAR EN ENTRENAMIENTO Y PRUEBA ===
n = len(df_weekly)
train = df_weekly[:int(n * 0.8)]
test = df_weekly[int(n * 0.8):]

# === ENTRENAR MODELO SARIMA(1,1,1)(1,1,1,7) ===
print("🚀 Entrenando modelo SARIMA(1,1,1)(1,1,1,7)...")
inicio = time.time()

model = SARIMAX(train,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False)
results = model.fit()

tiempo_total = time.time() - inicio
print(f"✅ Entrenamiento completado en {tiempo_total:.2f} segundos.")

# === PRONÓSTICO ===
forecast_result = results.get_forecast(steps=len(test))
forecast = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

# === MÉTRICAS ===
rmse = np.sqrt(mean_squared_error(test, forecast))
mape = mean_absolute_percentage_error(test, forecast) * 100

# === RESULTADOS EN FORMATO README ===
salida = f"""
📊 Evaluación del Modelo SARIMA(1,1,1)(1,1,1,7)
-----------------------------------------------
🔹 Tiempo de entrenamiento: {tiempo_total:.2f} segundos
🔹 RMSE: {rmse:.4f}
🔹 MAPE: {mape:.2f}%

Resumen del modelo:
{results.summary()}
"""

# === COPIAR RESULTADO AL PORTAPAPELES ===
pyperclip.copy(salida)
print("📋 Resultados copiados al portapapeles.")

# === GRAFICAR PRONÓSTICO vs REAL ===
plt.figure(figsize=(12, 6))
plt.plot(train[-20:], label="Entrenamiento (últimas semanas)", linestyle='--', color='gray')
plt.plot(test, label="Observado (test)", color='blue')
plt.plot(forecast, label="Pronóstico SARIMA", color='orange')
plt.fill_between(forecast.index,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 color='orange', alpha=0.3)
plt.title("Pronóstico vs Datos reales - SARIMA(1,1,1)(1,1,1,7)")
plt.xlabel("Fecha")
plt.ylabel("Temperatura semanal promedio (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
