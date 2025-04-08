
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# === Cargar datos ===
file_path = "datos/jena_climate_2009_2016.csv"  # Asegúrate que esté en esta ruta
df = pd.read_csv(file_path)
df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")

# === Configurar índice temporal ===
df.set_index("Date Time", inplace=True)

# === Serie diaria interpolada ===
df_daily = df["T (degC)"].resample("D").mean()
df_daily_interpolated = df_daily.interpolate()

# === Descomposición estacional ===
decomposition = seasonal_decompose(df_daily_interpolated, model="additive", period=365)

# === ADF Test para verificar estacionariedad ===
adf_test = adfuller(df_daily_interpolated.dropna())
print("Estadístico ADF:", adf_test[0])
print("p-valor:", adf_test[1])

# === División de datos ===
n = len(df_daily_interpolated)
train = df_daily_interpolated[:int(n * 0.8)]
test = df_daily_interpolated[int(n * 0.8):]

# === Ajuste del modelo SARIMA ===
modelo = SARIMAX(train,
                 order=(1,1,1),
                 seasonal_order=(1,1,1,365),
                 enforce_stationarity=False,
                 enforce_invertibility=False)
resultado = modelo.fit()

# === Pronóstico ===
forecast = resultado.get_forecast(steps=len(test))
predicted = forecast.predicted_mean

# === Evaluación ===
rmse = np.sqrt(mean_squared_error(test, predicted))
mape = mean_absolute_percentage_error(test, predicted) * 100
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
print(resultado.summary())

# === Visualización ===
plt.figure(figsize=(14, 6))
plt.plot(train.index, train, label="Entrenamiento", color="skyblue")
plt.plot(test.index, test, label="Real", color="blue")
plt.plot(predicted.index, predicted, label="Pronóstico", color="orange")
plt.title("SARIMA (1,1,1)(1,1,1,365) - Predicción de temperatura diaria")
plt.xlabel("Fecha")
plt.ylabel("Temperatura (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
