from statsmodels.tsa.statespace.sarimax import SARIMAX

# Asegúrate de que no haya valores NaN
serie_final = df_daily_interpolated.dropna()

# Dividir en entrenamiento y prueba
n = len(serie_final)
train = serie_final[:int(n * 0.8)]
test = serie_final[int(n * 0.8):]

# Ajustar modelo SARIMA con estacionalidad anual
modelo = SARIMAX(train,
                 order=(1,1,1),
                 seasonal_order=(1,1,1,365),  # Anual
                 enforce_stationarity=False,
                 enforce_invertibility=False)
resultado = modelo.fit()

# Predicción
forecast = resultado.get_forecast(steps=len(test))
predicted = forecast.predicted_mean

# Evaluación
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
rmse = np.sqrt(mean_squared_error(test, predicted))
mape = mean_absolute_percentage_error(test, predicted) * 100

print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
print(resultado.summary())
