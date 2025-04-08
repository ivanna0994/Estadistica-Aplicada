contenido = """
### 🌡️ Predicción de la Temperatura del Aire
## Comparación y Optimización de Modelos de Series de Tiempo y Aprendizaje Automático

📌 Descripción del Proyecto
Este proyecto analiza diferentes estrategias para predecir la temperatura del aire a partir de datos meteorológicos registrados por la estación del Instituto Max Planck de Bioquímica en Jena, Alemania.

Se busca responder:
¿Cuál es la estrategia más precisa y eficiente para predecir la temperatura del aire: modelos de series de tiempo que capturan autocorrelaciones o modelos de regresión basados en variables climáticas?

También se explora la posibilidad de desarrollar un modelo mejorado que combine ambos enfoques: series de tiempo + aprendizaje automático.

---

📊 Conjunto de Datos
Nombre: Jena Climate
Periodo: 2009-01-01 a 2016-12-31
Frecuencia: cada 10 minutos
Variables: temperatura del aire, presión, humedad, dirección del viento, entre otras (14 en total)

# Guardar como README.md en la raíz del proyecto
with open("README.md", "w", encoding="utf-8") as f:
    f.write(contenido)
