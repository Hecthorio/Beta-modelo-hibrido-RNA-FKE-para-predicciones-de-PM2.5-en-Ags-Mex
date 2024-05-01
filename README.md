# Beta modelo hibrido Red Neuronal Artificial - Filtro de Kalman Extendido para el pronostico de concentración de PM<sub>2.5

En este repositorio encontraras un script para generar un modelo de red neuronal artificial (RNA) para pronosticar la concentración de PM<sub>2.5</sub>. La base de datos utilizada se obtuvo del Sistema Nacional de Información de la Calidad del Aire (SINAICA) de la estación CBTIS que se encuentra en la ciudad de Aguascalientes, México. El modelo de RNA es acopado al algoritmo del Filtro de Kalman Extendido para generar correciones a la salida del modelo y mejorar las estimaciones de la concentración. A continuación se describe de manera resumida la obtención, tratamiento, entrenamiento y resultados obtenidos:

## Base de datos
Para el entrenamiento del modelo de RNA se utilizaron las bases de datos del Sistema Nacional de Información sobre la Calidad del Aire (SINAICA); en especifico de la estación CBTIS que se encuentra ubicada en la ciudad de Ags, México. La base de datos es del 2022 y solo se emplearon datos temporales y de concentración de PM<sub>2.5</sub> para el entrenamiento.

Los datos temporales se dividen en tres, hora del día ($hora \in \mathbb{Z} \cap [0-23]$), día de la semana ($dow \in \mathbb{Z} \cap [0-6]$) y día del año ($doy \in \mathbb{Z} \cap [1-365]$). Estos parámetros son variables discretas y periodicas, por lo que se aplicarion funciones seno y coseno para convertirlas a variables de tipo continuo. Esto ayuda al modelo a generar mejores pronosticos.

$$\sin(2\pi t/p)$$

$$\cos(2\pi t/p)$$

Donde $p$ define el periodo de cada variable temporal.

Los parámetros de entrada y salida fueron escalados entre [-1,1] utilizando las funciones siguientes:

$$X=2\frac{x-\min(x)}{\max(x)-\min(x)}-1$$

$$Y=2\frac{y-\min(y)}{\max(y)-\min(y)}-1$$

Donde $x$ es la matriz de datos de entrada, $X$ es la matriz de datos escalados de entrada, $y$ el vector de datos de salida y $Y$ es el vector de datos de salida escalados

## Entrenamiento del modelo
Para el diseño y entrenamiento del modelo se utilizo la libreria de Keras. Los hiperparámetros del modelo propuesto se describen a continuación

- **Topologia**: 7-100-1
- **Función de activación**: tanh (ocualtas) y lineal (salida)
- **Función perdida**: MSE
- **Épocas**: 15
- **Datos entrenamiento, validación y prueba**: 80%-10%-10%
- **Learning-rate** = 0.01
- **Optimizador**: SGD
