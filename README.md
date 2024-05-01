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

Los parametros de entrada al modelo son las variables temporales continuas: sin(hour), cos(hour), sin(dow), cos(dow), sin(doy) y cos(doy) y el estado previo del sistema (una hora antes): PM<sub>2.5</sub>(t-1); mientras que la salida del modelo es la concentración en la hora actual: PM<sub>2.5</sub>(t).

Las metricas para medir el desempeño del modelo fueron el Error Cuadratico Medio, MSE (por sus siglas en inglés) y el coeficiente de determinación $\text{R}^2$

$$\text{R}^2=1-\frac{\displaystyle\sum_{i=1}^{N}(y_{RNA,i}-y_i)^2}{\displaystyle\sum_{i=1}^{N}(\bar{y}-y_i)^2}$$

$$\text{MSE}=\frac{1}{N}\sum_{i=1}^{N}(y_{RNA,i}-y_i)^2$$

## Filtro de Kalman Extendido

El algorimo de Filtro de Kalman es ampliamente usado para el seguimiento de señales y localización, navegación, control automatico, procesamiento de señales, economia y finanzas, etc. Para este analisis de procedio a utilizar la versión de Filtro de Kalman Extendido (FKE) para generar correcciones del modelo base que en este caso es un modelo de RNA feedforward. El algoritmo de FKE consta del siguiente sistema de ecuaciones y que se puede divivir en dos parte

### Predicción

$$x_k|_ {k-1}=f(\hat{x}_ {k-1|k-1},u_{k-1})$$

$$P_k|_ {k-1}=F_{k-1}P_{k-1|k-1}F^T_{k-1}+Q_{k-1}$$

Donde:

- $\hat{x}_{k|k-1}$ es la estimación del estado en el tiempo $k$ basada en la información del tiempo $k-1$.
- $f$ es la función de transición de estado.
- $u_{k-1}$ es la entrada en el tiempo $k-1$.
- $F_{k-1}$ es la **matriz Jacobiana** de la función de transición de estados con respecto al estado en el tiempo $k-1$.
- $P_{k-1|k-1}$ es la matriz de covarianza de la estimación del error en el tiempo $k-1$.
- $Q_{k-1}$ es la matriz de covarianza del ruido del proceso en el tiempo $k-1$

### Corrección

$$K_k=P_{k|k-1}H^T_k(H_kP_{k|k-1}H^T_k+R_k)^{-1}$$

$$\hat{x}_ {k|k}=\hat{x}_ {k|k-1}+K_k(z_k-h(\hat{x}_{k|k-1}))$$

$$P_{k|k}=(I-K_kH_k)P_{k|k-1}$$

Donde:

- $K_k$ es la ganancia del FKE.
- $H_k$ es la **matriz Jacobiana** de la función de medición en el tiempo $k$.
- $z_k$ es la medición en el tiempo $k$.
- $h$ es la función que relaciona el estado estimado con la medición.
- $R_k$ es la matriz de covarianza del ruido de medición en el tiempo $k$.
- $I$ es la matriz identidad.

## Acoplamiento del modelos

# Resultados
