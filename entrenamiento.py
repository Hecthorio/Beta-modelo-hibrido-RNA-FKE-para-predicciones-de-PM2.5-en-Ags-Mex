# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 13:40:36 2023

@author: hecto
"""

#entrenamiento de un modelo de red neuronal para predicción de PM2.5 para despues
#acoplarlo a un sistema de filtro de Kalman extendido.

#agregamos librerias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD#, Adam, RMSprop
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

#usaremos la base de datos siguiente
ruta = 'C:/Users/Hector/OneDrive/Documentos/ITA/amidiq 2024/codigos/'
nom_arch = 'base_datos.csv'

#leemos la base de datos
df = pd.read_csv(ruta + nom_arch)

#filtramos para solo utilizar la información de una sola fase de monitoreo
coordenadas = {'CBTIS' : [21.873486, -102.319011],
               'CENTRO' : [21.883085, -102.295848],
               'IED' : [21.903161, -102.276098],
               'SMA' : [21.845283, -102.291389]}

#filtramos solo la información de una zona de monitoreo
df = df[df.latitud == coordenadas['CBTIS'][0]]

#boxplot de la concentración de PM2.5
sns.boxplot(x=df['Hora'], y=df['PM25'])
plt.xlabel('Hora del día (h)', fontsize = 12)
plt.ylabel('PM$_{2.5}$ ($\mu$g/m$^3$)', fontsize = 12)
plt.savefig(ruta + 'boxplot.pdf', bbox_inches='tight', dpi = 300)
plt.show()

#Histograma de PM2.5
sns.histplot(df['PM25'], alpha = 0, stat='density', bins = 20)
sns.kdeplot(df['PM25'], color= 'k', linestyle="--")
plt.xlabel('PM$_{2.5}$ ($\mu$g/m$^3$)')
plt.savefig(ruta + 'histograma.pdf', bbox_inches='tight', dpi = 300)
plt.show()

#evaluación de la prueba de shapiro-test
statistic, p_value = stats.shapiro(df['PM25'])

# Imprimir los resultados
print("Estadística de prueba:", statistic)
print("Valor p:", p_value)

#eliminamos las columnas que no vamos a necesitar para el entrenamiento
df = df[['sin_hour', 'cos_hour', 'sin_dow', 'cos_dow', 'sin_doy', 'cos_doy', 'PM25', 'PM25_t1']]

#separanos nuestros datos en los datos de entrada y salida del modelo de red
X = df[['sin_hour', 'cos_hour', 'sin_dow', 'cos_dow', 'sin_doy', 'cos_doy', 'PM25']]
Y = df['PM25_t1']

#graficamos algunos valores del los senos y cosenos de las variables temporales
plt.plot(df['sin_hour'][0:100], 'o', label = 'sin_hour')
plt.plot(df['cos_hour'][0:100], 'o', label = 'cos_hour')
plt.legend()
plt.xlabel('Dato')
plt.title('Hora')
plt.savefig(ruta + 'sin_cos_hour.pdf', bbox_inches='tight', dpi = 300)
plt.show()

plt.plot(df['sin_dow'][0:1000], 'o', label = 'sin_dow')
plt.plot(df['cos_dow'][0:1000], 'o', label = 'cos_dow')
plt.legend()
plt.xlabel('Dato')
plt.title('Día semana')
plt.savefig(ruta + 'sin_cos_dow.pdf', bbox_inches='tight', dpi = 300)
plt.show()

plt.plot(df['sin_doy'], 'o', label = 'sin_doy')
plt.plot(df['cos_doy'], 'o', label = 'cos_doy')
plt.legend()
plt.xlabel('Dato')
plt.title('Día año')
plt.savefig(ruta + 'sin_cos_doy.pdf', bbox_inches='tight', dpi = 300)
plt.show()

#separamos los datos de entrenamiento y los de prueba
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1, shuffle = False)

#escalamos los datos a partir de los maximos y minimos
#generamos los objetos
X_scaler = MinMaxScaler(feature_range=(-1,1))
Y_scaler = MinMaxScaler(feature_range=(-1,1))

#hacemos un reshape de las salidas del modelo (esto solo se hace si es una salida)
Y_train = Y_train.values.reshape(-1,1)
Y_test = Y_test.values.reshape(-1,1)

#le pasamos las matrices para generar 
X_scaler.fit(X_train)
Y_scaler.fit(Y_train)

#escalamos nuestros datos para el entrenamiento y la validación
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)
Y_train = Y_scaler.transform(Y_train)

#Ahora generamos la estructura de red
red = Sequential()
red.add(Dense(100, input_dim = X_train.shape[1], activation = 'tanh'))
#red.add(Dense(100, activation = 'tanh'))
red.add(Dense(1, activation='linear'))

#Compilar el modelo
optimizer = SGD(learning_rate = 0.01)
red.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae', 'mse'])

#entrenamiento del modelo de red
loss_curve = red.fit(X_train, Y_train, epochs = 15, validation_split = 0.1, shuffle = True)

#graficamos la función de perdida para los datos de validación y prueba
plt.plot(loss_curve.history['loss'], '-o', label = 'Entrenamiento',  markerfacecolor = 'None')
plt.plot(loss_curve.history['val_loss'], '-o', label = 'Validación',  markerfacecolor = 'None')
plt.legend()
plt.xlabel('Épocas')
plt.ylabel('Función pérdida (MSE)')
plt.savefig(ruta + 'perdida.pdf', bbox_inches='tight', dpi = 300)
plt.show()

#evaluamos el modelo de red usando los datos de prueba
Y_test_eval = red.predict(X_test)

#aplicamos la transformación inversa del escalamiento
Y_test_eval = Y_scaler.inverse_transform(Y_test_eval)

#imrpimimos el resultado del coeficiente de correlación del modelo
print(f'\nR2 = {r2_score(Y_test, Y_test_eval)}')
R2_rna = r2_score(Y_test, Y_test_eval)

#evaluamos el modelo pero usando el filtro de Kalman extendido
#generamos una lista vacia donde iremos guardarndo los resultados del filtro
#y todos las otras variables que se emplean en el algoritmo
Y_test_eval_k = []
P_k = [np.array(0.5).reshape(1,1)]      #covarianza del estado
P_kk = 0.5       #condición inicial de la covarianza (debe ser el mismo que P_k)
Q_k = 0.1        #covarianza del ruido del proceso 0.1
R_k = 0.001      #covarianza del ruido de la medición 0.001
A_k = 0.5
H = 0.1       
n = 0            #contador
delta = 0.000001   #incremento de la derivada
K_k = []         #ganancia de kalman
x_cor = 0        #valor de corrección (no es del todo necesario ponerlo aqui)

#antes de comenzar el algoritmo tenemos que escalar las salidas (z)
Y_test = Y_scaler.transform(Y_test)

#generamos un ciclo donde evaluaresmos el algoritmo de kalman
for i in X_test[0:-1]:
    #sacamos los datos con los que vamos a evaluar la red
    x = i
    x = x.reshape(-1,1).T
    #actualizamos el valor del estado previo
    if n > 0:
        x[:,-1] = x_cor
    #Predicción del estado (ecuación de transición)
    Y_test_eval_k.append(red.predict(x, verbose = False))
    #evaluamos la matriz jacobina (como solo hay una salida solo es una escalar)
    x[-1] = x[-1] + delta
    A_k = (red.predict(x, verbose = False) - Y_test_eval_k[n])/delta
    #Covarianza del estado
    #P_k.append(A_k*P_k[n]*A_k + Q_k)
    P_k.append(A_k*P_kk*A_k + Q_k)
    
    #evaluamos la matriz de ganancias de Kalman
    #para este caso A_k = H_k porque suponemos que el modelo
    #h es la identidad (h = f)
    K_k.append(P_k[n+1]*A_k/(A_k**2*P_k[n+1] + R_k))
    #evaluamos la corrección del estado (x[-1] tiene el estado real previo)
    x_cor = Y_test_eval_k[n] + K_k[n]*(Y_test[n+1] - Y_test_eval_k[n])
    #evaluamos la corrección de la covarianza del estado
    #P_k.append((1-K_k[n]*A_k)*P_k[n+1])
    P_kk = ((1-K_k[n]*A_k)*P_k[n+1])
    #aumentamos en 1 el contador
    n = n + 1

#graficamos el comportamiento de K y P
plt.figure()
plt.plot(np.squeeze(np.array(P_k)))
plt.xlabel('Evaluación')
plt.ylabel('P$_k$')
plt.title('Matriz de covarianza estimada del error')
plt.savefig(ruta + 'covarianza.pdf', bbox_inches='tight', dpi = 300)
plt.show()

plt.figure()
plt.plot(np.squeeze(np.array(K_k)))
plt.xlabel('Evaluación')
plt.ylabel('K$_k$')
plt.title('Ganancia de Kalman')
plt.savefig(ruta + 'ganancia.pdf', bbox_inches='tight', dpi = 300)
plt.plot()

#pasamoa la lista a un arreglo numpy
Y_test_eval_k = np.squeeze(np.array(Y_test_eval_k), axis = 2)
#K_k = np.squeeze(np.array(K_k), axis = 2)

#evaluamos el coeficiente de determinación para los datos
print(f'\nR2 = {r2_score(Y_test[0:-1], np.array(Y_test_eval_k))}')
R2_rnafke = r2_score(Y_test[0:-1], np.array(Y_test_eval_k))

#quitamos el escalamiento a las variables para graficar
Y_test = Y_scaler.inverse_transform(Y_test)
Y_test_eval_k = Y_scaler.inverse_transform(Y_test_eval_k)

#graficamos los datos para ver su comportamiento
plt.figure()
plt.plot(Y_test[0:100], 'ko--' ,label = 'Sensor', markerfacecolor = 'None')
plt.plot(Y_test_eval[0:100], label = 'RNA')
plt.plot(Y_test_eval_k[0:100], label = 'RNA-FKE')
plt.xlabel('Tiempo (h)')
plt.ylabel('PM$_{2.5}$ ($\mu$g/m$^3$)')
plt.ylim(0,110)
#plt.title('Pronosticos de concentración de PM$_{2.5}$ (RNA y RNA-FKE)')
plt.text(30,90, '$R^{2}_{RNA}$ =' + f'{R2_rna:.3}' + '\n$R^{2}_{RNA-FKE}$ =' + f'{R2_rnafke:.3}', bbox = dict(facecolor = 'None', alpha = 0.2))
plt.legend()
plt.savefig(ruta + 'evaluacion.pdf', bbox_inches='tight', dpi = 300)
plt.show()

#evaluación del error cuadratico medio
mse_rna = np.mean((Y_test - Y_test_eval) ** 2)
mse_rna_kf = np.mean((Y_test[0:-1] - Y_test_eval_k) ** 2)

#evaluar residuales del la red y red-kf
resid = Y_test - Y_test_eval
resid_fk = Y_test[0:-1] - Y_test_eval_k
resid_std = (resid - np.mean(resid))/np.std(resid)
resid_std_fk = (resid_fk - np.mean(resid_fk))/np.std(resid_fk)

#graficas de los histogramas de los residuales
plt.figure()
sns.histplot(resid, alpha = 0, stat='density', bins = 20)
sns.kdeplot(resid, color= 'k', linestyle="--")
plt.xlabel('Residuales estandarizados')
plt.title('RNA')
plt.savefig(ruta + 'hist_Residuales_rna.pdf', bbox_inches='tight', dpi = 300)
plt.show()

plt.figure()
sns.histplot(resid_fk, alpha = 0, stat='density', bins = 20)
sns.kdeplot(resid_fk, color= 'k', linestyle="--")
plt.xlabel('Residuales estandarizados')
plt.title('RNA-FKE')
plt.savefig(ruta + 'hist_Residuales_rnafke.pdf', bbox_inches='tight', dpi = 300)
plt.show()

#graficamos los residuales contra el valor los valores predichos y agregamos
#tambien las lineas de los intervalos de confianza
plt.figure()
plt.plot(Y_test, resid_std,'ko', markerfacecolor = 'None')
plt.axhline(1.96, color = 'b', linestyle='--')
plt.axhline(0, color = 'r', linestyle='--')
plt.axhline(-1.96, color = 'b', linestyle='--')
plt.legend(['RNA'])
plt.xlabel('PM$_{2.5}$ $\mu$g/m$^3$')
plt.ylabel('Residuales estandarizados')
plt.savefig(ruta + 'Residuales_rna.pdf', bbox_inches='tight', dpi = 300)
plt.show()

plt.figure()
plt.plot(Y_test[0:-1], resid_std_fk,'ko', markerfacecolor = 'None')
plt.axhline(1.96, color = 'b', linestyle='--')
plt.axhline(0, color = 'r', linestyle='--')
plt.axhline(-1.96, color = 'b', linestyle='--')
plt.legend(['RNA-FKE'])
plt.xlabel('PM$_{2.5}$ $\mu$g/m$^3$')
plt.ylabel('Residuales estandarizados')
plt.savefig(ruta + 'Residuales_rnafke.pdf', bbox_inches='tight', dpi = 300)
plt.show()

#mostramos la media de los datos
print(f'Media de residuales del modelo RNA: {np.mean(resid)}')
print(f'Media de residuales del modelo RNA_FKE: {np.mean(resid_fk)}')

#evaluación de la prueba de shapiro-test
statistic, p_value = stats.shapiro(resid)
print(f'Residuales, prueba de Shapiro, p-value = {p_value}')
statistic, p_value = stats.shapiro(resid_fk)
print(f'Residuales, prueba de Shapiro, p-value = {p_value}')

#evaluación de la homocedasticidad de residuales
bp_resid = sm.stats.diagnostic.het_breuschpagan(resid = resid, exog_het = sm.add_constant(X_test))
print(f'Prueba de homocedasticidad de Breauch-Pagan para residuales RNA, p-value = {bp_resid[3]}')
bp_resid = sm.stats.diagnostic.het_breuschpagan(resid = resid_fk, exog_het = sm.add_constant(X_test[0:-1]))
print(f'Prueba de homocedasticidad de Breauch-Pagan para residuales RNA_FK, p-value = {bp_resid[3]}')

#graficas de autocorrelación
plt.figure()
plot_acf(resid, lags=23)
plt.title('Función de autocorrelación (RNA)')
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.savefig(ruta + 'acf_rna.pdf', bbox_inches='tight', dpi = 300)
plt.show()

plt.figure()
plot_acf(resid_fk, lags=23)
plt.title('Función de autocorrelación (RNA-FKE)')
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.savefig(ruta + 'acf_rnafke.pdf', bbox_inches='tight', dpi = 300)
plt.show()

plt.figure()
plot_pacf(resid, lags=23)
plt.title('Función de autocorrelación parcial (RNA)')
plt.xlabel('Lag')
plt.ylabel('PACF')
plt.savefig(ruta + 'pacf_rna.pdf', bbox_inches='tight', dpi = 300)
plt.show()

plt.figure()
plot_pacf(resid_fk, lags=23)
plt.title('Función de autocorrelación parcial (RNA-FKE)')
plt.xlabel('Lag')
plt.ylabel('PACF')
plt.savefig(ruta + 'pacf_rnafke.pdf', bbox_inches='tight', dpi = 300)
plt.show()

#evaluación de media, desviación estandar, kurtosis y oblicuidad de la salida del modelo
print(f'media real {np.mean(Y_test)}')
print(f'media RNA {np.mean(Y_test_eval)}')
print(f'media RNAFKE {np.mean(Y_test_eval_k)}')

print(f'desviación estandar real {np.std(Y_test)}')
print(f'desviación estandar RNA {np.std(Y_test_eval)}')
print(f'desviación estandar RNA_FKE {np.std(Y_test_eval_k)}')

print(f'oblicuidad real {stats.skew(Y_test)}')
print(f'oblicuidad RNA {stats.skew(Y_test_eval)}')
print(f'oblicuidad RNA_FKE {stats.skew(Y_test_eval_k)}')

print(f'oblicuidad real {stats.kurtosis(Y_test)}')
print(f'oblicuidad RNA {stats.kurtosis(Y_test_eval)}')
print(f'oblicuidad RNA_FKE {stats.kurtosis(Y_test_eval_k)}')

#graficamos las tendencias de los datos de los modelos y el real
plt.figure()
sns.kdeplot(Y_test, fill = True, alpha = 0.3, palette = 'rocket', label = 'Prueba')
sns.kdeplot(Y_test_eval, fill = True, alpha = 0.3, palette = 'crest', label = 'RNA')
sns.kdeplot(Y_test_eval_k, fill = True, alpha = 0.3, palette = 'mako', label = 'RNA-FKE')
plt.xlabel('PM$_{2.5}$ ($\mu$g/m$^3$)')
plt.savefig(ruta + 'hist_real_rna_rnafke.pdf', bbox_inches='tight', dpi = 300)
plt.legend()