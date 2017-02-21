# Red Neuronal MLP para regresión
Programa escrito en C++ que simula el funcionamiento de una red neuronal MLP (perceptrón multicapa) para problemas de regresión. En la carpeta `dat` se incluyen algunos conjuntos de datos tanto de entrenamiento como de test.

# ¿Cómo se usa?
Será necesario descargar el contenido de github y posteriormente ejecutar el comando `make` (para compilar) dentro de la carpeta del proyecto previamente bajada.
Una vez compilado, se puede ejecutar el programa `mlpRegression.x` con distintos argumentos para personalizar nuestra red neuronal.

# Argumentos del programa
- `Argumento t`: Indica el nombre del fichero que contiene los datos de entrenamiento a utilizar. Sin este argumento, el programa no funciona.
- `Argumento T`: Indica el nombre del fichero que contiene los datos de test a utilizar. Si no se especifica este argumento, se utilizan los datos de entrenamiento como test.
- `Argumento i`: Indica el número de iteraciones del bucle externo a realizar. Si no se especifica, se realizan 1000 iteraciones.
- `Argumento l`: Indica el número de capas ocultas del modelo de red neuronal. Si no se especifica, se utiliza 1 capa oculta.
- `Argumento h`: Indica el número de neuronas a introducir en cada una de las capas ocultas. Si no se especifica, se utilizan 5 neuronas.
- `Argumento e`: Indica el valor del parámetro eta. Por defecto, eta = 0,1.
- `Argumento m`: Indica el valor del parámetro mu. Por defecto, mu = 0,9.
- `Argumento b`: Indica si se va a utilizar sesgo en las neuronas. Por defecto, no se utiliza sesgo.

# Ejemplo de ejecución
Un ejemplo de ejecución sería el siguiente:
```
./mlpRegression.x -t dat/train_xor.dat -T dat/test_xor.dat -i 1000 -l 1 -h 10 -e 0.1 -m 0.9 -b
```
