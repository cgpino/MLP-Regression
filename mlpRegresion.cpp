//============================================================================
// Introducción a los Modelos Computacionales
// Name        : mlpRegresion.cpp
// Author      : Carlos Gómez Pino
// Version     : 2016
// Copyright   : Universidad de Córdoba
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>    // Para coger la hora time()
#include <cstdlib>  // Para establecer la semilla srand() y generar números aleatorios rand()
#include <string.h>
#include <math.h>
#include <vector>

// Inclusión de la clase PerceptrónMulticapa
#include "imc/PerceptronMulticapa.h"

int main(int argc, char **argv) {

    /* Valores de entrada del programa */

	// Indican si se han proporcionado datos de entrenamiento y/o test
    bool tflag = false, Tflag = false;

    // Indican el nombre de los ficheros de entrenamiento y test
    char *tvalue = NULL, *Tvalue = NULL;

    // Nº de iteraciones del bucle externo a realizar
    int ivalue = 1000;

    // Nº de capas ocultas del modelo de red neuronal
    int lvalue = 1;

    // Nº de neuronas en cada una de las capas ocultas
    int hvalue = 5;

    // Valor del parámetro eta (tasa de aprendizaje)
    double evalue = 0.1;

    // Valor del parámetro mu (factor de momento)
    double mvalue = 0.9;

    // Indica si se va a utilizar sesgo en las neuronas
    bool bflag = false;

    // Variable para comprobar las opciones activadas
    int c;

    /* Procesamiento de la línea de comandos */

    while ((c = getopt (argc, argv, "t:T:i:l:h:e:m:b")) != -1) {
    	switch(c) {

    	// Fichero con datos de entrenamiento
    	case 't':
    		tflag = true;
    		tvalue = optarg;
    		break;

    	// Fichero con datos de test
    	case 'T':
    	    Tflag = true;
    	   	Tvalue = optarg;
    	 	break;

    	// Iteraciones del bucle externo
    	case 'i':
    		ivalue = atoi(optarg);
    		break;

    	// Capas ocultas del modelo
    	case 'l':
    		lvalue = atoi(optarg);
    		break;

    	// Neuronas por capa oculta
    	case 'h':
    		hvalue = atoi(optarg);
    		break;

    	// Valor de eta (tasa de aprendizaje)
    	case 'e':
    		evalue = atof(optarg);
    		break;

    	// Valor de mu (factor de momento)
    	case 'm':
    		mvalue = atof(optarg);
    		break;

    	// Uso de sesgo o no
    	case 'b':
    		bflag = true;
    		break;

    	// Tratamiento de errores
    	case '?':
    		if (optopt == 'n' || optopt == 'u')
    			fprintf (stderr, "\n # La opción %c requiere un argumento.\n", optopt);
    		else if (isprint (optopt))
    			fprintf (stderr, "\n # Opción desconocida '-%c'.\n", optopt);
    		else
    			fprintf (stderr, "\n # Caracter `\\x%x'.\n", optopt);
    		exit(-1);

    	// Si no ocurre ninguna de éstas se aborta el programa por seguridad
    	default:
    		abort();
    	}
    }

    // Si no hay datos de entrenamiento no se puede continuar con el programa
    if (!tflag) {
    	std::cout << "\n # Se debe de especificar un fichero con datos de entrenamiento." << std::endl;
    	exit(-1);
    }

    // Si no se especifican datos de test, se escogerán los de entrenamiento también para ello
    if (!Tflag) {
    	std::cout << "\n # Fichero con datos de test no especificado, se usarán los de entrenamiento." << std::endl;
    	Tvalue = tvalue;
    }

    /* Se imprimen los datos especificados por el usuario */

    std::cout << "\n***************************************************" << std::endl;
    std::cout << "*         Valores de entrada del programa         *" << std::endl;
    std::cout << "***************************************************" << std::endl;
    std::cout << " > Fichero de entrenamiento.......: " << tvalue << std::endl;
    std::cout << " > Fichero de test................: " << Tvalue << std::endl;
    std::cout << " > Nº de iteraciones externas.....: " << ivalue << std::endl;
    std::cout << " > Nº de capas ocultas............: " << lvalue << std::endl;
    std::cout << " > Nº de neuronas en capa oculta..: " << hvalue << std::endl;
    std::cout << " > Tasa de aprendizaje (eta)......: " << evalue << std::endl;
    std::cout << " > Factor de momento (mu).........: " << mvalue << std::endl;
    std::cout << " > Uso de sesgo...................: " << ((bflag)?"Activado":"Desactivado") << std::endl;
    std::cout << "***************************************************" << std::endl;

    // Declaración del perceptrón multicapa
    imc::PerceptronMulticapa mlp;

    // Se proceden a leer los datos de entrenamiento y test de fichero
    imc::Datos * pDatosTrain = mlp.leerDatos(tvalue);
    imc::Datos * pDatosTest = mlp.leerDatos(Tvalue);

    // Se ajusta el uso o no de sesgo a la red neuronal
    mlp.setSesgo(bflag);

    // Se ajusta el valor de eta a la red neuronal
    mlp.setEta(evalue);

    // Se ajusta el valor de mu a la red neuronal
    mlp.setMu(mvalue);

    // Declaración e inicialización del vector topología
    // (Nº de neuronas por cada capa, incluyendo entrada y salida)
    std::vector<int> vTopologia(lvalue+2);

    // Se añaden las neuronas de capa de entrada
    vTopologia[0] = pDatosTrain->nNumEntradas;

    // Se añaden las capas ocultas con sus correspondientes neuronas
    for(int i=1; i<=lvalue; i++)
    	vTopologia[i] = hvalue;

    // Se añaden las neuronas de capa de salida
    vTopologia[lvalue+1] = pDatosTrain->nNumSalidas;

    // Inicialización propiamente dicha
    mlp.inicializar(vTopologia.size(),vTopologia);

    // Semilla de los números aleatorios
    int semillas[] = {10,20,30,40,50};

    // Vectores con los errores medios de test y train en cada semilla
    std::vector<double> erroresTest(5);
    std::vector<double> erroresTrain(5);

    // Media y desviación típica de los errores de test y train
    double mediaErrorTrain = 0.0, desviacionTipicaErrorTrain = 0.0;
    double mediaErrorTest = 0.0, desviacionTipicaErrorTest = 0.0;

    for(int i=0; i<5; i++) {

    	// Se muestra la semilla usada para generar los primeros pesos aleatorios de la red neuronal
        srand(semillas[i]);
    	std::cout << "\n**************" << std::endl;
    	std::cout << " Semilla <" << semillas[i] << ">" << std::endl;
    	std::cout << "**************" << std::endl;

    	// Se ejecuta el algoritmo en línea y se obtienen los errores de train y test
        mlp.ejecutarAlgoritmoOnline(pDatosTrain,pDatosTest,ivalue,erroresTrain[i],erroresTest[i]);
    	std::cout << "\n # Finalizado => Error de test final: " << erroresTest[i] << std::endl;

    	// Se calcula la media y desviación típica de los errores de train y test
    	mediaErrorTrain += erroresTrain[i];
    	mediaErrorTest += erroresTest[i];
    	desviacionTipicaErrorTrain += pow(erroresTrain[i],2);
    	desviacionTipicaErrorTest += pow(erroresTest[i],2);
    }

    // Se terminan de calcular la media y desviación típica de los errores
    mediaErrorTrain /= 5;
    mediaErrorTest /= 5;
    desviacionTipicaErrorTrain = sqrt((desviacionTipicaErrorTrain/5) - pow(mediaErrorTrain,2));
    desviacionTipicaErrorTest = sqrt((desviacionTipicaErrorTest/5) - pow(mediaErrorTest,2));

    // Se avisa por pantalla de la finalización de las semillas
    std::cout << "\n -> Todas las semillas han terminado. <-" << std::endl;

    // Se muestra el informe final extraído de la ejecución
    std::cout << "\n***************" << std::endl;
    std::cout << " Informe final" << std::endl;
    std::cout << "***************" << std::endl;
    std::cout << "\n > Error de entrenamiento (Media +- DT): " << mediaErrorTrain << " +- " << desviacionTipicaErrorTrain << std::endl;
    std::cout << " > Error de test (Media +- DT):          " << mediaErrorTest << " +- " << desviacionTipicaErrorTest << std::endl;

    return EXIT_SUCCESS;
}

