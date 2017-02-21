# Makefile para generar el ejecutable de una red neuronal MLP para regresión

### Desarrollado por Carlos Gómez Pino ###

CPP = g++
CPPFLAGS = -Wall
OBJECT = -c
NAME = -o

destino: ejecutable clean

ejecutable: main PerceptronMulticapa
	@$(CPP) $(CPPFLAGS) main.o PerceptronMulticapa.o $(NAME) mlpRegresion.x $(THREADS)
	@echo Creando mlpRegresion.x

main: main.cpp
	@$(CPP) $(CPPFLAGS) $(OBJECT) main.cpp
	@echo Creando main.o

PerceptronMulticapa: PerceptronMulticapa.hpp PerceptronMulticapa.cpp
	@$(CPP) $(CPPFLAGS) $(OBJECT) PerceptronMulticapa.cpp
	@echo Creando PerceptronMulticapa.o

clean:
	@rm *.o
	@echo Borrando archivos *.o
