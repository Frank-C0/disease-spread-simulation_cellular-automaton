# Simulación de Propagación de Enfermedades (Automata Celular con OpenCL)

Este proyecto implementa una simulación de propagación de enfermedades utilizando un modelo de autómata celular estocástico. La simulación está escrita en OpenCL y permite visualizar la dinámica de la propagación de enfermedades en una cuadrícula bidimensional.

![Simulación de Propagación de Enfermedades](IllAutomaton_out_gif_R0_3.8__ID_16_f500.gif)

## Funcionalidades Principales

- Simulación de propagación de enfermedades mediante autómata celular estocástico.
- Implementación en OpenCL para paralelización y aceleración en GPUs.
- Visualización de la simulación en tiempo real con Pygame.
- Guardado de la simulación en formato GIF.

## Requisitos

- Python (3.7 o superior)
- OpenCL (instalado y configurado)
- Pygame (instalado)

## Instalación

1. Clona este repositorio: `git clone https://github.com/Frank-C0/disease-spread-simulation_cellular-automaton.git`
2. Instala las dependencias
3. Configura el entorno de OpenCL según las instrucciones del proyecto.

## Uso
Modifica las constantes como R0, ILL_DURATION, etc.
Ejecuta la simulación con el siguiente comando:

```bash
python ill_automaton_gif.py

o para visualización en tiempo real:

```bash
python ill_automaton_window.py