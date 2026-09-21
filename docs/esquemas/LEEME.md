# Lista de verificación para los esquemas

Antes de entregar, agregue los dos diagramas elaborados a mano:

- [ ] `qlearning-esquema.png`: debe mostrar la observación continua, su discretización, la tabla Q, la selección de acción ε-voraz, la interacción con `MountainCar-v0` y la actualización temporal de Q con recompensa, siguiente estado y máximo valor futuro.
- [ ] `dqn-esquema.png`: debe mostrar el estado como entrada de la red neuronal, los valores Q de las tres acciones como salida, la selección de acción, la interacción con el entorno, el búfer de repetición y el entrenamiento por lotes con la actualización del objetivo temporal.

## Correcciones que deben reflejarse

- [ ] El valor de ε disminuye a lo largo de los episodios; no se reinicia en cada paso.
- [ ] Diferencie entre alcanzar la bandera y llegar al límite de 200 pasos: el límite trunca el episodio, pero conserva el término de *bootstrap* en la actualización; alcanzar la meta sí finaliza el episodio.
- [ ] En DQN, durante la exploración se conserva la misma acción durante una racha de pasos antes de elegir otra acción.

> **Advertencia:** los diagramas enviados deben ser trabajo propio del estudiante. Cualquier figura generada con IA es solo una ayuda de estudio y no debe entregarse: la rúbrica asigna cero puntos a ese tipo de figura.
