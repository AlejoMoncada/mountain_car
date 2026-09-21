# Comparación de Q-Learning tabular y DQN en MountainCar-v0

Este documento compara los dos agentes con **resultados medidos en este repositorio**, no con cifras de referencia. Todas las métricas provienen de los artefactos versionados en:

- `artifacts/qlearning/seed-20250308/`
- `artifacts/dqn/seed-20250310/`
- `artifacts/dqn/random-exploration/seed-20250310/`

## 1. El problema formulado como decisión

| Elemento | Descripción |
|---|---|
| Estado | Posición del carrito (`-1.2` a `0.6`) y velocidad (`-0.07` a `0.07`) |
| Acciones | Acelerar a la izquierda, no acelerar, acelerar a la derecha |
| Recompensa | `-1` en cada paso |
| Meta | Alcanzar la posición `0.5` |
| Fin del episodio | Alcanzar la meta (final real) o agotar 200 pasos (corte por tiempo) |

Como todas las recompensas son `-1`, **el retorno es exactamente el negativo del número de pasos**. Menos negativo significa mejor. Este detalle es clave para explicar una limitación: hasta que el agente alcanza la bandera al menos una vez, **todas las transiciones son idénticas en recompensa**, así que no hay señal que distinga una buena acción de una mala.

## 2. Los dos agentes

**Q-Learning tabular.** Discretiza posición y velocidad en una grilla de `20 × 20` (400 celdas) y guarda un valor por acción en cada celda. Explora con ε-greedy y actualiza con la regla de Bellman.

**DQN.** Una red neuronal de dos capas ocultas recibe posición y velocidad y devuelve tres valores Q. Aprende de una memoria de experiencias (*replay buffer*), calcula el objetivo con una red congelada y actualiza el error de Bellman.

## 3. Protocolo de medición

| Aspecto | Q-Learning | DQN |
|---|---|---|
| Episodios de entrenamiento | 20.000 | 2.500 |
| Semilla de entrenamiento | 20250308 | 20250310 |
| Episodios de evaluación | 100 | 100 |
| Semilla de evaluación | 30250308 | 30250310 |
| Política de evaluación | Voraz (sin exploración) | Voraz (sin exploración) |

Las semillas de evaluación son **disjuntas** de las de entrenamiento, de modo que la evaluación usa estados iniciales que el agente no vio durante el aprendizaje.

## 4. Línea base: acciones aleatorias

Antes de corregir la exploración se midió qué logra un agente que actúa completamente al azar, con 500 episodios sembrados:

| Métrica | Valor medido |
|---|---|
| Episodios con bandera alcanzada | **0 de 500** |
| Episodios cortados por tiempo | 500 |
| Retorno promedio | `-200,0` |
| Mejor posición alcanzada | `-0,168` (la meta es `0,5`) |
| Racha más larga con la misma acción | 11 pasos |

La última fila explica el resto: el carrito necesita empujar en la misma dirección durante muchos pasos seguidos para tomar impulso, y el azar uniforme produce secuencias demasiado cortas. La probabilidad de repetir la misma acción 20 veces con tres acciones posibles es de aproximadamente `3 × 10⁻¹⁰`.

## 5. Resultados obtenidos

### 5.1 Evaluación voraz (métrica principal)

| Métrica | Q-Learning | **DQN** |
|---|---|---|
| Retorno promedio | −166,88 | **−112,34** |
| Desviación estándar | 23,11 | 31,74 |
| Episodios exitosos | 84 / 100 | **89 / 100** |
| Mejor retorno | −128 | **−84** |
| Peor retorno | −200 | −200 |

### 5.2 Entrenamiento

| Métrica | Q-Learning | **DQN** |
|---|---|---|
| Episodios | 20.000 | **2.500** |
| Retorno promedio | −163,56 | −150,88 |
| Desviación estándar | 29,01 | 42,62 |
| Éxitos durante entrenamiento | 15.301 | 1.706 |
| Mejor retorno observado | −89 | −84 |
| Duración real | **35,5 s** | 3 m 40 s |

**Aclaración importante:** el mejor retorno de entrenamiento del agente tabular («−89») se logró **explorando**, no explotando lo aprendido. No es comparable con el mejor resultado de evaluación («−128»). La métrica que mide la calidad de la política final es la evaluación voraz.

## 6. Comparación cualitativa

| Criterio | Q-Learning tabular | DQN |
|---|---|---|
| Velocidad de aprendizaje | Necesita ~20.000 episodios | Alcanza buena política en ~2.500 |
| Costo por episodio | Muy bajo (una tabla) | Alto (red neuronal y retropropagación) |
| Tiempo total | 35 s | 3 m 40 s |
| Estabilidad durante entrenamiento | Más estable (DE 29) | Más ruidoso (DE 42,6) |
| Desempeño final | −166,88 | **−112,34** |
| Consistencia en evaluación | Menos dispersa (DE 23,1) | Más dispersa (DE 31,7) |
| Dificultad de implementación | Baja: discretizar, elegir, actualizar | Alta: red, memoria, red objetivo, exploración |
| Generalización entre estados | No: cada celda es independiente | Sí: estados parecidos comparten aprendizaje |
| Interpretabilidad | Alta: se puede leer la tabla | Baja: los pesos no se leen directamente |
| Ajuste de hiperparámetros | Pocos (bins, α, γ, ε) | Muchos (red, lote, memoria, sincronización, ε) |
| Limitación principal | La grilla pierde precisión; celdas no visitadas quedan sin aprender | Más costoso, más sensible y más difícil de depurar |

## 7. Por qué DQN obtiene mejor resultado

**1. Aprovecha mejor cada experiencia.** La tabla aprende celda por celda: lo aprendido en una casilla no ayuda en la vecina. La red sí generaliza, así que una transición mejora la estimación de estados parecidos.

**2. No pierde precisión por discretizar.** La grilla de `20 × 20` mezcla posiciones y velocidades distintas dentro de la misma celda. El DQN trabaja con los valores continuos originales.

**3. Llega a una mejor política final.** Con la misma evaluación voraz y 100 episodios, el DQN consigue −112,34 y 89% de éxito frente a −166,88 y 84%.

**Pero no todo favorece al DQN:** fue **6 veces más lento** en tiempo real, su entrenamiento fue **más ruidoso** y su evaluación más dispersa. Además, el DQN **solo funcionó tras corregir la exploración**; el Q-Learning tabular no necesitó esa corrección porque su tabla, más tolerante, sí acumuló aprendizaje con exploración por paso.

## 8. Limitaciones de esta comparación

- **Una sola semilla por agente.** Es una línea base reproducible, no un estudio estadístico. Diferencias pequeñas podrían deberse al azar.
- **Presupuestos de episodios distintos** (20.000 frente a 2.500). El DQN es más eficiente en *muestras*, pero no se midió cuánto mejora con más episodios.
- **Sin barrido de hiperparámetros.** Se usaron los valores por defecto del repositorio del curso más `explore_run_length=20`.
- **Ninguno de los dos llega al umbral de −110** de forma consistente: el DQN lo roza (−112,34), el Q-Learning queda lejos.
- **Nuestro Q-Learning no reproduce la cifra de referencia del curso** (`≈ −133`). Con la misma configuración obtuvimos −166,88. Las causas posibles son la semilla única y diferencias de implementación; queda como punto abierto para una verificación adicional.
- **11 de 100 episodios del DQN aún agotan los 200 pasos**, es decir, la política final todavía falla en algunos estados iniciales.

## 9. Conclusión

Para este problema, **el DQN aprendió una mejor política y en menos episodios**, mientras que **el Q-Learning tabular fue mucho más rápido de entrenar, más estable y más fácil de implementar y de explicar**.

La lección más valiosa no fue «DQN es mejor», sino **por qué el DQN necesitaba algo distinto**: en un entorno donde la única recompensa es `-1` por paso y el éxito exige secuencias sostenidas de acciones, una exploración que sortea una acción nueva en cada paso no puede alcanzar el objetivo ni una sola vez. Corregir la exploración —y medirlo antes de creerlo— fue lo que hizo posible el aprendizaje.
