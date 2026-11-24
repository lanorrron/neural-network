# Neural Network

Pequeño proyecto educativo en Rust para entender, paso a paso, cómo se construyen algunos componentes de una red neuronal: neuronas, capas, funciones de pérdida, gradientes y operaciones con vectores y matrices.

> **Estado:** proyecto en aprendizaje. Los experimentos muestran los cálculos y sus límites; no son todavía un sistema de reconocimiento de escritura o de voz.

## Requisitos

- Rust y Cargo (edición 2024).

## Ejecutar

```bash
# Experimento de vocales (opción predeterminada)
cargo run

# Probar una vocal de entrada
cargo run -- vocal a

# Otros experimentos
cargo run -- lineal
cargo run -- booleano
```

El experimento de vocales muestra el progreso cada 10 épocas. Si le pasas una letra, acepta `a`, `e`, `i`, `o` o `u`, sin distinguir mayúsculas.

## Experimentos

### Vocales: cálculo matricial

Una vocal se representa internamente como una cuadrícula de 5 × 5. `#` es un píxel encendido y `.` uno apagado. El programa aplana esa cuadrícula en 25 valores y los procesa con cinco salidas, una para cada vocal.

Las dimensiones de la pasada hacia adelante son:

```text
entrada (1 × 25) · pesos (25 × 5) + bias (1 × 5) → puntuaciones (1 × 5)
```

Después aplica *softmax* para convertir las puntuaciones en probabilidades. Durante 500 épocas, compara la salida con la clase esperada y actualiza los pesos y el bias mediante descenso por gradiente. La salida esperada se representa con un vector *one-hot*: por ejemplo, A es `[1, 0, 0, 0, 0]`.

**Alcance actual:** el entrenamiento contiene un patrón fijo por vocal. `cargo run -- vocal a` convierte la letra `a` en el patrón estándar A definido en el código y evalúa ese patrón. Esto sirve para recorrer la pasada de la red, pero no reconoce una A escrita a mano ni aprende variaciones de escritura. Para eso hacen falta imágenes de entrada, etiquetas y muchos ejemplos variados por vocal.

### Lineal: aprender `y = 2x`

Entrena una neurona lineal con ejemplos numéricos. Usa MSE para medir el error y descenso por gradiente para ajustar peso y bias. Cada cierto número de épocas muestra los parámetros y predicciones.

### Booleano: falso y verdadero

Entrena una neurona con dos ejemplos: `0 → falso` y `1 → verdadero`. Usa sigmoid para producir una probabilidad y el umbral `0.5` para elegir la clase: menor que `0.5` es falso; igual o mayor es verdadero. Imprime los gradientes y cada actualización del peso y bias.

## Cómo se organiza el código

```text
src/
├── experiments/   # programas demostrativos
├── learning/      # pérdida, gradientes y descenso por gradiente
├── math/          # operaciones con Vector y Matrix
├── neural/        # neurona, capa, red y activaciones
└── main.rs        # selecciona el experimento desde la terminal
```

En el experimento de vocales, `Matrix::multiply` hace la multiplicación de la entrada por los pesos. El gradiente de pesos se forma como el producto externo `entradaᵀ · error`, con dimensiones `(25 × 1) · (1 × 5) = (25 × 5)`. Así se actualizan en una operación conceptual los pesos que conectan las 25 entradas con las cinco salidas.

## Dependencias

El proyecto solo declara `rand` en Cargo. Los experimentos descritos usan datos fijos; no necesitan descargar un conjunto de datos.
