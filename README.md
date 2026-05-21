[English](#english) | [Español](#español)

---

`<a name="english"></a>`

## English — BIOM Face Detection

Custom Viola-Jones face detection system with a parallelized Haar cascade classifier. Trained via AdaBoost with hard negative mining on [WIDER FACE](http://shuoyang1213.me/WIDERFACE/). Supports Numba JIT (5-10x speedup) with a threaded NumPy fallback.

### Quick Start

```bash
conda create --name BIOM_env python=3.13 -y && conda activate BIOM_env
uv pip install -r requirements.txt
pip install -e .
pip install numba          # Optional — 5-10x inference speedup
```

### Usage

All commands run from repo root via `app/main.py`:

```bash
python app/main.py detect-camera      # Real-time face detection from webcam
python app/main.py train_vj           # Train Viola-Jones cascade classifier
python app/main.py generate-filter    # Generate filtered label view
```

Use `--help` on any subcommand for the full argument list.

### Project Structure

```
.
├── app/
│   ├── main.py                     # CLI entry point
│   ├── scripts/                    # Runnable scripts
│   └── src/                        # Core library
│       ├── config/                 # Configuration dataclass
│       ├── data/                   # Feature extraction, crops, datasets
│       └── model/                  # AdaBoost, cascade classifier, IO
├── data/
│   ├── ViolaJones/                 # Preprocessed crops & numpy arrays
│   └── others/WIDER_train/         # Raw WIDER FACE dataset
├── models/
│   └── haar_cascades_computed_best/  # Final trained cascades
├── notebooks/                      # Jupyter notebooks
├── docs/                           # Plots, figures, Viola-Jones paper
├── logs/                           # Training logs
└── scripts/                        # Shell shortcuts
```

### Source Packages

**`src/config/`** — `Configuration` dataclass. All paths, hyperparameters, defaults. Auto-creates directories on init.

**`src/data/`**

| Module | Purpose |
|--------|---------|
| `filter.py` | Integral image, squared integral, local normalization |
| `features.py` | Haar-like feature generation & face feature precomputation |
| `crops.py` | Positive/negative crop extraction from images |
| `dataset.py` | WIDER FACE & VPC dataset loaders |
| `balance.py` | Per-stage dataset balancing |
| `augment.py` | Data augmentation (rotation, flip, brightness/contrast) |

**`src/model/`**

| Module | Purpose |
|--------|---------|
| `cascade_def.py` | Data structures: `Stage`, `WeakClassifier`, `Feature` |
| `adaboost.py` | AdaBoost stump classifier training with early stopping |
| `cascade_clasifier.py` | Main parallel detector — Numba JIT or ThreadPoolExecutor fallback |
| `cascade_parser.py` | Parse trained cascade XML back into objects |
| `cascade_serializer.py` | Serialize stages to OpenCV-compatible XML |
| `test.py` | Cascade evaluation — TPR, FPR, visualizations |

### Scripts

| Script | Purpose |
|--------|---------|
| `detect_camera.py` | Opens the webcam and runs real-time face detection using a trained cascade |
| `train_viola_jones.py` | Full training loop — feature generation, AdaBoost per stage, hard negative mining |
| `generate_labels.py` | Generates a filtered label overlay view for dataset images |

### Notebooks

| Notebook | Content |
|----------|---------|
| `Ej1-Filters.ipynb` | Integral images, local normalization, standard deviation fundamentals |
| `Ej2-CurvaRoc.ipynb` | ROC curve analysis across cascade stages |
| `Ej3-CheckCascadeStages.ipynb` | Inspect cascade internals — stages, features, thresholds |
| `Ej3-faces.ipynb` | Face dataset preparation and feature extraction |
| `Ej3-TrainViloaJones.ipynb` | Interactive AdaBoost training loop |
| `Ej3-test.ipynb` | Test a single trained cascade on sample images |
| `Ej3-test_several.ipynb` | Compare multiple cascade configurations side-by-side |

### Trained Models

All cascades in `models/haar_cascades_computed_best/` were trained with **15,000 face samples**:

| File | TPR | Prog. FPR | PFP | Stages |
|------|-----|-----------|-----|--------|
| `..._stage_18_fpr_0.0000000000_99_pfp.xml` | 0.99 | Yes | Yes | 18 |
| `..._stage_21_fpr_0.0000000000_999_pfp.xml` | 0.999 | Yes | Yes | 21 |
| `..._stage_21_fpr_0.0000004014_999_pfp_all.xml` | 0.999 | Yes | Yes | 21 |
| `..._stage_23_fpr_0.0000000000_99.xml` | 0.99 | No | No | 23 |
| `..._stage_28_fpr_0.0000000000_999.xml` | 0.999 | No | No | 28 |

> PFP = Preserve False Positives across stages. Prog. FPR = Progressive FPR target (relaxed in early stages).

### Dataset

[WIDER FACE](http://shuoyang1213.me/WIDERFACE/) — large-scale face detection benchmark.

*Maintained by [MiquelGomezCorral](https://miquelgc.net)*

---

`<a name="español"></a>`

## Español — BIOM Face Detection

Sistema personalizado de detección facial Viola-Jones con clasificador en cascada Haar paralelizado. Entrenado mediante AdaBoost con hard negative mining sobre [WIDER FACE](http://shuoyang1213.me/WIDERFACE/). Compatible con Numba JIT (aceleración 5-10x) y fallback con NumPy multihilo.

### Inicio Rápido

```bash
conda create --name BIOM_env python=3.13 -y && conda activate BIOM_env
uv pip install -r requirements.txt
pip install -e .
pip install numba          # Opcional — acelera inferencia 5-10x
```

### Uso

Todos los comandos se ejecutan desde la raíz del repositorio con `app/main.py`:

```bash
python app/main.py detect-camera      # Detección facial en tiempo real con webcam
python app/main.py train_vj           # Entrenar clasificador en cascada Viola-Jones
python app/main.py generate-filter    # Generar vista de etiquetas filtradas
```

Usa `--help` en cualquier subcomando para ver la lista completa de argumentos.

### Estructura del Proyecto

```
.
├── app/
│   ├── main.py                     # Punto de entrada CLI
│   ├── scripts/                    # Scripts ejecutables
│   └── src/                        # Librería principal
│       ├── config/                 # Dataclass de configuración
│       ├── data/                   # Extracción de características, recortes, datasets
│       └── model/                  # AdaBoost, clasificador en cascada, IO
├── data/
│   ├── ViolaJones/                 # Recortes preprocesados y arrays numpy
│   └── others/WIDER_train/         # Dataset WIDER FACE original
├── models/
│   └── haar_cascades_computed_best/  # Cascadas finales entrenadas
├── notebooks/                      # Jupyter notebooks
├── docs/                           # Gráficos, figuras, artículo de Viola-Jones
├── logs/                           # Logs de entrenamiento
└── scripts/                        # Atajos de shell
```

### Paquetes de Código

**`src/config/`** — Dataclass `Configuration`. Rutas, hiperparámetros y valores por defecto. Crea directorios automáticamente.

**`src/data/`**

| Módulo | Propósito |
|--------|-----------|
| `filter.py` | Imagen integral, integral cuadrada, normalización local |
| `features.py` | Generación de características Haar y precomputación de vectores faciales |
| `crops.py` | Extracción de recortes positivos (caras) y negativos (no-caras) |
| `dataset.py` | Cargadores de datasets WIDER FACE y VPC |
| `balance.py` | Balanceo de datasets por etapa |
| `augment.py` | Aumentación de datos (rotación, flip, brillo/contraste) |

**`src/model/`**

| Módulo | Propósito |
|--------|-----------|
| `cascade_def.py` | Estructuras de datos: `Stage`, `WeakClassifier`, `Feature` |
| `adaboost.py` | Entrenamiento AdaBoost con stumps y early stopping |
| `cascade_clasifier.py` | Detector paralelo principal — Numba JIT o ThreadPoolExecutor |
| `cascade_parser.py` | Lectura de cascadas XML a objetos Python |
| `cascade_serializer.py` | Serialización de etapas a XML compatible con OpenCV |
| `test.py` | Evaluación de cascada — TPR, FPR, visualizaciones |

### Scripts

| Script | Propósito |
|--------|-----------|
| `detect_camera.py` | Abre la webcam y ejecuta detección facial en tiempo real |
| `train_viola_jones.py` | Bucle completo de entrenamiento — características, AdaBoost por etapa, hard negative mining |
| `generate_labels.py` | Genera una vista con etiquetas filtradas sobre imágenes del dataset |

### Notebooks

| Notebook | Contenido |
|----------|-----------|
| `Ej1-Filters.ipynb` | Imágenes integrales, normalización local, desviación estándar |
| `Ej2-CurvaRoc.ipynb` | Análisis de curvas ROC entre etapas de la cascada |
| `Ej3-CheckCascadeStages.ipynb` | Inspección interna de la cascada — etapas, características, umbrales |
| `Ej3-faces.ipynb` | Preparación de dataset facial y extracción de características |
| `Ej3-TrainViloaJones.ipynb` | Bucle interactivo de entrenamiento AdaBoost |
| `Ej3-test.ipynb` | Prueba de una cascada entrenada sobre imágenes de muestra |
| `Ej3-test_several.ipynb` | Comparación de múltiples configuraciones de cascada |

### Modelos Entrenados

Todas las cascadas en `models/haar_cascades_computed_best/` fueron entrenadas con **15,000 muestras de caras**:

| Archivo | TPR | FPR prog. | PFP | Etapas |
|---------|-----|-----------|-----|--------|
| `..._stage_18_fpr_0.0000000000_99_pfp.xml` | 0.99 | Sí | Sí | 18 |
| `..._stage_21_fpr_0.0000000000_999_pfp.xml` | 0.999 | Sí | Sí | 21 |
| `..._stage_21_fpr_0.0000004014_999_pfp_all.xml` | 0.999 | Sí | Sí | 21 |
| `..._stage_23_fpr_0.0000000000_99.xml` | 0.99 | No | No | 23 |
| `..._stage_28_fpr_0.0000000000_999.xml` | 0.999 | No | No | 28 |

> PFP = Preservar falsos positivos entre etapas. FPR prog. = FPR objetivo progresivo (relajado en etapas iniciales).

### Dataset

[WIDER FACE](http://shuoyang1213.me/WIDERFACE/) — benchmark de detección facial a gran escala.

*Mantenido por [MiquelGomezCorral](https://miquelgc.net)*
