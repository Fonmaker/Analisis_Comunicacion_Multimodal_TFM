# 🧠 Análisis Multimodal de la Comunicación en Video – TFM

**Repositorio oficial del Trabajo Fin de Máster (TFM):**  
**“Análisis multimodal de la comunicación en video para evaluar la calidad comunicativa en discursos”**

Este proyecto aborda el análisis automatizado de discursos en video, combinando **texto, audio y video** para evaluar la calidad comunicativa. Incluye la extracción de características multimodales, su procesamiento, el entrenamiento de modelos explicables, análisis interpretativo y la generación de **feedback estructurado** para el orador.

> 📍 **Todo el desarrollo se ha realizado en Google Colab**, trabajando directamente sobre archivos alojados en Google Drive para facilitar la gestión, ejecución distribuida por grupos de vídeos y la interoperabilidad con los recursos de almacenamiento en la nube.

---

## 📁 Repositorio y Archivos

El proyecto se encuentra disponible en dos ubicaciones:

- **GitHub:** [🔗 Repositorio](https://github.com/Fonmaker/Analisis_Comunicacion_Multimodal_TFM)
- **Google Drive (Completo):** [🔗 Carpeta completa](https://drive.google.com/drive/folders/15LFR3rK3jT_EtLpjtoeKOWTUoHEcF-1d?usp=drive_link)

---

## ⚠️ Archivos grandes disponibles solo en Google Drive

Debido a las restricciones de tamaño de GitHub (máx. 100 MB por archivo), algunos archivos están **solo en Google Drive**:

| Archivo                      | Descripción                                                  | Enlace                                                               |
|-----------------------------|--------------------------------------------------------------|----------------------------------------------------------------------|
| `df_dummies.csv`            | Dataset intermedio con variables emocionales codificadas     | [🔗 Descargar](https://drive.google.com/file/d/1RXl31QiY2JExLEBKjoMEwKwBx7TRqyJV/view?usp=drive_link) |
| `df_seg_win_med.csv`        | Dataset final de segmentos tras limpieza y winsorización     | [🔗 Descargar](https://drive.google.com/file/d/1oUNcY3c7w1ZANAIzXGD0M5kG1-W62CHk/view?usp=drive_link) |
| `features_videos_ted.json`  | Características de los vídeos procesados en formato JSON     | [🔗 Descargar](https://drive.google.com/file/d/1PrpVz4tk0KlDS5e_tsS4rk3a6YVcLl-q/view?usp=drive_link) |

📌 **Ubicación esperada dentro del proyecto:**  
`Analisis_Multimodal_Comunicacion_TFM/data/folder_path/`

---

## 🎙 Dataset de emociones en audio (RAVDESS)

Este proyecto utiliza el dataset **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** para entrenar un modelo de detección de emociones acústicas.

- 📦 Dataset original: [🔗 Zenodo](https://zenodo.org/record/1188976)
- 📄 Licencia: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- 📥 Versión alternativa: [🔗 Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

📂 **Ubicación esperada de los archivos `.wav`:**  
`Analisis_Multimodal_Comunicacion_TFM/data/ravdess_path/`

---

## 🧱 Estructura del proyecto

```
Analisis_Multimodal_Comunicacion_TFM/
│
├── notebooks/
│   ├── 1_preparacion_dataset.ipynb
│   ├── 2_modelo_emociones_audio_ravdess.ipynb
│   ├── 3_extraccion_datos_videos_ted.ipynb
│   ├── 4_limpieza_estructuracion_dataset_final.ipynb
│   ├── 5_modelo.ipynb
│   └── 6_produccion.ipynb ← Notebook para usar el sistema con nuevos vídeos
│
├── models/
│   ├── deep_model.pth               ← Modelo de emociones en audio
│   ├── scaler.pkl, label_encoder.pkl ← Archivos auxiliares para el modelo de emociones
│   ├── randomforest_artifacts_modelo4.pkl ← Modelo 4 (seleccionado) + artefactos
│   └── randomforest_artifacts_modelo5.pkl ← Modelo alternativo + artefactos
│
├── data/
│   ├── ravdess_path/     ← Audios RAVDESS usados para el modelo de emociones
│   ├── json_path/        ← Archivos JSON por grupo de vídeos TED
│   └── folder_path/      ← Archivos intermedios del proceso de análisis
│
├── utils/
│   └── utils.py          ← Funciones auxiliares para producción en Colab
│
├── Analisis_Multimodal_Comunicacion_TFM.html     ← Memoria completa del TFM
├── Presentacion_Analisis_Multimodal_Comunicacion_TFM.mp4 ← Vídeo explicativo
├── README.md
└── LICENSE
```
## 🎥 Presentación del proyecto en video

Si prefieres una explicación visual del funcionamiento y los objetivos del TFM, puedes ver el video explicativo, disponible tanto en este repositorio como en el siguiente enlace de YouTube:

🔗 [Ver en YouTube](https://youtu.be/kRusKsHG1uc)

---

## 📌 Notebooks clave

- `2_modelo_emociones_audio_ravdess.ipynb`: Entrenamiento del modelo de clasificación emocional a partir de audio.
- `3_extraccion_datos_videos_ted.ipynb`: Descarga, segmentación y análisis de vídeos TED.
- `4_limpieza_estructuracion_dataset_final.ipynb`: Procesamiento, limpieza y generación del dataset final.
- `5_modelo.ipynb`: Entrenamiento del modelo Random Forest y análisis de interpretabilidad.
- `6_produccion.ipynb`: Aplicación del modelo entrenado a nuevos vídeos, usando `utils.py`.
