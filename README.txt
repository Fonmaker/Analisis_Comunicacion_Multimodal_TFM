# 🧠 Análisis Multimodal de la Comunicación en Video – TFM

**Repositorio oficial del Trabajo Fin de Máster (TFM):**  
**"Análisis multimodal de la comunicación en video para evaluar la calidad comunicativa en discursos"**

Este proyecto aborda el análisis multimodal de discursos en video, combinando texto, audio y video para evaluar la calidad comunicativa. Incluye la extracción de características, procesamiento, entrenamiento de modelos, análisis interpretativo y generación de feedback.

---

## 📁 Archivos grandes alojados externamente

Debido a las restricciones de tamaño de GitHub (máximo 100 MB por archivo), los siguientes archivos no se encuentran en este repositorio. Puedes descargarlos desde los enlaces indicados:

| Archivo                     | Descripción                                  | Enlace de descarga                                                |
|----------------------------|----------------------------------------------|-------------------------------------------------------------------|
| `modelo_emociones_cnn.pth` | Modelo CNN entrenado con datos de RAVDESS    | [🔗 Descargar](https://drive.google.com/file/d/ID_MODELO/view?usp=sharing) |
| `dataset_segmentado.csv`   | Dataset final con características multimodales | [🔗 Descargar](https://drive.google.com/file/d/ID_DATASET/view?usp=sharing) |
| `video_ted_ejemplo.mp4`    | Video TED usado como muestra de evaluación    | [🔗 Descargar](https://drive.google.com/file/d/ID_VIDEO/view?usp=sharing) |

📌 **Ubicación esperada de los archivos:**
│
├── models/
│ └── modelo_emociones_cnn.pth
├── data/
│ └── dataset_segmentado.csv
├── videos/
│ └── video_ted_ejemplo.mp4

Este proyecto utiliza el **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)** para entrenar y validar modelos de reconocimiento de emociones a partir de audio.

- Dataset original disponible en Zenodo:  
  🔗 https://zenodo.org/record/1188976  
  📄 Licencia: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

- Versión utilizada descargada desde Kaggle (espejo no oficial):  
  🔗 https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

📂 Una vez descargado, coloca los archivos `.wav` en:

Analisis_Multimodal_Comunicacion_TFM/data/ravdess_path/

Analisis_Multimodal_Comunicacion_TFM/
│
├── notebooks/ # Notebooks para procesamiento, modelado y análisis
├── models/ # Modelos entrenados (descargables)
├── data/
│ ├── ravdess_path/ # Audios de RAVDESS
│ └── dataset_segmentado.csv
├── videos/ # Videos TED usados para evaluación
├── scripts/ # Scripts para extracción de características y procesamiento
├── requirements.txt # Dependencias del proyecto
└── README.md