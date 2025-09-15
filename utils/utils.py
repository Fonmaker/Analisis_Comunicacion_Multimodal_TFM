# **Ajustes iniciales y configuración de librerías**
# ================================
# Librerías estándar de Python
# ================================
import os
import warnings
import math
import random
import time
import json
import re
import subprocess
import tempfile

# Ignorar warnings
warnings.filterwarnings("ignore")

# ================================
# Ciencia de datos y utilidades
# ================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

# ================================
# Procesamiento de audio
# ================================
import librosa
import torchaudio

# ================================
# Deep Learning (PyTorch)
# ================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ================================
# NLP y modelos preentrenados
# ================================
import nltk
from sentence_transformers import SentenceTransformer, util

# Descargar recursos necesarios de nltk
nltk.download("punkt_tab")


# ================================
# Visión por computador
# ================================
import cv2
import mediapipe as mp
from ultralytics import YOLO

# ================================
# Modelos de voz / ASR
# ================================
from faster_whisper import WhisperModel

# ================================
# Descarga de vídeos
# ================================
import yt_dlp

from google.colab import files

from google.colab import drive
drive.mount('/content/drive')






# =======================
# Configuración
# =======================
SAMPLE_RATE       = 16000
FRAME_LENGTH      = 2048
HOP_LENGTH        = 512
EMPHASIS_LEVELS   = 10

# =======================
# Dispositivo
# =======================
device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)

print(f"\n{'='*50}")
print("Configuración de Dispositivo:")
print(f"Tipo: {device_type.upper()}")
if device_type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Capacidad: {torch.cuda.get_device_capability()}")
    print(f"Memoria Total: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
print(f"{'='*50}\n")

torch.set_default_dtype(torch.float32)
compute_type = "float32"

# =======================
# Instanciar modelos
# =======================
yolo_model = YOLO("yolov8n.pt")
model_whisper = WhisperModel(
    "small",
    device=device_type,
    compute_type=compute_type
)
print(f"Whisper configurado en {device_type} con compute_type={compute_type}")

# =======================
# Instanciar modelo SentenceTransformer
# =======================


model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def clean_up():
    if device_type == "cuda":
        torch.cuda.empty_cache()

# =========================================
# Modelo para extracción sentimiento audio
# =========================================


class DeepModel(nn.Module):
    def __init__(self, input_dim=143, output_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# =====================
# Funciones
# =====================
def pad_audio_smart(data, sr, target_sec=2.5, prev_data=None):

    target_len = int(target_sec * sr)

    if len(data) < target_len:
        pad_len = target_len - len(data)
        if prev_data is not None and len(prev_data) >= pad_len:
            # Tomar del final del audio anterior
            pad = prev_data[-pad_len:]
        else:
            # Rellenar con ceros
            pad = np.zeros(pad_len, dtype=data.dtype)
        data = np.concatenate([pad, data])
    else:
        # Recortar al final
        data = data[-target_len:]

    return data


def extract_features_emotion(data, sample_rate):
    # Calcular características una sola vez
    rms = np.mean(librosa.feature.rms(y=data, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0], dtype=np.float32)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0], dtype=np.float32)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13), axis=1, dtype=np.float32)
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH), axis=1, dtype=np.float32)

    # Crear vector final
    features = np.concatenate([[zcr], mfcc, [rms], mel])
    return features

def predict_emotion(data, sr,model_path):

    # =========================================
    # rutas al modelo
    # =========================================
    modelo_emo= os.path.join(model_path, "deep_model.pth")
    Scaler= os.path.join(model_path, "scaler.pkl")
    encoder= os.path.join(model_path, "label_encoder.pkl")

    model_e = DeepModel(input_dim=143, output_dim=8)
    state_dict = torch.load(modelo_emo, map_location=device)
    model_e.load_state_dict(state_dict)
    model_e = model_e.to(device).to(torch.float32).eval()  # Forzar float32

    scaler = joblib.load(Scaler)
    le = joblib.load(encoder)


    features = extract_features_emotion(data, sr)
    x_features = np.array(features).reshape(1, -1)
    scaled = scaler.transform(x_features)
    with torch.no_grad():
      sample = torch.tensor(scaled, dtype=torch.float32).to(device)
      output = model_e(sample)
      pred = output.argmax(dim=1)
      predicted_label = le.inverse_transform([pred.item()])[0]
      return predicted_label

def segmentos_por_pausa_enfasis(
    audio_path,
    sr=SAMPLE_RATE,
    threshold=0.01,
    min_pause=0.4,
    min_seg=1.0,
    min_cambio_enfasis=2
):


    y, _ = librosa.load(audio_path, sr=sr)

    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]

    # Normalización para énfasis
    rms_norm = rms / (np.max(rms) + 1e-8)
    zcr_norm = zcr / (np.max(zcr) + 1e-8)
    combined = 0.7 * rms_norm + 0.3 * zcr_norm
    combined_norm = combined / (np.max(combined) + 1e-8)
    enfasis_levels = np.clip(np.ceil(combined_norm * EMPHASIS_LEVELS), 1, EMPHASIS_LEVELS).astype(int).tolist()

    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP_LENGTH)

    segmentos = []
    start_time = times[0]
    pause_time = 0.0
    prev_enfasis = enfasis_levels[0]
    prev_pause = 0.0

    for i in range(1, len(rms)):
        time = times[i]
        is_pause = rms[i] < threshold
        enfasis_actual = enfasis_levels[i]
        cambio_enfasis = abs(enfasis_actual - prev_enfasis) >= min_cambio_enfasis

        if is_pause:
            pause_time += times[i] - times[i - 1]
        else:
            if pause_time >= min_pause:
                end_time = time
                if end_time - start_time >= min_seg:
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)

                    seg_rms = rms[(times >= start_time) & (times <= end_time)]
                    seg_zcr = zcr[(times >= start_time) & (times <= end_time)]
                    seg_wave = y[start_sample:end_sample]

                    segmentos.append({
                        "inicio": round(start_time, 2),
                        "fin": round(end_time, 2),
                        "rms_mean": float(np.mean(seg_rms)),
                        "zcr_mean": float(np.mean(seg_zcr)),
                        "prev_pause": prev_pause
                    })
                start_time = end_time
                prev_pause = pause_time
                pause_time = 0.0
            else:
                if cambio_enfasis and (time - start_time >= min_seg):
                    end_time = time
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)

                    seg_rms = rms[(times >= start_time) & (times <= end_time)]
                    seg_zcr = zcr[(times >= start_time) & (times <= end_time)]
                    seg_wave = y[start_sample:end_sample]

                    segmentos.append({
                        "inicio": round(start_time, 2),
                        "fin": round(end_time, 2),
                        "rms_mean": float(np.mean(seg_rms)),
                        "zcr_mean": float(np.mean(seg_zcr)),
                        "rms_vector": seg_rms.tolist(),
                        "prev_pause": prev_pause
                    })
                    start_time = end_time
                    prev_pause = pause_time
                    pause_time = 0.0
        prev_enfasis = enfasis_actual

    # Último segmento
    if times[-1] - start_time >= min_seg:
        start_sample = int(start_time * sr)
        end_sample = len(y)

        seg_rms = rms[(times >= start_time) & (times <= times[-1])]
        seg_zcr = zcr[(times >= start_time) & (times <= times[-1])]
        seg_wave = y[start_sample:end_sample]

        segmentos.append({
            "inicio": round(start_time, 2),
            "fin": round(times[-1], 2),
            "rms_mean": float(np.mean(seg_rms)),
            "zcr_mean": float(np.mean(seg_zcr)),
            "prev_pause": prev_pause
        })

    return y, sr, segmentos

import torch
import torchaudio
import tempfile

def transcribir_con_segmentos(model, y, sr, segmentos,model_path):
    segmentos_lista = []
    texto_completo = ""
    idioma_detectado = None

    # Guardar todo el audio como archivo temporal
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        torchaudio.save(tmp.name, torch.tensor(y).unsqueeze(0), sample_rate=sr)

        # Transcripción con Whisper
        whisper_gen, info = model.transcribe(tmp.name, word_timestamps=True)
        idioma_detectado = getattr(info, "language", None)
        whisper_segments = list(whisper_gen)

        # Texto completo
        texto_completo = " ".join([seg.text for seg in whisper_segments])

        # Aplanar palabras
        todas_palabras = []
        for seg in whisper_segments:
            todas_palabras.extend(seg.words)

        seg_id = 0
        n_samples = len(y)

        for seg in segmentos:
            seg_id += 1
            inicio = seg["inicio"]
            fin = seg["fin"]
            rms_mean = seg.get("rms_mean", 0)
            zcr_mean = seg.get("zcr_mean", 0)
            prev_pause = seg.get("prev_pause", 0)

            # Asegurar que inicio <= fin
            if fin < inicio:
                inicio, fin = fin, inicio

            # Convertir a muestras y limitar al rango del audio
            start_sample = max(0, int(inicio * sr))
            end_sample = min(int(fin * sr), n_samples)
            corte_audio = y[start_sample:end_sample]

            # Palabras dentro del rango
            palabras_segmento = [
                w.word for w in todas_palabras
                if getattr(w, "start", None) is not None and getattr(w, "end", None) is not None
                and w.end > inicio and w.start < fin
            ]
            texto = " ".join(palabras_segmento).strip()

            if texto and len(corte_audio) > 0:
                tipo = "Habla"
                duracion_min = (fin - inicio) / 60
                pmm = len(texto.split()) / duracion_min if duracion_min > 0 else 0
                emotion = predict_emotion(corte_audio, sr,model_path)
            else:
                tipo = "Pausa"
                pmm = 0
                emotion = ""

            t_central = round((inicio + fin) / 2, 2)

            segmentos_lista.append({
                "seg_id": seg_id,
                "audio": {
                    "inicio": round(inicio, 2),
                    "fin": round(fin, 2),
                    "duracion": round(fin - inicio, 2),
                    "pausa_anterior": prev_pause,
                    "rms_mean": rms_mean,
                    "zcr_mean": zcr_mean,
                    "tipo": tipo,
                    "pmm": pmm,
                    "texto": texto,
                    "emocion": emotion
                },
                "video": {
                    "t_central": t_central
                }
            })

    return segmentos_lista, texto_completo, idioma_detectado


def analizar_audio(audio_path,model_path):
    y, sr, segmentos = segmentos_por_pausa_enfasis(audio_path)

    segmentos_lista, texto_completo,idioma = transcribir_con_segmentos(model_whisper, y, sr, segmentos,model_path)

    duracion_video = librosa.get_duration(y=y, sr=sr)

    return {
        "duracion_video": duracion_video,
        "texto_completo": texto_completo,
        "idioma": idioma,
        "segmentos": segmentos_lista
    }

"""# **VIDEO**"""

# Inicializa YOLO



# ---- Inicialización de Mediapipe ----
def init_solutions():
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    return mp_face_mesh, mp_pose, mp_hands

# ---- Función de ángulo ----
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# ---- Detección persona principal ----

def detectar_persona_principal(frame, yolo_model, conf_thresh=0.3, alpha=0.6, beta=0.2, gamma=0.2):
    detections = yolo_model.predict(frame, conf=conf_thresh, verbose=False)[0]
    person_class_id = 0
    persons = [box for box in detections.boxes if int(box.cls) == person_class_id]

    if not persons:
        return frame  # no hay personas detectadas

    areas, brightness, centrality = [], [], []
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2

    for box in persons:
        coords = box.xyxy.cpu().numpy().flatten()
        x1, y1, x2, y2 = map(int, coords)

        # 1. Área
        area = (x2 - x1) * (y2 - y1)
        areas.append(area)

        # 2. Brillo
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size > 0 else np.array([0])
        brightness.append(np.mean(gray))

        # 3. Centralidad
        person_center_x = (x1 + x2) // 2
        person_center_y = (y1 + y2) // 2
        distance_to_center = np.sqrt((person_center_x - center_x)**2 + (person_center_y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        centrality.append(1 - (distance_to_center / max_distance))

    # Normalizar scores
    areas_norm = np.array(areas) / (np.max(areas) + 1e-6)
    brightness_norm = np.array(brightness) / (np.max(brightness) + 1e-6)
    centrality_norm = np.array(centrality) / (np.max(centrality) + 1e-6)
    confidence_scores = [box.conf.cpu().numpy() for box in persons]
    confidence_norm = np.array(confidence_scores) / (np.max(confidence_scores) + 1e-6)

    # Ponderar factores
    scores = (alpha * areas_norm + beta * brightness_norm + gamma * centrality_norm +
              (1 - alpha - beta - gamma) * confidence_norm)

    idx = np.argmax(scores)
    coords = persons[idx].xyxy.cpu().numpy().flatten()
    x1, y1, x2, y2 = map(int, coords)

    # Expansión del bounding box para personas pequeñas
    pad = max(5, int(0.05 * (x2 - x1)))  # 5 píxeles o 5% del ancho
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    crop = frame[y1:y2, x1:x2]
    return crop

# ---- Análisis de frame ----
def analyze_frame_extended(frame, frame_id, face_mesh, pose, hands):
    frame_resize = cv2.resize(frame, (320, int(frame.shape[0] * 320 / frame.shape[1])))
    rgb_frame = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2RGB)
    result_dict = {
        "frame_id": frame_id,
        "cara_detectada": False,
        "inclinacion_cabeza": {"yaw": None, "pitch": None, "roll": None},
        "boca_abierta": None,
        "sonrisa": None,
        "sonrisa_detectada": False,
        "ceño_fruncido": None,
        "ceño_detectado": False,
        "ojos_abiertos": None,
        "asimetria_labios": None,
        "tension_facial": None,
        "estado_emocional": "desconocido",
        "apertura_brazos": None,
        "inclinacion_torso": None

    }

    # ---- Cara ----
    face_results = face_mesh.process(rgb_frame)
    if face_results.multi_face_landmarks:
        face = face_results.multi_face_landmarks[0]
        result_dict["cara_detectada"] = True

        # Referencias
        left_eye = face.landmark[33]
        right_eye = face.landmark[263]
        nose_tip = face.landmark[1]
        chin = face.landmark[152]

        # Distancia entre ojos para normalizar medidas
        eye_distance = np.sqrt(
            (right_eye.x - left_eye.x) ** 2 + (right_eye.y - left_eye.y) ** 2
        )

        # ---- Inclinación cabeza (yaw, pitch, roll) ----
        roll = math.degrees(math.atan2(
            right_eye.y - left_eye.y,
            right_eye.x - left_eye.x
        ))
        pitch = math.degrees(math.atan2(
            chin.y - nose_tip.y,
            chin.x - nose_tip.x
        ))
        eye_center_x = (left_eye.x + right_eye.x) / 2.0
        eye_center_y = (left_eye.y + right_eye.y) / 2.0
        yaw = math.degrees(math.atan2(
            nose_tip.x - eye_center_x,
            nose_tip.y - eye_center_y
        ))

        result_dict["inclinacion_cabeza"] = {"yaw": yaw, "pitch": pitch, "roll": roll}

        # ---- Boca, sonrisa, ceño, ojos ----
        mouth_open = abs(face.landmark[13].y - face.landmark[14].y)
        mouth_open_norm = mouth_open / (eye_distance + 1e-6)
        result_dict["boca_abierta"] = mouth_open_norm

        left_mouth, right_mouth = face.landmark[61], face.landmark[291]
        mouth_width = abs(right_mouth.x - left_mouth.x)
        mouth_height = abs(face.landmark[13].y - face.landmark[14].y)
        smile_ratio = mouth_width / (mouth_height + 1e-6)
        result_dict["sonrisa"] = smile_ratio
        result_dict["sonrisa_detectada"] = smile_ratio > 1.8

        brow_left_inner, brow_right_inner = face.landmark[70], face.landmark[300]
        brow_distance = abs(brow_right_inner.x - brow_left_inner.x) / (eye_distance + 1e-6)
        result_dict["ceño_fruncido"] = brow_distance
        result_dict["ceño_detectado"] = brow_distance < 0.04

        left_eye_open = abs(face.landmark[159].y - face.landmark[145].y)
        right_eye_open = abs(face.landmark[386].y - face.landmark[374].y)
        eye_open_avg = (left_eye_open + right_eye_open) / 2 / (eye_distance + 1e-6)
        result_dict["ojos_abiertos"] = eye_open_avg


        result_dict["asimetria_labios"] = abs(left_mouth.y - right_mouth.y)

        # ---- Tensión facial ----
        tension_score = 0
        if brow_distance < 0.04: tension_score += 1
        if eye_open_avg > 0.06: tension_score += 1
        if mouth_open_norm < 0.02: tension_score += 1
        if smile_ratio > 2.5: tension_score += 1
        result_dict["tension_facial"] = tension_score

        # ---- Estado emocional ----
        if result_dict["sonrisa_detectada"] and tension_score <= 1:
            estado = "sonriente"
        elif tension_score >= 3:
            estado = "tenso"
        elif eye_open_avg > 0.08 and mouth_open_norm > 0.08:
            estado = "sorprendido"
        else:
            estado = "neutral"
        result_dict["estado_emocional"] = estado

    # ---- Postura ----
    pose_results = pose.process(rgb_frame)
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark
        left_angle = calculate_angle(lm[mp.solutions.pose.PoseLandmark.LEFT_ELBOW],
                                     lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER],
                                     lm[mp.solutions.pose.PoseLandmark.LEFT_HIP])
        right_angle = calculate_angle(lm[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW],
                                      lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER],
                                      lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP])
        result_dict["apertura_brazos"] = (left_angle + right_angle) / 2
        torso_angle = calculate_angle(lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER],
                                     lm[mp.solutions.pose.PoseLandmark.LEFT_HIP],
                                     lm[mp.solutions.pose.PoseLandmark.LEFT_KNEE])
        result_dict["inclinacion_torso"] = torso_angle



    return result_dict

# ---- Procesar video ----
def procesar_video(video_path, segmentos_lista):
    mp_face_mesh, mp_pose, mp_hands = init_solutions()

    with mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh, \
         mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose, \
         mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        for segmento in segmentos_lista:
            t_central = segmento["video"]["t_central"]
            frame_id = int(t_central * fps)

            # Mover puntero del video al frame deseado
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                print(f"No se pudo leer el frame {frame_id}")
                continue

            # Detectar persona principal y analizar frame
            principal_frame = detectar_persona_principal(frame, yolo_model)
            resultado = analyze_frame_extended(principal_frame, frame_id, face_mesh, pose, hands)

            # Guardar resultado en el segmento
            segmento["video"]["frame_id"] = frame_id
            segmento["video"]["analisis"] = resultado

        cap.release()

    return segmentos_lista  # ← Devolvemos la lista modificada

def crear_df_segmentos(resultados):
    data = []
    for video_id, info in resultados.items():
        link=info["link"]
        duracion_video=info["duracion_video"]
        texto_completo=info["texto_completo"]
        idioma=info["idioma"]
        for seg in info["segmentos"]:
            audio = seg["audio"]
            video_info = seg.get("video", {})
            analisis = video_info.get("analisis", {})

            fila = {
                "video_id": video_id,
                "link": link,
                "duracion_video": duracion_video,
                "texto_completo": texto_completo,
                "idioma": idioma,
                "seg_id": seg["seg_id"],
                "texto": audio.get("texto", ""),
                "inicio": audio.get("inicio", None),
                "fin": audio.get("fin", None),
                "duracion": audio.get("duracion", None),
                "pausa_anterior": audio.get("pausa_anterior", None),
                "rms_mean": audio.get("rms_mean", None),
                "zcr_mean": audio.get("zcr_mean", None),
                "tipo_audio": audio.get("tipo", None),
                "pmm": audio.get("pmm", None),
                "emocion_audio": audio.get("emocion", None),
                # Información del video
                "t_central": video_info.get("t_central", None),
                "frame_id": video_info.get("frame_id", None),
                "cara_detectada": analisis.get("cara_detectada", False),
                "yaw": analisis.get("inclinacion_cabeza", {}).get("yaw", None),
                "pitch": analisis.get("inclinacion_cabeza", {}).get("pitch", None),
                "roll": analisis.get("inclinacion_cabeza", {}).get("roll", None),
                "boca_abierta": analisis.get("boca_abierta", None),
                "sonrisa": analisis.get("sonrisa", None),

                "ceño_fruncido": analisis.get("ceño_fruncido", None),

                "ojos_abiertos": analisis.get("ojos_abiertos", None),
                "asimetria_labios": analisis.get("asimetria_labios", None),
                "tension_facial": analisis.get("tension_facial", None),

                "apertura_brazos": analisis.get("apertura_brazos", None),
                "inclinacion_torso": analisis.get("inclinacion_torso", None)

            }


            data.append(fila)

    df_segmentos = pd.DataFrame(data)
    return df_segmentos

def eliminar_pausas(df_segmentos):
    # 1. Eliminar filas donde tipo_audio == "Pausa"
    df_segmentos = df_segmentos[df_segmentos["tipo_audio"] != "Pausa"].copy()

    # 2. Ordenar por video_id y seg_id (o por inicio si es más fiable)
    df_segmentos = df_segmentos.sort_values(by=["video_id", "seg_id"]).copy()

    # 3. Reasignar seg_id por cada video_id
    df_segmentos["seg_id"] = df_segmentos.groupby("video_id").cumcount() + 1

    def recalcular_pausa(grupo):
        # Ordenar por seg_id para asegurar secuencia correcta
        grupo = grupo.sort_values(by="seg_id").copy()

        nuevas_pausas = []
        fin_anterior = None

        for _, fila in grupo.iterrows():
            if fin_anterior is None:
                # Primer segmento del video
                nueva_pausa = fila["pausa_anterior"] + fila["inicio"]
            else:
                # Diferencia con el segmento anterior
                nueva_pausa = fila["pausa_anterior"] + (fila["inicio"] - fin_anterior)


            nuevas_pausas.append(nueva_pausa)
            fin_anterior = fila["fin"]

        grupo["pausa_anterior"] = nuevas_pausas
        return grupo

    # 4. Aplicar por cada video_id
    df_segmentos= df_segmentos.groupby("video_id", group_keys=False).apply(recalcular_pausa)

    # 5. Eliminar columna tipo_audio
    if "tipo_audio" in df_segmentos.columns:
        df_segmentos= df_segmentos.drop(columns=["tipo_audio"])

    # 6. Resetear índice
    df_segmentos=df_segmentos.reset_index(drop=True)
    return df_segmentos

def winsorizar(df_segmentos,model_path):

    artifacts_path = os.path.join(model_path , "randomforest_artifacts_modelo4.pkl")
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"No se encontró el archivo del modelo en {artifacts_path}")
    artifacts = joblib.load(artifacts_path)


    winsor_limits_df = artifacts['winsor_limits']


    limites_guardados = {}
    for _, row in winsor_limits_df.iterrows():
        col_name = row['Unnamed: 0']  # nombre de columna
        limites_guardados[col_name] = {"min": row['lower_limit'], "max": row['upper_limit']}

    # ---- Winsorizar columnas presentes en df ----
    for col, limites in limites_guardados.items():
        if col in df_segmentos.columns:
            df_segmentos[col] = np.clip(df_segmentos[col], limites["min"], limites["max"])

    #crear columna para controlar nulos por segmento
    df_segmentos['sin_nulos'] =  (~df_segmentos.isnull().any(axis=1)).astype(int)

    # Calcular porcentaje de frames con cara_detectada=True por video_id
    porcentaje_no_nulos = ( df_segmentos
                                .groupby("video_id")['sin_nulos']
                                  .transform(lambda x: x.mean() * 100) # mean de booleanos = proporción de True
                                  )
    df_segmentos['porcentaje_no_nulos']=porcentaje_no_nulos
    # Resumen por video_id → la clase máxima
    resumen1 = ( df_segmentos .groupby("video_id")["porcentaje_no_nulos"]
                .max()
                .reset_index() )
    # print(resumen1["porcentaje_no_nulos"].value_counts().sort_index())
    # print(df_segmentos["sin_nulos"].value_counts().sort_index())

    #imputar por la mediana de cada fearure
    # Columnas numéricas
    num_cols = df_segmentos.select_dtypes(include=np.number).columns

    # Imputar cada columna numérica con la mediana por video_id
    df_segmentos[num_cols] = (
        df_segmentos
        .groupby("video_id")[num_cols]
        .transform(lambda x: x.fillna(x.median()))
    )
    return df_segmentos

def features_texto_video(df_segmentos):
    # Seleccionar columnas
    columnas = ['video_id', 'link', 'duracion_video', 'texto_completo', 'idioma']

    # Extraer la primera fila de esas columnas

    df_video = df_segmentos[columnas].iloc[[0]]


    # -----------------------------
    # Configuración inicial
    # -----------------------------




    # -----------------------------
    # Patrones lingüísticos
    # -----------------------------
    muletillas = {
        "es": ["eh", "este", "o sea", "vale", "bueno", "ehm"],
        "en": ["um", "uh", "like", "you know", "so", "well"]
    }

    indicadores_anecdota = {
        "es": ["una vez", "recuerdo que", "me pasó", "cuando", "estaba", "fui", "tenía", "en mi vida"],
        "en": ["once", "i remember", "it happened", "when", "i was", "i went", "i had"]
    }

    conectores_ejemplo = {
        "es": ["por ejemplo", "como", "tal como", "a modo de ejemplo", "imagina que", "supongamos que"],
        "en": ["for example", "such as", "like", "as an example", "imagine that", "suppose that"]
    }

    # Para el impacto
    PODER_ES = ["clave", "crítico", "potente", "nuevo", "probado", "garantiza", "impulsa", "mejora",
                "transforma", "aumenta", "reduce", "rápido", "fácil", "eficaz", "evidencia", "récord",
                "resultados", "ahora", "demostrado"]
    POWER_EN = ["key", "critical", "powerful", "new", "proven", "guaranteed", "boost", "improves",
                "transforms", "increases", "reduces", "fast", "easy", "effective", "evidence",
                "record", "results", "now", "demonstrated"]

    VOWELS = set("aeiouáéíóúüAEIOUÁÉÍÓÚÜ")

    # -----------------------------
    # Tokenización
    # -----------------------------
    def split_sentences(text: str):
        text = re.sub(r"\s+", " ", text.strip())
        parts = re.split(r"(?<=[.!?¿¡])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def split_words(text: str):
        return re.findall(r"[\wÁÉÍÓÚÜáéíóúüñÑ'-]+", text, flags=re.UNICODE)

    # -----------------------------
    # Métricas de claridad e impacto
    # -----------------------------
    def count_syllables(word: str):
        groups = 0
        prev = False
        for ch in word:
            is_v = ch in VOWELS
            if is_v and not prev:
                groups += 1
            prev = is_v
        return max(1, groups)

    def flesch_es(text: str):
        sents = split_sentences(text)
        words = split_words(text)
        if not sents or not words:
            return 0.0
        syllables = sum(count_syllables(w) for w in words)
        spw = syllables / len(words)
        wps = len(words) / len(sents)
        return 206.84 - 62.3 * spw - wps

    def flesch_en(text: str):
        sents = split_sentences(text)
        words = split_words(text)
        if not sents or not words:
            return 0.0
        syllables = sum(count_syllables(w) for w in words)
        spw = syllables / len(words)
        wps = len(words) / len(sents)
        return 206.835 - 1.015 * wps - 84.6 * spw

    def score_clarity(text: str, lang: str):
        sents = split_sentences(text)
        readability = flesch_es(text) if lang=="es" else flesch_en(text)
        return {"oraciones": len(sents), "readability": readability}

    def score_impact(text: str, lang: str):
        words = split_words(text)
        power = PODER_ES if lang=="es" else POWER_EN
        n_poder = sum(1 for w in words if w.lower() in power)
        anecdote_markers = ["yo","mi","nosotros","nuestro"] if lang=="es" else ["i","my","we","our"]
        anecdote_hits = sum(1 for w in words if w.lower() in anecdote_markers)
        example_hits = len(re.findall(r"por ejemplo", text, re.IGNORECASE)) if lang=="es" else len(re.findall(r"for example", text, re.IGNORECASE))
        return {"palabras_poder": n_poder, "anecdotas_detectadas": anecdote_hits, "ejemplos_detectados": example_hits}

    # -----------------------------
    # Funciones de análisis tipo embeddings
    # -----------------------------
    def contar_muletillas(oraciones, patrones):
        count = 0
        total = 0
        for o in oraciones:
            total += len(o.split())
            for p in patrones:
                if len(p) <= 3:
                    regex = r"\b" + re.escape(p) + r"+\b"
                else:
                    regex = r"\b" + re.escape(p) + r"\b"
                if re.search(regex, o.lower()):
                    count += 1
        return count / total if total > 0 else 0

    def contar_frases(oraciones, lista_patrones):
        return sum(any(p in o.lower() for p in lista_patrones) for o in oraciones)

    def detectar_por_embeddings(oraciones_emb, seed_patterns, threshold=0.6):
        if oraciones_emb.shape[0] == 0:
            return 0
        seeds_emb = model.encode(seed_patterns, convert_to_tensor=True)
        sims = util.cos_sim(oraciones_emb, seeds_emb)
        return sum((sims.max(dim=1).values > threshold).cpu().numpy())

    # -----------------------------
    # Función unificada de análisis
    # -----------------------------
    def analizar_video(row):
        texto = row['texto_completo']
        idioma = row.get('idioma', None)
        if not idioma:
            # detectar idioma
            words = [w.lower() for w in split_words(texto)]
            es_hits = sum(1 for w in words if w in set("""y de la que el en los se del las un por con no una su para es al lo como más o pero sus le ya o u e si porque muy sin sobre también me hasta hay donde quien desde todo nos durante todos uno les ni contra otros ese eso ante ellos e esto mi antes algunos qué unos yo otro otras otra él tanto esa estos mucho quienes nada muchos cual cuales sea poco ella estar estas algún cómo dos haber aquí mío tuyo suyo nuestra nuestras vuestro vuestros quién cuándo cuáles cuál cuál es son fue eran sería sería serán ser estar estoy estás está estamos están estaban estuvieron estará estaría fui fuiste fue fuimos fueron será serán""".split()))
            en_hits = sum(1 for w in words if w in set("""the of and to in a is that it for on you with as at this but by from they we be have not are or an was if can all will your which their more one about when so what there use up said do out who get she he them his her into than some could been just like now only its also then may our should after over such where how new other these see two first any because me my would did has those before being having""".split()))
            idioma = "es" if es_hits >= en_hits else "en"

        oraciones = split_sentences(texto)
        oraciones_emb = model.encode(oraciones, convert_to_tensor=True) if oraciones else None

        # Métricas de embeddings
        num_anecdotas = (
            contar_frases(oraciones, indicadores_anecdota[idioma]) +
            (detectar_por_embeddings(oraciones_emb, indicadores_anecdota[idioma]) if oraciones_emb is not None else 0)
        )
        num_ejemplos = (
            contar_frases(oraciones, conectores_ejemplo[idioma]) +
            (detectar_por_embeddings(oraciones_emb, conectores_ejemplo[idioma]) if oraciones_emb is not None else 0)
        )

        # Métricas de claridad e impacto
        c = score_clarity(texto, idioma)
        i = score_impact(texto, idioma)



        return pd.Series({
            "claridad_oraciones": c["oraciones"],
            "impacto_palabras_poder": i["palabras_poder"],
            "impacto_anecdotas_detectadas": i["anecdotas_detectadas"],
            "anecdotas": num_anecdotas,
            "ejemplos": num_ejemplos,

         })

    # -----------------------------
    # Aplicar al DataFrame
    # -----------------------------
    df_result = df_video.apply(analizar_video, axis=1)
    df_video = pd.concat([df_video, df_result], axis=1)
    return df_video

def calculos_agregados_video(df_segmentos):

      #EMOCION AUDIO A DUMMIES
      df_segmentos_dummies = pd.get_dummies(df_segmentos, columns=['emocion_audio'])
      # -------------------------
      # 1. Columnas de segmento
      # -------------------------
      segment_features = [
          'pausa_anterior','rms_mean', 'zcr_mean', 'pmm','yaw', 'pitch', 'roll',
          'boca_abierta', 'sonrisa', 'ceño_fruncido', 'ojos_abiertos', 'asimetria_labios',
          'tension_facial', 'apertura_brazos', 'inclinacion_torso',
          'emocion_audio_angry', 'emocion_audio_calm', 'emocion_audio_disgust',
          'emocion_audio_fearful', 'emocion_audio_happy', 'emocion_audio_neutral',
          'emocion_audio_sad', 'emocion_audio_surprised','porcentaje_no_nulos'
      ]

      for col in segment_features:
          if col not in df_segmentos_dummies.columns:
              df_segmentos_dummies[col] = 0

      # -------------------------
      # 2. Estadísticas agregadas por video
      # -------------------------
      agg_funcs = ['mean', 'var', 'max', 'min']
      df_videos = df_segmentos_dummies.groupby('video_id')[segment_features].agg(agg_funcs)


      # Aplanar MultiIndex
      df_videos.columns = ['_'.join(col).strip() for col in df_videos.columns.values]

      # Añadir etiqueta de tipo de comunicador
      labels = df_segmentos_dummies.groupby('video_id').first()
      df_videos.reset_index(inplace=True)

      # -------------------------
      # 3. Feature engineering: combinaciones inteligentes
      # -------------------------
      df_videos['rms_sonrisa'] = df_videos['rms_mean_mean'] * df_videos['sonrisa_mean']
      df_videos['pitch_ojos'] = df_videos['pitch_mean'] * df_videos['ojos_abiertos_mean']
      df_videos['zcr_ceño'] = df_videos['zcr_mean_mean'] * df_videos['ceño_fruncido_mean']

      df_videos['rms_brazos'] = df_videos['rms_mean_mean'] * df_videos['apertura_brazos_mean']
      df_videos['pitch_torso'] = df_videos['pitch_mean'] * df_videos['inclinacion_torso_mean']

      df_videos['rms_zcr_var_ratio'] = df_videos['rms_mean_var'] / (df_videos['zcr_mean_var'] + 1e-6)
      df_videos['boca_apertura_diff'] = df_videos['boca_abierta_mean'] - np.sqrt(df_videos['boca_abierta_var'] + 1e-6)

      # -------------------------
      # 4. Medida de expresividad
      # -------------------------
      # Parámetro de mezcla: 40% gestual, 60% facial
      alpha = 0.4

      # Expresividad gestual (movimientos de cabeza)
      df_videos['expresividad_gestual'] = np.sqrt(
          df_videos['yaw_mean']**2 + df_videos['pitch_mean']**2 + df_videos['roll_mean']**2
      )

      # Expresividad facial (media de intensidades faciales)
      df_videos['expresividad_facial'] = df_videos[[
          'boca_abierta_mean','sonrisa_mean','ceño_fruncido_mean',
          'ojos_abiertos_mean','asimetria_labios_mean','tension_facial_mean'
      ]].mean(axis=1)

      # Índice global
      df_videos['expresividad'] = alpha * df_videos['expresividad_gestual'] + (1-alpha) * df_videos['expresividad_facial']
      df_calculos_agregados_video = df_videos.copy()

      return df_calculos_agregados_video

def probabilidades(df_video,df_calculos_agregados_video, model_path):

    # =======================
    # Cargar modelo Random Forest y artefactos
    # =======================
    artifacts_path = os.path.join(model_path , "randomforest_artifacts_modelo4.pkl")
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"No se encontró el archivo del modelo en {artifacts_path}")
    artifacts = joblib.load(artifacts_path)

    model_artifacts      = artifacts["model"]
    explainer            = artifacts["explainer"]
    features             = artifacts["features"]
    selected_features    = artifacts["features"]
    frases_por_feature   = artifacts["frases_por_feature"]
    ideales_globales     = artifacts["ideales_globales"]
    # -----------------------------
    # Combinar DataFrames
    # -----------------------------
    df_combined = pd.concat([df_video, df_calculos_agregados_video], axis=1)

    # -----------------------------
    # Asegurarse de que todas las columnas existen
    # -----------------------------
    for col in selected_features:
        if col not in df_combined.columns:
            df_combined[col] = 0

    # -----------------------------
    # Extraer columnas y convertir a lista
    # -----------------------------
    X = df_combined[selected_features].copy()
    x_sample = X
    i = 0

    CLASS_CHOICE = "positive"
    ROW_INDEX = None
    DATA_SOURCE = "all"





    # -----------------------------
    # 3) Predicción del modelo
    # -----------------------------
    probs = model_artifacts.predict_proba(X)[0]
    classes = list(model_artifacts.classes_)
    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]

    prob_1 = probs[1]
    nivel_comunicador = f"🎯 Tus cualidades comunicativas según el modelo: {(prob_1)*100:.0f} %"


    # Clasificación cualitativa
    if prob_1 < 0.20:
        categoria = "🔴 Nivel de comunicador **bajo**"
    elif prob_1 < 0.40:
        categoria = "🟠 Nivel de comunicador **medio-bajo**"
    elif prob_1 < 0.60:
        categoria = "🟡 Nivel de comunicador **medio**"
    elif prob_1 < 0.80:
        categoria = "🟢 Nivel de comunicador **medio-alto**"
    else:
        categoria = "🔵 Nivel de comunicador **alto**"

    print(nivel_comunicador)
    print(categoria)


    # -----------------------------
    # 4) Obtener SHAP local (robusto a formas)
    # -----------------------------
    def compute_shap_for_one(explainer, x_df):
        """
        Devuelve un objeto que puede ser shap.Explanation o array/list,
        dependiendo de la versión/Explainer.
        """
        try:
            return explainer(x_df)  # API moderna suele devolver Explanation
        except Exception:
            # Fallback a .shap_values clásico
            return explainer.shap_values(x_df)

    sv = compute_shap_for_one(explainer, x_sample)

    def to_1d_for_class(sv_obj, features, classes, class_choice, pred_class):
        """
        Normaliza SHAP a un vector 1D (n_features,) para UNA clase.
        Soporta:
          - shap.Explanation con .values de formas (f,), (1,f), (f,c), (c,f),
            (1,f,c), (1,c,f)
          - list por clase
          - ndarray con las formas anteriores
        """
        class_to_idx = {c:i for i,c in enumerate(classes)}
        if class_choice == "positive" and 1 in class_to_idx:
            target_idx = class_to_idx[1]
        elif class_choice == "predicted":
            target_idx = class_to_idx[pred_class]
        else:
            # fallback: predicha
            target_idx = class_to_idx[pred_class]

        # Extraer valores crudos
        arr = np.asarray(getattr(sv_obj, "values", sv_obj))

        # Casos
        if isinstance(sv_obj, (list, tuple)):
            # Lista por clase: tomar directamente la del target
            part = sv_obj[target_idx]
            arr = np.asarray(getattr(part, "values", part))
            if arr.ndim == 2 and arr.shape[0] == 1:  # (1, f)
                arr = arr[0]
            assert arr.ndim == 1 or (arr.ndim == 2 and arr.shape[0] == 1) or (arr.ndim == 2 and arr.shape[1] == len(features)), \
                f"SHAP list forma inesperada: {arr.shape}"
            if arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 2 and arr.shape[1] == len(features):
                # (?, f). Si batch=1 -> coger fila 0
                if arr.shape[0] == 1:
                    arr = arr[0]
                else:
                    raise ValueError("Se esperaba batch=1 para la explicación local.")
            return arr.reshape(-1)

        # Explanation/ndarray
        if arr.ndim == 1 and arr.shape[0] == len(features):
            return arr  # (f,)
        if arr.ndim == 2:
            if arr.shape == (1, len(features)):            # (1,f)
                return arr[0]
            if arr.shape == (len(features), len(classes)):  # (f,c)
                return arr[:, target_idx]
            if arr.shape == (len(classes), len(features)):  # (c,f)
                return arr[target_idx, :]
            raise ValueError(f"Forma SHAP 2D inesperada: {arr.shape}")

        if arr.ndim == 3:
            # Esperamos batch=1
            if arr.shape[0] != 1:
                raise ValueError(f"Se esperaba batch=1, obtuve {arr.shape}")
            if arr.shape[1] == len(features) and arr.shape[2] == len(classes):   # (1,f,c)
                return arr[0, :, target_idx]
            if arr.shape[1] == len(classes) and arr.shape[2] == len(features):   # (1,c,f)
                return arr[0, target_idx, :]
            raise ValueError(f"Forma SHAP 3D inesperada: {arr.shape}")

        raise ValueError(f"Forma SHAP no soportada: {arr.shape}")

    shap_val_1d = to_1d_for_class(sv, features, classes, CLASS_CHOICE, pred_class)
    assert shap_val_1d.shape[0] == len(features), f"Incongruencia SHAP vs features: {shap_val_1d.shape} vs {len(features)}"

    # -----------------------------
    # 5) Agrupación por tipo de feature (según tu lista)
    # -----------------------------
    feature_groups = {
        "Texto": [
            'claridad_oraciones', 'impacto_palabras_poder', 'impacto_anecdotas_detectadas',
            'anecdotas', 'ejemplos'
        ],
        "Audio - Energía/Ritmo": [
            'rms_mean_mean', 'rms_mean_var', 'rms_zcr_var_ratio', 'zcr_mean_min'
        ],
        "Audio - Emociones": [
            'emocion_audio_happy_mean', 'emocion_audio_fearful_mean', 'emocion_audio_angry_mean',
            'emocion_audio_surprised_mean', 'emocion_audio_disgust_mean', 'emocion_audio_calm_mean'
        ],
        "Postura/Gestos": [
            'yaw_max', 'pitch_min', 'apertura_brazos_var', 'expresividad_facial',
            'expresividad_gestual', 'rms_brazos'
        ],
        "otros": ['duracion_video', 'pmm_mean']
    }

    # Mapear feature -> grupo (las que no estén en el mapeo van a "Otros")
    feature_to_group = {}
    for g, cols in feature_groups.items():
        for col in cols:
            feature_to_group[col] = g
    for f in features:
        if f not in feature_to_group:
            feature_to_group[f] = "Otros"
            if "Otros" not in feature_groups:
                feature_groups["Otros"] = []
            if f not in feature_groups["Otros"]:
                feature_groups["Otros"].append(f)

    # Acumular por grupo
    group_shap = {}
    group_details = {}
    for feat, s_val, x_val in zip(features, shap_val_1d, x_sample.values[0]):
        g = feature_to_group.get(feat, "Otros")
        group_shap[g] = group_shap.get(g, 0.0) + float(s_val)
        group_details.setdefault(g, []).append((feat, x_val, float(s_val)))

    def generar_feedback(
        x_sample,
        shap_val_1d,
        ideales_globales,
        feature_to_group,
        features,
        frases_por_feature
    ):
        feedback_global = ""
        feedback_grupo = {}
        feedback_individual = {}

        grupo_shap_acumulado = {}
        grupo_frases = {}

        for f, val_actual, shap_val in zip(features, x_sample.values[0], shap_val_1d):
            ideal = ideales_globales.get(f, None)
            grupo = feature_to_group.get(f, "Otros")
            desc = frases_por_feature.get(f, {}).get("descripcion", "")

            diff = None
            if ideal is not None:
                diff = val_actual - ideal

            # Elegir frase según diferencia y dirección SHAP
            if ideal is not None and abs(diff) < 0.05:
                frase_detalle = frases_por_feature[f].get("alineado", "")
            elif diff is not None and diff > 0:
                if shap_val > 0:
                    frase_detalle = frases_por_feature[f].get("mas_positivo", "")
                else:
                    frase_detalle = frases_por_feature[f].get("mas_negativo", "")
            elif diff is not None and diff < 0:
                if shap_val > 0:
                    frase_detalle = frases_por_feature[f].get("menos_positivo", "")
                else:
                    frase_detalle = frases_por_feature[f].get("menos_negativo", "")
            else:
                frase_detalle = "No se pudo interpretar el impacto de esta variable."

            # Construir frase completa
            frase = f" {desc}\n"
            frase += f"Tiene un valor de `{val_actual:.2f}`"
            if ideal is not None:
                frase += f" frente a un ideal de `{ideal:.2f}` (diferencia `{diff:+.2f}`)"
            frase += f". SHAP={shap_val:+.3f}. {frase_detalle}"

            feedback_individual[f] = frase

            grupo_shap_acumulado[grupo] = grupo_shap_acumulado.get(grupo, 0.0) + shap_val
            grupo_frases.setdefault(grupo, []).append(frase)

        # Feedback por grupo
        for grupo, total_shap in grupo_shap_acumulado.items():
            if total_shap > 0.05:
                resumen = f"🔵 El grupo **{grupo}** tiene una influencia positiva destacada en la predicción."
            elif total_shap < -0.05:
                resumen = f"🔴 El grupo **{grupo}** tiene una influencia negativa significativa en la predicción."
            else:
                resumen = f"⚪ El grupo **{grupo}** tiene un impacto neutro o leve en la predicción."

            feedback_grupo[grupo] = resumen #+ "\n" + "\n".join(grupo_frases[grupo])

        # Feedback global
        total = sum(grupo_shap_acumulado.values())
        if total > 0.1:
            feedback_global = "🟢 En general, las características del orador favorecen positivamente la predicción del modelo."
        elif total < -0.1:
            feedback_global = "🔴 En general, las características del orador restan probabilidad a la clase positiva del modelo."
        else:
            feedback_global = "⚪ El perfil del orador presenta un equilibrio entre elementos positivos y negativos."

        return {
            "feedback_global": feedback_global,
            "feedback_grupo": feedback_grupo,
            "feedback_individual": feedback_individual
        }

    feedback = generar_feedback(
        x_sample=X,
        shap_val_1d=shap_val_1d,
        ideales_globales=ideales_globales,
        feature_to_group=feature_to_group,
        features=features,
        frases_por_feature=frases_por_feature
    )

    print("\n🔷 FEEDBACK GLOBAL")
    print(feedback["feedback_global"])
    print("\n")
    # -----------------------------
    # 6) Gráfico agrupado
    # -----------------------------
    groups = list(group_shap.keys())
    values = [group_shap[g] for g in groups]

    # Asignar color según signo
    colors = ['#5DADE2' if v >= 0 else '#E74C3C' for v in values]  # azul / rojo
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(groups, values, color=colors)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_title(f"Contribución por grupo a la predicción (fila {i} / clase {'predicha' if CLASS_CHOICE=='predicted' else 'positiva'})")
    ax.set_xlabel("Contribución SHAP (en probabilidad)")
    plt.grid(True, axis='x')

    for bar in bars:
        w = bar.get_width()
        ax.text(w + (0.001 if w >= 0 else -0.001),
                bar.get_y() + bar.get_height()/2,
                f"{w:.3f}",
                ha="left" if w >= 0 else "right",
                va="center")
    fig.subplots_adjust(left=0.35)
    # plt.tight_layout()
    plt.show()



    for grupo, feats in feature_groups.items():

        print(f"\n🟩 FEEDBACK — Grupo: {grupo}")
        print(feedback["feedback_grupo"].get(grupo, "Sin resumen disponible."))
        print("")


        # Solo conservar las features que están en el modelo
        feats_in_model = [f for f in feats if f in features]
        if not feats_in_model:
            continue

        # === 1) Comparativa valor actual vs valor ideal ===
        vals_actual = [X.iloc[0][f] for f in feats_in_model]
        vals_ideal = [ideales_globales.get(f, np.nan) for f in feats_in_model]

        x = np.arange(len(feats_in_model))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - width/2, vals_actual, width, label='Valor actual')
        ax.bar(x + width/2, vals_ideal, width, label='Valor ideal grupo 1')
        ax.set_xticks(x)
        ax.set_xticklabels(feats_in_model, rotation=45, ha='right')
        ax.set_title(f"Comparativa: segmento vs ideal — Grupo: {grupo}")
        ax.set_ylabel("Valor")
        ax.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        # === 2) Gráfico SHAP individual por característica del grupo ===
        detalles_grupo = group_details.get(grupo, [])
        if detalles_grupo:
            detalles_sorted = sorted(detalles_grupo, key=lambda x: abs(x[2]), reverse=True)
            nombres = [f[0] for f in detalles_sorted]
            shap_vals = [f[2] for f in detalles_sorted]

            fig, ax = plt.subplots(figsize=(10, max(4, len(nombres)*0.4)))
            bars = ax.barh(nombres, shap_vals, color=["#5DADE2" if v >= 0 else "#E74C3C" for v in shap_vals])
            ax.axvline(0, color='black', linewidth=1)
            ax.set_title(f"Contribución SHAP individual — Grupo: {grupo} (fila {i})")
            ax.set_xlabel("Contribución SHAP (en probabilidad)")
            ax.invert_yaxis()
            plt.grid(True, axis='x', linestyle='--', alpha=0.5)

            for bar, val in zip(bars, shap_vals):
                w = bar.get_width()
                ax.text(w + (0.001 if w >= 0 else -0.001),
                        bar.get_y() + bar.get_height()/2,
                        f"{w:.3f}",
                        ha="left" if w >= 0 else "right",
                        va="center")
            fig.subplots_adjust(left=0.35)
            # plt.tight_layout()
            plt.show()

        # === 3) Feedback textual ===


        for f in feats_in_model:
            if f in feedback["feedback_individual"]:

                print(f"   🔹 {f}:{feedback['feedback_individual'][f]}\n")

"""# **RUN**"""

def download_video():

    VIDEO_DIR = "videos"
    os.makedirs(VIDEO_DIR, exist_ok=True)
    opcion = input("Elige opción:\n1. Descargar desde URL\n2. Subir desde tu disco\nSelecciona 1 o 2: ")

    if opcion == "1":
        url = input("Introduce la URL del video: ")
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': os.path.join(VIDEO_DIR, '%(id)s.%(ext)s'),
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = os.path.join(VIDEO_DIR, f"{info['id']}.mp4")
            return video_path, info['id']

    elif opcion == "2":
        uploaded = files.upload()
        for filename in uploaded.keys():
            file_id, ext = os.path.splitext(filename)
            video_path = os.path.join(VIDEO_DIR, filename)
            os.rename(filename, video_path)
            return video_path, file_id

    else:
        print("Opción no válida.")
        return None, None


def check_and_convert(video_path, n_frames=5):
    """Comprueba si OpenCV puede leer los primeros n_frames del video e imprime el resultado."""
    cap = cv2.VideoCapture(video_path)
    success_any = False

    for i in range(n_frames):
        success, frame = cap.read()
        if success:

            success_any = True
            break

    cap.release()

    if not success_any:
        print("No se pudieron leer frames, convirtiendo video formato compatible...")
        base, ext = os.path.splitext(video_path)
        new_path = base + "_converted.mp4"
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "28",
            "-vf", "scale=640:-2",
            "-r", "24",
            "-an",  # sin audio
            "-y",
            new_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return new_path
    else:
        return video_path



def extract_audio_from_video(video_path, audio_path):
    command = [
        'ffmpeg', '-i', video_path, '-vn',
        '-acodec', 'pcm_s16le', '-ar', str(SAMPLE_RATE),
        '-ac', '1', audio_path, '-y'
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def procesar_url(model_path):
    inicio = time.time()

    video_path, audio_path, video_id = None, None, None
    resultados = {}

    try:
        video_path, video_id = download_video()
        audio_path = f"{video_id}.wav"

        extract_audio_from_video(video_path, audio_path)

        #chequear video
        video_path = check_and_convert(video_path)


        resultado_audio = analizar_audio(audio_path,model_path)


        segmentos = procesar_video(video_path, resultado_audio["segmentos"])

        resultados[video_id] = {
            "video_id": video_id,
            "link": video_path,
            "duracion_video": resultado_audio["duracion_video"],
            "texto_completo": resultado_audio["texto_completo"],
            "idioma": resultado_audio["idioma"],
            "segmentos": segmentos
        }

    except Exception as e:
        if video_id:
            print(f"❌ Error procesando {video_id}: {e}")
        else:
            print(f"❌ Error procesando video: {e}")

    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        clean_up()

    fin = time.time()
    duracion = round(fin - inicio, 2)

    if video_id:
        print(f"✅ Video {video_id} procesado en {duracion} segundos.")
    else:
        print(f"⚠️ Proceso terminado en {duracion} segundos sin resultados.")

    return resultados


def analisis_multimodal(model_path):

    resultados=procesar_url(model_path)

    df_segmentos = crear_df_segmentos(resultados)
    df_segmentos=eliminar_pausas(df_segmentos)
    df_segmentos=winsorizar(df_segmentos,model_path)
    df_video=features_texto_video(df_segmentos)
    df_calculos_agregados_video=calculos_agregados_video(df_segmentos)
    probabilidades(df_video,df_calculos_agregados_video,model_path)
    return df_video,df_calculos_agregados_video


if __name__ == "__main__":

    df_video,df_calculos_agregados_video=analisis_multimodal(model_path)