# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 20:34:46 2025

@author: Hugo
"""

# main.py
import os, re, difflib
from io import BytesIO
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import numpy as np
import cv2

# En Linux (Render) Tesseract está en PATH cuando lo instalamos en Docker
os.environ.pop("TESSDATA_PREFIX", None)
TESS_CONFIG = "--psm 6"  # si instalas 'spa' en Docker, puedes agregar lang="spa"

app = FastAPI(title="OCR Boletas API")

# CORS: ajusta origins a tu dominio de FF (o deja * para pruebas)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # en prod pon tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- helpers de prepro (sin rotación) ---
def _to_uint8(img):
    return img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)

def _unsharp(gray, sigma=1.0, strength=1.0):
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return _to_uint8(cv2.addWeighted(gray, 1.0 + strength, blur, -strength, 0))

def enhance_for_ocr_pil(pil_img, target_height=1800):
    img = pil_img.convert("RGB")
    w, h = img.size
    if h < target_height:
        s = target_height / float(h)
        img = img.resize((int(w * s), target_height), resample=Image.BICUBIC)

    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)
    sharp = _unsharp(binary, sigma=1.0, strength=1.0)
    return Image.fromarray(sharp)

# --- fuzzy + extracción ---
def buscar_valor_near(texto, patrones_clave, regex_valor, look_ahead=2, cutoff=0.6):
    lineas = texto.splitlines()
    bajas = [l.lower() for l in lineas]
    for patron in patrones_clave:
        m = difflib.get_close_matches(patron.lower(), bajas, n=1, cutoff=cutoff)
        if not m:
            continue
        idx = bajas.index(m[0])
        fin = min(len(lineas), idx + 1 + look_ahead)
        for ln in lineas[idx:fin]:
            mm = re.search(regex_valor, ln)
            if mm:
                return mm.group(1)
    return None

REGEX_MONTO = r"(\$?\s?\d{1,3}(?:[.\s]\d{3})+|\$?\s?\d+(?:[.,]\d{2})?)"
REGEX_FECHA_GLOBAL = re.compile(
    r"(?P<fecha>(?:\d{4}[/-]\d{1,2}[/-]\d{1,2})|(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|(?:\d{1,2}[.]\d{1,2}[.]\d{2,4}))"
    r"(?:\s*[\/\-]?\s*(?P<hora>\d{1,2}[:.]\d{2}(?::\d{2})?))?",
    re.I
)

def normalizar_fecha(fecha_str, _hora_str=None):
    f = fecha_str.replace('.', '-').replace('/', '-')
    parts = f.split('-')
    try:
        if len(parts[0]) == 4:
            y, m, d = map(int, parts)
        else:
            d, m, y = map(int, parts)
            if y < 100:
                y += 2000 if y < 70 else 1900
        return f"{d:02d}-{m:02d}-{str(y)[-2:]}"  # DD-MM-YY
    except Exception:
        return None

def extraer_fecha(texto):
    cerca = buscar_valor_near(
        texto,
        patrones_clave=["fecha emision", "fecha emisión", "fecha compra", "fecha", "f. emision", "f. emisión"],
        regex_valor=r"(\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        look_ahead=2,
        cutoff=0.55,
    )
    if cerca:
        return normalizar_fecha(cerca, None)
    for ln in texto.splitlines():
        g = REGEX_FECHA_GLOBAL.search(ln)
        if g:
            return normalizar_fecha(g.group("fecha"), g.group("hora"))
    return None

PRIORIDAD_ID = [
    ("numero operacion", r"(\d{6,})"),
    ("comprobante", r"(\d{6,})"),
    ("nro transaccion", r"([A-Za-z0-9]{8,})"),
    ("boleta electronica", r"(\d{6,})"),
]

def extraer_campos(texto_boleta: str):
    total = buscar_valor_near(
        texto_boleta,
        patrones_clave=["total a pagar", "total", "monto", "importe total", "total compra"],
        regex_valor=REGEX_MONTO,
        look_ahead=2,
        cutoff=0.55
    )
    fecha = extraer_fecha(texto_boleta)
    id_boleta = None
    id_label = None
    for etiqueta, rgx in PRIORIDAD_ID:
        id_boleta = buscar_valor_near(
            texto_boleta,
            patrones_clave=[etiqueta],
            regex_valor=rgx,
            look_ahead=3,
            cutoff=0.55
        )
        if id_boleta:
            id_label = etiqueta
            break
    return total, fecha, id_boleta, id_label

@app.post("/ocr")
async def ocr_boleta(file: UploadFile = File(...)):
    try:
        pil_img = Image.open(BytesIO(await file.read()))
    except Exception:
        return JSONResponse({"error": "No pude abrir la imagen. Usa jpg/png/tif/webp."}, status_code=400)

    img_proc = enhance_for_ocr_pil(pil_img)
    texto = pytesseract.image_to_string(img_proc, config=TESS_CONFIG).strip()
    total, fecha, id_boleta, id_label = extraer_campos(texto)

    return {
        "filename": file.filename,
        "total": total,
        "fecha": fecha,  # DD-MM-YY
        "id_boleta": id_boleta,
        "id_label_prioritario": id_label,
        "texto_ocr": texto
    }
