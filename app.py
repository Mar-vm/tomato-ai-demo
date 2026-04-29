from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import sys
import os
import cv2
import numpy as np
from PIL import Image
import io

# Agregar yolov7 al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yolov7'))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

app = FastAPI(title="LOTUS Plant Disease AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clases del modelo (en el mismo orden que se entrenó)
CLASES = [
    "Apple Scab Leaf",
    "Apple leaf",
    "Apple rust leaf",
    "Bell_pepper leaf spot",
    "Bell_pepper leaf",
    "Blueberry leaf",
    "Cherry leaf",
    "Corn Gray leaf spot",
    "Corn leaf blight",
    "Corn rust leaf",
    "Peach leaf",
    "Potato leaf early blight",
    "Potato leaf late blight",
    "Potato leaf",
    "Raspberry leaf",
    "Soyabean leaf",
    "Soybean leaf",
    "Squash Powdery mildew leaf",
    "Strawberry leaf",
    "Tomato Early blight leaf",
    "Tomato Septoria leaf spot",
    "Tomato leaf bacterial spot",
    "Tomato leaf late blight",
    "Tomato leaf mosaic virus",
    "Tomato leaf yellow virus",
    "Tomato leaf",
    "Tomato mold leaf",
    "Tomato two spotted spider mites leaf",
    "grape leaf black rot",
    "grape leaf"
]

# Traducciones y recomendaciones por clase
INFO_CLASES = {
    "Apple Scab Leaf": {
        "nombre_es": "Sarna de la hoja del manzano",
        "descripcion": "Provoca manchas oscuras y ásperas en las hojas y frutos del manzano.",
        "solucion": "Podar ramas infectadas, eliminar hojas caídas y aplicar fungicidas específicos."
    },
    "Apple rust leaf": {
        "nombre_es": "Roya de la hoja del manzano",
        "descripcion": "Provoca manchas amarillas o anaranjadas en las hojas del manzano.",
        "solucion": "Retirar hojas infectadas, evitar cercanía con cedros y aplicar fungicidas preventivos."
    },
    "Bell_pepper leaf spot": {
        "nombre_es": "Mancha foliar del pimiento morrón",
        "descripcion": "Pequeñas manchas oscuras que pueden necrosarse y causar caída prematura del follaje.",
        "solucion": "Usar semillas sanas, rotar cultivos y aplicar bactericidas/fungicidas."
    },
    "Corn Gray leaf spot": {
        "nombre_es": "Mancha gris de la hoja de maíz",
        "descripcion": "Manchas rectangulares gris-marrón que reducen la fotosíntesis.",
        "solucion": "Rotación de cultivos, eliminar rastrojos y fungicidas si es severo."
    },
    "Corn leaf blight": {
        "nombre_es": "Tizón foliar del maíz",
        "descripcion": "Manchas rectangulares de color gris o marrón que afectan el rendimiento.",
        "solucion": "Rotación de cultivos, híbridos resistentes y fungicidas si hay alta presión."
    },
    "Corn rust leaf": {
        "nombre_es": "Roya del maíz",
        "descripcion": "Pústulas anaranjadas o marrones que afectan la fotosíntesis.",
        "solucion": "Usar híbridos resistentes y aplicar fungicidas preventivos."
    },
    "Potato leaf early blight": {
        "nombre_es": "Tizón temprano de la papa",
        "descripcion": "Manchas marrones con patrón concéntrico que causan defoliación prematura.",
        "solucion": "Rotación de cultivos, eliminar follaje infectado y aplicar fungicidas cada 7-14 días."
    },
    "Potato leaf late blight": {
        "nombre_es": "Tizón tardío de la papa",
        "descripcion": "Manchas oscuras con moho gris-blanquecino. Puede destruir rápidamente las plantas.",
        "solucion": "Usar variedades resistentes, fungicidas preventivos y eliminar plantas enfermas."
    },
    "Soybean leaf": {
        "nombre_es": "Roya de la soya",
        "descripcion": "Pústulas rojo-anaranjadas que provocan defoliación prematura.",
        "solucion": "Aplicar fungicidas preventivos, rotar cultivos y eliminar restos infectados."
    },
    "Squash Powdery mildew leaf": {
        "nombre_es": "Oídio de la calabaza",
        "descripcion": "Polvo blanco sobre hojas y tallos que reduce la fotosíntesis.",
        "solucion": "Mejorar ventilación, evitar riego por aspersión y aplicar fungicidas."
    },
    "Tomato Early blight leaf": {
        "nombre_es": "Tizón temprano del tomate",
        "descripcion": "Manchas marrones con anillos concéntricos que causan defoliación.",
        "solucion": "Rotar cultivos, eliminar follaje enfermo y aplicar fungicidas preventivos."
    },
    "Tomato Septoria leaf spot": {
        "nombre_es": "Mancha foliar de Septoria en tomate",
        "descripcion": "Manchas circulares con centro gris y borde oscuro.",
        "solucion": "Retirar hojas infectadas, usar riego por goteo y aplicar fungicidas."
    },
    "Tomato leaf bacterial spot": {
        "nombre_es": "Mancha bacteriana del tomate",
        "descripcion": "Manchas oscuras y acuosas que pueden causar necrosis.",
        "solucion": "Usar semillas sanas, eliminar hojas infectadas y aplicar bactericidas."
    },
    "Tomato leaf late blight": {
        "nombre_es": "Tizón tardío del tomate",
        "descripcion": "Manchas oscuras con moho gris-blanquecino. Destruye rápidamente la planta.",
        "solucion": "Retirar hojas infectadas, aplicar fungicidas preventivos y rotar cultivos."
    },
    "Tomato leaf mosaic virus": {
        "nombre_es": "Mosaico del tomate",
        "descripcion": "Hojas con patrones amarillos en mosaico y deformación.",
        "solucion": "Usar semillas libres de virus, controlar insectos vectores y eliminar plantas infectadas."
    },
    "Tomato leaf yellow virus": {
        "nombre_es": "Enrollamiento y amarillamiento del tomate",
        "descripcion": "Hojas amarillentas, enrolladas y deformadas.",
        "solucion": "Usar semillas sanas, controlar mosca blanca y eliminar plantas infectadas."
    },
    "Tomato mold leaf": {
        "nombre_es": "Mildiu del tomate",
        "descripcion": "Manchas húmedas o mohosas que se vuelven marrones.",
        "solucion": "Evitar exceso de humedad, ventilar y aplicar fungicidas específicos."
    },
    "Tomato two spotted spider mites leaf": {
        "nombre_es": "Ácaros de dos manchas en tomate",
        "descripcion": "Manchas amarillas en hojas, pérdida de vigor y defoliación en infestaciones graves.",
        "solucion": "Controlar con acaricidas o depredadores naturales, evitar estrés hídrico."
    },
    "grape leaf black rot": {
        "nombre_es": "Tizón negro de la vid",
        "descripcion": "Manchas negras circulares en hojas que afectan también los racimos.",
        "solucion": "Podar ramas infectadas, eliminar restos vegetales y aplicar fungicidas."
    },
}

INFO_DEFAULT = {
    "nombre_es": "Hoja saludable",
    "descripcion": "No se detectaron enfermedades significativas.",
    "solucion": "Continúa con el mantenimiento preventivo regular."
}

# Cargar modelo al iniciar
MODEL_PATH = "modelo/plantdoc_300_epochs3/weights/best.pt"
device = select_device('')
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = attempt_load(MODEL_PATH, map_location=device)
        model.eval()
        print("✅ Modelo cargado correctamente")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")

@app.get("/")
def root():
    return {
        "message": "LOTUS Plant Disease AI running",
        "modelo": "YOLOv7 - PlantDoc 300 epochs",
        "clases": len(CLASES)
    }

@app.get("/health")
def health():
    return {"status": "ok", "modelo_cargado": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    # Leer imagen
    contents = await file.read()
    img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(img_pil)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Preprocesar
    img_resized = cv2.resize(img_cv, (640, 640))
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    # Inferencia
    with torch.no_grad():
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Procesar resultados
    detecciones = []
    for det in pred:
        if det is not None and len(det):
            for *xyxy, conf, cls in det:
                clase_idx = int(cls)
                clase_nombre = CLASES[clase_idx] if clase_idx < len(CLASES) else "Desconocido"
                info = INFO_CLASES.get(clase_nombre, INFO_DEFAULT)
                detecciones.append({
                    "clase": clase_nombre,
                    "nombre_es": info["nombre_es"],
                    "confianza": round(float(conf) * 100, 2),
                    "descripcion": info["descripcion"],
                    "solucion": info["solucion"]
                })

    if not detecciones:
        return {
            "success": True,
            "detecciones": [],
            "mensaje": "No se detectaron enfermedades en la imagen",
            "modelo": "YOLOv7-PlantDoc"
        }

    # Ordenar por confianza
    detecciones.sort(key=lambda x: x["confianza"], reverse=True)

    return {
        "success": True,
        "detecciones": detecciones,
        "modelo": "YOLOv7-PlantDoc"
    }
