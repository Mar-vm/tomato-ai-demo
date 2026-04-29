from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import sys
import os
import cv2
import numpy as np
from PIL import Image
import io

# Agregar yolov7 al path ANTES de importar
YOLOV7_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov7')
if YOLOV7_PATH not in sys.path:
    sys.path.insert(0, YOLOV7_PATH)

app = FastAPI(title="LOTUS Plant Disease AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASES = [
    "Apple Scab Leaf", "Apple leaf", "Apple rust leaf",
    "Bell_pepper leaf spot", "Bell_pepper leaf", "Blueberry leaf",
    "Cherry leaf", "Corn Gray leaf spot", "Corn leaf blight",
    "Corn rust leaf", "Peach leaf", "Potato leaf early blight",
    "Potato leaf late blight", "Potato leaf", "Raspberry leaf",
    "Soyabean leaf", "Soybean leaf", "Squash Powdery mildew leaf",
    "Strawberry leaf", "Tomato Early blight leaf", "Tomato Septoria leaf spot",
    "Tomato leaf bacterial spot", "Tomato leaf late blight",
    "Tomato leaf mosaic virus", "Tomato leaf yellow virus", "Tomato leaf",
    "Tomato mold leaf", "Tomato two spotted spider mites leaf",
    "grape leaf black rot", "grape leaf"
]

INFO_CLASES = {
    "Apple Scab Leaf": {"nombre_es": "Sarna de la hoja del manzano", "descripcion": "Manchas oscuras y ásperas en hojas y frutos.", "solucion": "Podar ramas infectadas y aplicar fungicidas específicos."},
    "Apple rust leaf": {"nombre_es": "Roya de la hoja del manzano", "descripcion": "Manchas amarillas o anaranjadas en las hojas.", "solucion": "Retirar hojas infectadas y aplicar fungicidas preventivos."},
    "Bell_pepper leaf spot": {"nombre_es": "Mancha foliar del pimiento morrón", "descripcion": "Manchas oscuras que se necrosan y causan caída del follaje.", "solucion": "Rotar cultivos y aplicar bactericidas/fungicidas."},
    "Corn Gray leaf spot": {"nombre_es": "Mancha gris del maíz", "descripcion": "Manchas rectangulares gris-marrón que reducen la fotosíntesis.", "solucion": "Rotación de cultivos y fungicidas si es severo."},
    "Corn leaf blight": {"nombre_es": "Tizón foliar del maíz", "descripcion": "Manchas grises o marrones que afectan el rendimiento.", "solucion": "Híbridos resistentes y fungicidas si hay alta presión."},
    "Corn rust leaf": {"nombre_es": "Roya del maíz", "descripcion": "Pústulas anaranjadas que afectan la fotosíntesis.", "solucion": "Usar híbridos resistentes y fungicidas preventivos."},
    "Potato leaf early blight": {"nombre_es": "Tizón temprano de la papa", "descripcion": "Manchas marrones con patrón concéntrico.", "solucion": "Eliminar follaje infectado y aplicar fungicidas cada 7-14 días."},
    "Potato leaf late blight": {"nombre_es": "Tizón tardío de la papa", "descripcion": "Manchas oscuras con moho gris-blanquecino.", "solucion": "Fungicidas preventivos y eliminar plantas enfermas."},
    "Soybean leaf": {"nombre_es": "Roya de la soya", "descripcion": "Pústulas rojo-anaranjadas que causan defoliación.", "solucion": "Fungicidas preventivos y rotar cultivos."},
    "Squash Powdery mildew leaf": {"nombre_es": "Oídio de la calabaza", "descripcion": "Polvo blanco sobre hojas que reduce la fotosíntesis.", "solucion": "Mejorar ventilación y aplicar fungicidas."},
    "Tomato Early blight leaf": {"nombre_es": "Tizón temprano del tomate", "descripcion": "Manchas marrones con anillos concéntricos.", "solucion": "Eliminar follaje enfermo y aplicar fungicidas preventivos."},
    "Tomato Septoria leaf spot": {"nombre_es": "Mancha de Septoria en tomate", "descripcion": "Manchas circulares con centro gris y borde oscuro.", "solucion": "Riego por goteo y fungicidas específicos."},
    "Tomato leaf bacterial spot": {"nombre_es": "Mancha bacteriana del tomate", "descripcion": "Manchas oscuras y acuosas que causan necrosis.", "solucion": "Semillas sanas y aplicar bactericidas."},
    "Tomato leaf late blight": {"nombre_es": "Tizón tardío del tomate", "descripcion": "Manchas oscuras con moho gris-blanquecino.", "solucion": "Fungicidas preventivos y rotar cultivos."},
    "Tomato leaf mosaic virus": {"nombre_es": "Mosaico del tomate", "descripcion": "Hojas con patrones amarillos en mosaico.", "solucion": "Controlar insectos vectores y eliminar plantas infectadas."},
    "Tomato leaf yellow virus": {"nombre_es": "Amarillamiento del tomate", "descripcion": "Hojas amarillentas, enrolladas y deformadas.", "solucion": "Controlar mosca blanca y eliminar plantas infectadas."},
    "Tomato mold leaf": {"nombre_es": "Mildiu del tomate", "descripcion": "Manchas húmedas o mohosas en las hojas.", "solucion": "Evitar exceso de humedad y aplicar fungicidas."},
    "Tomato two spotted spider mites leaf": {"nombre_es": "Ácaros de dos manchas en tomate", "descripcion": "Manchas amarillas y defoliación en infestaciones graves.", "solucion": "Acaricidas o depredadores naturales."},
    "grape leaf black rot": {"nombre_es": "Tizón negro de la vid", "descripcion": "Manchas negras circulares en hojas y racimos.", "solucion": "Podar ramas infectadas y aplicar fungicidas."},
}

INFO_DEFAULT = {
    "nombre_es": "Hoja saludable",
    "descripcion": "No se detectaron enfermedades significativas.",
    "solucion": "Continúa con mantenimiento preventivo regular."
}

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo", "plantdoc_300_epochs3", "weights", "best.pt")
device = None
model = None

@app.on_event("startup")
def load_model():
    global model, device
    try:
        from models.experimental import attempt_load
        from utils.torch_utils import select_device
        device = select_device('')
        model = attempt_load(MODEL_PATH, map_location=device)
        model.eval()
        print(f"✅ Modelo cargado desde {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")

@app.get("/")
def root():
    return {
        "message": "LOTUS Plant Disease AI running",
        "modelo": "YOLOv7 - PlantDoc 300 epochs",
        "clases": len(CLASES),
        "modelo_cargado": model is not None
    }

@app.get("/health")
def health():
    return {"status": "ok", "modelo_cargado": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    try:
        from utils.general import non_max_suppression

        contents = await file.read()
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(img_pil)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        img_resized = cv2.resize(img_cv, (640, 640))
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            pred = model(img_tensor)[0]
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

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

        detecciones.sort(key=lambda x: x["confianza"], reverse=True)

        return {
            "success": True,
            "detecciones": detecciones if detecciones else [],
            "mensaje": "Detección completada" if detecciones else "No se detectaron enfermedades",
            "modelo": "YOLOv7-PlantDoc"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")
