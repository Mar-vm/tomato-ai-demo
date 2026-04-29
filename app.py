from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import os

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
    "Bell_pepper leaf spot": {"nombre_es": "Mancha foliar del pimiento morrón", "descripcion": "Manchas oscuras que se necrosan.", "solucion": "Rotar cultivos y aplicar bactericidas/fungicidas."},
    "Corn Gray leaf spot": {"nombre_es": "Mancha gris del maíz", "descripcion": "Manchas rectangulares gris-marrón.", "solucion": "Rotación de cultivos y fungicidas."},
    "Corn leaf blight": {"nombre_es": "Tizón foliar del maíz", "descripcion": "Manchas grises o marrones.", "solucion": "Híbridos resistentes y fungicidas."},
    "Corn rust leaf": {"nombre_es": "Roya del maíz", "descripcion": "Pústulas anaranjadas.", "solucion": "Fungicidas preventivos."},
    "Potato leaf early blight": {"nombre_es": "Tizón temprano de la papa", "descripcion": "Manchas marrones con patrón concéntrico.", "solucion": "Fungicidas cada 7-14 días."},
    "Potato leaf late blight": {"nombre_es": "Tizón tardío de la papa", "descripcion": "Manchas oscuras con moho.", "solucion": "Fungicidas preventivos y eliminar plantas enfermas."},
    "Soybean leaf": {"nombre_es": "Roya de la soya", "descripcion": "Pústulas rojo-anaranjadas.", "solucion": "Fungicidas preventivos y rotar cultivos."},
    "Squash Powdery mildew leaf": {"nombre_es": "Oídio de la calabaza", "descripcion": "Polvo blanco sobre hojas.", "solucion": "Mejorar ventilación y aplicar fungicidas."},
    "Tomato Early blight leaf": {"nombre_es": "Tizón temprano del tomate", "descripcion": "Manchas marrones con anillos concéntricos.", "solucion": "Eliminar follaje enfermo y aplicar fungicidas."},
    "Tomato Septoria leaf spot": {"nombre_es": "Mancha de Septoria en tomate", "descripcion": "Manchas circulares con centro gris.", "solucion": "Riego por goteo y fungicidas."},
    "Tomato leaf bacterial spot": {"nombre_es": "Mancha bacteriana del tomate", "descripcion": "Manchas oscuras y acuosas.", "solucion": "Semillas sanas y bactericidas."},
    "Tomato leaf late blight": {"nombre_es": "Tizón tardío del tomate", "descripcion": "Manchas oscuras con moho.", "solucion": "Fungicidas preventivos y rotar cultivos."},
    "Tomato leaf mosaic virus": {"nombre_es": "Mosaico del tomate", "descripcion": "Hojas con patrones amarillos.", "solucion": "Controlar insectos vectores."},
    "Tomato leaf yellow virus": {"nombre_es": "Amarillamiento del tomate", "descripcion": "Hojas amarillentas y enrolladas.", "solucion": "Controlar mosca blanca."},
    "Tomato mold leaf": {"nombre_es": "Mildiu del tomate", "descripcion": "Manchas húmedas o mohosas.", "solucion": "Evitar humedad y aplicar fungicidas."},
    "Tomato two spotted spider mites leaf": {"nombre_es": "Ácaros de dos manchas en tomate", "descripcion": "Manchas amarillas y defoliación.", "solucion": "Acaricidas o depredadores naturales."},
    "grape leaf black rot": {"nombre_es": "Tizón negro de la vid", "descripcion": "Manchas negras circulares.", "solucion": "Podar ramas infectadas y aplicar fungicidas."},
}

INFO_DEFAULT = {
    "nombre_es": "Hoja saludable",
    "descripcion": "No se detectaron enfermedades significativas.",
    "solucion": "Continúa con mantenimiento preventivo regular."
}

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo", "plantdoc_300_epochs3", "weights", "best.onnx")
session = None

@app.on_event("startup")
def load_model():
    global session
    try:
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        print(f"✅ Modelo ONNX cargado desde {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")

@app.get("/")
def root():
    return {
        "message": "LOTUS Plant Disease AI running",
        "modelo": "YOLOv7 ONNX - PlantDoc 300 epochs",
        "clases": len(CLASES),
        "modelo_cargado": session is not None
    }

@app.get("/health")
def health():
    return {"status": "ok", "modelo_cargado": session is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if session is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((640, 640))
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = img_np.transpose(2, 0, 1)
        img_np = np.expand_dims(img_np, axis=0)

        inputs = {session.get_inputs()[0].name: img_np}
        outputs = session.run(None, inputs)

        detecciones = []
        pred = outputs[0]  # shape: [1, N, 85] o [N, 6]
        
        # Normalizar forma del tensor
        if pred.ndim == 3:
            pred = pred[0]  # quitar batch dimension -> [N, 85]

        for det in pred:
            # Obtener confianza y clase
            if len(det) >= 6:
                box_conf = float(det[4])
                class_scores = det[5:]
                clase_idx = int(np.argmax(class_scores))
                conf = box_conf * float(class_scores[clase_idx])
            else:
                continue
                
            if conf > 0.25:
                clase_nombre = CLASES[clase_idx] if clase_idx < len(CLASES) else "Desconocido"
                info = INFO_CLASES.get(clase_nombre, INFO_DEFAULT)
                detecciones.append({
                    "clase": clase_nombre,
                    "nombre_es": info["nombre_es"],
                    "confianza": round(conf * 100, 2),
                    "descripcion": info["descripcion"],
                    "solucion": info["solucion"]
                })

        detecciones.sort(key=lambda x: x["confianza"], reverse=True)
        detecciones = detecciones[:5]  # top 5

        return {
            "success": True,
            "detecciones": detecciones,
            "mensaje": "Detección completada" if detecciones else "No se detectaron enfermedades",
            "modelo": "YOLOv7-ONNX-PlantDoc"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")
