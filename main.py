from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps, ImageEnhance
import pandas as pd
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import gdown

app = FastAPI()

# --- SEGURIDAD ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. DESCARGA AUTOMÁTICA Y CARGA DE IA ---
print("--- INICIANDO SERVIDOR APPLONCITO CLOUD ---")

file_id = '13hF-8e0E1AAqvq4LNoHk46K90B99_9b4'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'modelo_huellas_v2.h5'

# Descargar si no existe
if not os.path.exists(output):
    print("📥 Descargando modelo desde Google Drive...")
    try:
        gdown.download(url, output, quiet=False)
        print("✅ Descarga finalizada.")
    except Exception as e:
        print(f"❌ Error descargando de Drive: {e}")

# Intentar Cargar el modelo
try:
    modelo_base = load_model(output)
    nombre_capa_features = modelo_base.layers[-2].name 
    modelo = Model(inputs=modelo_base.input, outputs=modelo_base.get_layer(nombre_capa_features).output)
    print("✅ IA Lista: MODO CLOUD (Modelo Propio)")
except Exception as e:
    print(f"⚠️ Error cargando modelo .h5: {e}")
    print("🔄 Usando ResNet50 como respaldo...")
    from tensorflow.keras.applications import ResNet50
    modelo = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# --- 2. PREPARACIÓN DE IMAGEN ---
def preparar_imagen(imagen_bytes):
    img = Image.open(BytesIO(imagen_bytes))
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    img = ImageOps.equalize(img)
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# --- 3. MATEMÁTICA DEL PARENTESCO ---
def calcular_similitud_estricta(vector1, vector2):
    v1, v2 = vector1.flatten(), vector2.flatten()
    raw = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    print(f"📊 DIAGNÓSTICO RAW: {raw:.5f}")

    corte = 0.9585 
    if raw < corte:
        porcentaje = ((raw / corte) ** 50) * 35 
    else:
        porcentaje = 78.0 + ((raw - corte) * 1000)
        if raw > 0.97: porcentaje += 10

    return float(np.clip(round(porcentaje, 2), 0.0, 99.9))

# --- 4. SEMÁFORO ---
def generar_diagnostico(probabilidad):
    if probabilidad >= 70:
        return {"color": "#28a745", "titulo": "🟢 ALTA PROBABILIDAD", "consejo": "Herencia visual fuerte. Es casi seguro que comparten genética."}
    elif probabilidad >= 40:
        return {"color": "#ffc107", "titulo": "🟡 ZONA INCIERTA", "consejo": "Rasgos compartidos pero no dominantes. Recomendado: Triangulación."}
    else:
        return {"color": "#dc3545", "titulo": "🔴 BAJA COINCIDENCIA", "consejo": "Poco parecido visual. Alerta: Podría ser Fenotipo Materno."}

# --- 5. GUARDAR HISTORIAL ---
def guardar_historial(probabilidad, diagnostico):
    archivo_csv = "/tmp/historial_pruebas.csv" 
    datos = {
        "Fecha": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Probabilidad (%)": [probabilidad],
        "Veredicto": [diagnostico["titulo"]]
    }
    df = pd.DataFrame(datos)
    header = not os.path.exists(archivo_csv)
    df.to_csv(archivo_csv, mode='a', header=header, index=False, sep=';', encoding='utf-8-sig')

# --- 6. ENDPOINT ---
@app.post("/analizar_parentesco")
async def analizar_parentesco(foto_padre: UploadFile = File(...), foto_hijo: UploadFile = File(...)):
    print(f"\n📸 Procesando nueva solicitud desde la App...")
    try:
        img_p = preparar_imagen(await foto_padre.read())
        img_h = preparar_imagen(await foto_hijo.read())
        
        f_p = modelo.predict(img_p, verbose=0)
        f_h = modelo.predict(img_h, verbose=0)
        
        resultado = calcular_similitud_estricta(f_p, f_h)
        diag = generar_diagnostico(resultado)
        guardar_historial(resultado, diag)

        return JSONResponse(content={
            "probabilidad": resultado,
            "titulo": diag["titulo"],
            "mensaje": diag["consejo"],
            "color_hex": diag["color"]
        })
    except Exception as e:
        print(f"❌ Error: {e}")
        return JSONResponse(content={"probabilidad": 0, "mensaje": f"Error: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


