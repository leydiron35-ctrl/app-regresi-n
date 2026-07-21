""
api/predict.py
-----------------------------------------------------------------------------
Endpoint serverless para Vercel (runtime Python).

Recibe por POST el mismo conjunto de campos que ya captura el formulario en
aprobacion_de_creditos.html y devuelve la MISMA probabilidad y el MISMO
desglose de factores que hoy calcula computePrediction() en el <script> del
HTML. No se cambia la lógica del modelo: este endpoint solo la expone por
HTTP para quien quiera consumirla desde un backend en vez de (o además de)
calcularla en el navegador.

Body esperado (JSON):
{
  "gender":   "hombre" | "mujer",
  "marital":  "casado" | "soltero" | "separado",
  "ingresos": <number>,
  "gastos":   <number>,   // 0-100, puntos porcentuales
  "deuda":    <number>    // 0-100, puntos porcentuales
}

Respuesta (JSON):
{
  "prob": 0.588,
  "approved": true,
  "contributions": {
    "Género (Hombre vs. Mujer)": -0.01,
    "Estado civil": -0.02,
    "Ingresos": 3.28,
    "Ratio Gastos/Ingreso": -1.375,
    "Ratio Deuda/Ingreso": -1.815
  }
}

Configuración en vercel.json: este archivo se despliega automáticamente
como función en /api/predict (convención de carpeta api/ de Vercel).
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler

import joblib

# model_lib.py y modelo.joblib viven en la raíz del repo; este archivo está
# en api/, así que se sube un nivel para encontrarlos. Es necesario importar
# CreditApprovalModel explícitamente ANTES de joblib.load(): joblib guarda
# solo la referencia a la clase (módulo + nombre), no su código, así que el
# módulo tiene que estar ya cargado en sys.modules para poder deserializarla.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model_lib import CreditApprovalModel  # noqa: F401  (necesario para joblib.load)

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "modelo.joblib")
_modelo = joblib.load(_MODEL_PATH)


def _validar_numero(valor, nombre):
    try:
        return float(valor)
    except (TypeError, ValueError):
        raise ValueError(f"'{nombre}' debe ser un número válido.")


def calcular_prediccion(payload):
    gender = payload.get("gender")
    marital = payload.get("marital")

    if gender not in ("hombre", "mujer"):
        raise ValueError("'gender' debe ser 'hombre' o 'mujer'.")
    if marital not in ("casado", "soltero", "separado"):
        raise ValueError("'marital' debe ser 'casado', 'soltero' o 'separado'.")

    ingresos = _validar_numero(payload.get("ingresos"), "ingresos")
    gastos = _validar_numero(payload.get("gastos"), "gastos")
    deuda = _validar_numero(payload.get("deuda"), "deuda")

    # Misma codificación de rango completo que en el HTML: referencia = Mujer / Soltero
    hombre = 1 if gender == "hombre" else 0
    casados = 1 if marital == "casado" else 0
    separados = 1 if marital == "separado" else 0

    prob, contributions = _modelo.predict_proba(
        hombre=hombre, casados=casados, separados=separados,
        ingresos=ingresos, gastos=gastos, deuda=deuda,
    )

    return {
        "prob": prob,
        "approved": prob >= 0.5,
        "contributions": contributions,
    }


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status, body):
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self._send_json(204, {})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw or b"{}")
            resultado = calcular_prediccion(payload)
            self._send_json(200, resultado)
        except ValueError as e:
            self._send_json(400, {"error": str(e)})
        except Exception as e:
            self._send_json(500, {"error": f"Error interno: {e}"})

    def do_GET(self):
        # Ping simple para verificar que el endpoint está desplegado
        self._send_json(200, {"status": "ok", "mensaje": "Usar POST con los campos del formulario."})
        handler = CreditHandler
