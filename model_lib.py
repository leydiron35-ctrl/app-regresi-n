"""
model_lib.py
-----------------------------------------------------------------------------
Define la clase CreditApprovalModel en un módulo propio (no en __main__ ni
dentro de api/predict.py) para que joblib pueda serializarla/deserializarla
de forma confiable: pickle necesita poder hacer
`from model_lib import CreditApprovalModel` tanto al guardar modelo.joblib
como al cargarlo en el entorno serverless de Vercel.

Misma fórmula que computePrediction() en aprobacion_de_creditos.html y que
predict_proba() en entrenar_modelo.py. No se modifica ninguna lógica aquí,
solo se reubica la clase para que el pickle sea portable.
"""
import numpy as np


class CreditApprovalModel:
    """Envoltorio con predict_proba().

    Variables de entrada (mismo orden/semántica que la UI):
      hombre, casados, separados : 0/1 (referencia = Mujer / Soltero)
      ingresos                   : valor libre, entra como ln(1 + ingresos)
      gastos, deuda              : puntos porcentuales (0-100), sin tope
    """

    def __init__(self, intercept, coef):
        self.intercept = intercept
        self.coef = coef  # dict con las mismas llaves que MODEL.coef en el HTML

    def _ingreso_transformado(self, ingresos):
        return np.log1p(max(ingresos, 0))

    def predict_proba(self, hombre, casados, separados, ingresos, gastos, deuda):
        contributions = {
            "Género (Hombre vs. Mujer)": self.coef["hombre"] * hombre,
            "Estado civil": (self.coef["casados"] * casados)
                             + (self.coef["separados"] * separados),
            "Ingresos": self.coef["ingresos"] * self._ingreso_transformado(ingresos),
            "Ratio Gastos/Ingreso": self.coef["ratio_gastos"] * gastos,
            "Ratio Deuda/Ingreso": self.coef["ratio_deuda"] * deuda,
        }
        z = self.intercept + sum(contributions.values())
        prob = 1 / (1 + np.exp(-z))
        return prob, contributions


# Mismos valores que MODEL en el <script> del HTML / INTERCEPTO-COEFICIENTES
# en entrenar_modelo.py.
INTERCEPT = 0.286
COEF = {
    "hombre": -0.01,
    "casados": -0.02,
    "separados": -0.01,
    "ingresos": 0.8,
    "ratio_gastos": -0.055,
    "ratio_deuda": -0.055,
}
