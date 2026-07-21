"""
build_model.py
-----------------------------------------------------------------------------
Genera modelo.joblib a partir de la clase y los coeficientes definidos en
model_lib.py (los mismos que usa aprobacion_de_creditos.html y
entrenar_modelo.py). No reentrena nada: solo serializa el objeto.

Ejecutar una vez (local) para (re)generar modelo.joblib:
    python build_model.py
"""
import joblib
from model_lib import CreditApprovalModel, INTERCEPT, COEF

if __name__ == "__main__":
    modelo = CreditApprovalModel(INTERCEPT, COEF)
    joblib.dump(modelo, "modelo.joblib")
    print("modelo.joblib generado correctamente.")

    p, contrib = modelo.predict_proba(hombre=1, casados=1, separados=0,
                                       ingresos=60, gastos=25, deuda=33)
    print(f"Caso de control (60 ingreso, 25/33 ratios) -> {p*100:.1f}%")
