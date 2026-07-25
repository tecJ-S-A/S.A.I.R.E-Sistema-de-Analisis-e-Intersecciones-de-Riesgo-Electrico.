import os
import json

RUTA_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "postes_test", "casos_referencia.json")

def _cargar_banco():
    if not os.path.exists(RUTA_JSON): return {}
    try:
        with open(RUTA_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

_banco = _cargar_banco()

def buscar_caso_verificado(nombre_archivo):
    if not nombre_archivo: return None
    return _banco.get(nombre_archivo)

def texto_comparacion(caso_verificado, poste):
    msg = "== COMPARACIÓN CON CASO VERIFICADO ==\n\n"
    msg += f"Rango Cables Esperado: {caso_verificado['cables_min']} a {caso_verificado['cables_max']}\n"
    msg += f"Rango IA Obtenido: {poste['cables_min']} a {poste['cables_max']}\n\n"
    
    p_ia = "Sí" if poste["riesgo_contacto_persona"] else "No"
    p_real = "Sí" if caso_verificado["persona"] else "No"
    msg += f"Riesgo Persona: Esperado={p_real} | IA={p_ia}\n"
    
    n_ia = poste["num_nudos_aprox"]
    n_real = caso_verificado["nudos_reales"]
    msg += f"Nudos detectados: Esperado={n_real} | IA={n_ia}\n"
    
    if "nota" in caso_verificado:
        msg += f"\nNota de referencia: {caso_verificado['nota']}"
        
    return msg