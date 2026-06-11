# -*- coding: utf-8 -*-
"""
Lanzador unificado del pipeline Walk-Forward CV.
Selecciona la version y ejecuta entrenamiento + fan charts + exportacion CSV.

Uso:
    python step005_iteracion.py
    runfile('...step005_iteracion.py', wdir='...')
"""

###############################################################################
# SELECTOR — unico parametro a cambiar
###############################################################################

#  Version  | "v3" → sin normalizacion de target (pipeline anterior)
#            | "v4" → con normalizacion de target (recomendado)
VERSION = "v4"

###############################################################################

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

if VERSION == "v4":
    pipeline = importlib.import_module("step005_walk_forward_cv_4")
elif VERSION == "v3":
    pipeline = importlib.import_module("step005_walk_forward_cv_3")
else:
    raise ValueError(f"VERSION no reconocida: {VERSION!r}. Usa 'v3' o 'v4'.")

pipeline.main()
