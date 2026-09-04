import os

_base = r"H:\DPINV\CARPETAS PERSONALES\DIEGO\2. Python\Paquetes python\entornosistema"
_dll_dirs = [
    os.path.join(_base, "Library", "bin"),
    os.path.join(_base, "Library", "mingw-w64", "bin"),
    os.path.join(_base, "Library", "usr", "bin"),
]
for _d in _dll_dirs:
    if os.path.isdir(_d):
        try:
            os.add_dll_directory(_d)
        except Exception:
            pass
