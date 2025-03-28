# Pasos para instalar API (Windows):
## 1. Instalar [Python 3.10.11](https://www.python.org/downloads/release/python-31011/)
## 2. Clonar repositorio de la API
```
git clone https://github.com/Ramiro-DG/super_ai_api
```
## 3. Crear entorno virtual para instalar dependencias
```
cd super_ai_api
python -m venv venv
```
## 4. Activar entorno virtual
```
venv/Scripts/activate
```
## 5. Instalar dependencias (puede ser lento)
```
pip install -r .\requirements.txt
```
## 6. Iniciar API
```
fastapi run .\main.py
```
> ### Iniciar API en modo dev
> Para realizar modificaciones en el código también se puede correr en modo dev, que automáticamente detectará los cambios en el código y volverá a cargar la API.
> ```
> fastapi dev .\main.py
> ```
