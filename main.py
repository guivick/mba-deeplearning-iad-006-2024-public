from fastapi import FastAPI, File
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pickle as pk
import warnings

import base64
from PIL import Image
import io

warnings.simplefilter(action='ignore', category=DeprecationWarning)

app = FastAPI()

# Definição dos tipos de dados
class PredictionResponse(BaseModel):
    prediction: float

class ImageRequest(BaseModel):
    image: str

# Carregamento do modelo de machine learning
def load_model():
    global xgb_model_carregado
    with open('./notebooks/modelo.pkl', 'rb') as f:
        xgb_model_carregado = pk.load(f)

# Inicialização da aplicação:
@app.on_event('startup')
async def startup_event():
    load_model()

# Definição do endpoint /predict que aceita as requisições via POST
# Esse endpoint que irá receber a imagem em base64 e irá convertê-la para fazer inferência
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageRequest):
    print('predição')
    # Processamento da Imagem
    img_bytes = base64.b64decode(request.image)
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((8,8))
    img_array = np.array(img)
    print('imagem processada!')
    # Converter a imagem para escala de cinza
    img_array = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])

    img_array = img_array.reshape(1, -1)
    print('imagem convertida para escala de cinza!')

    # Predição do Modelo de Machine Learning
    prediction = xgb_model_carregado.predict(img_array)
    print('realizada a predição')
    print(prediction)
    return {"prediction": prediction}

# Endpoint de Healthcheck
@app.get("/healthcheck")
async def healthcheck():
    # retorna um objeto com um campo status com valor "ok" se a aplicação estiver funcionando corretamente
    return {"status": "ok"}
