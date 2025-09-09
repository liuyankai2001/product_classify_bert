from typing import Union

import uvicorn
from fastapi import FastAPI, HTTPException


from web.predict_router import predict_router




app = FastAPI()

app.include_router(predict_router)

def run_app():
    uvicorn.run("web.app:app",port=8000)
