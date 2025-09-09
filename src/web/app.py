from typing import Union

from fastapi import FastAPI, HTTPException


from web.predict_router import predict_router




app = FastAPI()

app.include_router(predict_router)
