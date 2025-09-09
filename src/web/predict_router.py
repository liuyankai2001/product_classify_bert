from fastapi import APIRouter, HTTPException

from web.schemes import PredictRequest, PredictResponse
from web.service import predict_titile

predict_router = APIRouter()



@predict_router.post('/predict')
def predict(req: PredictRequest) -> PredictResponse:
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=401,detail='title不能为空')
    label = predict_titile(title)
    return PredictResponse(title=title,label=label)