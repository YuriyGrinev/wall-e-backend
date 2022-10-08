from fastapi import FastAPI, Request, File, Form, UploadFile

import cv2
import numpy as np

import base64

app = FastAPI()


@app.get("/")
async def home():
    return {"massage": "None"}


@app.post("/bitwise")
async def bitwise(
    request: Request,
    file_list: list[UploadFile] = File(...),
):
    img_batch = [
        cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
        for file in file_list
    ]
    img_batch_bitwise = [cv2.bitwise_not(img) for img in img_batch]
    return encode_json(img_batch_bitwise)


@app.post("/colorize")
async def colorize(
    request: Request,
    file_list: list[UploadFile] = File(...),
):
    img_batch = get_img_batch(file_list)
    img_batch_colorize = [img for img in img_batch]
    return encode_json(img_batch_colorize)


@app.post("/upscale")
async def upscale(
    request: Request,
    file_list: list[UploadFile] = File(...),
    scale_percent: int = Form(200),
):
    scale_percent = scale_percent
    img_batch = get_img_batch(file_list)
    img_batch_upscale = [upscale_image(img, scale_percent) for img in img_batch]
    return encode_json(img_batch_upscale)


def base64EncodeImage(img):
    _, im_arr = cv2.imencode(".jpg", img)
    im_b64 = base64.b64encode(im_arr.tobytes()).decode("utf-8")
    return im_b64


def encode_json(img_batch):
    json_results = [{"image_base64": base64EncodeImage(img)} for img in img_batch]
    return str(json_results).replace("'", r'"')


def get_img_batch(file_list):
    return [
        cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
        for file in file_list
    ]


def upscale_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


if __name__ == "__main__":
    import uvicorn

    app_str = "server_minimal:app"
    uvicorn.run(app_str, host="0.0.0.0", port=8000, reload=True, workers=1)
