from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import cv2
import os
from car_detector import CarDetector

app = FastAPI()
detector = CarDetector()


@app.post("/detect_cars/")
async def detect_cars(file: UploadFile = File(...)):
    image_path = "temp_image.jpg"
    with open(image_path, "wb") as buffer:
        buffer.write(await file.read())

    output_image, dominant_colors = detector.detect_cars(image_path)

    output_image_path = "output_image.jpg"
    cv2.imwrite(output_image_path, output_image)

    # Возвращаем результат
    return JSONResponse(content={"dominant_colors": dominant_colors})


@app.get("/get_image/")
async def get_image():
    output_image_path = "output_image.jpg"
    if os.path.exists(output_image_path):
        return FileResponse(output_image_path)
    else:
        return JSONResponse(content={"error": "Image not found"}, status_code=404)