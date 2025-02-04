import streamlit as st
import requests
from PIL import Image
import io

st.title("Распознавание машин и их цветов")

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
##
if uploaded_file is not None:
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("http://127.0.0.1:8000/detect_cars/", files=files)

    if response.status_code == 200:
        dominant_colors = response.json()["dominant_colors"]
        st.subheader("Доминирующие цвета машин:")
        for color in dominant_colors:
            st.write(f"Цвет: {color}")

        image_response = requests.get("http://127.0.0.1:8000/get_image/")
        if image_response.status_code == 200:
            image = Image.open(io.BytesIO(image_response.content))
            st.image(image, caption="Обработанное изображение", use_column_width=True)
        else:
            st.error("Не удалось загрузить обработанное изображение.")
    else:
        st.error("Ошибка при обработке изображения.")