import io as std_io

import streamlit as st
import numpy as np
from skimage import io
from PIL import Image

uploaded_img = st.sidebar.file_uploader('Загрузи изображение в ч/б', type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    try:
        image = io.imread(uploaded_img)
        if len(image.shape) ==3:
            image= image[:,:, 0]
        image_show = Image.open(uploaded_img)
        st.image(image_show, caption='Загруженное изображение', use_column_width=True)
        top_k = st.sidebar.number_input("Введи количество сингулярных чисел:", min_value=0, value=10, step=1)

        U, sing_values, V = np.linalg.svd(image)

        sigma = np.zeros(shape=image.shape)
        np.fill_diagonal(sigma, sing_values)

        trunc_U = U[:, :top_k]
        trunc_sigma = sigma[:top_k, :top_k]
        trunc_V = V[:top_k, :]
        image_svd = trunc_U @ trunc_sigma @ trunc_V
        img_download = Image.fromarray(image_svd)
        img_download = img_download.convert('RGB')
        st.image(img_download, caption=f'{top_k} сингулярных чисел', use_column_width=True)

        img_buffer = std_io.BytesIO()
        img_download.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        st.download_button(
            label="Скачать сжатое изображение",
            data=img_buffer,
            file_name='img_svd.png',
            mime="image/png",
        )
    except Exception as e:
            st.error(f"Не удалось открыть изображение. Ошибка: {e}")
