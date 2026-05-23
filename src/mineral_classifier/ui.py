from __future__ import annotations

import tempfile
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from mineral_classifier.config import load_config
from mineral_classifier.inference import load_model, predict, preprocess_image


@st.cache_resource(show_spinner=False)
def get_model(model_path: str):
    return load_model(model_path)


def build_confidence_chart(class_names: list[str], probabilities: list[float]) -> go.Figure:
    chart = go.Figure([go.Bar(x=class_names, y=probabilities)])
    chart.update_layout(
        title="Tingkat Keyakinan Prediksi",
        xaxis_title="Mineral",
        yaxis_title="Keyakinan",
        yaxis=dict(range=[0, 1]),
    )
    return chart


def main() -> None:
    config = load_config()
    st.set_page_config(page_title=config.title)

    st.title("Program Pengenalan Gambar Mineral")
    st.caption("Upload gambar mineral untuk mendapatkan prediksi kelas dan confidence score.")

    uploaded_file = st.file_uploader("Upload gambar mineral", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.info("Silakan upload gambar mineral terlebih dahulu.")
        return

    image = preprocess_image(Image.open(uploaded_file), config.image_size)
    st.image(image, caption="Gambar yang diupload")

    if st.button("Deteksi Gambar", type="primary"):
        with st.spinner("Sedang diproses..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / "input.jpg"
                image.save(temp_path)

                try:
                    model = get_model(str(config.model_path))
                    result = predict(model, temp_path)
                except Exception as exc:
                    st.error(f"Prediksi gagal: {exc}")
                    return

        st.success(
            f"Mineral terdeteksi: **{result.predicted_class}** "
            f"({result.confidence:.2%} confidence)"
        )
        st.plotly_chart(
            build_confidence_chart(result.class_names, result.probabilities),
            use_container_width=True,
        )

    st.caption("Program aplikasi deteksi batuan © 2025")
