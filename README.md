---
title: Mineral Classification Demo
emoji: 🪨
colorFrom: blue
colorTo: green
sdk: streamlit
python_version: "3.10"
app_file: app.py
pinned: false
---

# Mineral Classification Demo

Deep learning-based mineral classification system for identifying common minerals from image data.

Currently, the demo version can accurately classify 7 mineral types: biotite, malachite, muscovite, pyrite, quartz, bornite, and chrysocolla.

# Mineral Klasifikasi

Aplikasi Streamlit untuk klasifikasi gambar mineral menggunakan model Ultralytics YOLO classification.

## Fitur

- Upload gambar `jpg`, `jpeg`, atau `png`
- Prediksi kelas mineral
- Visualisasi confidence score per kelas
- Struktur project siap untuk linting, testing, dan CI

## Struktur Project

```text
.
├── app.py
├── configs/app.yaml
├── models/.gitkeep
├── src/mineral_classifier/
│   ├── config.py
│   ├── inference.py
│   └── ui.py
├── tests/test_inference.py
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── .github/workflows/ci.yml
```

## Setup Lokal

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements-dev.txt
```

Simpan model hasil training sebagai:

```text
models/best.pt
```

Atau ubah path model di `configs/app.yaml`.

## Menjalankan Aplikasi

```bash
streamlit run app.py
```

## Quality Check

```bash
ruff check .
pytest
```

## Catatan ML Engineering

- File model besar seperti `best.pt` sebaiknya tidak disimpan langsung di Git.
- Simpan dataset dan model artifact di storage terpisah, misalnya DVC, MLflow, Google Drive, S3, atau Hugging Face Hub.
- Tambahkan laporan training berisi sumber dataset, pembagian train/validation/test, metrik evaluasi, confusion matrix, dan versi model.
