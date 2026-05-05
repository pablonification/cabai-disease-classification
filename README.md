# CabAI - Sprint Demo Progress

CabAI adalah prototype tugas besar II4012 untuk klasifikasi awal penyakit daun cabai berbasis computer vision. Sprint ini menyiapkan artefak demo progress Jumat, 8 Mei 2026: pipeline dataset, training awal, evaluasi, dan simulasi inference melalui notebook.

## Target Demo Jumat

- Menunjukkan sumber data dan scope 5 kelas: `healthy`, `leaf curl`, `leaf spot`, `whitefly`, `yellowish`.
- Menjalankan training awal model EfficientNet-B0 berbasis transfer learning.
- Menampilkan metrik awal: accuracy, macro F1, confusion matrix, serta kurva loss/accuracy.
- Menampilkan prototype inference: gambar daun cabai -> prediksi -> confidence -> rekomendasi tindakan awal.
- Menjelaskan kendala dan rencana final dengan bahasa aman: prototype ini membantu identifikasi awal, bukan diagnosis pasti.

## Struktur

```text
Tubes/
  data/
    raw/          # dataset asli hasil download
    interim/      # manifest dan hasil audit sementara
    processed/    # data olahan bila diperlukan
  notebooks/
    01_data_audit.ipynb
    02_train_baseline.ipynb
    03_demo_progress.ipynb
  src/cabai/
    config.py
    data.py
    evaluate.py
    model.py
    recommend.py
    train.py
  outputs/
    checkpoints/  # model .pt
    figures/      # confusion matrix, curves, sample grid
    reports/      # manifest/report/screenshot fallback
```

## Dataset

Rencana utama:

1. Roboflow `Chili leaves disease classification` sebagai dataset training utama.
2. Kaggle `penyakit-cabai` sebagai external benchmark yang tidak ikut training.

Letakkan dataset dalam bentuk folder-based image classification, misalnya:

```text
Tubes/data/raw/roboflow_chili/
  train/
    healthy/
    leaf curl/
    leaf spot/
    whitefly/
    yellowish/
  valid/
    ...
  test/
    ...

Tubes/data/raw/kaggle_penyakit_cabai/
  train/
  validation/
  test/
```

Jika Roboflow belum siap, gunakan Kaggle sebagai fallback training awal untuk demo progress. Catat keterbatasannya di notebook.

## Menjalankan di Colab/Kaggle

```bash
pip install -r requirements.txt
```

Urutan notebook:

1. `notebooks/01_data_audit.ipynb`
2. `notebooks/02_train_baseline.ipynb`
3. `notebooks/03_demo_progress.ipynb`

Notebook ketiga adalah notebook utama untuk demo Jumat.

## Catatan Klaim

Gunakan klaim berikut saat demo:

- "Membantu identifikasi awal penyakit cabai."
- "Prototype/proof of concept berbasis dataset publik."
- "Model masih perlu evaluasi eksternal dan validasi lebih lanjut."

Hindari klaim:

- "Diagnosis pasti."
- "Menggantikan ahli pertanian."
- "Siap produksi untuk semua kondisi lahan."
