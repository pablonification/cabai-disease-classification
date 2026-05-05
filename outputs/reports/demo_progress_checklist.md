# Checklist Demo Progress CabAI

Tanggal demo progress: Jumat, 8 Mei 2026.

## Artefak yang Ditunjukkan

- `notebooks/01_data_audit.ipynb`: audit dataset dan contoh gambar.
- `notebooks/02_train_baseline.ipynb`: training awal EfficientNet-B0.
- `notebooks/03_demo_progress.ipynb`: alur presentasi utama.
- `outputs/figures/sample_grid.png`: contoh gambar per kelas.
- `outputs/figures/training_curves.png`: kurva loss dan accuracy.
- `outputs/figures/confusion_matrix_internal.png`: confusion matrix awal.
- `outputs/checkpoints/efficientnet_b0_demo.pt`: checkpoint model awal.

## Narasi Singkat

CabAI adalah prototype sistem klasifikasi awal penyakit daun cabai. Sistem menerima gambar daun, memprediksi salah satu dari lima kelas, lalu memberi rekomendasi tindakan awal. Prototype ini ditujukan untuk screening awal dan edukasi, bukan pengganti diagnosis ahli pertanian.

## Kendala yang Aman Disampaikan

- Dataset publik masih heterogen dan perlu audit lebih lanjut.
- Compute gratis membatasi lama training dan eksperimen.
- External benchmark Kaggle akan diselesaikan setelah model utama Roboflow stabil.
- Grad-CAM dan deployment cloud masuk rencana final.

## Fallback Demo

Jika runtime notebook bermasalah:

1. Buka `03_demo_progress.ipynb`.
2. Tunjukkan markdown problem statement, dataset scope, dan model.
3. Tunjukkan screenshot/gambar di `outputs/figures/` bila sudah tersedia.
4. Jalankan cell rekomendasi simulasi untuk memperlihatkan output prototype.
