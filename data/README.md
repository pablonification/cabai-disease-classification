# Data CabAI

Folder ini menyimpan dataset lokal untuk sprint demo progress.

## Dataset Utama

Roboflow `Chili leaves disease classification`:

- Link: https://universe.roboflow.com/chili-leaves-disease-classification/chili-v2
- Peran: training/validation/test internal utama.
- Scope: gunakan 5 kelas bersama dan keluarkan `powdery mildew`.
- Folder target: `data/raw/roboflow_chili/`

Contoh struktur:

```text
data/raw/roboflow_chili/
  train/
    healthy/
    leaf curl/
    leaf spot/
    whitefly/
    yellowish/
  valid/
  test/
```

## External Benchmark

Kaggle `penyakit-cabai`:

- Link: https://www.kaggle.com/datasets/bryanrizqiprakosa/penyakit-cabai
- Peran: external benchmark, tidak ikut training utama.
- Folder target: `data/raw/kaggle_penyakit_cabai/`

Contoh struktur:

```text
data/raw/kaggle_penyakit_cabai/
  train/
  validation/
  test/
```

## Fallback Demo Progress

Jika Roboflow belum bisa diunduh sebelum Jumat, pakai Kaggle dulu untuk training awal. Saat demo, jelaskan bahwa:

- Roboflow tetap dataset utama final.
- Kaggle sementara dipakai untuk membuktikan pipeline training.
- External benchmark akan dijalankan setelah dataset utama lengkap.
