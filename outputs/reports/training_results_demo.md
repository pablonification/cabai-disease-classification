# Hasil Training Awal Demo Progress

Eksperimen ini dijalankan pada Selasa, 5 Mei 2026 menggunakan dataset Kaggle `penyakit-cabai` sebagai fallback training awal untuk demo progress.

## Dataset

- Total gambar: 486
- Train: 386
- Validation: 50
- Test: 50
- Kelas: `healthy`, `leaf curl`, `leaf spot`, `whitefly`, `yellowish`

Distribusi test seimbang: 10 gambar per kelas.

## Konfigurasi Model

- Model: EfficientNet-B0
- Pretrained weights: ImageNet
- Input size: 224 x 224
- Optimizer: AdamW
- Learning rate: 1e-3
- Epoch: 5
- Batch size: 16
- Strategi: freeze backbone, train classifier/head
- Device lokal: MPS

## Hasil Training

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|---:|---:|---:|---:|---:|
| 1 | 1.3825 | 0.4689 | 1.1594 | 0.6400 |
| 2 | 0.9837 | 0.6736 | 0.8131 | 0.7800 |
| 3 | 0.6543 | 0.8109 | 0.6317 | 0.7200 |
| 4 | 0.5357 | 0.8005 | 0.5960 | 0.7200 |
| 5 | 0.4680 | 0.8420 | 0.4761 | 0.8200 |

## Hasil Evaluasi Test

- Accuracy: 0.9000
- Macro F1-score: 0.9009
- Macro precision: 0.9175
- Macro recall: 0.9000

Per kelas:

| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| healthy | 0.7692 | 1.0000 | 0.8696 | 10 |
| leaf curl | 0.8182 | 0.9000 | 0.8571 | 10 |
| leaf spot | 1.0000 | 1.0000 | 1.0000 | 10 |
| whitefly | 1.0000 | 0.8000 | 0.8889 | 10 |
| yellowish | 1.0000 | 0.8000 | 0.8889 | 10 |

## Artefak Lokal

File berikut tersedia lokal tetapi tidak dipush karena berisi output/generated artifacts:

- `outputs/checkpoints/efficientnet_b0_demo.pt`
- `outputs/figures/training_curves.png`
- `outputs/figures/confusion_matrix_internal.png`
- `outputs/figures/sample_grid.png`
- `outputs/reports/metrics_internal_demo.json`
- `outputs/reports/prediction_preview_internal.csv`

## Catatan Demo

Hasil ini adalah bukti progress awal, bukan klaim final. Untuk final project, rencana tetap mengikuti master brief: Roboflow sebagai dataset training utama, Kaggle sebagai external benchmark, baseline pembanding, Grad-CAM, dan deployment web.
