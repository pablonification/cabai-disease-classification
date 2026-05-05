# Script Demo Progress CabAI

## Pembuka

CabAI adalah prototype untuk membantu identifikasi awal penyakit daun cabai dari gambar. Masalah yang ingin diselesaikan adalah sulitnya pengguna non-ahli membedakan gejala penyakit secara cepat, terutama pada tahap awal. Sistem ini bukan pengganti ahli pertanian, tetapi alat bantu screening awal.

## Data

Scope label kami dibatasi ke lima kelas: sehat, keriting daun, bercak daun, whitefly/kutu kebul, dan daun menguning. Dataset utama yang ditargetkan adalah Roboflow karena ukurannya lebih besar dan task-nya image classification. Dataset Kaggle dipisahkan sebagai external benchmark agar evaluasi final lebih jujur.

## Pipeline

Pipeline yang sudah disiapkan adalah audit dataset, standardisasi label, preprocessing gambar ke 224x224, training awal EfficientNet-B0, evaluasi metrik, dan simulasi rekomendasi tindakan awal.

## Model

Model awal menggunakan EfficientNet-B0 dengan transfer learning. Pendekatan ini dipilih karena dataset publik cabai ukurannya moderat, sehingga training from scratch terlalu berisiko dan terlalu mahal secara compute.

## Evaluasi Awal

Untuk progress demo, metrik yang ditampilkan adalah accuracy, macro F1, confusion matrix, serta kurva loss dan accuracy. Angka ini belum diklaim sebagai performa final karena eksperimen tuning dan external benchmark masih berjalan.

## Prototype

Prototype menerima gambar daun cabai, menghasilkan label prediksi dan confidence, lalu menampilkan rekomendasi tindakan awal berdasarkan kelas prediksi. Pada final project, bagian ini akan dilengkapi Grad-CAM agar pengguna dapat melihat area gambar yang memengaruhi prediksi model.

## Kendala dan Rencana

Kendala utama saat ini adalah waktu sprint pendek, dataset publik yang heterogen, dan keterbatasan compute gratis. Rencana berikutnya adalah menyelesaikan training utama Roboflow, menjalankan benchmark Kaggle, membandingkan baseline, menambahkan Grad-CAM, dan deploy demo ke Streamlit atau Hugging Face Spaces.
