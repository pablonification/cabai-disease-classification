# Debrief Update Progress CabAI

Dokumen ini dipakai untuk briefing internal sebelum update progress dengan asisten II4012. Tujuannya agar semua anggota kelompok bisa menjawab pertanyaan dengan narasi yang konsisten.

## Ringkasan Proyek

**CabAI** adalah prototype sistem klasifikasi awal penyakit daun cabai berbasis computer vision. Sistem menerima gambar daun cabai, memprediksi kelas penyakit, menampilkan confidence score, lalu memberi rekomendasi tindakan awal.

Posisi solusi:

- Prototype ini membantu **identifikasi awal**, bukan diagnosis pasti.
- Prototype ini tidak menggantikan ahli pertanian.
- Fokus progress saat ini adalah membuktikan pipeline data, model, evaluasi, dan demo awal sudah berjalan.

## Jawaban Cepat Jika Ditanya Asisten

### Model AI yang Dipakai

Model utama awal yang dipakai adalah **EfficientNet-B0** dengan pendekatan **transfer learning**.

Detail:

- Framework: PyTorch
- Model: EfficientNet-B0
- Pretrained weights: ImageNet
- Input size: 224 x 224
- Loss: CrossEntropyLoss
- Optimizer: AdamW
- Learning rate: 1e-3
- Epoch progress demo: 5
- Batch size: 16
- Strategi: backbone di-freeze, classifier/head dilatih untuk 5 kelas cabai

Alasan memilih EfficientNet-B0:

- Ringan dan realistis untuk compute gratis/laptop.
- Cocok untuk dataset kecil-menengah.
- Transfer learning lebih masuk akal dibanding training from scratch karena data publik cabai masih terbatas.

### Dataset yang Dipakai

Untuk progress demo saat ini, dataset yang sudah dipakai adalah **Kaggle `penyakit-cabai`**.

Detail dataset progress:

- Total gambar: 486
- Train: 386
- Validation: 50
- Test: 50
- Kelas: `healthy`, `leaf curl`, `leaf spot`, `whitefly`, `yellowish`

Rencana final:

- Dataset utama training final: Roboflow `Chili leaves disease classification`
- Dataset Kaggle akan diposisikan sebagai **external benchmark**

Catatan penting:

Saat progress demo, Kaggle dipakai dulu sebagai fallback karena datasetnya sudah siap, kecil, dan labelnya sesuai dengan lima kelas final. Roboflow tetap menjadi rencana dataset utama untuk final.

### Progress Sekarang Berapa Persen?

Untuk demo progress, status proyek bisa disebut sekitar **50%**.

Yang sudah selesai:

- Struktur project dan repo GitHub
- Dataset pipeline
- Audit dataset Kaggle
- Training awal EfficientNet-B0
- Evaluasi awal
- Confusion matrix
- Training curve
- Checkpoint model awal
- Prototype inference di notebook
- Recommendation layer berbasis knowledge base

Yang belum selesai:

- Training utama dengan dataset Roboflow
- Baseline pembanding ResNet50/MobileNetV3
- External benchmark final yang benar-benar dipisah dari training
- Grad-CAM
- Deployment ke Streamlit/Hugging Face Spaces
- Laporan final berbasis CRISP-DM

### Hasil Evaluasi Awal

Hasil training awal pada test set Kaggle:

- Accuracy: **90.00%**
- Macro F1-score: **90.09%**

Per kelas:

| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| healthy | 0.7692 | 1.0000 | 0.8696 | 10 |
| leaf curl | 0.8182 | 0.9000 | 0.8571 | 10 |
| leaf spot | 1.0000 | 1.0000 | 1.0000 | 10 |
| whitefly | 1.0000 | 0.8000 | 0.8889 | 10 |
| yellowish | 1.0000 | 0.8000 | 0.8889 | 10 |

Catatan:

Angka ini adalah hasil awal untuk progress demo, bukan klaim final. Dataset Kaggle kecil, sehingga hasil perlu divalidasi lagi pada dataset utama dan benchmark eksternal.

## Fungsi Tiap Notebook

### 1. `01_data_audit.ipynb`

Notebook ini dipakai untuk **Data Understanding** dan audit dataset.

Yang dilakukan:

- Membaca dataset dari folder lokal.
- Mengecek jumlah gambar.
- Mengecek split train, validation, dan test.
- Mengecek distribusi label.
- Menampilkan contoh gambar tiap kelas.
- Memastikan label dataset sesuai scope lima kelas.

Output penting:

- Tabel distribusi data.
- Sample grid gambar per kelas.
- Manifest dataset.

Hal yang perlu dijelaskan:

Notebook ini membuktikan bahwa dataset sudah terbaca dengan benar dan struktur datanya siap dipakai untuk training.

Kalimat siap pakai:

> Notebook pertama kami gunakan untuk audit dataset. Di sini kami memastikan struktur folder, jumlah gambar, split data, dan label sudah sesuai dengan scope proyek, yaitu lima kelas penyakit cabai.

### 2. `02_train_baseline.ipynb`

Notebook ini dipakai untuk **training model awal**.

Yang dilakukan:

- Membaca manifest dataset.
- Membuat dataloader.
- Membuat model EfficientNet-B0.
- Melakukan transfer learning.
- Melatih classifier/head selama 5 epoch.
- Menyimpan checkpoint model.
- Membuat training curve.
- Mengevaluasi model pada test set.
- Membuat confusion matrix.
- Menyimpan prediction preview.

Konfigurasi penting:

- Model: EfficientNet-B0
- Input size: 224 x 224
- Epoch: 5
- Batch size: 16
- Optimizer: AdamW
- Learning rate: 1e-3
- Loss: CrossEntropyLoss

Output penting:

- `outputs/checkpoints/efficientnet_b0_demo.pt`
- `outputs/figures/training_curves.png`
- `outputs/figures/confusion_matrix_internal.png`
- `outputs/reports/metrics_internal_demo.json`
- `outputs/reports/prediction_preview_internal.csv`

Hal yang perlu dijelaskan:

Notebook ini belum dimaksudkan sebagai eksperimen final. Tujuannya adalah membuktikan bahwa pipeline training dan evaluasi sudah berjalan.

Kalimat siap pakai:

> Notebook kedua berisi training awal model. Kami menggunakan EfficientNet-B0 dengan transfer learning karena dataset publik cabai ukurannya masih moderat. Untuk progress ini, model sudah bisa dilatih dan menghasilkan accuracy 90% serta macro F1 sekitar 90% pada test set Kaggle.

### 3. `03_demo_progress.ipynb`

Notebook ini adalah notebook utama untuk **demo progress**.

Yang dilakukan:

- Menjelaskan problem statement.
- Menjelaskan dataset dan scope label.
- Menjelaskan preprocessing.
- Menjelaskan arsitektur model.
- Menampilkan hasil training awal.
- Menampilkan confusion matrix.
- Menjalankan simulasi inference.
- Menampilkan prediksi kelas.
- Menampilkan confidence score.
- Menampilkan rekomendasi tindakan awal.
- Menjelaskan kendala dan rencana final.

Output penting:

- Alur demo end-to-end.
- Prediksi dari checkpoint model awal.
- Recommendation layer untuk hasil prediksi.

Hal yang perlu dijelaskan:

Notebook ini dirancang untuk ditunjukkan ke asisten karena alurnya sudah menyerupai presentasi: masalah, data, model, hasil, demo, kendala, dan rencana final.

Kalimat siap pakai:

> Notebook ketiga adalah notebook demo progress. Di sini kami menunjukkan alur dari problem bisnis, dataset, model, evaluasi, sampai prototype inference. Output prototype berupa prediksi kelas, confidence, dan rekomendasi tindakan awal untuk pengguna.

## Fitur yang Sudah Ada

Fitur progress saat ini:

- Klasifikasi gambar daun cabai ke 5 kelas.
- Confidence score.
- Training curve.
- Confusion matrix.
- Evaluasi accuracy, precision, recall, dan macro F1.
- Recommendation layer berbasis knowledge base.
- Notebook demo end-to-end.

Fitur final yang masih direncanakan:

- Grad-CAM untuk explainability visual.
- Baseline pembanding.
- Training dataset Roboflow.
- External benchmark Kaggle.
- Deployment web berbasis Streamlit atau Hugging Face Spaces.

## Pertanyaan yang Mungkin Muncul

### Kenapa menggunakan EfficientNet-B0?

Karena EfficientNet-B0 ringan, cocok untuk dataset kecil-menengah, dan efisien untuk compute gratis. Transfer learning juga lebih realistis dibanding training from scratch.

### Kenapa dataset progress pakai Kaggle, bukan Roboflow?

Kaggle dipakai dulu untuk progress karena datasetnya sudah siap dan labelnya sesuai lima kelas final. Roboflow tetap direncanakan sebagai dataset training utama final karena ukurannya lebih besar.

### Apakah hasil 90% sudah final?

Belum. Hasil 90% adalah hasil awal pada test set Kaggle. Karena datasetnya kecil, hasil ini perlu divalidasi lagi dengan strategi final: Roboflow sebagai training utama dan Kaggle sebagai external benchmark.

### Apa komponen AIaaS-nya?

Untuk final, AIaaS akan berupa deployment demo ke Hugging Face Spaces atau Streamlit Community Cloud. Progress saat ini masih notebook demo lokal.

### Apa komponen AI bebasnya?

Saat ini sudah ada recommendation layer berbasis knowledge base. Untuk final, akan ditambahkan Grad-CAM sebagai explainability visual.

### Apa risiko terbesar saat ini?

Risiko utama:

- Dataset kecil dan heterogen.
- Potensi overfitting.
- Perbedaan distribusi antara dataset Kaggle dan Roboflow.
- Compute gratis terbatas.

Mitigasi:

- Pakai transfer learning.
- Gunakan benchmark eksternal.
- Tambahkan baseline model.
- Bahas limitation secara jujur.

## Update Progress 1 Menit

Gunakan versi ini jika diminta menjelaskan cepat:

> Progress kami saat ini sudah sampai pipeline end-to-end. Dataset Kaggle penyakit cabai sudah diaudit, total 486 gambar dengan lima kelas. Kami sudah training awal EfficientNet-B0 menggunakan transfer learning di PyTorch selama 5 epoch. Hasil awal pada test set adalah accuracy 90% dan macro F1 90.09%. Kami juga sudah punya notebook demo yang menampilkan prediksi gambar, confidence score, dan rekomendasi tindakan awal. Untuk final, kami akan menggunakan Roboflow sebagai dataset training utama, menjadikan Kaggle sebagai external benchmark, menambahkan baseline pembanding, Grad-CAM, dan deployment web.

## Update Progress 3 Menit

Gunakan versi ini jika asisten meminta lebih detail:

> CabAI adalah prototype sistem klasifikasi awal penyakit daun cabai berbasis computer vision. Masalah yang kami angkat adalah pengguna non-ahli sulit membedakan gejala penyakit cabai secara cepat. Untuk progress saat ini, kami sudah membuat pipeline end-to-end dari data sampai demo inference.
>
> Dataset yang sudah kami gunakan untuk progress adalah Kaggle `penyakit-cabai`, total 486 gambar dengan split 386 train, 50 validation, dan 50 test. Kelasnya ada lima: healthy, leaf curl, leaf spot, whitefly, dan yellowish. Dataset ini kami pakai sebagai fallback progress karena sudah siap dan labelnya sesuai scope final. Untuk final, rencana kami tetap menggunakan Roboflow sebagai dataset training utama, sedangkan Kaggle akan dipakai sebagai external benchmark.
>
> Model awal yang kami pakai adalah EfficientNet-B0 dengan transfer learning dari ImageNet. Kami freeze backbone dan melatih classifier head selama 5 epoch menggunakan AdamW. Hasil awal pada test set Kaggle adalah accuracy 90% dan macro F1 90.09%. Namun, kami belum menganggap ini sebagai hasil final karena dataset masih kecil dan perlu validasi lintas sumber.
>
> Dari sisi artefak, kami punya tiga notebook. Notebook pertama untuk audit dataset, notebook kedua untuk training dan evaluasi, dan notebook ketiga untuk demo progress. Di notebook demo, sistem sudah bisa menampilkan prediksi kelas, confidence score, dan rekomendasi tindakan awal. Berikutnya kami akan melanjutkan training dengan dataset Roboflow, menambahkan baseline pembanding, Grad-CAM, dan deployment web.

## Hal yang Harus Dihindari Saat Menjawab

Jangan mengatakan:

- "Model sudah final."
- "Sistem sudah bisa diagnosis pasti."
- "Akurasi 90% berarti siap dipakai di lapangan."
- "Model menggantikan ahli pertanian."

Gunakan kalimat aman:

- "Prototype untuk identifikasi awal."
- "Hasil awal pada dataset kecil."
- "Masih perlu validasi eksternal."
- "Akan ditingkatkan pada final project."
