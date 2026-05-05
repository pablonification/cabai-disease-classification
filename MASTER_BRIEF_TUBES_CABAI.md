# Master Brief — Single Source of Truth

Dokumen ini adalah **patokan utama tim** untuk proyek tugas besar II4012 dengan topik **klasifikasi penyakit cabai menggunakan computer vision**. Gunakan dokumen ini sebagai sumber acuan tunggal saat menyusun proposal, presentasi progress, laporan akhir, demo, dan narasi bisnis. Jika ada perubahan keputusan besar, ubah dokumen ini terlebih dahulu agar seluruh tim tetap sinkron.

## 1. Tujuan dokumen

- Menyatukan arah proyek agar semua anggota memakai **narasi, istilah, dan keputusan teknis yang sama**.
- Menjelaskan **apa yang akan dibangun**, **mengapa solusi itu dipilih**, dan **apa batasannya**.
- Menyediakan kerangka yang bisa langsung dipakai untuk:
  - video proposal,
  - demo progress,
  - laporan akhir berbasis CRISP-DM,
  - demo produk.
- Menjadi sumber rujukan saat ada perbedaan pendapat soal scope, model, atau deployment.

## 2. Ringkasan eksekutif

### Judul kerja

**CabAI — Sistem Klasifikasi Penyakit Cabai Berbasis Computer Vision dengan Pendekatan AI Gratis dan Siap Demo**

### Inti ide

Masalah utama yang ingin diselesaikan adalah **sulitnya identifikasi dini penyakit cabai oleh petani atau pengguna non-ahli**. Dampaknya adalah keterlambatan penanganan, penurunan kualitas hasil panen, dan potensi kerugian ekonomi. Solusi yang diusulkan adalah sistem yang menerima foto daun cabai, memprediksi kelas penyakit, lalu menampilkan penjelasan dan rekomendasi tindak lanjut.

### Keputusan utama yang dibekukan

Ini adalah keputusan default tim, kecuali nanti ada alasan kuat untuk merevisi:

- **Semua solusi harus gratis** untuk dibangun, dilatih, dan didemokan.
- **Task utama** adalah **image classification**, bukan object detection atau segmentation.
- **Strategi data resmi**: dataset klasifikasi cabai Roboflow sebagai sumber utama training/validation, sedangkan dataset Kaggle dipakai terpisah sebagai **external benchmark**.
- **Ruang lingkup label final** dibekukan ke **5 kelas bersama**: `healthy`, `leaf curl`, `leaf spot`, `whitefly`, `yellowish`.
- **Model utama**: `EfficientNet-B0` berbasis transfer learning.
- **Baseline pembanding**: `ResNet50` dan/atau `MobileNetV3-Large`.
- **Model modern pembanding opsional**: `MobileViT-XXS` atau `DeiT-Tiny` bila waktu dan compute cukup.
- **Framework utama**: **PyTorch**.
- **Deployment demo utama**: **Hugging Face Spaces** atau **Streamlit Community Cloud**.
- **Solusi AI bebas tambahan**: **sistem rekomendasi berbasis knowledge base + explainability (Grad-CAM)**.

## 3. Constraint tugas besar yang wajib dipenuhi

Berdasarkan `Spesifikasi.md`, proyek harus mencakup:

- **minimal 1 solusi berbasis AIaaS**,
- **minimal 1 solusi model ML yang dilatih sendiri**,
- **minimal 1 solusi AI bebas**.

Selain itu, laporan akhir perlu mengikuti alur **CRISP-DM**:

- Business Understanding
- Data Understanding
- Data Preparation
- Modelling
- Evaluation
- Deployment

### Cara proyek ini memenuhi requirement

#### A. Solusi AIaaS

**Pilihan resmi tim**: **Hugging Face Spaces** sebagai layanan AI hosted gratis untuk demo publik, dengan opsi memakai **Hugging Face Inference** untuk skala demo jika dibutuhkan.

Cara framing yang aman untuk presentasi:

- Model hasil training diunggah ke ekosistem Hugging Face.
- Sistem diakses melalui layanan hosted berbasis cloud.
- Pengguna cukup membuka URL publik tanpa instalasi lokal.
- Dari sisi pengguna, ini berfungsi sebagai **AI service yang dikonsumsi secara online**.

> Catatan penting: jangan mengklaim “enterprise-grade AIaaS berkapasitas besar”. Yang kita klaim adalah **hosted AI service untuk skala edukasi/demo**.

#### B. Self-trained ML model

**Pilihan resmi tim**: fine-tuning model `EfficientNet-B0` dengan **dataset klasifikasi cabai utama yang lebih besar**, lalu mengevaluasinya ulang pada **dataset Kaggle yang dipisah** sebagai external benchmark.

### Klarifikasi penting tentang istilah “dilatih sendiri”

Untuk konteks tugas besar ini, **fine-tuning transfer learning tetap masuk ke kategori model ML yang dilatih sendiri**, selama:

- tim melakukan proses training/fine-tuning sendiri,
- tim memakai dataset proyek sendiri,
- tim menentukan eksperimen, hyperparameter, dan evaluasi sendiri,
- hasil model akhir diperoleh dari proses training ulang tersebut.

Yang **tidak** sebaiknya diklaim sebagai “dilatih sendiri”:

- hanya memakai model pretrained apa adanya tanpa training ulang,
- hanya memanggil API model pihak ketiga tanpa proses pelatihan,
- hanya melakukan inference pada model publik yang tidak disesuaikan dengan dataset proyek.

Framing resmi tim:

> Solusi self-trained kami menggunakan transfer learning melalui fine-tuning model pretrained pada dataset klasifikasi cabai yang kami kurasi sendiri untuk training dan validasi. Meskipun backbone awal berasal dari model pretrained, model final tetap dilatih ulang, dievaluasi, dan disesuaikan sendiri untuk kasus proyek ini.

#### C. Solusi AI bebas

**Pilihan resmi tim**: sistem rekomendasi penanganan penyakit dan explainability berbasis:

- **knowledge base / aturan sederhana**,
- **Grad-CAM** untuk menunjukkan area gambar yang paling berpengaruh,
- opsi tambahan FAQ assistant berbasis dokumen jika waktu cukup.

## 4. Problem statement dan sudut pandang bisnis

### Masalah bisnis

Identifikasi penyakit tanaman cabai di lapangan sering bergantung pada pengalaman manual. Pada kondisi nyata, petani bisa kesulitan membedakan gejala yang mirip, terutama jika:

- gejala baru muncul di tahap awal,
- pencahayaan foto buruk,
- latar belakang ramai,
- daun yang difoto bukan kondisi ideal.

Akibatnya:

- penanganan bisa terlambat,
- tindakan yang diambil bisa tidak tepat,
- hasil panen menurun,
- biaya perawatan meningkat.

### Problem statement formal

**Bagaimana membangun sistem berbasis AI yang mampu mengklasifikasikan penyakit cabai dari citra daun secara cepat, murah, dan mudah diakses untuk membantu pengguna non-ahli melakukan identifikasi awal?**

### Nilai bisnis yang ditawarkan

- Membantu **screening awal** penyakit cabai.
- Mengurangi ketergantungan penuh pada observasi manual.
- Memberi **rekomendasi tindakan awal** yang lebih cepat.
- Menjadi dasar produk agritech sederhana yang bisa dipakai untuk edukasi dan demo bisnis.

### Posisi solusi

Solusi ini **bukan pengganti diagnosis ahli pertanian**. Solusi ini adalah **alat bantu identifikasi awal**.

Kalimat yang aman dipakai:

- “membantu identifikasi awal,”
- “memberikan prediksi kelas berdasarkan citra,”
- “mendukung pengambilan keputusan awal.”

Kalimat yang sebaiknya dihindari:

- “mendiagnosis secara pasti,”
- “menggantikan ahli,”
- “akurasi sempurna di kondisi lapangan.”

## 5. Ruang lingkup dan non-ruang lingkup

### In scope

- Klasifikasi citra daun cabai ke kelas penyakit tertentu.
- Training model klasifikasi berbasis transfer learning.
- Visualisasi evaluasi model.
- Demo aplikasi berbasis web.
- Explainability sederhana dengan Grad-CAM.
- Rekomendasi penanganan awal berbasis kelas prediksi.

### Out of scope

- Deteksi lokasi penyakit pada daun dalam bentuk bounding box.
- Segmentasi area penyakit secara pixel-level.
- Diagnosis multi-penyakit per gambar.
- Sistem agronomi lengkap dengan sensor IoT.
- Uji lapangan besar-besaran di kebun nyata.
- Validasi medis/agronomis formal.

## 6. Strategi data resmi proyek

Bagian ini menggantikan asumsi lama bahwa proyek hanya bertumpu pada satu dataset Kaggle. Keputusan final tim adalah memakai **strategi data bertahap** agar hasil lebih kuat secara akademik, tetapi tetap realistis untuk proyek mahasiswa.

### 6.1 Ringkasan keputusan data

- **Dataset utama untuk training dan validation**: Roboflow **Chili leaves disease classification**.
- **Dataset eksternal untuk benchmark terpisah**: Kaggle `penyakit-cabai` oleh Bryan Rizqi Prakosa.
- **Label final yang dipakai model utama**: `healthy`, `leaf curl`, `leaf spot`, `whitefly`, `yellowish`.
- **Kelas `powdery mildew` tidak masuk scope utama** walaupun muncul di dataset Roboflow.
- **Dataset detection, single-disease, atau pepper umum tidak dicampur ke training utama** kecuali untuk eksperimen tambahan yang jelas dipisahkan dari klaim utama.

### 6.2 Dataset utama — Roboflow classification dataset

- Roboflow Universe: `Chili leaves disease classification`  
  Link: <https://universe.roboflow.com/chili-leaves-disease-classification/chili-leaves-disease-classification>
- Tipe task: **image classification**
- Ukuran yang tertera: **1.1k images**
- Kelas yang tertera: `healthy`, `leaf spot`, `whitefly`, `leaf curl`, `powdery mildew`, `yellowish`
- Lisensi yang tertera: **CC BY 4.0**

### Kenapa dataset ini dijadikan sumber utama

- Lebih besar dari dataset Kaggle yang sebelumnya menjadi kandidat utama.
- Tetap spesifik ke **cabai/chili leaf disease classification**.
- Task-nya langsung sesuai dengan scope proyek, yaitu **classification**, bukan detection.
- Memberi sinyal training yang lebih baik tanpa memaksa tim menggabungkan terlalu banyak sumber yang heterogen.

### 6.3 Dataset benchmark eksternal — Kaggle dataset

- Kaggle: `penyakit-cabai` oleh Bryan Rizqi Prakosa  
  Link: <https://www.kaggle.com/datasets/bryanrizqiprakosa/penyakit-cabai>
- Tipe task: **folder-based image classification**
- Kelas: `healthy`, `leaf curl`, `leaf spot`, `whitefly`, `yellowish`
- Total gambar yang sudah teridentifikasi: **486 JPG**

Rincian split yang tertera:

Train:

- healthy: 80
- leaf curl: 79
- leaf spot: 78
- whitefly: 72
- yellowish: 77

Validation:

- masing-masing 10 gambar per kelas

Test:

- masing-masing 10 gambar per kelas

### Kenapa dataset Kaggle tidak dicampur ke training utama

- Ukurannya relatif kecil.
- Ada risiko **near-duplicate** atau overlap visual dengan dataset publik lain.
- Menyimpannya terpisah membuat evaluasi lebih defensible karena tim bisa menunjukkan performa model pada **sumber data berbeda**.

### Framing resmi tim

> Model utama kami dilatih pada dataset klasifikasi cabai yang lebih besar, lalu diuji ulang pada dataset Kaggle yang disimpan terpisah sebagai external benchmark untuk melihat generalisasi lintas sumber.

### 6.4 Scope label final

Scope label proyek dibekukan ke lima kelas berikut:

- `healthy`
- `leaf curl`
- `leaf spot`
- `whitefly`
- `yellowish`

### Kenapa `powdery mildew` dikeluarkan

- Kelas tersebut ada di dataset Roboflow, tetapi **tidak punya pasangan yang jelas** di dataset benchmark Kaggle.
- Memasukkannya ke scope utama akan membuat narasi training, evaluasi, dan benchmarking menjadi tidak simetris.
- Menjaganya tetap di luar scope membuat eksperimen utama lebih rapi dan lebih mudah dipertanggungjawabkan.

### 6.5 Dataset yang tidak dipakai untuk training utama

#### A. Roboflow object detection dataset

- Link: <https://universe.roboflow.com/chili-leaf-disease-uf14c/projectta-chili-leaf-disease>
- Tipe task: **object detection**
- Ukuran yang tertera: **620 images**

Status dalam proyek:

- **bukan sumber training classifier utama**,
- boleh dipakai hanya jika tim membuat eksperimen detection terpisah sebagai stretch goal.

#### B. Zenodo Cercospora dataset

- Link: <https://zenodo.org/records/13272039>
- Deskripsi publik: **1,738 preprocessed images** untuk **Cercospora leaf spot** pada chili pepper leaves

Status dalam proyek:

- **tidak dicampur** ke multiclass training utama,
- hanya mungkin dipakai untuk eksperimen tambahan yang jelas bersifat **single-disease / binary / robustness check**.

#### C. PlantVillage dan dataset pepper umum lain

Status dalam proyek:

- **bukan dataset final utama**,
- dapat dipakai sebagai konteks literatur, benchmark tambahan, atau eksperimen pendukung,
- tidak menjadi dasar klaim utama karena domain-nya lebih bersih/terkontrol dan tidak sepenuhnya selaras dengan kelas cabai proyek.

### 6.6 Implikasi teknis dari strategi data ini

- Model utama tidak lagi semata-mata diasumsikan bekerja pada dataset yang sangat kecil.
- Namun, data tetap berasal dari **sumber publik yang heterogen**, sehingga risiko label noise dan domain shift masih ada.
- Karena itu, pendekatan terbaik tetap **transfer learning**, bukan training from scratch.
- Evaluasi harus dilaporkan dalam dua lapis:
  - performa pada data train/validation internal,
  - performa pada **external benchmark** Kaggle.

### 6.7 Posisi dataset dalam narasi presentasi

Narasi yang disarankan:

- tim sengaja memilih strategi data bertahap agar tidak terjebak pada dataset kecil tunggal,
- training memakai sumber klasifikasi cabai yang lebih besar,
- evaluasi utama tetap dijaga jujur dengan benchmark lintas sumber,
- proyek menekankan **kelayakan solusi yang defensible**, bukan angka tinggi yang sulit dipercaya.

## 7. Standardisasi label dan istilah

Gunakan penamaan yang konsisten di semua slide, laporan, demo, dan notebook.

| Label dataset | Nama Indonesia yang dipakai tim | Catatan |
|---|---|---|
| healthy | sehat | daun sehat / tanpa gejala utama |
| leaf curl | keriting daun | gejala daun melengkung atau keriting |
| leaf spot | bercak daun | gejala bercak pada daun |
| whitefly | whitefly / kutu kebul | bisa disebut gejala terkait whitefly |
| yellowish | virus kuning / daun menguning | jelaskan bahwa label dataset memakai istilah `yellowish` |
| powdery mildew | tidak masuk scope utama | hanya ada di dataset utama, tidak dipakai dalam benchmark final |

### Aturan penggunaan istilah

- Di tabel teknis, boleh pakai **label asli dataset**.
- Di UI dan presentasi, tampilkan **Bahasa Indonesia**.
- Jika perlu, tulis keduanya: `leaf curl (keriting daun)`.
- Untuk scope utama proyek, **jangan** menambahkan `powdery mildew` ke confusion matrix final atau klaim utama kecuali seluruh strategi benchmark ikut diubah.

## 8. Arsitektur solusi yang disepakati

## 8.1 Gambaran umum

Arsitektur akhir proyek terdiri dari tiga lapis:

1. **Model klasifikasi utama** untuk memprediksi kelas penyakit.
2. **Explainability layer** untuk menunjukkan alasan visual model.
3. **Recommendation layer** untuk memberi tindakan awal berbasis knowledge base.

Alur sederhananya:

`Gambar daun → preprocessing → model klasifikasi → prediksi + confidence → Grad-CAM → rekomendasi tindakan`

## 8.2 Solusi 1 — Self-trained ML model

### Pilihan resmi

**EfficientNet-B0** dengan transfer learning di PyTorch.

### Alasan memilih EfficientNet-B0

- Kuat untuk dataset kecil-menengah dan masih relevan saat data utama diperbesar secara moderat.
- Kompromi yang bagus antara akurasi dan ukuran model.
- Lebih realistis untuk train di Google Colab/Kaggle gratis.
- Lebih mudah direproduksi dibanding arsitektur yang terlalu eksperimental.

### Model pembanding

Minimal salah satu dari berikut ini digunakan sebagai baseline pembanding:

- `ResNet50` — baseline akademik yang aman dan mudah dijelaskan.
- `MobileNetV3-Large` — baseline ringan jika deployment/mobile jadi fokus.

### Model modern opsional

Jika resource dan waktu cukup, tambahkan satu model modern ringan:

- `MobileViT-XXS`, atau
- `DeiT-Tiny`.

Tujuan model opsional ini adalah **membuktikan bahwa tim mempertimbangkan pendekatan latest tech**, namun tetap realistis untuk dataset cabai publik yang ukurannya masih terbatas.

### Keputusan penting

- **Jangan** mulai dari transformer besar.
- **Jangan** training from scratch.
- **Jangan** mengejar model paling kompleks hanya demi terlihat canggih.

Argumen resminya:

> Untuk dataset cabai publik yang masih terbatas dan heterogen, transfer learning pada model yang efisien lebih cocok daripada arsitektur besar yang membutuhkan data dan compute jauh lebih besar.

## 8.3 Solusi 2 — AIaaS gratis

### Pilihan resmi

**Hugging Face Spaces** sebagai hosted AI demo service gratis.

### Kenapa ini dipilih

- Mudah dipublikasikan melalui URL.
- Tidak perlu infrastruktur server sendiri.
- Cocok untuk demo edukasi dan presentasi.
- Gratis untuk public Space dengan resource dasar CPU.

### Cara menjelaskannya di laporan/presentasi

- Model yang telah dilatih diunggah ke environment hosted.
- Pengguna bisa mengakses layanan inferensi melalui antarmuka web.
- Sistem berjalan sebagai **AI service berbasis cloud untuk skala demo**.

### Caveat yang harus jujur disebutkan

- Free hardware akan tidur jika tidak dipakai.
- Performa inferensi CPU lebih lambat dibanding GPU.
- Ini cocok untuk **demo dan validasi konsep**, bukan deployment produksi skala besar.

## 8.4 Solusi 3 — AI bebas

### Pilihan resmi

**Recommendation engine + explainability**.

Komponennya:

- Rule-based/knowledge-based recommendation.
- Grad-CAM untuk visual explanation.
- Opsi FAQ assistant berbasis dokumen jika waktu cukup.

### Kenapa ini dipilih

- Memberi nilai tambah nyata di luar sekadar klasifikasi.
- Mudah dijelaskan ke audiens bisnis dan teknis.
- Tidak bergantung pada API berbayar.
- Lebih relevan untuk pengguna akhir karena prediksi saja belum cukup membantu.

### Bentuk output yang diharapkan

Setelah model memprediksi kelas, sistem juga menampilkan:

- nama penyakit,
- confidence score,
- area daun yang paling diperhatikan model,
- saran tindakan awal,
- catatan bahwa hasil ini adalah bantuan awal, bukan diagnosis final.

## 9. Kenapa pendekatan ini cocok untuk kasus cabai

### Dari sisi data

Karena data proyek berasal dari sumber publik yang ukurannya masih moderat dan kualitasnya tidak sepenuhnya seragam, solusi harus:

- hemat compute,
- kuat pada transfer learning,
- tidak terlalu sensitif terhadap keterbatasan ukuran dan variasi domain.

### Dari sisi pengguna

Karena target use case bersifat praktis, solusi harus:

- mudah dijelaskan,
- mudah didemokan,
- bisa diakses dari browser,
- tidak mahal.

### Dari sisi akademik

Karena ini tugas besar, solusi harus:

- memenuhi requirement AIaaS + self-trained + AI bebas,
- mengikuti CRISP-DM,
- punya narasi evaluasi yang jujur,
- cukup modern tanpa overengineering.

## 10. Detail pendekatan modelling

### Framework utama

- **PyTorch**
- Model zoo: **timm**

### Konfigurasi awal yang direkomendasikan

- ukuran input: `224 x 224`
- pretrained weights: **ImageNet pretrained**
- optimizer: `AdamW`
- learning rate tahap head-only: `1e-3`
- learning rate tahap fine-tuning: `1e-4` atau `3e-5`
- loss utama: `CrossEntropyLoss`
- opsi jika perlu: weighted cross entropy

### Strategi training yang direkomendasikan

Strategi ini berlaku untuk **dataset utama training**, sedangkan dataset Kaggle diposisikan sebagai benchmark terpisah.

Tahap 1:

- freeze backbone,
- train classifier head beberapa epoch.

Tahap 2:

- unfreeze sebagian/top layers,
- fine-tune dengan learning rate lebih kecil.

### Data augmentation yang direkomendasikan

Gunakan augmentasi ringan-menengah yang masih menjaga ciri penyakit:

- horizontal flip,
- vertical flip,
- random rotation `10–30°`,
- random resized crop,
- brightness/contrast jitter,
- hue/saturation jitter secara moderat,
- light blur.

Opsional:

- `MixUp`,
- `CutMix`.

### Augmentasi yang perlu hati-hati

- color transform berlebihan bisa merusak gejala warna,
- blur terlalu besar bisa menghapus tekstur bercak,
- random erasing besar bisa menutup area penyakit.

## 11. Evaluasi yang wajib ada

Jangan hanya melaporkan accuracy.

### Metrik utama

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion matrix

### Kenapa Macro F1 penting

Karena distribusi data tidak sepenuhnya seimbang dan beberapa kelas bisa lebih sulit dibedakan. Macro F1 membantu menunjukkan performa antar kelas dengan lebih adil.

### Visualisasi yang wajib ditampilkan

- training loss vs validation loss,
- training accuracy vs validation accuracy,
- confusion matrix,
- contoh prediksi benar,
- contoh prediksi salah,
- visualisasi Grad-CAM.

### Struktur evaluasi yang disepakati

Laporkan hasil minimal dalam dua blok:

1. **Internal evaluation** pada split training/validation/test dari dataset utama.
2. **External benchmark evaluation** pada dataset Kaggle yang tidak ikut dipakai saat training utama.

Tujuannya agar tim bisa menunjukkan tidak hanya performa in-domain, tetapi juga indikasi generalisasi lintas sumber.

### Target evaluasi yang realistis

Jangan mengunci target angka terlalu agresif. Posisi yang aman:

- target performa yang layak: **cukup tinggi dan stabil**,
- kisaran yang masuk akal untuk dataset seperti ini: **sekitar 90–96%** pada split yang bersih,
- jika hasil sangat tinggi, tim harus tetap membahas kemungkinan split mudah atau adanya near-duplicate.

## 12. Explainability dan recommendation layer

### Explainability

**Grad-CAM** dipilih karena:

- mudah dipahami audiens non-teknis,
- cocok untuk model klasifikasi visual,
- dapat menunjukkan area gambar yang dianggap penting oleh model.

### Recommendation layer

Setiap kelas harus dipetakan minimal ke:

- deskripsi singkat,
- indikasi gejala umum,
- saran tindakan awal,
- saran pencegahan,
- disclaimer.

Contoh struktur knowledge base:

```json
{
  "leaf_spot": {
    "nama_id": "bercak daun",
    "deskripsi": "Gejala berupa bercak pada daun cabai.",
    "tindakan_awal": [
      "Pisahkan daun yang tampak parah bila memungkinkan.",
      "Jaga kebersihan area tanam.",
      "Pertimbangkan konsultasi dengan penyuluh atau ahli pertanian."
    ],
    "disclaimer": "Prediksi ini adalah bantuan awal dan tidak menggantikan pemeriksaan ahli."
  }
}
```

## 13. Deployment yang disepakati

### Pilihan utama

- **Hugging Face Spaces** untuk demo publik, atau
- **Streamlit Community Cloud** bila alur deploy lebih sederhana.

### UI yang disarankan

Fitur minimal:

- upload gambar,
- tampilkan prediksi kelas,
- tampilkan confidence,
- tampilkan heatmap/Grad-CAM,
- tampilkan rekomendasi tindakan awal.

### Bahasa UI

Gunakan **Bahasa Indonesia**. Bila perlu, tambahkan subtitle istilah Inggris kecil di bawahnya.

## 14. MLOps dan manajemen eksperimen

### Pendekatan yang dipilih

Tetap ringan. Jangan membangun pipeline yang terlalu berat.

Pilihan resmi:

- Git/GitHub untuk versioning code,
- **MLflow** untuk tracking eksperimen jika sempat,
- atau minimal tabel eksperimen manual bila waktu sangat terbatas.

### Yang wajib dicatat per eksperimen

- nama model,
- ukuran input,
- augmentasi yang dipakai,
- learning rate,
- epoch,
- hasil accuracy,
- hasil macro F1,
- catatan error atau insight.

## 15. Risiko proyek dan mitigasinya

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Dataset utama tetap moderat, bukan besar | overfitting / model kurang stabil | transfer learning, augmentasi, early stopping |
| Mismatch label antar sumber | evaluasi tidak fair | bekukan scope ke 5 kelas bersama |
| Near-duplicate / leakage | skor terlalu optimistis | lakukan pengecekan visual seperlunya dan bahas limitation |
| Background bervariasi | model belajar background, bukan penyakit | augmentasi dan analisis error |
| Compute gratis terbatas | training terputus | pakai checkpoint, simpan model per epoch |
| Waktu tim mepet | scope melebar | kunci fitur inti sejak awal |
| AIaaS gratis terbatas | demo tidak stabil | siapkan fallback local/Streamlit |

## 16. Klaim yang boleh dan tidak boleh dibuat

### Boleh

- “Sistem membantu identifikasi awal penyakit cabai.”
- “Model dilatih menggunakan transfer learning pada dataset citra cabai.”
- “Sistem menyediakan rekomendasi tindakan awal berbasis hasil klasifikasi.”
- “Solusi ditujukan sebagai proof of concept yang murah dan mudah diakses.”

### Jangan dibuat tanpa bukti kuat

- “Siap dipakai di semua kondisi lahan Indonesia.”
- “Menggantikan pakar pertanian.”
- “Mampu diagnosis pasti.”
- “Akurat sempurna.”

## 17. Struktur narasi proposal dan presentasi

Gunakan urutan ini agar semua presentasi konsisten.

### Slide 1 — Latar belakang

- cabai adalah komoditas penting,
- penyakit daun memengaruhi hasil,
- identifikasi manual punya keterbatasan.

### Slide 2 — Problem statement

- butuh sistem identifikasi awal berbasis gambar yang cepat dan murah.

### Slide 3 — Solusi yang diusulkan

- model klasifikasi,
- explainability,
- rekomendasi tindakan,
- deployment web gratis.

### Slide 4 — Kenapa pendekatan ini

- data publik cabai masih terbatas dan heterogen → transfer learning,
- butuh gratis → pakai tool free tier/open source,
- butuh mudah didemo → web app hosted.

### Slide 5 — Dataset

- sumber,
- strategi data utama vs benchmark,
- kelas,
- jumlah dan pembagian per sumber,
- limitation.

### Slide 6 — Arsitektur sistem

- upload gambar → prediksi → heatmap → rekomendasi.

### Slide 7 — Rencana modelling

- EfficientNet-B0 sebagai model utama,
- baseline pembanding,
- evaluasi utama.

### Slide 8 — Nilai bisnis

- akses mudah,
- bantuan keputusan awal,
- potensi edukasi dan agritech murah.

### Slide 9 — Demo/progress/final plan

- apa yang sudah ada,
- apa yang akan dibangun berikutnya.

## 18. Struktur laporan akhir berbasis CRISP-DM

## 18.1 Business Understanding

Isi minimum:

- konteks bisnis cabai,
- masalah utama,
- tujuan solusi,
- stakeholder,
- manfaat.

## 18.2 Data Understanding

Isi minimum:

- sumber dataset utama dan benchmark,
- deskripsi kelas,
- jumlah data per sumber,
- contoh data,
- insight EDA sederhana,
- limitation data.

## 18.3 Data Preparation

Isi minimum:

- harmonisasi label 5 kelas,
- preprocessing,
- resize,
- normalisasi,
- augmentasi,
- split yang dipakai dan alasan kenapa benchmark Kaggle dipisah,
- alasan tiap langkah.

## 18.4 Modelling

Isi minimum:

- alasan pilih EfficientNet-B0,
- baseline pembanding,
- hyperparameter,
- skema transfer learning,
- kurva training.

## 18.5 Evaluation

Isi minimum:

- accuracy, precision, recall, macro F1,
- confusion matrix,
- contoh salah klasifikasi,
- interpretasi hasil,
- Grad-CAM.

## 18.6 Deployment

Isi minimum:

- bentuk aplikasi,
- use case diagram sederhana,
- fitur sistem,
- batasan deployment gratis.

## 19. Prioritas pengerjaan tim

Urutan prioritas resmi:

1. Dataset siap dipakai dan label mapping jelas.
2. Baseline model jalan.
3. Model utama jalan dan terukur.
4. Evaluasi lengkap.
5. Web demo jalan.
6. Explainability jalan.
7. Recommendation layer rapi.
8. Fitur tambahan opsional.

## 20. Definisi selesai per milestone

### Proposal selesai jika

- problem statement jelas,
- solusi 3 komponen jelas,
- dataset dan pendekatan sudah dipilih,
- mockup / gambaran produk tersedia.

### Demo progress selesai jika

- model minimal satu sudah train,
- ada hasil evaluasi awal,
- ada prototype antarmuka atau alur demo.

### Final selesai jika

- model utama terlatih,
- evaluasi lengkap,
- demo bisa diakses,
- laporan lengkap,
- semua narasi konsisten.

## 21. Sumber utama yang boleh dirujuk

Gunakan sumber ini sebagai rujukan resmi tim. Jika anggota ingin menambah sumber lain, usahakan tetap memakai sumber primer atau dokumentasi resmi.

### A. Spesifikasi tugas

- Dokumen lokal: `Spesifikasi.md`

### B. Dataset

- Roboflow `Chili leaves disease classification`:  
  <https://universe.roboflow.com/chili-leaves-disease-classification/chili-leaves-disease-classification>
- Kaggle dataset `penyakit-cabai`:  
  <https://www.kaggle.com/datasets/bryanrizqiprakosa/penyakit-cabai>
- Roboflow `ProjectTA Chili Leaf Disease` (reference only, detection):  
  <https://universe.roboflow.com/chili-leaf-disease-uf14c/projectta-chili-leaf-disease>
- Zenodo `Cercospora Leaf Spot in Chili Pepper Leaves Image Dataset` (reference only, single disease):  
  <https://zenodo.org/records/13272039>

### C. Platform training dan notebook gratis

- Google Colab FAQ:  
  <https://research.google.com/colaboratory/faq.html>
- Kaggle Notebooks documentation:  
  <https://www.kaggle.com/docs/notebooks>

### D. Deployment gratis

- Hugging Face Spaces overview:  
  <https://huggingface.co/docs/hub/en/spaces-overview>
- Streamlit deployment docs:  
  <https://docs.streamlit.io/deploy>

### E. Framework dan library utama

- PyTorch tutorials:  
  <https://docs.pytorch.org/tutorials/>
- `timm` / PyTorch Image Models repository:  
  <https://github.com/huggingface/pytorch-image-models>
- Albumentations docs:  
  <https://albumentations.ai/docs/>
- Hugging Face Transformers docs:  
  <https://huggingface.co/docs/transformers/index>

### F. Explainability dan experiment tracking

- Grad-CAM package:  
  <https://pypi.org/project/grad-cam/>
- PyTorch Grad-CAM repository:  
  <https://github.com/jacobgil/pytorch-grad-cam>
- MLflow tracking docs:  
  <https://mlflow.org/docs/latest/ml/tracking/>

### G. Referensi deployment edge/mobile

- TensorFlow Lite overview:  
  <https://www.tensorflow.org/lite>

### H. Referensi riset terbaru untuk positioning model

Berikut referensi yang berguna untuk mendukung argumen bahwa model ringan/hybrid/transformer sedang aktif diteliti pada plant disease classification. Gunakan ini untuk landasan narasi, **bukan untuk copy-claim mentah**:

- ViT-RoT: benchmark Vision Transformer untuk tomato leaf disease recognition (2025):  
  <https://www.mdpi.com/2624-7402/7/6/185>
- Multi-kernel inception-enhanced vision transformer for plant leaf disease recognition (Scientific Reports, 2025):  
  <https://www.nature.com/articles/s41598-025-16142-x>
- A novel lightweight hybrid CNN–ViT for maize leaf disease classification (Scientific Reports, 2026):  
  <https://www.nature.com/articles/s41598-026-41190-2>
- Advancing multi-label tomato leaf disease identification using Vision Transformer and EfficientNet with XAI techniques (2025):  
  <https://www.mdpi.com/2079-9292/14/23/4762>

## 22. Penutup — kalimat posisi resmi tim

Kalimat ini bisa dipakai hampir di semua konteks sebagai ringkasan:

> Proyek ini mengusulkan sistem klasifikasi penyakit cabai berbasis computer vision yang seluruh komponennya dirancang tetap gratis dan realistis untuk skala mahasiswa. Solusi utama menggunakan transfer learning pada model EfficientNet-B0 yang dilatih pada dataset klasifikasi cabai publik yang lebih besar, lalu diuji ulang pada benchmark Kaggle yang dipisah agar hasilnya lebih defensible. Sistem kemudian dilengkapi deployment berbasis layanan hosted gratis dan recommendation layer agar hasil prediksi lebih mudah dipahami dan ditindaklanjuti.

## 23. Next action yang disarankan setelah dokumen ini

- Tetapkan nama kelompok dan judul final.
- Bagi tugas per anggota berdasarkan bagian CRISP-DM.
- Unduh, audit, dan harmonisasi label dataset utama serta benchmark.
- Buat baseline notebook pertama.
- Siapkan template slide dan template laporan berdasarkan isi dokumen ini.
