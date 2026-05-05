from __future__ import annotations

from .config import DISPLAY_NAMES

RECOMMENDATIONS = {
    "healthy": {
        "nama_id": "sehat",
        "deskripsi": "Daun tidak menunjukkan gejala utama pada kelas penyakit yang menjadi scope prototype.",
        "gejala_umum": [
            "Warna daun relatif hijau merata.",
            "Tidak tampak bercak dominan atau perubahan bentuk ekstrem.",
        ],
        "tindakan_awal": [
            "Lanjutkan pemantauan rutin.",
            "Jaga kebersihan area tanam dan sirkulasi udara.",
            "Ambil foto ulang jika muncul gejala baru.",
        ],
        "pencegahan": [
            "Gunakan penyiraman dan pemupukan yang konsisten.",
            "Hindari kelembapan berlebih pada daun.",
        ],
    },
    "leaf curl": {
        "nama_id": "keriting daun",
        "deskripsi": "Gejala daun tampak melengkung, menggulung, atau berubah bentuk.",
        "gejala_umum": [
            "Daun keriting atau menggulung.",
            "Pertumbuhan daun muda dapat terlihat tidak normal.",
        ],
        "tindakan_awal": [
            "Pisahkan atau tandai tanaman yang menunjukkan gejala berat.",
            "Periksa kemungkinan vektor seperti kutu kebul.",
            "Konsultasikan dengan penyuluh pertanian untuk penanganan spesifik.",
        ],
        "pencegahan": [
            "Kontrol hama vektor secara rutin.",
            "Bersihkan gulma di sekitar tanaman.",
        ],
    },
    "leaf spot": {
        "nama_id": "bercak daun",
        "deskripsi": "Gejala berupa bercak pada permukaan daun cabai.",
        "gejala_umum": [
            "Bercak cokelat/kehitaman atau area nekrotik pada daun.",
            "Pada kasus berat, bercak dapat menyebar dan daun mengering.",
        ],
        "tindakan_awal": [
            "Buang daun yang tampak sangat terinfeksi bila memungkinkan.",
            "Kurangi kelembapan berlebih di sekitar tanaman.",
            "Pertimbangkan konsultasi untuk penggunaan fungisida yang tepat.",
        ],
        "pencegahan": [
            "Hindari percikan air berlebih ke daun.",
            "Jaga jarak tanam dan sanitasi lahan.",
        ],
    },
    "whitefly": {
        "nama_id": "whitefly / kutu kebul",
        "deskripsi": "Gejala terkait serangan whitefly atau kutu kebul pada daun cabai.",
        "gejala_umum": [
            "Daun dapat menguning, melemah, atau terlihat terganggu.",
            "Kadang ditemukan serangga kecil berwarna putih di bagian bawah daun.",
        ],
        "tindakan_awal": [
            "Periksa bagian bawah daun untuk keberadaan kutu kebul.",
            "Gunakan perangkap kuning atau kontrol hama sesuai rekomendasi lokal.",
            "Pisahkan tanaman dengan gejala berat jika memungkinkan.",
        ],
        "pencegahan": [
            "Pantau populasi hama secara rutin.",
            "Jaga sanitasi dan kendalikan gulma inang.",
        ],
    },
    "yellowish": {
        "nama_id": "virus kuning / daun menguning",
        "deskripsi": "Daun tampak menguning sesuai label dataset `yellowish`.",
        "gejala_umum": [
            "Warna daun berubah kekuningan.",
            "Pertumbuhan tanaman dapat melemah bila gejala meluas.",
        ],
        "tindakan_awal": [
            "Amati apakah gejala menyebar ke daun lain.",
            "Periksa kemungkinan hama vektor dan kondisi nutrisi tanaman.",
            "Konsultasikan dengan penyuluh pertanian untuk membedakan penyakit dan defisiensi nutrisi.",
        ],
        "pencegahan": [
            "Jaga keseimbangan nutrisi tanaman.",
            "Kontrol hama vektor dan bersihkan gulma sekitar lahan.",
        ],
    },
}

DISCLAIMER = "Prediksi ini adalah bantuan identifikasi awal dan tidak menggantikan pemeriksaan ahli pertanian."


def get_recommendation(label: str) -> dict:
    recommendation = dict(RECOMMENDATIONS.get(label, {}))
    recommendation.setdefault("nama_id", DISPLAY_NAMES.get(label, label))
    recommendation.setdefault("deskripsi", "Belum ada deskripsi untuk label ini.")
    recommendation["disclaimer"] = DISCLAIMER
    return recommendation


def format_recommendation_markdown(label: str, confidence: float | None = None) -> str:
    rec = get_recommendation(label)
    confidence_text = f" ({confidence:.1%})" if confidence is not None else ""
    lines = [
        f"### {rec['nama_id']}{confidence_text}",
        "",
        rec["deskripsi"],
        "",
        "**Tindakan awal:**",
    ]
    lines.extend(f"- {item}" for item in rec.get("tindakan_awal", []))
    lines.extend(["", f"_{rec['disclaimer']}_"])
    return "\n".join(lines)
