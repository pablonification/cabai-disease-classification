from __future__ import annotations

import numpy as np

from .config import DISPLAY_NAMES

# Deskripsi visual per penyakit — konteks tambahan untuk LLM
_DISEASE_VISUAL_CONTEXT = {
    "healthy": "Daun cabai sehat biasanya berwarna hijau merata tanpa bercak atau deformasi.",
    "leaf curl": "Keriting daun ditandai dengan daun yang menggulung atau melengkung, sering disebabkan oleh virus atau serangan kutu.",
    "leaf spot": "Bercak daun menunjukkan lesi nekrotik berwarna cokelat atau hitam pada permukaan daun.",
    "whitefly": "Serangan kutu kebul menyebabkan bintik-bintik kuning kecil dan daun melemah; serangga kecil putih sering ditemukan di bagian bawah daun.",
    "yellowish": "Virus kuning menyebabkan perubahan warna daun menjadi kekuningan, menandakan gangguan pada klorofil.",
}


def _describe_heatmap_region(grayscale_cam: np.ndarray) -> str:
    """
    Analisis programatik posisi area 'panas' pada heatmap Grad-CAM.
    Mengembalikan deskripsi tekstual posisi fokus model (atas/bawah, kiri/kanan, tepi/tengah).
    """
    h, w = grayscale_cam.shape
    threshold = 0.5
    hot = grayscale_cam >= threshold

    if hot.sum() == 0:
        # Fallback: ambil top 20% nilai tertinggi
        threshold = np.percentile(grayscale_cam, 80)
        hot = grayscale_cam >= threshold

    rows, cols = np.where(hot)

    if len(rows) == 0:
        return "seluruh area daun"

    # Posisi vertikal
    center_row = rows.mean() / h
    if center_row < 0.35:
        vertical = "bagian atas"
    elif center_row > 0.65:
        vertical = "bagian bawah"
    else:
        vertical = "bagian tengah"

    # Posisi horizontal
    center_col = cols.mean() / w
    if center_col < 0.35:
        horizontal = "sisi kiri"
    elif center_col > 0.65:
        horizontal = "sisi kanan"
    else:
        horizontal = "area tengah"

    # Seberapa tersebar
    spread = hot.sum() / hot.size
    if spread > 0.4:
        spread_desc = "hampir merata di seluruh daun"
    elif spread > 0.2:
        spread_desc = f"{vertical} dan {horizontal}"
    else:
        spread_desc = f"titik-titik spesifik di {vertical} {horizontal}"

    return spread_desc


def build_explanation_prompt(
    label: str,
    confidence: float,
    grayscale_cam: np.ndarray,
) -> str:
    display_name = DISPLAY_NAMES.get(label, label)
    region = _describe_heatmap_region(grayscale_cam)
    visual_context = _DISEASE_VISUAL_CONTEXT.get(label, "")
    confidence_pct = f"{confidence:.0%}"

    prompt = f"""Kamu adalah sistem AI yang membantu petani cabai memahami hasil analisis penyakit daun.

Model computer vision mendeteksi: **{display_name}** (confidence: {confidence_pct})
Area fokus model pada heatmap Grad-CAM: {region}
Konteks visual penyakit ini: {visual_context}

Tulis penjelasan singkat dalam 2-3 kalimat bahasa Indonesia yang mudah dipahami petani.
Jelaskan:
1. Apa yang dilihat model di gambar (kaitkan area fokus dengan gejala penyakit)
2. Mengapa model menyimpulkan penyakit tersebut

Gunakan bahasa yang sederhana. Jangan gunakan jargon teknis AI."""

    return prompt


def generate_explanation_gemini(
    label: str,
    confidence: float,
    grayscale_cam: np.ndarray,
    api_key: str,
) -> str:
    """
    Panggil Gemini API untuk menghasilkan narasi penjelasan prediksi.
    Mengembalikan string penjelasan, atau pesan fallback jika gagal.
    """
    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        prompt = build_explanation_prompt(label, confidence, grayscale_cam)
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
        )
        return response.text.strip()

    except ImportError:
        return _fallback_explanation(label, confidence, grayscale_cam)
    except Exception:
        return _fallback_explanation(label, confidence, grayscale_cam)


def build_recommendation_prompt(label: str, confidence: float) -> str:
    from .recommend import RECOMMENDATIONS
    display_name = DISPLAY_NAMES.get(label, label)
    rec = RECOMMENDATIONS.get(label, {})
    gejala = "; ".join(rec.get("gejala_umum", []))
    confidence_pct = f"{confidence:.0%}"

    prompt = f"""Kamu adalah asisten pertanian yang membantu petani cabai menangani penyakit tanaman.

Penyakit terdeteksi: {display_name} (confidence: {confidence_pct})
Gejala umum: {gejala}

Berikan rekomendasi penanganan praktis dalam 3-5 poin singkat, dalam bahasa Indonesia yang mudah dipahami petani.
Sertakan:
1. Tindakan segera yang perlu dilakukan
2. Cara mencegah penyebaran
3. Kapan perlu konsultasi ke ahli

Gunakan bahasa sederhana. Awali setiap poin dengan tanda "-". Tambahkan disclaimer singkat di akhir bahwa ini bukan pengganti saran ahli agronomi."""

    return prompt


def generate_recommendation_gemini(
    label: str,
    confidence: float,
    api_key: str,
) -> str:
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        prompt = build_recommendation_prompt(label, confidence)
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
        )
        return response.text.strip()

    except ImportError:
        return ""
    except Exception:
        return ""


def answer_followup_question(
    question: str,
    label: str,
    confidence: float,
    recommendation_context: dict,
    api_key: str,
) -> str:
    """
    Jawab pertanyaan lanjutan user tentang penyakit yang terdeteksi.
    Dikasih context penyakit + rekomendasi KB yang sudah ditampilkan supaya LLM tidak ngasal.
    """
    display_name = DISPLAY_NAMES.get(label, label)
    tindakan = "\n".join(f"- {t}" for t in recommendation_context.get("tindakan_awal", []))
    pencegahan = "\n".join(f"- {t}" for t in recommendation_context.get("pencegahan", []))
    deskripsi = recommendation_context.get("deskripsi", "")

    system_context = f"""Kamu adalah asisten pertanian untuk aplikasi CabAI yang membantu petani cabai.

Konteks prediksi saat ini:
- Penyakit terdeteksi: {display_name} (confidence: {confidence:.0%})
- Deskripsi: {deskripsi}
- Rekomendasi tindakan yang sudah diberikan ke petani:
{tindakan}
- Rekomendasi pencegahan yang sudah diberikan:
{pencegahan}

Jawab pertanyaan petani berdasarkan konteks di atas. Gunakan bahasa Indonesia yang sederhana.
Jika pertanyaan di luar topik penyakit cabai, arahkan kembali ke topik tersebut.
Di akhir jawaban, ingatkan singkat bahwa untuk keputusan penting sebaiknya konsultasi ke ahli agronomi."""

    prompt = f"{system_context}\n\nPertanyaan petani: {question}"

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
        )
        return response.text.strip()

    except ImportError:
        return "Maaf, layanan chatbot tidak tersedia saat ini."
    except Exception:
        return "Maaf, terjadi kesalahan saat memproses pertanyaan. Silakan coba lagi."


def _fallback_explanation(
    label: str,
    confidence: float,
    grayscale_cam: np.ndarray,
) -> str:
    """Penjelasan statis jika API tidak tersedia."""
    display_name = DISPLAY_NAMES.get(label, label)
    region = _describe_heatmap_region(grayscale_cam)
    visual_context = _DISEASE_VISUAL_CONTEXT.get(label, "")
    return (
        f"Model mendeteksi **{display_name}** dengan keyakinan {confidence:.0%}. "
        f"Area perhatian utama model berada di {region}. "
        f"{visual_context}"
    )
