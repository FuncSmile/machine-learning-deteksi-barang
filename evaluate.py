"""
evaluate.py
-----------
Evaluasi lengkap model YOLOv8 setelah training.

Yang dilakukan:
  1. Load model dari runs/train/deteksi_barang_v1/weights/best.pt
  2. Evaluasi pada TEST set (split='test') memakai data.yaml.
  3. Tampilkan metrik global + per class: mAP50, mAP50-95, Precision, Recall, F1.
  4. Identifikasi class dengan Precision < 90% dan beri rekomendasi.
  5. Simpan laporan rapi ke evaluation_report.txt.
  6. Confusion matrix otomatis dibuat ultralytics, lalu disalin ke runs/evaluate/.
"""

from pathlib import Path
import shutil
import traceback

# === KONFIGURASI PATH DASAR ===
BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
DATA_YAML = DATASETS_DIR / "data.yaml"
BEST_MODEL = BASE_DIR / "runs" / "deteksi-barang-yolov8s" / "weights" / "best.pt"
EVAL_DIR = BASE_DIR / "runs" / "evaluate"
REPORT_FILE = BASE_DIR / "evaluation_report.txt"

# Ambang precision target
PRECISION_TARGET = 0.90


def f1_score(p: float, r: float) -> float:
    """Hitung F1 dari precision & recall (aman terhadap pembagian nol)."""
    if (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r)


def build_report(metrics, names):
    """Susun teks laporan evaluasi. Return (report_str, low_precision_list)."""
    lines = []
    lines.append("=" * 78)
    lines.append("LAPORAN EVALUASI - deteksi_barang (TEST SET)")
    lines.append("=" * 78)

    box = metrics.box

    # --- Metrik global ---
    lines.append("\n[ METRIK GLOBAL ]")
    lines.append(f"  Precision (rata-rata) : {box.mp:.4f}")
    lines.append(f"  Recall    (rata-rata) : {box.mr:.4f}")
    lines.append(f"  F1        (rata-rata) : {f1_score(box.mp, box.mr):.4f}")
    lines.append(f"  mAP50                 : {box.map50:.4f}")
    lines.append(f"  mAP50-95              : {box.map:.4f}")
    status = "TERCAPAI ✓" if box.mp >= PRECISION_TARGET else "BELUM tercapai ✗"
    lines.append(f"  Target Precision >= {PRECISION_TARGET:.2f} : {status}")

    # --- Metrik per class ---
    lines.append("\n[ METRIK PER CLASS ]")
    header = f"  {'CLASS':<16} {'P':>7} {'R':>7} {'F1':>7} {'mAP50':>8} {'mAP50-95':>9}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    low_precision = []

    # box.ap_class_index = indeks class yang punya data di evaluasi
    # Gunakan API per-class ultralytics: box.class_result(i)
    ap_index = list(box.ap_class_index)
    for i, cidx in enumerate(ap_index):
        cname = names[cidx] if cidx < len(names) else f"id_{cidx}"
        try:
            # class_result(i) -> (precision, recall, ap50, ap50-95)
            p, r, ap50, ap = box.class_result(i)
        except Exception:
            # Fallback bila API berbeda versi
            p = float(box.p[i]) if hasattr(box, "p") else 0.0
            r = float(box.r[i]) if hasattr(box, "r") else 0.0
            ap50 = float(box.ap50[i]) if hasattr(box, "ap50") else 0.0
            ap = float(box.ap[i]) if hasattr(box, "ap") else 0.0

        f1 = f1_score(p, r)
        lines.append(f"  {cname:<16} {p:>7.3f} {r:>7.3f} {f1:>7.3f} "
                     f"{ap50:>8.3f} {ap:>9.3f}")

        if p < PRECISION_TARGET:
            low_precision.append((cname, p, r))

    # Class yang sama sekali tidak muncul di hasil (tidak ada prediksi/data)
    missing = [names[c] for c in range(len(names)) if c not in ap_index]
    if missing:
        lines.append("\n  [CATATAN] Class tanpa hasil evaluasi (tidak ada data/"
                     "prediksi di test set):")
        for m in missing:
            lines.append(f"            - {m}")

    # --- Rekomendasi class precision rendah ---
    lines.append("\n" + "=" * 78)
    lines.append("REKOMENDASI")
    lines.append("=" * 78)
    if low_precision:
        lines.append(f"Class dengan Precision < {PRECISION_TARGET:.0%} "
                     f"({len(low_precision)} class):")
        for cname, p, r in low_precision:
            lines.append(f"  - {cname:<16} P={p:.3f}  R={r:.3f}")
        lines.append("")
        lines.append("Saran perbaikan:")
        lines.append("  1. Tambah & perbaiki anotasi untuk class precision rendah")
        lines.append("     (precision rendah = banyak false positive / label salah).")
        lines.append("  2. Periksa apakah ada class yang mirip secara visual dan")
        lines.append("     sering tertukar -> lihat confusion_matrix.png.")
        lines.append("  3. Naikkan confidence threshold saat inference untuk menekan")
        lines.append("     false positive (mis. dari 0.15 ke 0.25-0.35).")
        lines.append("  4. Tambah data untuk class minim (HP, dompet, jam tangan, mouse).")
        lines.append("  5. Pertimbangkan training lebih lama / unfreeze + fine-tune.")
    else:
        lines.append("Semua class sudah mencapai Precision >= "
                     f"{PRECISION_TARGET:.0%}. Mantap! ✓")

    lines.append("=" * 78)
    return "\n".join(lines), low_precision


def copy_confusion_matrix(save_dir: Path):
    """Salin confusion matrix yang dibuat ultralytics ke runs/evaluate/."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for fname in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        src = save_dir / fname
        if src.exists():
            dst = EVAL_DIR / fname
            try:
                shutil.copy(src, dst)
                copied.append(dst)
            except Exception as e:
                print(f"[WARNING] Gagal menyalin {fname}: {e}")
    return copied


def evaluate():
    """Jalankan evaluasi penuh pada test set."""
    print("=" * 70)
    print("EVALUATE YOLOv8 - deteksi_barang")
    print("=" * 70)

    # --- Validasi prasyarat ---
    if not BEST_MODEL.exists():
        print(f"[ERROR] Model best.pt tidak ditemukan: {BEST_MODEL}")
        print("        Jalankan training dulu: python train.py")
        return False
    if not DATA_YAML.exists():
        print(f"[ERROR] data.yaml tidak ditemukan: {DATA_YAML}")
        print("        Jalankan: python fix_yaml.py")
        return False

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Package 'ultralytics' belum terinstall.")
        print("        Jalankan: pip install -r requirements.txt")
        return False

    try:
        import yaml
        with DATA_YAML.open("r", encoding="utf-8") as f:
            names = yaml.safe_load(f).get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]

        print(f"[INFO] Memuat model: {BEST_MODEL}")
        model = YOLO(str(BEST_MODEL))

        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        print("[INFO] Menjalankan evaluasi pada TEST set...\n")

        # split='test' -> evaluasi pada test set. plots=True -> confusion matrix.
        metrics = model.val(
            data=str(DATA_YAML),
            split="test",
            project=str(BASE_DIR / "runs"),
            name="evaluate",
            exist_ok=True,
            plots=True,
        )

        # --- Susun & cetak laporan ---
        report, low_precision = build_report(metrics, names)
        print("\n" + report)

        # --- Simpan laporan ---
        REPORT_FILE.write_text(report, encoding="utf-8")
        print(f"\n[OK] Laporan disimpan ke: {REPORT_FILE}")

        # --- Confusion matrix ---
        save_dir = Path(metrics.save_dir) if hasattr(metrics, "save_dir") else EVAL_DIR
        copied = copy_confusion_matrix(save_dir)
        if copied:
            for c in copied:
                print(f"[OK] Confusion matrix: {c}")
        else:
            print(f"[INFO] Confusion matrix ada di: {save_dir}")

        print("\n[NEXT] Coba inference: python predict.py --mode webcam")
        return True

    except Exception as e:
        print(f"[ERROR] Evaluasi gagal: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    ok = evaluate()
    if not ok:
        raise SystemExit(1)
