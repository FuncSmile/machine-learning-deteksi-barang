"""
train.py
--------
Training pipeline YOLOv8 dengan best practice untuk target Precision >= 90%.

Alur:
  1. Pastikan data.yaml memakai absolute path (panggil fix_yaml otomatis).
  2. Deteksi device (CUDA / CPU) otomatis.
  3. Cetak konfigurasi training.
  4. Train dengan fallback batch (16 -> 8 -> 4) jika terjadi CUDA OOM.
  5. Simpan ringkasan ke results_summary.txt.
  6. Cetak lokasi best.pt.

Catatan target precision:
  - yolov8m + augmentasi + 150 epoch biasanya cukup untuk dataset terkurasi.
  - Class dengan data minim (dompet, jam tangan, mouse, HP) adalah penentu
    apakah precision rata-rata bisa tembus 90%. Lihat check_dataset.py.
"""

from pathlib import Path
import traceback

# === KONFIGURASI PATH DASAR ===
BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
DATA_YAML = DATASETS_DIR / "data.yaml"
PROJECT_DIR = BASE_DIR / "runs"
RUN_NAME = "deteksi-barang-yolov8s"
SUMMARY_FILE = BASE_DIR / "results_summary.txt"

# === HYPERPARAMETER / KONFIGURASI TRAINING ===
# Disusun sebagai dict agar mudah dibaca, di-print, dan diubah.
TRAIN_CONFIG = {
    "model": "yolov8s.pt",     # small: sesuai model deteksi-barang-yolov8s
    "imgsz": 640,
    "epochs": 150,
    "patience": 30,            # early stopping
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "warmup_epochs": 5,
    # Kurangi workers agar RAM dataloader tidak meledak di laptop
    "cache": False,
    # --- Augmentasi ---
    "mosaic": 1.0,
    "mixup": 0.1,
    "copy_paste": 0.1,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 10,
    "translate": 0.1,
    "scale": 0.5,
    "fliplr": 0.5,
    "flipud": 0.0,
    # --- Sistem ---
    "workers": 2,   # 4 → 2: kurangi proses dataloader di RAM
    "pretrained": True,
    "val": True,
    "save_period": 10,
    "exist_ok": True,
}

# Urutan batch yang dicoba bila terjadi CUDA Out-Of-Memory
BATCH_FALLBACK = [16, 8, 4]


def detect_device():
    """Deteksi device otomatis. Return 'cuda' atau 'cpu' (string untuk YOLO)."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[INFO] CUDA terdeteksi: {gpu_name}")
            return "cuda"
    except Exception as e:
        print(f"[WARNING] Gagal cek CUDA ({e}). Memakai CPU.")
    print("[INFO] CUDA tidak tersedia. Memakai CPU (training akan lebih lambat).")
    return "cpu"


def ensure_yaml_fixed():
    """Pastikan data.yaml absolute dengan memanggil fix_yaml.py."""
    try:
        import fix_yaml
        ok = fix_yaml.fix_yaml()
        if not ok:
            print("[WARNING] fix_yaml gagal. Melanjutkan dengan data.yaml apa adanya.")
    except Exception as e:
        print(f"[WARNING] Tidak bisa menjalankan fix_yaml otomatis: {e}")


def print_config(device: str, batch: int):
    """Cetak konfigurasi training yang akan dipakai."""
    print("\n" + "=" * 70)
    print("KONFIGURASI TRAINING")
    print("=" * 70)
    print(f"  data         : {DATA_YAML}")
    print(f"  device       : {device}")
    print(f"  batch        : {batch}")
    print(f"  project      : {PROJECT_DIR}")
    print(f"  name         : {RUN_NAME}")
    for k, v in TRAIN_CONFIG.items():
        print(f"  {k:<12} : {v}")
    print("=" * 70 + "\n")


def save_summary(results, best_path: Path, batch_used: int, device: str):
    """Tulis ringkasan hasil training ke results_summary.txt."""
    try:
        lines = []
        lines.append("=" * 70)
        lines.append("RINGKASAN HASIL TRAINING - deteksi_barang")
        lines.append("=" * 70)
        lines.append(f"Model        : {TRAIN_CONFIG['model']}")
        lines.append(f"Device       : {device}")
        lines.append(f"Batch dipakai: {batch_used}")
        lines.append(f"Epochs (maks): {TRAIN_CONFIG['epochs']} "
                     f"(patience {TRAIN_CONFIG['patience']})")
        lines.append(f"Image size   : {TRAIN_CONFIG['imgsz']}")
        lines.append(f"Best model   : {best_path}")
        lines.append("")

        # Metrik akhir (jika tersedia dari objek results)
        try:
            box = results.box  # ultralytics Metrics
            lines.append("--- METRIK VALIDASI (akhir) ---")
            lines.append(f"  Precision (mp)  : {box.mp:.4f}")
            lines.append(f"  Recall    (mr)  : {box.mr:.4f}")
            lines.append(f"  mAP50           : {box.map50:.4f}")
            lines.append(f"  mAP50-95        : {box.map:.4f}")
            target = "TERCAPAI" if box.mp >= 0.90 else "BELUM tercapai"
            lines.append(f"  Target precision >= 0.90 : {target}")
        except Exception:
            lines.append("(Metrik detail tidak tersedia dari objek results; "
                         "lihat folder run untuk results.csv)")

        lines.append("")
        lines.append("Langkah berikutnya: python evaluate.py")
        lines.append("=" * 70)

        SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")
        print(f"[OK] Ringkasan disimpan ke: {SUMMARY_FILE}")
    except Exception as e:
        print(f"[WARNING] Gagal menyimpan summary: {e}")


def is_oom_error(err: Exception) -> bool:
    """Deteksi apakah sebuah error berkaitan dengan Out-Of-Memory GPU."""
    msg = str(err).lower()
    return ("out of memory" in msg or "cuda error" in msg
            or "cublas" in msg or "oom" in msg)


def train():
    """Jalankan training dengan fallback batch saat OOM."""
    print("=" * 70)
    print("TRAIN YOLOv8 - deteksi_barang")
    print("=" * 70)

    # 1. Fix path data.yaml otomatis
    ensure_yaml_fixed()

    if not DATA_YAML.exists():
        print(f"[ERROR] data.yaml tidak ditemukan: {DATA_YAML}")
        return False

    # 2. Import ultralytics (di sini agar fix_yaml tetap jalan walau lib belum ada)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Package 'ultralytics' belum terinstall.")
        print("        Jalankan: pip install -r requirements.txt")
        return False

    import torch  # untuk membersihkan cache saat OOM

    # 3. Deteksi device
    device = detect_device()

    # 4. Loop fallback batch
    last_error = None
    for batch in BATCH_FALLBACK:
        print_config(device, batch)
        try:
            # Model baru tiap percobaan agar state bersih
            model = YOLO(TRAIN_CONFIG["model"])

            results = model.train(
                data=str(DATA_YAML),
                device=device,
                batch=batch,
                project=str(PROJECT_DIR),
                name=RUN_NAME,
                **TRAIN_CONFIG,
            )

            # --- Sukses ---
            best_path = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
            print("\n" + "=" * 70)
            print("[OK] TRAINING SELESAI")
            print("=" * 70)
            print(f"  Best model : {best_path}")
            print(f"  Last model : {PROJECT_DIR / RUN_NAME / 'weights' / 'last.pt'}")

            save_summary(results, best_path, batch, device)
            print("\n[NEXT] Jalankan evaluasi: python evaluate.py")
            return True

        except RuntimeError as e:
            last_error = e
            if is_oom_error(e) and device == "cuda":
                print(f"\n[OOM] CUDA kehabisan memori pada batch={batch}.")
                # Bersihkan cache GPU sebelum mencoba batch lebih kecil
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                if batch != BATCH_FALLBACK[-1]:
                    print(f"[INFO] Mencoba ulang dengan batch lebih kecil...")
                    continue
                else:
                    print("[ERROR] Sudah mencapai batch terkecil namun tetap OOM.")
                    print("        Saran: turunkan imgsz (mis. 512) atau pakai yolov8s.")
            else:
                # Error lain (bukan OOM) -> hentikan & laporkan
                print(f"\n[ERROR] Training gagal (bukan OOM): {e}")
                traceback.print_exc()
            break
        except Exception as e:
            last_error = e
            print(f"\n[ERROR] Terjadi kesalahan tak terduga: {e}")
            traceback.print_exc()
            break

    print(f"\n[GAGAL] Training tidak selesai. Error terakhir: {last_error}")
    return False


if __name__ == "__main__":
    ok = train()
    if not ok:
        raise SystemExit(1)
