"""
predict.py
----------
Inference YOLOv8 untuk webcam real-time, file video, atau gambar.

Mode:
  --mode webcam              : ambil dari kamera (default index 0)
  --mode video --source x.mp4: proses file video, simpan ke runs/predict/
  --mode image --source x.jpg: proses satu gambar, simpan ke runs/predict/

Fitur overlay:
  - Bounding box + label + confidence score.
  - FPS di pojok KIRI ATAS.
  - Jumlah objek terdeteksi per class di pojok KANAN ATAS.

Kontrol:
  - Tekan 'q' untuk keluar (mode webcam / video).

Contoh:
  python predict.py --mode webcam
  python predict.py --mode webcam --camera 1
  python predict.py --mode video --source datasets/test/sample.mp4
  python predict.py --mode image --source datasets/test/images/contoh.jpg
"""

from pathlib import Path
from collections import Counter
import argparse
import time
import sys

# === KONFIGURASI PATH DASAR ===
BASE_DIR = Path(__file__).resolve().parent
BEST_MODEL = BASE_DIR / "runs" / "train" / "deteksi_barang_v1" / "weights" / "best.pt"
PREDICT_DIR = BASE_DIR / "runs" / "predict"

# === THRESHOLD INFERENCE ===
CONF_THRESHOLD = 0.15   # optimal threshold dari evaluasi
IOU_THRESHOLD = 0.45

# === WARNA & FONT OVERLAY ===
BOX_COLOR = (0, 200, 0)        # hijau (BGR)
TEXT_COLOR = (255, 255, 255)   # putih
BG_COLOR = (0, 0, 0)           # latar teks hitam
FPS_COLOR = (0, 255, 255)      # kuning


def load_model():
    """Muat model YOLO best.pt. Return objek model atau None jika gagal."""
    if not BEST_MODEL.exists():
        print(f"[ERROR] Model tidak ditemukan: {BEST_MODEL}")
        print("        Jalankan training dulu: python train.py")
        return None
    try:
        from ultralytics import YOLO
        print(f"[INFO] Memuat model: {BEST_MODEL}")
        return YOLO(str(BEST_MODEL))
    except ImportError:
        print("[ERROR] Package 'ultralytics' belum terinstall. "
              "Jalankan: pip install -r requirements.txt")
        return None
    except Exception as e:
        print(f"[ERROR] Gagal memuat model: {e}")
        return None


def draw_overlay(cv2, frame, result, names, fps):
    """Gambar box, label, FPS, dan ringkasan jumlah objek per class."""
    class_counter = Counter()

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            # Koordinat & info
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cname = names[cls_id] if cls_id < len(names) else f"id_{cls_id}"
            class_counter[cname] += 1

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

            # Label + confidence dengan latar agar mudah dibaca
            label = f"{cname} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), BOX_COLOR, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    # --- FPS pojok KIRI ATAS ---
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, FPS_COLOR, 2, cv2.LINE_AA)

    # --- Jumlah objek per class pojok KANAN ATAS ---
    h, w = frame.shape[:2]
    total = sum(class_counter.values())
    lines = [f"Objek: {total}"] + [f"{c}: {n}" for c, n in class_counter.most_common()]
    y = 25
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        x = w - tw - 12
        cv2.rectangle(frame, (x - 4, y - th - 4), (w - 4, y + 4), BG_COLOR, -1)
        cv2.putText(frame, line, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)
        y += th + 10

    return frame


def run_stream(cv2, model, names, cap, save_path=None):
    """Loop umum untuk webcam / video. Jika save_path diisi -> rekam output."""
    if not cap.isOpened():
        print("[ERROR] Tidak bisa membuka sumber video/kamera.")
        return False

    # Siapkan VideoWriter jika perlu menyimpan
    writer = None
    if save_path is not None:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps_src = cap.get(cv2.CAP_PROP_FPS)
        fps_out = fps_src if fps_src and fps_src > 0 else 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        PREDICT_DIR.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(save_path), fourcc, fps_out, (w, h))
        print(f"[INFO] Output video -> {save_path}")

    print("[INFO] Tekan 'q' untuk keluar.")
    prev_t = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[INFO] Stream selesai / frame habis.")
                break

            # Inference
            results = model.predict(frame, conf=CONF_THRESHOLD,
                                    iou=IOU_THRESHOLD, verbose=False)
            result = results[0]

            # Hitung FPS
            now = time.time()
            fps = 1.0 / (now - prev_t) if now > prev_t else 0.0
            prev_t = now

            frame = draw_overlay(cv2, frame, result, names, fps)

            if writer is not None:
                writer.write(frame)

            cv2.imshow("Deteksi Barang - YOLOv8", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Dihentikan oleh user (q).")
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    return True


def predict_webcam(cv2, model, names, camera_index):
    """Mode webcam real-time."""
    print(f"[INFO] Membuka kamera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    return run_stream(cv2, model, names, cap, save_path=None)


def predict_video(cv2, model, names, source):
    """Mode video file -> simpan ke runs/predict/output_video.mp4."""
    src = Path(source)
    if not src.exists():
        print(f"[ERROR] File video tidak ditemukan: {src}")
        return False
    cap = cv2.VideoCapture(str(src))
    save_path = PREDICT_DIR / "output_video.mp4"
    return run_stream(cv2, model, names, cap, save_path=save_path)


def predict_image(cv2, model, names, source):
    """Mode gambar -> simpan ke runs/predict/output_image.jpg."""
    src = Path(source)
    if not src.exists():
        print(f"[ERROR] File gambar tidak ditemukan: {src}")
        return False

    frame = cv2.imread(str(src))
    if frame is None:
        print(f"[ERROR] Gagal membaca gambar: {src}")
        return False

    t0 = time.time()
    results = model.predict(frame, conf=CONF_THRESHOLD,
                            iou=IOU_THRESHOLD, verbose=False)
    fps = 1.0 / (time.time() - t0) if time.time() > t0 else 0.0
    frame = draw_overlay(cv2, frame, results[0], names, fps)

    PREDICT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PREDICT_DIR / "output_image.jpg"
    cv2.imwrite(str(save_path), frame)
    print(f"[OK] Hasil disimpan: {save_path}")

    # Tampilkan (tutup dengan tombol apa saja)
    try:
        cv2.imshow("Deteksi Barang - YOLOv8", frame)
        print("[INFO] Tekan tombol apa saja pada jendela untuk menutup.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        # Lingkungan tanpa display (headless) -> lewati tampilan
        pass
    return True


def load_class_names():
    """Ambil names dari data.yaml. Return list (boleh kosong)."""
    try:
        import yaml
        data_yaml = BASE_DIR / "datasets" / "data.yaml"
        with data_yaml.open("r", encoding="utf-8") as f:
            names = yaml.safe_load(f).get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        return names
    except Exception:
        return []


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(
        description="Inference YOLOv8 deteksi_barang (webcam/video/image).")
    parser.add_argument("--mode", required=True,
                        choices=["webcam", "video", "image"],
                        help="Mode inference: webcam | video | image")
    parser.add_argument("--source", default=None,
                        help="Path file untuk mode video/image.")
    parser.add_argument("--camera", type=int, default=0,
                        help="Index kamera untuk mode webcam (default 0).")
    return parser.parse_args()


def main():
    args = parse_args()

    # Import OpenCV di sini agar pesan error jelas jika belum terinstall
    try:
        import cv2
    except ImportError:
        print("[ERROR] Package 'opencv-python' belum terinstall. "
              "Jalankan: pip install -r requirements.txt")
        sys.exit(1)

    model = load_model()
    if model is None:
        sys.exit(1)

    # names dari model lebih akurat; fallback ke data.yaml
    names = model.names if hasattr(model, "names") and model.names else load_class_names()
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    print(f"[INFO] Mode: {args.mode} | conf={CONF_THRESHOLD} | iou={IOU_THRESHOLD}")

    if args.mode == "webcam":
        ok = predict_webcam(cv2, model, names, args.camera)
    elif args.mode == "video":
        if not args.source:
            print("[ERROR] Mode video membutuhkan --source path/ke/video.mp4")
            sys.exit(1)
        ok = predict_video(cv2, model, names, args.source)
    else:  # image
        if not args.source:
            print("[ERROR] Mode image membutuhkan --source path/ke/gambar.jpg")
            sys.exit(1)
        ok = predict_image(cv2, model, names, args.source)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
