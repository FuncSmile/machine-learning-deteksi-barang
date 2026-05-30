# Deteksi Barang — YOLOv8 Object Detection

Proyek deteksi objek berbasis **YOLOv8m (Ultralytics)** untuk mengenali **26 kelas barang rumah tangga** khas Indonesia secara real-time menggunakan webcam, video, maupun gambar statis.

## Kelas yang Dideteksi

| # | Kelas | # | Kelas | # | Kelas |
|---|-------|---|-------|---|-------|
| 0 | HP | 9 | gunting | 18 | pulpen |
| 1 | asbak | 10 | jam tangan | 19 | ransel |
| 2 | bangku | 11 | kaca mata | 20 | sapu |
| 3 | botol | 12 | kalkulator | 21 | sendok |
| 4 | buku | 13 | korek | 22 | solasi |
| 5 | bungkus rokok | 14 | kunci | 23 | spidol |
| 6 | dompet | 15 | lampu | 24 | steples |
| 7 | galon | 16 | laptop | 25 | tas |
| 8 | gelas | 17 | mouse | | |

**Dataset:** [Roboflow — deteksi-barang-qhjji v2](https://app.roboflow.com/muhamads-workspace-o5w3d/deteksi-barang-qhjji/2) (~2.569 gambar)  
**Target kualitas:** Precision per-kelas ≥ 0.90

---

## Struktur Proyek

```
deteksi_barang/
├── datasets/
│   ├── data.yaml.example   # template path (jalankan fix_yaml.py untuk generate)
│   ├── train/              # gambar & label training
│   ├── valid/              # gambar & label validasi
│   └── test/               # gambar & label pengujian
├── fix_yaml.py             # tulis ulang path absolut di data.yaml
├── check_dataset.py        # validasi dataset sebelum training
├── train.py                # training model YOLOv8
├── evaluate.py             # evaluasi per-kelas di split test
├── predict.py              # inferensi (webcam / video / gambar)
├── eda.py                  # eksplorasi dan analisis dataset
├── clean_dataset.py        # pembersihan dataset
├── verify_cleaning.py      # verifikasi hasil pembersihan
├── requirements.txt        # dependensi Python
└── keep.txt                # catatan setup environment
```

> **Catatan:** Folder `runs/`, file `*.pt`, dan `datasets/data.yaml` (berisi path absolut) tidak dimasukkan ke repositori — lihat `.gitignore`.

---

## Setup

### 1. Clone & Buat Virtual Environment

```bash
git clone <url-repo>
cd deteksi_barang

python3 -m venv OBJECT_DETECTION
source OBJECT_DETECTION/bin/activate
```

### 2. Install Dependensi

```bash
# Jika punya GPU NVIDIA, install torch versi CUDA terlebih dahulu:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### 3. Siapkan Dataset

Letakkan dataset Roboflow (format YOLO) ke folder `datasets/`, lalu jalankan:

```bash
python fix_yaml.py
```

Script ini menulis ulang path di `data.yaml` menjadi path absolut sesuai mesin yang dipakai (wajib dijalankan setelah clone atau pindah mesin).

---

## Pipeline (jalankan berurutan)

```bash
# 1. Validasi dataset
python check_dataset.py

# 2. Training (~150 epoch, model yolov8m)
python train.py
# Hasil: runs/train/deteksi_barang_v1/weights/best.pt

# 3. Evaluasi pada split test
python evaluate.py
# Laporan: evaluation_report.txt

# 4. Inferensi
python predict.py --mode webcam                      # kamera langsung
python predict.py --mode video  --source path/video  # file video
python predict.py --mode image  --source path/gambar # file gambar
```

Tekan `q` untuk menghentikan stream webcam/video.

---

## Hyperparameter Training

| Parameter | Nilai |
|-----------|-------|
| Model | `yolov8m.pt` |
| Epoch | 150 |
| Image size | 640 |
| Optimizer | AdamW |
| Batch | 32 (auto-turun jika OOM) |
| Batch fallback | 16 → 8 → 4 |

---

## Konfigurasi Inferensi

| Parameter | Nilai |
|-----------|-------|
| Confidence threshold | 0.15 |
| IoU threshold | 0.45 |

Nilai threshold dituning rendah agar recall tinggi — sesuaikan di `predict.py` jika diperlukan.

---

## Dependensi Utama

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) `>=8.2.0`
- PyTorch `>=2.1.0`
- OpenCV `>=4.8.0`
- Python 3.14 (venv di `OBJECT_DETECTION/`)
