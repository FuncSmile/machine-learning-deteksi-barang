# PRESENTASI — Sistem Deteksi Seragam Sekolah Berbasis YOLOv8

---

## SLIDE 1 — HALAMAN JUDUL

# Sistem Deteksi Kepatuhan Seragam Sekolah
## Berbasis Computer Vision YOLOv8 Secara Real-Time

**Teknologi:** YOLOv8s · Python · OpenCV · Ultralytics  
**Tanggal:** Mei–Juni 2026

---

## SLIDE 2 — LATAR BELAKANG MASALAH

### Masalah yang Ditemukan

> Pelanggaran seragam sekolah masih sering terjadi dan sulit dipantau secara menyeluruh.

| # | Masalah |
|---|---------|
| 1 | Pemeriksaan seragam dilakukan **manual** oleh guru / satpam — tidak konsisten |
| 2 | Cakupan pemantauan **terbatas** waktu dan sumber daya manusia |
| 3 | Tidak ada **catatan digital** pelanggaran yang terstruktur |
| 4 | Sulit membedakan jenis pakaian yang mirip secara visual (rok abu-abu vs rok hitam, dll.) |

**Dampak:** Penegakan aturan seragam tidak merata → kedisiplinan siswa menurun

---

## SLIDE 3 — RUMUSAN MASALAH & TUJUAN

### Rumusan Masalah
- Bagaimana membangun sistem yang mampu **mendeteksi dan mengklasifikasikan** jenis pakaian seragam sekolah secara otomatis?
- Bagaimana sistem dapat beroperasi **real-time** dari kamera tanpa bergantung pada tenaga manusia?

### Tujuan
- Membangun model deteksi objek untuk **15 kategori pakaian** seragam sekolah
- Mengimplementasikan inference real-time melalui **webcam / video / IP kamera HP**
- Menghasilkan output visual berupa **bounding box + label + confidence score**

---

## SLIDE 4 — SOLUSI & METODOLOGI

### Pendekatan: Object Detection dengan YOLOv8

```
[Kamera / Video]
      │
      ▼
[YOLOv8s Model]  ──→  Deteksi & Klasifikasi 15 Kelas Seragam
      │
      ▼
[Overlay Real-Time]  ──→  Bounding Box + Label + FPS + Jumlah per Kelas
```

### Mengapa YOLOv8s?

| Kriteria | YOLOv8s |
|----------|---------|
| Kecepatan | Tinggi — cocok real-time |
| Akurasi | Cukup baik untuk 15 kelas |
| Resource | Efisien di GPU maupun CPU |
| Ekosistem | Ultralytics — mudah deploy |

---

## SLIDE 5 — ALUR KERJA SISTEM (PIPELINE LENGKAP)

### Tahapan dari Awal sampai Sistem Berjalan

```
┌─────────────────────────────────────────────────────────────────────┐
│  TAHAP 1          TAHAP 2          TAHAP 3          TAHAP 4         │
│                                                                     │
│  📷 Kumpul       🏷️  Anotasi       ⚙️  Preprocessing   🧠 Training  │
│  Dataset    →   & Labeling    →   & Augmentasi   →   Model         │
│                                                                     │
│  TAHAP 5          TAHAP 6                                           │
│                                                                     │
│  📊 Evaluasi  →  🚀 Inference &                                     │
│  Model           Deployment                                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## SLIDE 5A — TAHAP 1: PENGUMPULAN DATASET

### Mengumpulkan Data Gambar Seragam

**Tujuan:** Menyediakan data visual nyata sebagai bahan belajar model

#### Sumber Data yang Digunakan

| Sumber | Keterangan |
|--------|------------|
| Foto & video siswa nyata | Diambil di lingkungan sekolah |
| Berbagai sudut pengambilan | Depan, samping, jarak dekat & jauh |
| Berbagai kondisi pencahayaan | Dalam ruang & luar ruangan |
| Roboflow (platform dataset) | Upload, simpan, dan kelola anotasi |

#### Apa yang Dikumpulkan?

- Gambar siswa **mengenakan 15 kategori** seragam yang berbeda
- **2.569 gambar** sumber sebelum augmentasi
- Format file: `.jpg` / `.png` resolusi bervariasi

#### Tantangan
- Beberapa seragam **mirip secara visual** (contoh: celana abu-abu vs rok abu-abu)
- Perbedaan pencahayaan mempengaruhi warna seragam
- Butuh banyak variasi posisi agar model tidak overfitting

---

## SLIDE 5B — TAHAP 2: ANOTASI & PELABELAN

### Memberi Label pada Setiap Objek di Gambar

**Tujuan:** Memberitahu model "di mana" objek berada dan "apa" kategorinya

#### Proses Anotasi dengan Roboflow

```
Gambar asli                Hasil anotasi
┌──────────────┐           ┌──────────────┐
│              │           │ ┌──────────┐ │
│   [foto      │   ──→     │ │ seragam  │ │  ← bounding box
│    siswa]    │  Roboflow │ │ putih    │ │     + label kelas
│              │           │ └──────────┘ │
└──────────────┘           └──────────────┘
```

#### Format Label YOLOv8

Setiap objek dicatat dalam file `.txt` berisi:
```
<class_id>  <x_center>  <y_center>  <width>  <height>
     3         0.512       0.641      0.230     0.410
```
*(semua nilai dinormalisasi 0–1 terhadap ukuran gambar)*

#### 15 Kelas yang Dianotasi

> Batik · Almamater · Celana Abu-Abu · Celana Hitam · Dasi Abu-Abu  
> Dasi Bebas · Rok Abu-Abu · Rok Bebas · Rok Hitam · Sepatu Bebas  
> Sepatu Hitam · Seragam Bebas · Seragam Pramuka · Seragam Putih · Tali Pinggang

---

## SLIDE 5C — TAHAP 3: PREPROCESSING & AUGMENTASI

### Menyiapkan Data agar Siap Dilatihkan

**Tujuan:** Menstandardisasi input dan memperbanyak variasi data secara buatan

#### Preprocessing (Normalisasi)

| Langkah | Detail |
|---------|--------|
| Resize gambar | Semua diubah ke **640 × 640** piksel (standar YOLO) |
| Auto-orientasi | Koreksi rotasi berdasarkan metadata EXIF kamera |
| Auto-kontras | Penyesuaian kecerahan otomatis |

#### Augmentasi (Perbanyak Variasi)

```
Gambar Asli  ──→  ┌── Flip Horizontal (50%)
                  ├── Rotasi Acak (±15°)
                  ├── Brightness ±15%
                  ├── Gaussian Blur
                  ├── Mosaic (gabung 4 gambar)
                  └── MixUp (blend 2 gambar)
                  
Hasil: Variasi data lebih banyak → model lebih robust
```

#### Hasil Pembagian Dataset

| Split | Jumlah Gambar | Fungsi |
|-------|--------------|--------|
| Training | **762** | Bahan belajar model |
| Validasi | **42** | Pantau selama training |
| Testing | **40** | Uji akhir model final |

---

## SLIDE 5D — TAHAP 4: TRAINING MODEL YOLOv8

### Melatih Model agar Bisa Mendeteksi Seragam

**Tujuan:** Model belajar mengenali pola visual dari setiap kelas seragam

#### Alur Training

```
Data Training (762 gambar)
        │
        ▼
┌───────────────────────────────┐
│        YOLOv8s Model          │
│                               │
│  Forward Pass → Prediksi      │
│  Hitung Loss (kesalahan)      │
│  Backward Pass → Update bobot │
│                               │
│  Ulangi per Epoch (maks. 100) │
└───────────────────────────────┘
        │
        ▼
 Simpan bobot terbaik → best.pt
```

#### Konfigurasi Training

| Parameter | Nilai |
|-----------|-------|
| Model dasar | `yolov8s.pt` (pre-trained COCO) |
| Epochs | 100 (berhenti di epoch 50 via Early Stopping) |
| Batch size | 32 |
| Optimizer | AdamW / SGD (otomatis) |
| Platform | Google Colab (GPU CUDA) |

#### Early Stopping
> Jika metrik validasi tidak membaik selama **20 epoch berturut-turut**, training otomatis berhenti.  
> **→ Model terbaik tersimpan di:** `runs/train/deteksi_barang_v1/weights/best.pt`

---

## SLIDE 5E — TAHAP 5: EVALUASI MODEL

### Mengukur Performa Model pada Data Baru

**Tujuan:** Memastikan model benar-benar bisa generalisasi, bukan hanya hafal data training

#### Proses Evaluasi

```
Data Testing (40 gambar — belum pernah dilihat model)
        │
        ▼
   Model best.pt
        │
        ▼
   Hasil Prediksi
        │
        ▼
 Bandingkan dengan Label Asli
        │
        ▼
 Hitung: Precision · Recall · F1 · mAP
```

#### Metrik Evaluasi yang Digunakan

| Metrik | Arti Sederhana |
|--------|---------------|
| **Precision** | Seberapa sering prediksi model itu benar |
| **Recall** | Seberapa banyak objek nyata yang berhasil ditemukan |
| **mAP@50** | Rata-rata akurasi di semua kelas pada IoU ≥ 50% |
| **mAP@50-95** | Akurasi lebih ketat di berbagai threshold IoU |

#### Hasil Evaluasi Final

| Metrik | Nilai |
|--------|-------|
| Precision | **73.6%** |
| Recall | **75.1%** |
| mAP@50 | **74.9%** |
| mAP@50-95 | **42.9%** |

---

## SLIDE 5F — TAHAP 6: INFERENCE & DEPLOYMENT

### Menjalankan Model di Dunia Nyata

**Tujuan:** Model yang sudah terlatih digunakan untuk mendeteksi seragam secara langsung

#### Alur Inference Real-Time

```
Input (Webcam / Video / Gambar / IP Kamera HP)
        │
        ▼
  Ambil frame per frame
        │
        ▼
  Kirim ke Model YOLOv8s (best.pt)
        │
        ▼
  Hasil: Bounding box + Label + Confidence Score
        │
        ▼
  Gambar overlay dengan OpenCV
        │
        ▼
  Tampilkan ke layar (real-time / simpan output)
```

#### Mode Penggunaan

```bash
# Deteksi langsung dari webcam laptop
python predict.py --mode webcam

# Deteksi dari file gambar
python predict.py --mode image --source foto_siswa.jpg

# Deteksi dari file video
python predict.py --mode video --source rekaman.mp4
```

#### Output yang Dihasilkan

| Output | Keterangan |
|--------|------------|
| Bounding box berwarna | Kotak di setiap objek terdeteksi |
| Label + confidence | Nama kelas dan tingkat keyakinan model |
| FPS (frame per detik) | Kecepatan pemrosesan real-time |
| Jumlah per kelas | Hitungan objek tiap kategori di frame |

---

## SLIDE 6 — DATASET

### Sumber Data
- **Asal:** Rekaman video & foto siswa nyata (Roboflow export, Mei 2026)
- **Total gambar:** 2.569 gambar sumber → di-augmentasi menjadi dataset final
- **Format:** YOLOv8 (bounding box annotation)

### Pembagian Dataset

| Split | Jumlah File |
|-------|-------------|
| Training | **762** gambar |
| Validasi | **42** gambar |
| Testing | **40** gambar |
| **Total** | **844** file berlabel |

### Pre-processing & Augmentasi (Roboflow)
- Resize ke **640 × 640** piksel
- Auto-orientasi + auto-contrast
- Flip horizontal (50%), rotasi acak, brightness ±15%, Gaussian blur

---

## SLIDE 7 — KATEGORI YANG DIDETEKSI (15 KELAS)

| # | Kelas | # | Kelas |
|---|-------|---|-------|
| 1 | Batik | 9 | Rok Hitam |
| 2 | Alamamater | 10 | Sepatu Bebas |
| 3 | Celana Abu-Abu | 11 | Sepatu Hitam |
| 4 | Celana Hitam | 12 | Seragam Bebas |
| 5 | Dasi Abu-Abu | 13 | Seragam Pramuka |
| 6 | Dasi Bebas | 14 | Seragam Putih |
| 7 | Rok Abu-Abu | 15 | Tali Pinggang |
| 8 | Rok Bebas | | |

> Mencakup seragam harian, pramuka, batik, almamater, dan aksesori wajib

---

## SLIDE 8 — KONFIGURASI TRAINING

### Hyperparameter Model

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| Model | YOLOv8s | Small — cepat & ringan |
| Epochs | 100 (early stop di 50) | Patience = 20 epoch |
| Batch Size | 32 | GPU Google Colab |
| Image Size | 640 × 640 | Resolusi standar YOLO |
| Optimizer | Auto (SGD/AdamW) | Dipilih otomatis |
| Augmentasi | Mosaic, MixUp, Flip | Variasi data training |
| Device | GPU (CUDA) | Google Colab GPU |

### Tools & Platform
- **Training:** Google Colab (GPU)
- **Framework:** Ultralytics YOLOv8
- **Anotasi Dataset:** Roboflow
- **Inference:** Python + OpenCV (lokal / webcam / IP kamera)

---

## SLIDE 9 — PROSES TRAINING (KURVA BELAJAR)

### Perkembangan Metrik per Epoch

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| 1 | 0.507 | 0.516 | 0.467 | 0.199 |
| 10 | 0.602 | 0.632 | 0.653 | 0.336 |
| 20 | ~0.69 | ~0.70 | ~0.72 | ~0.40 |
| 30 | ~0.72 | ~0.72 | ~0.76 | ~0.43 |
| 50 | **0.736** | **0.751** | **0.749** | **0.429** |

### Tren
- Loss training **terus menurun** → model belajar dengan baik
- Precision & Recall **stabil naik** dari epoch 1 → 50
- Model berhenti (early stopping) di epoch 50 sebelum 100 epoch

---

## SLIDE 10 — HASIL EVALUASI MODEL

### Metrik Final (Epoch Terbaik)

| Metrik | Nilai | Interpretasi |
|--------|-------|--------------|
| **Precision** | **73.6%** | Dari semua prediksi positif, 73.6% benar |
| **Recall** | **75.1%** | Dari semua objek nyata, 75.1% berhasil terdeteksi |
| **mAP@50** | **74.9%** | Akurasi rata-rata di threshold IoU 50% |
| **mAP@50-95** | **42.9%** | Akurasi ketat di berbagai threshold IoU |

### Konfigurasi Inference

| Parameter | Nilai | Alasan |
|-----------|-------|--------|
| Confidence Threshold | 0.15 | Recall tinggi — tidak melewatkan objek |
| IoU Threshold | 0.45 | Mengurangi duplikasi deteksi |

---

## SLIDE 11 — DEMO SISTEM

### Cara Penggunaan

```bash
# Mode webcam (kamera laptop)
python predict.py --mode webcam

# Mode gambar
python predict.py --mode image --source foto_siswa.jpg

# Mode video
python predict.py --mode video --source rekaman.mp4

# Mode IP kamera HP (via Wi-Fi)
python predict.py --mode ip --ip 192.168.1.5 --port 8080
```

### Output Sistem
- **Bounding box** berwarna di setiap objek terdeteksi
- **Label + confidence score** di atas setiap kotak
- **FPS** real-time di pojok kiri atas
- **Jumlah objek per kelas** di pojok kanan atas

---

## SLIDE 12 — KESIMPULAN

### Yang Berhasil Dicapai

| # | Capaian |
|---|---------|
| ✅ | Model berhasil mendeteksi **15 kategori** seragam sekolah |
| ✅ | Sistem berjalan **real-time** (webcam, video, IP kamera) |
| ✅ | Precision **73.6%**, Recall **75.1%**, mAP50 **74.9%** |
| ✅ | Dataset dibangun dari data nyata siswa (762 gambar training) |
| ✅ | Dapat dijalankan di laptop biasa (CPU) maupun GPU |

### Keterbatasan & Rencana Perbaikan

| Keterbatasan | Solusi ke Depan |
|--------------|-----------------|
| mAP50 belum mencapai 90% | Tambah data training per kelas |
| Dataset validasi kecil (42 gambar) | Perbanyak data & augmentasi |
| Beberapa kelas mirip visual | Fine-tune threshold per kelas |
| Training terhenti di 50 epoch | Perpanjang training + lebih banyak data |

---

## SLIDE 13 — PENUTUP

> ### *"Teknologi computer vision dapat membantu sekolah menegakkan aturan seragam secara otomatis, konsisten, dan tanpa bergantung penuh pada tenaga manusia."*

### Langkah Selanjutnya
1. Perbanyak dataset → target **>2.000 gambar per kelas**
2. Training ulang hingga Precision ≥ 90% per kelas
3. Integrasi dengan sistem absensi / portal sekolah
4. Deploy di kamera CCTV sekolah

---

**Terima Kasih**

*Dibangun dengan: Python · YOLOv8 · Ultralytics · OpenCV · Roboflow*
