<!-- ═══════════════════════════════════════════════════════════════════
     PPT.md  —  Panduan Desain Slide Presentasi
     Sistem Deteksi Kepatuhan Seragam Sekolah Berbasis YOLOv8
     Audien: Guru, Staf Sekolah, Komite / Non-teknis & Semi-teknis
     Palet warna: Biru Navy #1A3C6E  |  Biru Aksen #2E8FD4  |  Abu-Abu #F4F6F9
     Font: Judul → Poppins Bold 40 pt  |  Konten → Poppins Regular 20 pt
     ═══════════════════════════════════════════════════════════════════ -->

---

# SLIDE 1  ·  HALAMAN JUDUL

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│          [Logo Sekolah / Institusi — pojok kiri atas]               │
│                                                                     │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│                                                                     │
│      SISTEM DETEKSI KEPATUHAN SERAGAM SEKOLAH                       │
│      Berbasis Computer Vision & YOLOv8 Secara Real-Time             │
│                                                                     │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│                                                                     │
│      Nama Penyaji      :  [Nama Lengkap]                            │
│      Program Studi     :  [Nama Prodi / Instansi]                   │
│      Tanggal           :  Juni 2026                                 │
│                                                                     │
│      Dibangun dengan:  Python  ·  YOLOv8  ·  OpenCV  ·  Roboflow   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

> **Catatan desainer:** Gunakan foto siswa berseragam (buram/samar) sebagai background dengan overlay warna navy 70% opacity. Logo di pojok kanan bawah.

---

# SLIDE 2  ·  AGENDA PRESENTASI

```
┌─────────────────────────────────────────────────────────────────────┐
│  AGENDA                                              [01 dari 13]   │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│   01  Latar Belakang Masalah                                        │
│   02  Rumusan Masalah & Tujuan                                      │
│   03  Solusi — Teknologi YOLOv8                                     │
│   04  Alur Kerja Sistem (Pipeline)                                  │
│   05  Dataset & Kategori Seragam                                    │
│   06  Proses Training Model                                         │
│   07  Hasil Evaluasi                                                │
│   08  Demo Sistem Nyata                                             │
│   09  Kesimpulan & Rencana ke Depan                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

> **Catatan desainer:** Setiap angka agenda diberi kotak berwarna biru aksen. Saat slide aktif, beri highlight kuning pada nomor yang sedang dibahas.

---

# SLIDE 3  ·  LATAR BELAKANG MASALAH

```
┌─────────────────────────────────────────────────────────────────────┐
│  MASALAH YANG ADA SAAT INI                          [02 dari 13]   │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  "Pemeriksaan seragam masih dilakukan secara manual,                │
│   tidak konsisten, dan tidak terdokumentasi."                       │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │  MASALAH 1   │  │  MASALAH 2   │  │  MASALAH 3   │             │
│  │              │  │              │  │              │             │
│  │  Pemeriksaan │  │  Cakupan     │  │  Tidak ada   │             │
│  │  MANUAL oleh │  │  pemantauan  │  │  catatan     │             │
│  │  guru/satpam │  │  TERBATAS    │  │  pelanggaran │             │
│  │  (tdk merata)│  │  waktu & SDM │  │  DIGITAL     │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  DAMPAK: Penegakan aturan tidak konsisten                    │  │
│  │          → Kedisiplinan siswa menurun                        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

> **Catatan desainer:** 3 kotak masalah pakai ikon (clipboard, kamera-coret, folder-x). Kotak DAMPAK pakai warna merah muda agar mencolok. Animasi: kotak muncul satu per satu (fly-in dari bawah).

---

# SLIDE 4  ·  RUMUSAN MASALAH & TUJUAN

```
┌─────────────────────────────────────────────────────────────────────┐
│  RUMUSAN MASALAH & TUJUAN                           [03 dari 13]   │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  RUMUSAN MASALAH                 TUJUAN PENELITIAN                  │
│  ────────────────────            ─────────────────                  │
│                                                                     │
│  Bagaimana membangun             Membangun model deteksi            │
│  sistem yang mampu               objek untuk 15 KATEGORI           │
│  mendeteksi & mengklasifikasi    pakaian seragam sekolah            │
│  seragam secara OTOMATIS?                                           │
│                                                                     │
│  Bagaimana sistem dapat          Mengimplementasikan inference      │
│  beroperasi REAL-TIME            real-time via webcam,              │
│  tanpa bergantung pada           video, maupun IP kamera            │
│  tenaga manusia?                                                    │
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│  HASIL YANG DIHARAPKAN:                                             │
│  Output visual → Bounding Box + Label Kelas + Confidence Score      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

> **Catatan desainer:** Bagi slide menjadi dua kolom (kiri: masalah dengan ikon tanda tanya, kanan: tujuan dengan ikon target/checklist). Gunakan warna berbeda untuk setiap kolom (biru muda vs biru tua).

---

# SLIDE 5  ·  SOLUSI — MENGAPA YOLOV8?

```
┌─────────────────────────────────────────────────────────────────────┐
│  SOLUSI: OBJECT DETECTION dengan YOLOv8                [04 dari 13] │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  YOLOv8 (You Only Look Once, v8)                                    │
│  Algoritma deteksi objek yang memindai seluruh gambar               │
│  SEKALI SAJA — menghasilkan deteksi yang cepat & akurat.            │
│                                                                     │
│  ┌──────────────┬─────────────────────────────────────────────┐    │
│  │  KRITERIA    │  KEUNGGULAN YOLOv8s                         │    │
│  ├──────────────┼─────────────────────────────────────────────┤    │
│  │  Kecepatan   │  Tinggi — mampu proses 30+ FPS (real-time)  │    │
│  │  Akurasi     │  Baik untuk 15 kelas seragam                │    │
│  │  Sumber Daya │  Efisien di laptop biasa (CPU) & GPU        │    │
│  │  Kemudahan   │  Ekosistem Ultralytics — mudah digunakan    │    │
│  │  Open Source │  Gratis, komunitas besar, aktif dikembangkan│    │
│  └──────────────┴─────────────────────────────────────────────┘    │
│                                                                     │
│  [Kamera] ──> [YOLOv8s] ──> [Deteksi & Kelas] ──> [Overlay Visual] │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

> **Catatan desainer:** Sertakan logo Ultralytics di sudut. Gunakan diagram sederhana di bawah tabel: 4 kotak terhubung panah dengan ikon (kamera → otak/AI → label → layar monitor).

---

# SLIDE 6  ·  PIPELINE LENGKAP SISTEM

```
┌─────────────────────────────────────────────────────────────────────┐
│  ALUR KERJA SISTEM (6 TAHAP)                        [05 dari 13]   │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│   TAHAP 1        TAHAP 2        TAHAP 3        TAHAP 4              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐         │
│  │ Kumpul  │    │ Anotasi │    │ Prepro- │    │Training │         │
│  │ Dataset │───>│ & Label │───>│ cessing │───>│  Model  │         │
│  │         │    │         │    │         │    │ YOLOv8s │         │
│  │2.569 img│    │Roboflow │    │640x640px│    │150 epoch│         │
│  └─────────┘    └─────────┘    └─────────┘    └────┬────┘         │
│                                                     │              │
│                              TAHAP 6      TAHAP 5  │              │
│                          ┌──────────┐   ┌──────────┤              │
│                          │Inference │   │ Evaluasi │              │
│                          │Real-Time │<──│  Model   │<─────────────│
│                          │          │   │ best.pt  │              │
│                          │Webcam/Vid│   │mAP50:74.9│              │
│                          └──────────┘   └──────────┘              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

> **Catatan desainer:** Setiap tahap dibuat sebagai kartu (card) dengan ikon unik. Gunakan animasi "Sequential Appear" saat presentasi agar audien mengikuti alur per tahap. Warna tahap 1-3 = biru muda, tahap 4 = biru tua, tahap 5-6 = hijau (hasil).

---

# SLIDE 7  ·  DATASET & KATEGORI SERAGAM

```
┌─────────────────────────────────────────────────────────────────────┐
│  DATASET & 15 KATEGORI SERAGAM                      [06 dari 13]   │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  SUMBER DATA                     PEMBAGIAN DATASET                  │
│  ──────────────────────          ──────────────────                 │
│  Foto & video siswa nyata        Training   :  762 gambar  (90.3%)  │
│  Berbagai sudut & jarak          Validasi   :   42 gambar  ( 5.0%)  │
│  Kondisi dalam & luar ruang      Testing    :   40 gambar  ( 4.7%)  │
│  Platform: Roboflow              ─────────────────────────────────  │
│                                  Total sumber:  2.569 gambar        │
│  ─────────────────────────────────────────────────────────────────  │
│  15 KATEGORI YANG DIDETEKSI                                         │
│  ─────────────────────────────────────────────────────────────────  │
│  Batik           │ Almamater        │ Celana Abu-Abu                 │
│  Celana Hitam    │ Dasi Abu-Abu     │ Dasi Bebas                     │
│  Rok Abu-Abu     │ Rok Bebas        │ Rok Hitam                      │
│  Sepatu Bebas    │ Sepatu Hitam     │ Seragam Bebas                  │
│  Seragam Pramuka │ Seragam Putih    │ Tali Pinggang                  │
│                                                                     │
│  Mencakup: Seragam harian · Pramuka · Batik · Almamater · Aksesori  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

> **Catatan desainer:** Kiri: info dataset + pie chart sederhana (train/val/test). Kanan: 15 kelas ditampilkan sebagai chip/badge berwarna. Setiap badge bisa diberi foto kecil contoh seragamnya.

---

# SLIDE 8  ·  PREPROCESSING & AUGMENTASI DATA

```
┌─────────────────────────────────────────────────────────────────────┐
│  PERSIAPAN DATA                                     [07 dari 13]   │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  PREPROCESSING (Standarisasi)    AUGMENTASI (Perbanyak Variasi)     │
│  ────────────────────────────    ──────────────────────────────     │
│                                                                     │
│  Resize → 640 x 640 piksel       Flip Horizontal (50%)             │
│  Auto-orientasi (EXIF)           Rotasi Acak (±10°)                 │
│  Normalisasi kontras             Brightness ±15%                    │
│                                  Mosaic (gabung 4 gambar)           │
│                                  MixUp (blend 2 gambar)             │
│                                  HSV Color Jitter                   │
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│  Gambar Asli       Setelah Preprocessing      Setelah Augmentasi    │
│  [foto-asli]   →   [640×640, auto-orient]  →  [variasi berganda]   │
│  ─────────────────────────────────────────────────────────────────  │
│  Tujuan: Model lebih TANGGUH terhadap variasi kondisi nyata         │
│  (pencahayaan berbeda, posisi kamera, jarak, sudut pandang)         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

> **Catatan desainer:** Tampilkan contoh visual nyata: 1 gambar asli → versi preprocessing → 4 versi augmentasi yang berbeda. Ini sangat membantu audien non-teknis memahami prosesnya.

---

# SLIDE 9  ·  PROSES TRAINING MODEL

```
┌─────────────────────────────────────────────────────────────────────┐
│  TRAINING MODEL YOLOv8s                             [08 dari 13]   │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  KONFIGURASI TRAINING                                               │
│  ┌─────────────────┬──────────────────────────────────────────┐    │
│  │  Model Dasar    │  YOLOv8s  (pre-trained ImageNet/COCO)    │    │
│  │  Epoch Maks.    │  150  (berhenti di 50 via Early Stopping) │    │
│  │  Batch Size     │  16 – 32  (otomatis sesuai RAM GPU)       │    │
│  │  Resolusi Input │  640 × 640 piksel                         │    │
│  │  Optimizer      │  AdamW  (lr = 0.001)                      │    │
│  │  Platform       │  Google Colab / Laptop GPU                │    │
│  └─────────────────┴──────────────────────────────────────────┘    │
│                                                                     │
│  KURVA BELAJAR                                                      │
│  ─────────────────────────────────────────────────────────────────  │
│  Epoch   1  :  Precision 50.7%   Recall 51.6%   mAP50 46.7%        │
│  Epoch  10  :  Precision 60.2%   Recall 63.2%   mAP50 65.3%        │
│  Epoch  30  :  Precision 72.0%   Recall 72.0%   mAP50 76.0%        │
│  Epoch  50  :  Precision 73.6%   Recall 75.1%   mAP50 74.9%        │
│  (BERHENTI — Early Stopping, tidak ada peningkatan 20 epoch)        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

> **Catatan desainer:** Kurva belajar ditampilkan sebagai line chart animasi (epoch bertambah satu per satu). Beri garis putus-putus merah di epoch 50 bertulis "Model Terbaik Tersimpan". Audien bisa melihat secara visual bagaimana model "belajar".

---

# SLIDE 10  ·  HASIL EVALUASI MODEL

```
┌─────────────────────────────────────────────────────────────────────┐
│  HASIL EVALUASI — MODEL FINAL                       [09 dari 13]   │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  ┌─────────────┬────────────┬──────────────────────────────────┐   │
│  │  METRIK     │   NILAI    │  APA ARTINYA?                    │   │
│  ├─────────────┼────────────┼──────────────────────────────────┤   │
│  │  Precision  │   73.6%    │  Dari semua deteksi, 73.6% BENAR │   │
│  │  Recall     │   75.1%    │  75.1% objek nyata DITEMUKAN     │   │
│  │  mAP@50     │   74.9%    │  Akurasi rata-rata semua kelas   │   │
│  │  mAP@50-95  │   42.9%    │  Akurasi dengan standar ketat    │   │
│  └─────────────┴────────────┴──────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  KONFIGURASI INFERENCE                                       │  │
│  │  Confidence Threshold  :  0.15  (prioritaskan menemukan obj) │  │
│  │  IoU Threshold         :  0.45  (kurangi deteksi ganda)      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Target Precision >= 90%  :  BELUM tercapai → perlu data lebih     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

> **Catatan desainer:** Setiap metrik ditampilkan sebagai gauge/speedometer berwarna: 73-75% = kuning-hijau. Beri konteks sederhana: "Artinya: dari 100 seragam yang ada di frame, sistem berhasil menemukan 75 di antaranya." Jangan hanya angka — jelaskan maknanya dalam kalimat sehari-hari.

---

# SLIDE 11  ·  DEMO SISTEM NYATA

```
┌─────────────────────────────────────────────────────────────────────┐
│  DEMO SISTEM DETEKSI SERAGAM                        [10 dari 13]   │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  TAMPILAN OUTPUT SISTEM (contoh screenshot)                         │
│  ─────────────────────────────────────────────────────────────────  │
│  ┌─────────────────────────────────┐  ┌────────────────────────┐   │
│  │  FPS: 24.3                      │  │  OBJEK TERDETEKSI:     │   │
│  │                                 │  │  Seragam Putih  :  2   │   │
│  │  ┌──────────────────────┐       │  │  Celana Abu-Abu :  2   │   │
│  │  │  seragam_putih  0.87 │       │  │  Sepatu Hitam   :  1   │   │
│  │  └──────────────────────┘       │  │  Tali Pinggang  :  1   │   │
│  │  [foto/video siswa]             │  └────────────────────────┘   │
│  │  ┌──────────────────────┐       │                               │
│  │  │  celana_abu_abu 0.79 │       │  CARA PENGGUNAAN:             │
│  │  └──────────────────────┘       │  ─────────────────────────    │
│  └─────────────────────────────────┘  Webcam  : python predict.py  │
│                                         --mode webcam              │
│                                       Video   : --mode video       │
│                                         --source rekaman.mp4       │
│                                       Gambar  : --mode image       │
│                                         --source foto.jpg          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

> **Catatan desainer:** Ini slide paling penting bagi audien non-teknis. Jika memungkinkan, jalankan demo LIVE saat presentasi. Jika tidak, tampilkan GIF atau video pendek rekaman sistem berjalan (15-30 detik). Screenshot harus nyata dari sistem, bukan mockup.

---

# SLIDE 12  ·  KESIMPULAN

```
┌─────────────────────────────────────────────────────────────────────┐
│  KESIMPULAN                                         [11 dari 13]   │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  YANG BERHASIL DIBANGUN                                             │
│  ─────────────────────────────────────────────────────────────────  │
│  [OK]  Model mendeteksi 15 kategori seragam sekolah                 │
│  [OK]  Sistem berjalan REAL-TIME (webcam, video, IP kamera)         │
│  [OK]  Precision 73.6% · Recall 75.1% · mAP50 74.9%                │
│  [OK]  Dataset dibangun dari data nyata siswa (762 gambar training)  │
│  [OK]  Berjalan di laptop biasa maupun GPU                          │
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│  KETERBATASAN & LANGKAH PERBAIKAN                                   │
│  ─────────────────────────────────────────────────────────────────  │
│  Precision belum 90%    →  Tambah data training per kelas           │
│  Dataset validasi kecil →  Perbanyak data & variasi augmentasi      │
│  Kelas mirip visual     →  Fine-tune threshold per kelas            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

> **Catatan desainer:** Bagian "Yang Berhasil" gunakan ikon centang hijau. Bagian "Keterbatasan" gunakan ikon panah kanan berwarna oranye (bukan merah — agar tidak terkesan negatif, melainkan "peluang perbaikan"). Animasi: muncul per baris.

---

# SLIDE 13  ·  RENCANA KE DEPAN & PENUTUP

```
┌─────────────────────────────────────────────────────────────────────┐
│  RENCANA KE DEPAN                                   [12 dari 13]   │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  FASE 1 — Peningkatan Data               (Jangka Pendek)            │
│  Perbanyak dataset → target > 500 gambar per kelas                  │
│                                                                     │
│  FASE 2 — Peningkatan Akurasi            (Jangka Menengah)          │
│  Training ulang hingga Precision ≥ 90% per kelas                    │
│                                                                     │
│  FASE 3 — Integrasi Sistem               (Jangka Panjang)           │
│  Hubungkan dengan sistem absensi / portal administrasi sekolah      │
│                                                                     │
│  FASE 4 — Deploy di Sekolah              (Final)                    │
│  Instalasi di kamera CCTV pintu gerbang / gerbang kelas             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────────┐
│  PENUTUP                                            [13 dari 13]   │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│   "Teknologi computer vision dapat membantu sekolah                 │
│    menegakkan aturan seragam secara otomatis,                       │
│    konsisten, dan tanpa bergantung penuh pada                       │
│    tenaga manusia."                                                 │
│                                                                     │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│                                                                     │
│              TERIMA KASIH                                           │
│              Ada pertanyaan?                                        │
│                                                                     │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│  Kontak: [email penyaji]    GitHub: [link repo jika ada]            │
│  Stack : Python · YOLOv8 · Ultralytics · OpenCV · Roboflow         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

> **Catatan desainer:** Slide penutup gunakan foto real output sistem di background (buram). Kutipan ditulis besar di tengah dengan warna putih. "Terima Kasih" ukuran sangat besar. Kontak info kecil di bawah.

---

## PANDUAN DESAIN VISUAL (untuk PowerPoint / Google Slides)

### Palet Warna

| Elemen | Warna | Kode Hex |
|--------|-------|----------|
| Background utama | Putih bersih | `#FFFFFF` |
| Header slide | Biru Navy | `#1A3C6E` |
| Aksen / highlight | Biru Aksen | `#2E8FD4` |
| Background konten | Abu-abu sangat muda | `#F4F6F9` |
| Teks utama | Abu-abu gelap | `#2D2D2D` |
| Status sukses | Hijau | `#27AE60` |
| Status peringatan | Oranye | `#E67E22` |

### Tipografi

| Elemen | Font | Ukuran | Berat |
|--------|------|--------|-------|
| Judul slide | Poppins / Montserrat | 36–40 pt | Bold |
| Sub-judul | Poppins | 24 pt | SemiBold |
| Konten / tabel | Poppins / Open Sans | 18–20 pt | Regular |
| Keterangan kecil | Poppins | 14 pt | Light |
| Nomor slide | Poppins | 12 pt | Regular |

### Prinsip Layout

- **Rule of thirds:** Judul di atas 1/3 slide, konten mengisi 2/3 sisanya
- **Satu pesan per slide:** Jangan tumpuk terlalu banyak informasi
- **Konsistensi:** Setiap slide punya header, nomor, dan footer yang sama
- **Ruang putih:** Beri margin cukup — slide sesak membuat audien sulit fokus
- **Animasi:** Gunakan hanya "Appear" atau "Fade" — hindari animasi berputar/terbang berlebihan

### Ikon yang Disarankan

Gunakan ikon dari [Flaticon](https://www.flaticon.com) atau [Iconify](https://iconify.design) — style **Outlined / Line**, warna biru aksen:

| Slide | Ikon |
|-------|------|
| Masalah | clipboard-x, camera-off, folder-minus |
| Dataset | image-gallery, database, layers |
| Training | brain, cpu, chart-line |
| Evaluasi | bar-chart, target, checkmark |
| Demo | play-circle, monitor, webcam |
| Kesimpulan | check-circle, lightbulb |
