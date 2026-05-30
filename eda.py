"""
eda.py
------
Exploratory Data Analysis (EDA) menyeluruh untuk dataset deteksi objek YOLOv8.

Menghasilkan laporan teks, visualisasi (PNG), dan daftar file bermasalah (JSON)
yang nantinya dikonsumsi oleh clean_dataset.py.

Analisis yang dilakukan:
  a) Dataset overview          (jumlah gambar/anotasi per split, rasio split)
  b) Class distribution        (instance per class, imbalance ratio, warning)
  c) Bounding box analysis     (ukuran bbox px, kategori S/M/L, aspect ratio)
  d) Image quality analysis    (gelap, terlalu terang, blur)
  e) Duplicate detection       (exact via MD5, near-duplicate frame video)
  f) Label integrity           (koordinat invalid, bbox nol, class_id invalid,
                                gambar tanpa label, orphan label)
  g) Co-occurrence matrix      (heatmap pasangan class dalam satu gambar)

Output ada di runs/eda/:
  - class_distribution.png
  - bbox_size_distribution.png
  - bbox_scatter.png
  - brightness_distribution.png
  - cooccurrence_heatmap.png
  - eda_report.txt
  - problematic_files.json

Jalankan: python eda.py
"""

from pathlib import Path
from collections import Counter, defaultdict
import json
import hashlib
import re
import sys

import numpy as np

# === KONFIGURASI PATH DASAR ===
BASE_DIR = Path("/home/funcsmile/Desktop/SandBox/Kuliah/deteksi_barang")
DATASET_DIR = BASE_DIR / "datasets"
DATA_YAML = DATASET_DIR / "data.yaml"
EDA_DIR = BASE_DIR / "runs" / "eda"

# Split: nama_split -> nama_folder di disk
SPLITS = {"train": "train", "valid": "valid", "test": "test"}

# Ekstensi gambar yang dianggap valid
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# === AMBANG BATAS ANALISIS ===
WARN_INSTANCES = 30        # class di bawah ini -> "kurang"
CRITICAL_INSTANCES = 10    # class di bawah ini -> "kritis"
DARK_THRESHOLD = 50        # mean brightness < ini -> gelap
BRIGHT_THRESHOLD = 200     # mean brightness > ini -> terlalu terang
BLUR_THRESHOLD = 100.0     # variance Laplacian < ini -> blur
SMALL_BBOX_PX = 32         # bbox < ini (sisi terpanjang) -> Small
LARGE_BBOX_PX = 96         # bbox > ini -> Large
NEAR_DUP_FRAME_GAP = 5     # selisih nomor frame <= ini -> near-duplicate

# Pola nama frame hasil ekstrak video: "<source>_mp4-<frame>_jpg" atau "-mp4-<frame>"
VIDEO_FRAME_RE = re.compile(r"^(?P<source>.+?)[-_]mp4-(?P<frame>\d+)")
# Pola suffix Roboflow: "<nama_asli>.rf.<hash>.<ext>"
ROBOFLOW_RE = re.compile(r"\.rf\.[0-9a-fA-F]+$")


def load_class_names():
    """Ambil daftar nama class dari data.yaml. Mengembalikan list nama (urut id)."""
    import yaml
    if not DATA_YAML.exists():
        print(f"[ERROR] data.yaml tidak ditemukan: {DATA_YAML}")
        sys.exit(1)
    try:
        with open(DATA_YAML, "r") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Gagal membaca data.yaml: {e}")
        sys.exit(1)

    names = data.get("names")
    if names is None:
        print("[ERROR] Key 'names' tidak ada di data.yaml")
        sys.exit(1)

    # Normalisasi: bisa berupa list atau dict {id: nama}
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    elif not isinstance(names, list):
        print("[ERROR] Format 'names' tidak dikenali (bukan list/dict)")
        sys.exit(1)
    return list(names)


def img_label_pairs(split_folder):
    """Kembalikan dict {stem: {'image': Path|None, 'label': Path|None}} untuk satu split."""
    images_dir = DATASET_DIR / split_folder / "images"
    labels_dir = DATASET_DIR / split_folder / "labels"
    pairs = defaultdict(lambda: {"image": None, "label": None})

    if images_dir.is_dir():
        for p in images_dir.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                pairs[p.stem]["image"] = p
    if labels_dir.is_dir():
        for p in labels_dir.iterdir():
            if p.suffix.lower() == ".txt":
                pairs[p.stem]["label"] = p
    return pairs


def parse_label_file(label_path, nc):
    """
    Baca satu file label YOLO.
    Mengembalikan (valid_rows, invalid_rows) di mana:
      valid_rows   = list (class_id, xc, yc, w, h)
      invalid_rows = list (nomor_baris, isi_baris, alasan)
    """
    valid_rows, invalid_rows = [], []
    try:
        text = label_path.read_text().splitlines()
    except Exception as e:
        invalid_rows.append((0, "", f"gagal membaca file: {e}"))
        return valid_rows, invalid_rows

    for i, line in enumerate(text, start=1):
        stripped = line.strip()
        if not stripped:
            continue  # baris kosong diabaikan, bukan error
        parts = stripped.split()
        if len(parts) != 5:
            invalid_rows.append((i, stripped, f"jumlah kolom = {len(parts)} (harus 5)"))
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:])
        except ValueError:
            invalid_rows.append((i, stripped, "nilai non-numerik"))
            continue

        reason = None
        if cls < 0 or cls >= nc:
            reason = f"class_id {cls} di luar [0,{nc - 1}]"
        elif not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            reason = "koordinat pusat di luar [0,1]"
        elif not (0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
            reason = "lebar/tinggi di luar [0,1]"
        elif w <= 0.0 or h <= 0.0:
            reason = "lebar/tinggi <= 0"

        if reason:
            invalid_rows.append((i, stripped, reason))
        else:
            valid_rows.append((cls, xc, yc, w, h))
    return valid_rows, invalid_rows


def original_name(stem):
    """Buang suffix Roboflow '.rf.<hash>' agar dapat nama asli sebelum augmentasi."""
    return ROBOFLOW_RE.sub("", stem)


def video_source_and_frame(stem):
    """
    Jika stem berasal dari ekstrak frame video, kembalikan (source, frame_int).
    Jika bukan, kembalikan (None, None).
    """
    base = original_name(stem)
    m = VIDEO_FRAME_RE.match(base)
    if not m:
        return None, None
    return m.group("source"), int(m.group("frame"))


def scan_images(all_pairs):
    """
    Satu kali pass membaca setiap gambar untuk menghitung: md5, dimensi (w,h),
    brightness rata-rata, dan variance Laplacian (blur).

    Mengembalikan dict {stem: {...metrik...}} hanya untuk gambar yang bisa dibaca.
    """
    import cv2
    from tqdm import tqdm

    metrics = {}
    # Kumpulkan semua gambar dari semua split
    items = []
    for split, pairs in all_pairs.items():
        for stem, fp in pairs.items():
            if fp["image"] is not None:
                items.append((split, stem, fp["image"]))

    for split, stem, img_path in tqdm(items, desc="Scan gambar", unit="img"):
        try:
            data = np.fromfile(str(img_path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                metrics[stem] = {"split": split, "path": img_path, "unreadable": True}
                continue
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness = float(gray.mean())
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            md5 = hashlib.md5(data.tobytes()).hexdigest()
            metrics[stem] = {
                "split": split, "path": img_path, "unreadable": False,
                "w": w, "h": h, "brightness": brightness, "blur": blur, "md5": md5,
            }
        except Exception as e:
            metrics[stem] = {"split": split, "path": img_path,
                             "unreadable": True, "error": str(e)}
    return metrics


def rel(path):
    """Path relatif terhadap BASE_DIR untuk laporan yang ringkas & portabel."""
    try:
        return str(Path(path).relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def run_eda():
    print("=" * 70)
    print(" EDA DATASET DETEKSI BARANG (YOLOv8)")
    print("=" * 70)

    EDA_DIR.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names()
    nc = len(class_names)
    print(f"Jumlah class (nc) : {nc}")
    print(f"Output EDA        : {rel(EDA_DIR)}\n")

    # --- Kumpulkan pasangan image/label per split ---
    all_pairs = {split: img_label_pairs(folder) for split, folder in SPLITS.items()}

    report = []  # baris-baris untuk eda_report.txt
    def log(line=""):
        print(line)
        report.append(line)

    log("=" * 70)
    log(" a) DATASET OVERVIEW")
    log("=" * 70)

    split_counts = {}
    total_images = 0
    for split, pairs in all_pairs.items():
        n_img = sum(1 for v in pairs.values() if v["image"] is not None)
        n_lbl = sum(1 for v in pairs.values() if v["label"] is not None)
        split_counts[split] = n_img
        total_images += n_img
        log(f"  {split:6s} : {n_img:5d} gambar | {n_lbl:5d} label")
    log(f"  {'TOTAL':6s} : {total_images:5d} gambar")
    log("")
    log("  Rasio split aktual vs ideal (80/10/10):")
    ideal = {"train": 80.0, "valid": 10.0, "test": 10.0}
    for split in SPLITS:
        pct = (split_counts[split] / total_images * 100) if total_images else 0.0
        log(f"    {split:6s} : {pct:5.1f}%  (ideal {ideal[split]:4.0f}%)")
    log("")

    # --- Integritas label + kumpulkan anotasi (untuk class dist, bbox, co-occurrence) ---
    problematic = {
        "exact_duplicates": [],
        "near_duplicates_by_video": {},
        "dark_images": [],
        "bright_images": [],
        "blurry_images": [],
        "corrupt_labels": [],
        "images_without_labels": [],
        "orphan_labels": [],
    }

    # class_dist[split][class_id] = jumlah instance
    class_dist = {split: Counter() for split in SPLITS}
    # Simpan anotasi valid per stem untuk analisis bbox & co-occurrence
    annotations = {}  # stem -> list (cls, xc, yc, w, h)

    from tqdm import tqdm
    log("=" * 70)
    log(" f) LABEL INTEGRITY")
    log("=" * 70)

    for split, pairs in all_pairs.items():
        for stem, fp in tqdm(list(pairs.items()), desc=f"Cek label [{split}]", unit="file"):
            img_p, lbl_p = fp["image"], fp["label"]

            # Orphan label (label tanpa gambar)
            if img_p is None and lbl_p is not None:
                problematic["orphan_labels"].append(rel(lbl_p))
                continue
            # Gambar tanpa label
            if img_p is not None and lbl_p is None:
                problematic["images_without_labels"].append(rel(img_p))
                continue

            valid_rows, invalid_rows = parse_label_file(lbl_p, nc)
            if invalid_rows:
                problematic["corrupt_labels"].append({
                    "label": rel(lbl_p),
                    "image": rel(img_p),
                    "total_rows": len(valid_rows) + len(invalid_rows),
                    "invalid": [
                        {"line": ln, "content": content, "reason": reason}
                        for (ln, content, reason) in invalid_rows
                    ],
                })
            annotations[stem] = valid_rows
            for (cls, *_rest) in valid_rows:
                class_dist[split][cls] += 1

    log(f"  Gambar tanpa label   : {len(problematic['images_without_labels'])}")
    log(f"  Orphan label         : {len(problematic['orphan_labels'])}")
    log(f"  File label corrupt   : {len(problematic['corrupt_labels'])}")
    if problematic["corrupt_labels"]:
        log("    Contoh (maks 10):")
        for item in problematic["corrupt_labels"][:10]:
            first = item["invalid"][0]
            log(f"      {item['label']}  baris {first['line']}: {first['reason']}")
    if problematic["images_without_labels"]:
        log("    Gambar tanpa label (maks 10):")
        for x in problematic["images_without_labels"][:10]:
            log(f"      {x}")
    if problematic["orphan_labels"]:
        log("    Orphan label (maks 10):")
        for x in problematic["orphan_labels"][:10]:
            log(f"      {x}")
    log("")

    # --- b) CLASS DISTRIBUTION ---
    log("=" * 70)
    log(" b) CLASS DISTRIBUTION")
    log("=" * 70)
    total_per_class = Counter()
    for split in SPLITS:
        for cls, cnt in class_dist[split].items():
            total_per_class[cls] += cnt

    log(f"  {'class':18s} {'train':>7s} {'valid':>7s} {'test':>7s} {'TOTAL':>7s}  status")
    log("  " + "-" * 64)
    # Urutkan descending berdasarkan total
    order = sorted(range(nc), key=lambda c: total_per_class.get(c, 0), reverse=True)
    for cls in order:
        name = class_names[cls]
        tr = class_dist["train"].get(cls, 0)
        va = class_dist["valid"].get(cls, 0)
        te = class_dist["test"].get(cls, 0)
        tot = total_per_class.get(cls, 0)
        if tot < CRITICAL_INSTANCES:
            status = "KRITIS"
        elif tot < WARN_INSTANCES:
            status = "kurang"
        else:
            status = "ok"
        log(f"  {name:18s} {tr:7d} {va:7d} {te:7d} {tot:7d}  {status}")

    nonzero = [total_per_class.get(c, 0) for c in range(nc) if total_per_class.get(c, 0) > 0]
    if nonzero:
        imb = max(nonzero) / min(nonzero)
        log("")
        log(f"  Instance terbanyak  : {max(nonzero)}")
        log(f"  Instance tersedikit : {min(nonzero)}")
        log(f"  Imbalance ratio     : {imb:.1f}x")
    missing = [class_names[c] for c in range(nc) if total_per_class.get(c, 0) == 0]
    if missing:
        log(f"  Class TANPA instance: {', '.join(missing)}")
    log("")

    # --- Scan gambar (md5, dimensi, brightness, blur) ---
    log("Membaca seluruh gambar untuk analisis kualitas & duplikat ...")
    metrics = scan_images(all_pairs)
    unreadable = [rel(m["path"]) for m in metrics.values() if m.get("unreadable")]
    if unreadable:
        log(f"  [WARNING] {len(unreadable)} gambar tidak bisa dibaca/corrupt:")
        for x in unreadable[:10]:
            log(f"      {x}")
    log("")

    # --- c) BOUNDING BOX ANALYSIS ---
    log("=" * 70)
    log(" c) BOUNDING BOX ANALYSIS")
    log("=" * 70)
    bbox_w_px, bbox_h_px, aspect_ratios = [], [], []
    size_cat = Counter()  # Small / Medium / Large
    for stem, rows in annotations.items():
        m = metrics.get(stem)
        if not m or m.get("unreadable"):
            continue
        iw, ih = m["w"], m["h"]
        for (_cls, _xc, _yc, w, h) in rows:
            wpx, hpx = w * iw, h * ih
            bbox_w_px.append(wpx)
            bbox_h_px.append(hpx)
            if hpx > 0:
                aspect_ratios.append(wpx / hpx)
            longest = max(wpx, hpx)
            if longest < SMALL_BBOX_PX:
                size_cat["Small"] += 1
            elif longest <= LARGE_BBOX_PX:
                size_cat["Medium"] += 1
            else:
                size_cat["Large"] += 1

    if bbox_w_px:
        arr_w, arr_h = np.array(bbox_w_px), np.array(bbox_h_px)
        log(f"  Total bbox dianalisis : {len(bbox_w_px)}")
        log(f"  Lebar  (px)  min/mean/median/max : "
            f"{arr_w.min():.0f} / {arr_w.mean():.0f} / {np.median(arr_w):.0f} / {arr_w.max():.0f}")
        log(f"  Tinggi (px)  min/mean/median/max : "
            f"{arr_h.min():.0f} / {arr_h.mean():.0f} / {np.median(arr_h):.0f} / {arr_h.max():.0f}")
        log(f"  Kategori ukuran (sisi terpanjang):")
        total_bbox = sum(size_cat.values())
        for cat in ("Small", "Medium", "Large"):
            c = size_cat.get(cat, 0)
            pct = c / total_bbox * 100 if total_bbox else 0
            log(f"    {cat:7s} (<{SMALL_BBOX_PX}px / {SMALL_BBOX_PX}-{LARGE_BBOX_PX}px / >{LARGE_BBOX_PX}px): {c:6d} ({pct:4.1f}%)")
        if aspect_ratios:
            ar = np.array(aspect_ratios)
            log(f"  Aspect ratio (w/h) min/mean/median/max : "
                f"{ar.min():.2f} / {ar.mean():.2f} / {np.median(ar):.2f} / {ar.max():.2f}")
    else:
        log("  [WARNING] Tidak ada bbox valid untuk dianalisis.")
    log("")

    # --- d) IMAGE QUALITY ANALYSIS ---
    log("=" * 70)
    log(" d) IMAGE QUALITY ANALYSIS")
    log("=" * 70)
    brightness_vals = []
    for stem, m in metrics.items():
        if m.get("unreadable"):
            continue
        brightness_vals.append(m["brightness"])
        if m["brightness"] < DARK_THRESHOLD:
            problematic["dark_images"].append(
                {"image": rel(m["path"]), "brightness": round(m["brightness"], 1)})
        elif m["brightness"] > BRIGHT_THRESHOLD:
            problematic["bright_images"].append(
                {"image": rel(m["path"]), "brightness": round(m["brightness"], 1)})
        if m["blur"] < BLUR_THRESHOLD:
            problematic["blurry_images"].append(
                {"image": rel(m["path"]), "blur": round(m["blur"], 1)})

    log(f"  Gambar gelap   (brightness < {DARK_THRESHOLD})   : {len(problematic['dark_images'])}")
    log(f"  Gambar terang  (brightness > {BRIGHT_THRESHOLD})  : {len(problematic['bright_images'])}")
    log(f"  Gambar blur    (laplacian  < {BLUR_THRESHOLD:.0f}) : {len(problematic['blurry_images'])}")
    for key, label in (("dark_images", "Gelap"), ("bright_images", "Terang"), ("blurry_images", "Blur")):
        if problematic[key]:
            log(f"    {label} (maks 10):")
            for item in sorted(problematic[key], key=lambda d: list(d.values())[1])[:10]:
                val_key = "brightness" if "brightness" in item else "blur"
                log(f"      {item['image']}  ({val_key}={item[val_key]})")
    log("")

    # --- e) DUPLICATE DETECTION ---
    log("=" * 70)
    log(" e) DUPLICATE DETECTION")
    log("=" * 70)

    # Exact duplicate via MD5
    md5_map = defaultdict(list)
    for stem, m in metrics.items():
        if m.get("unreadable"):
            continue
        md5_map[m["md5"]].append(rel(m["path"]))
    exact_groups = [sorted(g) for g in md5_map.values() if len(g) > 1]
    problematic["exact_duplicates"] = exact_groups
    n_exact_redundant = sum(len(g) - 1 for g in exact_groups)
    log(f"  Grup exact-duplicate (MD5)     : {len(exact_groups)}")
    log(f"  Gambar redundan (exact)        : {n_exact_redundant}")
    for g in exact_groups[:10]:
        log(f"    [{len(g)} file] {g[0]}  (+{len(g) - 1} lainnya)")

    # Near-duplicate dari frame video
    # Kelompokkan per source -> [(frame, rel_path, jumlah_label)]
    by_source = defaultdict(list)
    for split, pairs in all_pairs.items():
        for stem, fp in pairs.items():
            if fp["image"] is None:
                continue
            source, frame = video_source_and_frame(stem)
            if source is None:
                continue
            n_lbl = len(annotations.get(stem, []))
            by_source[source].append((frame, rel(fp["image"]), n_lbl))

    near_dup = {}
    total_near_redundant = 0
    for source, items in by_source.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda t: t[0])
        redundant = []
        last_kept_frame = None
        for frame, path, _n in items:
            if last_kept_frame is not None and (frame - last_kept_frame) <= NEAR_DUP_FRAME_GAP:
                redundant.append(path)  # terlalu dekat dgn frame yg dipertahankan
            else:
                last_kept_frame = frame  # jadikan acuan baru
        if redundant:
            near_dup[source] = {
                "total_frames": len(items),
                "estimated_redundant": len(redundant),
                "redundant_files": redundant,
            }
            total_near_redundant += len(redundant)
    problematic["near_duplicates_by_video"] = near_dup

    log(f"  Sumber video terdeteksi        : {len(by_source)}")
    log(f"  Sumber dgn near-duplicate      : {len(near_dup)}")
    log(f"  Estimasi frame redundan (near) : {total_near_redundant}")
    if near_dup:
        log("    Top sumber redundan (maks 10):")
        top = sorted(near_dup.items(), key=lambda kv: kv[1]["estimated_redundant"], reverse=True)
        for source, info in top[:10]:
            log(f"      {source[:48]:48s} {info['estimated_redundant']:4d}/{info['total_frames']:4d} redundan")
    log("")

    # --- g) CO-OCCURRENCE MATRIX ---
    log("=" * 70)
    log(" g) CO-OCCURRENCE MATRIX")
    log("=" * 70)
    cooc = np.zeros((nc, nc), dtype=np.int64)
    for stem, rows in annotations.items():
        present = sorted({cls for (cls, *_r) in rows})
        for i in range(len(present)):
            for j in range(i, len(present)):
                a, b = present[i], present[j]
                cooc[a][b] += 1
                if a != b:
                    cooc[b][a] += 1
    # Pasangan yang paling sering muncul bersama
    pairs_freq = []
    for i in range(nc):
        for j in range(i + 1, nc):
            if cooc[i][j] > 0:
                pairs_freq.append((cooc[i][j], class_names[i], class_names[j]))
    pairs_freq.sort(reverse=True)
    log(f"  Pasangan class co-occur (>0)   : {len(pairs_freq)}")
    if pairs_freq:
        log("    Top pasangan (maks 10):")
        for cnt, a, b in pairs_freq[:10]:
            log(f"      {a} + {b}: {cnt}x")
    log("")

    # === VISUALISASI ===
    log("Membuat visualisasi ...")
    make_plots(class_names, total_per_class, class_dist, bbox_w_px, bbox_h_px,
               size_cat, aspect_ratios, brightness_vals, cooc, order)

    # === SIMPAN JSON FILE BERMASALAH ===
    json_path = EDA_DIR / "problematic_files.json"
    with open(json_path, "w") as f:
        json.dump(problematic, f, indent=2, ensure_ascii=False)
    log(f"  -> {rel(json_path)}")

    # === SIMPAN LAPORAN TEKS ===
    # Tambahkan baris ringkasan total_per_class agar bisa dibaca verify_cleaning.py
    summary_lines = ["", "=" * 70, " RINGKASAN TOTAL INSTANCE PER CLASS (untuk verifikasi)", "=" * 70]
    for cls in range(nc):
        summary_lines.append(f"  CLASS_TOTAL\t{class_names[cls]}\t{total_per_class.get(cls, 0)}")
    summary_lines.append(f"  TOTAL_IMAGES\t{total_images}")
    report.extend(summary_lines)

    report_path = EDA_DIR / "eda_report.txt"
    report_path.write_text("\n".join(report) + "\n")

    print("\n" + "=" * 70)
    print(" EDA SELESAI")
    print("=" * 70)
    print(f"  Laporan         : {rel(report_path)}")
    print(f"  File bermasalah : {rel(json_path)}")
    print(f"  Visualisasi     : {rel(EDA_DIR)}/*.png")


def make_plots(class_names, total_per_class, class_dist, bbox_w_px, bbox_h_px,
               size_cat, aspect_ratios, brightness_vals, cooc, order):
    """Buat semua chart dan simpan ke EDA_DIR. Error plot tidak menggagalkan EDA."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # backend non-interaktif (aman headless)
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as e:
        print(f"  [WARNING] matplotlib/seaborn tidak tersedia, lewati plot: {e}")
        return

    nc = len(class_names)

    # 1) Class distribution (sorted descending) ----------------------------
    try:
        names_sorted = [class_names[c] for c in order]
        totals_sorted = [total_per_class.get(c, 0) for c in order]
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = ["#d62728" if v < CRITICAL_INSTANCES else
                  "#ff7f0e" if v < WARN_INSTANCES else "#2ca02c" for v in totals_sorted]
        ax.bar(range(nc), totals_sorted, color=colors)
        ax.set_xticks(range(nc))
        ax.set_xticklabels(names_sorted, rotation=60, ha="right")
        ax.axhline(WARN_INSTANCES, ls="--", c="#ff7f0e", lw=1, label=f"warn ({WARN_INSTANCES})")
        ax.axhline(CRITICAL_INSTANCES, ls="--", c="#d62728", lw=1, label=f"kritis ({CRITICAL_INSTANCES})")
        ax.set_ylabel("Jumlah instance (semua split)")
        ax.set_title("Distribusi Class (sorted descending)")
        ax.legend()
        for i, v in enumerate(totals_sorted):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=7)
        fig.tight_layout()
        fig.savefig(EDA_DIR / "class_distribution.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  [WARNING] gagal class_distribution.png: {e}")

    # 2) Bbox size distribution (kategori + histogram) ---------------------
    try:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        cats = ["Small", "Medium", "Large"]
        vals = [size_cat.get(c, 0) for c in cats]
        axes[0].bar(cats, vals, color=["#1f77b4", "#2ca02c", "#ff7f0e"])
        axes[0].set_title(f"Kategori ukuran bbox\n(<{SMALL_BBOX_PX} / {SMALL_BBOX_PX}-{LARGE_BBOX_PX} / >{LARGE_BBOX_PX} px)")
        axes[0].set_ylabel("Jumlah bbox")
        for i, v in enumerate(vals):
            axes[0].text(i, v, str(v), ha="center", va="bottom")
        if aspect_ratios:
            axes[1].hist(np.clip(aspect_ratios, 0, 5), bins=50, color="#9467bd")
            axes[1].set_title("Histogram aspect ratio (w/h, clip 0-5)")
            axes[1].set_xlabel("aspect ratio")
            axes[1].set_ylabel("jumlah bbox")
        fig.tight_layout()
        fig.savefig(EDA_DIR / "bbox_size_distribution.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  [WARNING] gagal bbox_size_distribution.png: {e}")

    # 3) Bbox scatter width vs height (px) ---------------------------------
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        if bbox_w_px:
            ax.scatter(bbox_w_px, bbox_h_px, s=4, alpha=0.25, color="#1f77b4")
        ax.set_xlabel("lebar bbox (px)")
        ax.set_ylabel("tinggi bbox (px)")
        ax.set_title("Scatter ukuran bbox (width vs height)")
        fig.tight_layout()
        fig.savefig(EDA_DIR / "bbox_scatter.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  [WARNING] gagal bbox_scatter.png: {e}")

    # 4) Brightness distribution -------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        if brightness_vals:
            ax.hist(brightness_vals, bins=50, color="#8c564b")
        ax.axvline(DARK_THRESHOLD, ls="--", c="#d62728", label=f"gelap (<{DARK_THRESHOLD})")
        ax.axvline(BRIGHT_THRESHOLD, ls="--", c="#ff7f0e", label=f"terang (>{BRIGHT_THRESHOLD})")
        ax.set_xlabel("mean brightness (0-255)")
        ax.set_ylabel("jumlah gambar")
        ax.set_title("Distribusi brightness gambar")
        ax.legend()
        fig.tight_layout()
        fig.savefig(EDA_DIR / "brightness_distribution.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  [WARNING] gagal brightness_distribution.png: {e}")

    # 5) Co-occurrence heatmap ---------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(13, 11))
        sns.heatmap(cooc, xticklabels=class_names, yticklabels=class_names,
                    cmap="viridis", annot=False, square=True, ax=ax,
                    cbar_kws={"label": "jumlah gambar"})
        ax.set_title("Co-occurrence matrix (pasangan class dalam 1 gambar)")
        fig.tight_layout()
        fig.savefig(EDA_DIR / "cooccurrence_heatmap.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  [WARNING] gagal cooccurrence_heatmap.png: {e}")


if __name__ == "__main__":
    try:
        run_eda()
    except KeyboardInterrupt:
        print("\n[DIBATALKAN] EDA dihentikan oleh user.")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] EDA gagal: {e}")
        traceback.print_exc()
        sys.exit(1)
