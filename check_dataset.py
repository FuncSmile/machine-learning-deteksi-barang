"""
check_dataset.py
----------------
Validasi dataset SEBELUM training agar tidak membuang waktu / resource.

Pemeriksaan yang dilakukan:
  1. Cek keberadaan semua folder & file penting.
  2. Hitung jumlah gambar per split (train / valid / test).
  3. Hitung distribusi class dari file label (.txt YOLO format).
  4. Beri WARNING untuk class dengan jumlah instance < 30.
  5. Cek gambar tanpa label, dan label tanpa gambar (orphan files).
  6. Tampilkan summary yang informatif.

Catatan: HANYA memakai library standar + yaml (sesuai constraint).
"""

from pathlib import Path
from collections import Counter, defaultdict
import os
import yaml

# === KONFIGURASI PATH DASAR ===
BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
DATA_YAML = DATASETS_DIR / "data.yaml"

# Ekstensi gambar yang dianggap valid
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Ambang batas minimum instance per class sebelum diberi warning
MIN_INSTANCES = 30

# Split yang diperiksa: nama_split -> nama_folder
SPLITS = {"train": "train", "valid": "valid", "test": "test"}


def load_class_names():
    """Ambil daftar nama class dari data.yaml. Return list atau None jika gagal."""
    if not DATA_YAML.exists():
        print(f"[ERROR] data.yaml tidak ditemukan: {DATA_YAML}")
        return None
    try:
        with DATA_YAML.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names")
        if isinstance(names, dict):  # format {0: 'a', 1: 'b'} -> list
            names = [names[k] for k in sorted(names)]
        return names
    except Exception as e:
        print(f"[ERROR] Gagal membaca data.yaml: {e}")
        return None


def list_images(images_dir: Path):
    """Kembalikan list file gambar (berdasarkan ekstensi) di sebuah folder."""
    if not images_dir.exists():
        return []
    return [p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS]


def parse_label_classes(label_path: Path):
    """Baca file label YOLO dan kembalikan list class_id (int) di dalamnya."""
    class_ids = []
    try:
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format YOLO: <class_id> <x> <y> <w> <h>
                cls = line.split()[0]
                class_ids.append(int(float(cls)))
    except Exception as e:
        print(f"[WARNING] Gagal membaca label {label_path.name}: {e}")
    return class_ids


def check_split(split_name: str, folder_name: str, num_classes: int):
    """Periksa satu split. Kembalikan dict berisi statistiknya."""
    images_dir = DATASETS_DIR / folder_name / "images"
    labels_dir = DATASETS_DIR / folder_name / "labels"

    result = {
        "exists": True,
        "n_images": 0,
        "n_labels": 0,
        "class_counts": Counter(),
        "images_without_label": [],
        "labels_without_image": [],
        "invalid_class_ids": [],
    }

    # --- Cek folder ---
    if not images_dir.exists():
        print(f"[ERROR] Folder images hilang: {images_dir}")
        result["exists"] = False
    if not labels_dir.exists():
        print(f"[ERROR] Folder labels hilang: {labels_dir}")
        result["exists"] = False
    if not result["exists"]:
        return result

    images = list_images(images_dir)
    result["n_images"] = len(images)

    # Set stem (nama file tanpa ekstensi) untuk pencocokan image <-> label
    image_stems = {p.stem for p in images}
    label_files = [p for p in labels_dir.iterdir()
                   if p.is_file() and p.suffix.lower() == ".txt"]
    result["n_labels"] = len(label_files)
    label_stems = {p.stem for p in label_files}

    # --- Distribusi class & validasi class_id ---
    for lbl in label_files:
        for cid in parse_label_classes(lbl):
            if 0 <= cid < num_classes:
                result["class_counts"][cid] += 1
            else:
                result["invalid_class_ids"].append((lbl.name, cid))

    # --- Orphan detection ---
    # Gambar tanpa label (kecuali label kosong yang valid = background image)
    for stem in image_stems:
        if stem not in label_stems:
            result["images_without_label"].append(stem)
    # Label tanpa gambar
    for stem in label_stems:
        if stem not in image_stems:
            result["labels_without_image"].append(stem)

    return result


def check_dataset():
    """Jalankan seluruh pemeriksaan dataset dan cetak summary."""
    print("=" * 70)
    print("CHECK DATASET  ->  validasi sebelum training")
    print("=" * 70)

    if not DATASETS_DIR.exists():
        print(f"[ERROR] Folder datasets tidak ditemukan: {DATASETS_DIR}")
        return False

    names = load_class_names()
    if not names:
        print("[ERROR] Tidak bisa memuat nama class. Jalankan fix_yaml.py dulu?")
        return False
    num_classes = len(names)
    print(f"[INFO] Jumlah class (nc): {num_classes}")

    # Akumulasi distribusi class dari SEMUA split untuk ringkasan akhir
    total_class_counts = Counter()
    per_split_results = {}
    has_error = False

    # --- Periksa tiap split ---
    for split_name, folder_name in SPLITS.items():
        print("\n" + "-" * 70)
        print(f"SPLIT: {split_name.upper()}  (folder: {folder_name})")
        print("-" * 70)

        res = check_split(split_name, folder_name, num_classes)
        per_split_results[split_name] = res

        if not res["exists"]:
            has_error = True
            continue

        print(f"  Gambar : {res['n_images']}")
        print(f"  Label  : {res['n_labels']}")

        # Orphan report
        n_img_no_lbl = len(res["images_without_label"])
        n_lbl_no_img = len(res["labels_without_image"])
        if n_img_no_lbl:
            print(f"  [WARNING] {n_img_no_lbl} gambar TANPA label "
                  f"(mis. background / lupa anotasi).")
            for s in res["images_without_label"][:5]:
                print(f"            - {s}")
            if n_img_no_lbl > 5:
                print(f"            ... dan {n_img_no_lbl - 5} lainnya")
        if n_lbl_no_img:
            print(f"  [WARNING] {n_lbl_no_img} label TANPA gambar (orphan).")
            for s in res["labels_without_image"][:5]:
                print(f"            - {s}")
            if n_lbl_no_img > 5:
                print(f"            ... dan {n_lbl_no_img - 5} lainnya")
        if res["invalid_class_ids"]:
            has_error = True
            print(f"  [ERROR] Ditemukan {len(res['invalid_class_ids'])} "
                  f"class_id di luar rentang 0..{num_classes - 1}:")
            for fname, cid in res["invalid_class_ids"][:5]:
                print(f"            - {fname}: class_id={cid}")

        if not n_img_no_lbl and not n_lbl_no_img and not res["invalid_class_ids"]:
            print("  [OK] Semua gambar & label cocok, class_id valid.")

        total_class_counts.update(res["class_counts"])

    # --- Distribusi class gabungan (fokus: train sangat penting) ---
    print("\n" + "=" * 70)
    print("DISTRIBUSI CLASS (instance count - SEMUA split digabung)")
    print("=" * 70)
    print(f"{'ID':>3}  {'CLASS':<16} {'TOTAL':>7}   STATUS")
    print("-" * 70)

    low_classes = []
    for cid in range(num_classes):
        count = total_class_counts.get(cid, 0)
        cname = names[cid] if cid < len(names) else f"id_{cid}"
        if count == 0:
            status = "[!!] TIDAK ADA DATA"
            low_classes.append((cname, count))
        elif count < MIN_INSTANCES:
            status = f"[!] < {MIN_INSTANCES} (kurang)"
            low_classes.append((cname, count))
        else:
            status = "OK"
        print(f"{cid:>3}  {cname:<16} {count:>7}   {status}")

    # --- Summary akhir ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for split_name in SPLITS:
        res = per_split_results[split_name]
        print(f"  {split_name:5s}: {res['n_images']} gambar, "
              f"{res['n_labels']} label")
    total_instances = sum(total_class_counts.values())
    print(f"  Total instance objek (semua split): {total_instances}")

    if low_classes:
        print(f"\n  [WARNING] {len(low_classes)} class memiliki data < "
              f"{MIN_INSTANCES} instance:")
        for cname, count in low_classes:
            print(f"            - {cname}: {count}")
        print("\n  REKOMENDASI untuk class minim data:")
        print("    * Tambah gambar untuk class tsb (target >= 30-50 instance).")
        print("    * Gunakan augmentasi (sudah aktif di train.py: mosaic, mixup,")
        print("      copy_paste) untuk memperkaya variasi.")
        print("    * Pertimbangkan oversampling / class weighting jika tetap timpang.")

    if has_error:
        print("\n[GAGAL] Ada ERROR pada dataset. Perbaiki dulu sebelum training.")
        return False

    print("\n[OK] Dataset siap. Lanjut ke: python train.py")
    return True


if __name__ == "__main__":
    ok = check_dataset()
    if not ok:
        raise SystemExit(1)
