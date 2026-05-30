"""
clean_dataset.py
----------------
Membersihkan dataset YOLOv8 berdasarkan runs/eda/problematic_files.json
(dihasilkan oleh eda.py).

Cleaning yang dilakukan:
  a) Hapus exact duplicate          (pertahankan 1 per grup)
  b) Near-duplicate reduction       (frame thinning per sumber video)
  c) Hapus / perbaiki corrupt label (semua corrupt -> hapus; sebagian -> sisakan valid)
  d) Hapus gambar tanpa label
  e) Hapus orphan label
  f) Kualitas gambar (gelap/blur)   -> HANYA jika --remove-dark / --remove-blurry

KESELAMATAN:
  - Default DRY RUN (tidak menghapus apa pun). Tampilkan rencana.
  - Eksekusi nyata: python clean_dataset.py --execute
  - Manifest penghapusan selalu ditulis ke runs/cleaning/deletion_manifest.txt
    SEBELUM ada file yang disentuh.
  - Saat --execute, wajib konfirmasi ketik 'yes' (kecuali diberi --yes).
  - Tidak pernah menghapus tanpa konfirmasi.

Jalankan:
  python clean_dataset.py              # dry run (sama dgn --dry-run)
  python clean_dataset.py --execute    # eksekusi (dgn konfirmasi)
  python clean_dataset.py --execute --remove-blurry --remove-dark
"""

from pathlib import Path
from collections import defaultdict
import argparse
import json
import math
import re
import sys
import datetime

# === KONFIGURASI PATH DASAR ===
BASE_DIR = Path("/home/funcsmile/Desktop/SandBox/Kuliah/deteksi_barang")
DATASET_DIR = BASE_DIR / "datasets"
DATA_YAML = DATASET_DIR / "data.yaml"
EDA_DIR = BASE_DIR / "runs" / "eda"
CLEANING_DIR = BASE_DIR / "runs" / "cleaning"
PROBLEMATIC_JSON = EDA_DIR / "problematic_files.json"

SPLITS = {"train": "train", "valid": "valid", "test": "test"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# === PARAMETER FRAME THINNING (near-duplicate) ===
KEEP_FRACTION = 0.20   # pertahankan maksimal 20% frame per sumber video
MIN_KEEP_PER_VIDEO = 10  # tapi minimal 10 frame per sumber

# Pola nama (sama dengan eda.py, sengaja diduplikasi agar script standalone)
VIDEO_FRAME_RE = re.compile(r"^(?P<source>.+?)[-_]mp4-(?P<frame>\d+)")
ROBOFLOW_RE = re.compile(r"\.rf\.[0-9a-fA-F]+$")


def load_nc():
    """Ambil jumlah class dari data.yaml (untuk validasi ulang corrupt label)."""
    import yaml
    with open(DATA_YAML, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    return len(names)


def abspath(rel_str):
    """Ubah path relatif (dari JSON) menjadi absolut berbasis BASE_DIR."""
    p = Path(rel_str)
    return p if p.is_absolute() else (BASE_DIR / p)


def rel(path):
    try:
        return str(Path(path).relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def original_name(stem):
    return ROBOFLOW_RE.sub("", stem)


def video_source_and_frame(stem):
    base = original_name(stem)
    m = VIDEO_FRAME_RE.match(base)
    if not m:
        return None, None
    return m.group("source"), int(m.group("frame"))


def label_for_image(image_path):
    """Path label (.txt) pasangan dari sebuah gambar."""
    return image_path.parent.parent / "labels" / (image_path.stem + ".txt")


def image_for_label(label_path):
    """Cari gambar pasangan dari sebuah label (.txt). None jika tidak ada."""
    images_dir = label_path.parent.parent / "images"
    for ext in IMG_EXTS:
        cand = images_dir / (label_path.stem + ext)
        if cand.exists():
            return cand
    return None


def parse_valid_invalid(label_path, nc):
    """Kembalikan (valid_lines, n_invalid). valid_lines = baris asli yang lolos validasi."""
    valid_lines, n_invalid = [], 0
    try:
        lines = label_path.read_text().splitlines()
    except Exception:
        return [], 0
    for line in lines:
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 5:
            n_invalid += 1
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:])
        except ValueError:
            n_invalid += 1
            continue
        ok = (0 <= cls < nc and 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0
              and 0.0 < w <= 1.0 and 0.0 < h <= 1.0)
        if ok:
            valid_lines.append(s)
        else:
            n_invalid += 1
    return valid_lines, n_invalid


def count_labels(label_path):
    """Hitung jumlah baris label non-kosong (untuk prioritas frame thinning)."""
    if not label_path.exists():
        return 0
    try:
        return sum(1 for ln in label_path.read_text().splitlines() if ln.strip())
    except Exception:
        return 0


def build_plan(args):
    """
    Bangun rencana cleaning.
    Kembalikan dict dgn:
      delete_images : list (path, reason)
      delete_labels : list (path, reason)
      rewrite_labels: list (path, valid_lines, n_removed, reason)
    """
    if not PROBLEMATIC_JSON.exists():
        print(f"[ERROR] {rel(PROBLEMATIC_JSON)} tidak ditemukan. Jalankan eda.py dulu.")
        sys.exit(1)
    with open(PROBLEMATIC_JSON, "r") as f:
        prob = json.load(f)

    nc = load_nc()

    # Gunakan set agar tidak menghapus file yang sama dua kali
    del_images = {}   # str(abs path) -> reason  (urutan pertama yg menang)
    del_labels = {}
    rewrites = {}     # str(abs path) -> (valid_lines, n_removed, reason)

    def mark_image(p, reason):
        key = str(p)
        del_images.setdefault(key, reason)
        lp = label_for_image(p)
        del_labels.setdefault(str(lp), f"pasangan dari gambar dihapus ({reason})")

    def mark_label(p, reason):
        del_labels.setdefault(str(p), reason)

    # --- a) Exact duplicates: pertahankan file pertama tiap grup ---
    for group in prob.get("exact_duplicates", []):
        for dup in group[1:]:  # group[0] dipertahankan
            mark_image(abspath(dup), "exact-duplicate (MD5)")

    # --- c) Corrupt labels ---
    for item in prob.get("corrupt_labels", []):
        lbl = abspath(item["label"])
        img = abspath(item["image"]) if item.get("image") else image_for_label(lbl)
        if not lbl.exists():
            continue
        valid_lines, n_invalid = parse_valid_invalid(lbl, nc)
        if not valid_lines:
            # Semua baris corrupt -> hapus gambar + label
            if img is not None:
                mark_image(img, "semua label corrupt")
            else:
                mark_label(lbl, "semua label corrupt (gambar tdk ada)")
        elif n_invalid > 0:
            # Sebagian corrupt -> sisakan baris valid saja
            rewrites[str(lbl)] = (valid_lines, n_invalid, f"buang {n_invalid} baris invalid")

    # --- d) Images without labels ---
    for img_rel in prob.get("images_without_labels", []):
        mark_image(abspath(img_rel), "gambar tanpa label")

    # --- e) Orphan labels ---
    for lbl_rel in prob.get("orphan_labels", []):
        mark_label(abspath(lbl_rel), "orphan label (tanpa gambar)")

    # --- b) Near-duplicate reduction (frame thinning per sumber video) ---
    # Hitung ulang dari disk; abaikan gambar yang sudah ditandai untuk dihapus.
    by_source = defaultdict(list)
    for folder in SPLITS.values():
        images_dir = DATASET_DIR / folder / "images"
        if not images_dir.is_dir():
            continue
        for img in images_dir.iterdir():
            if img.suffix.lower() not in IMG_EXTS:
                continue
            if str(img) in del_images:
                continue  # sudah akan dihapus (mis. exact-dup)
            source, frame = video_source_and_frame(img.stem)
            if source is None:
                continue
            n_lbl = count_labels(label_for_image(img))
            by_source[source].append((frame, img, n_lbl))

    thinning_summary = []  # (source, total, keep, removed)
    for source, items in by_source.items():
        total = len(items)
        keep_target = max(MIN_KEEP_PER_VIDEO, math.ceil(total * KEEP_FRACTION))
        if total <= keep_target:
            continue  # tidak perlu dikurangi
        # Prioritaskan frame dengan jumlah label terbanyak (lebih informatif),
        # tie-break dengan nomor frame agar tersebar & deterministik.
        ranked = sorted(items, key=lambda t: (-t[2], t[0]))
        keep_set = {id(x) for x in ranked[:keep_target]}
        removed_here = 0
        for it in items:
            if id(it) not in keep_set:
                _frame, img, _n = it
                mark_image(img, f"near-duplicate video thinning (sumber {source[:30]})")
                removed_here += 1
        thinning_summary.append((source, total, keep_target, removed_here))

    # --- f) Kualitas gambar (opsional) ---
    if args.remove_dark:
        for item in prob.get("dark_images", []):
            mark_image(abspath(item["image"]), f"gelap (brightness={item.get('brightness')})")
    if args.remove_blurry:
        for item in prob.get("blurry_images", []):
            mark_image(abspath(item["image"]), f"blur (laplacian={item.get('blur')})")

    return {
        "delete_images": [(Path(k), v) for k, v in del_images.items()],
        "delete_labels": [(Path(k), v) for k, v in del_labels.items()],
        "rewrite_labels": [(Path(k), v[0], v[1], v[2]) for k, v in rewrites.items()],
        "thinning_summary": thinning_summary,
        "prob": prob,
    }


def write_manifest(plan, args):
    """Tulis deletion manifest SEBELUM ada file yang disentuh."""
    CLEANING_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CLEANING_DIR / "deletion_manifest.txt"
    lines = []
    lines.append("=" * 70)
    lines.append(" DELETION MANIFEST - clean_dataset.py")
    lines.append(f" Dibuat        : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f" Mode          : {'EXECUTE' if args.execute else 'DRY-RUN'}")
    lines.append(f" remove_dark   : {args.remove_dark}")
    lines.append(f" remove_blurry : {args.remove_blurry}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"GAMBAR akan dihapus : {len(plan['delete_images'])}")
    lines.append(f"LABEL  akan dihapus : {len(plan['delete_labels'])}")
    lines.append(f"LABEL  akan ditulis ulang (sebagian baris dibuang) : {len(plan['rewrite_labels'])}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("[HAPUS GAMBAR]")
    for p, reason in sorted(plan["delete_images"], key=lambda x: str(x[0])):
        lines.append(f"  DEL  {rel(p)}\t# {reason}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("[HAPUS LABEL]")
    for p, reason in sorted(plan["delete_labels"], key=lambda x: str(x[0])):
        lines.append(f"  DEL  {rel(p)}\t# {reason}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("[TULIS ULANG LABEL]")
    for p, valid_lines, n_removed, reason in sorted(plan["rewrite_labels"], key=lambda x: str(x[0])):
        lines.append(f"  EDIT {rel(p)}\t# {reason} (sisa {len(valid_lines)} baris)")
    lines.append("")

    manifest_path.write_text("\n".join(lines) + "\n")
    return manifest_path


def execute_plan(plan):
    """Lakukan penghapusan & penulisan ulang. Kembalikan statistik aktual."""
    stats = {"images_deleted": 0, "labels_deleted": 0, "labels_rewritten": 0,
             "lines_removed": 0, "errors": []}

    # Rewrite dulu (sebelum file label terhapus oleh langkah lain — meski sudah dijaga set)
    for p, valid_lines, n_removed, _reason in plan["rewrite_labels"]:
        try:
            p.write_text("\n".join(valid_lines) + ("\n" if valid_lines else ""))
            stats["labels_rewritten"] += 1
            stats["lines_removed"] += n_removed
        except Exception as e:
            stats["errors"].append(f"rewrite {rel(p)}: {e}")

    for p, _reason in plan["delete_images"]:
        try:
            if p.exists():
                p.unlink()
                stats["images_deleted"] += 1
        except Exception as e:
            stats["errors"].append(f"del img {rel(p)}: {e}")

    for p, _reason in plan["delete_labels"]:
        try:
            if p.exists():
                p.unlink()
                stats["labels_deleted"] += 1
        except Exception as e:
            stats["errors"].append(f"del lbl {rel(p)}: {e}")

    return stats


def write_cleaning_report(plan, args, stats=None):
    CLEANING_DIR.mkdir(parents=True, exist_ok=True)
    report_path = CLEANING_DIR / "cleaning_report.txt"
    lines = []
    lines.append("=" * 70)
    lines.append(" CLEANING REPORT - clean_dataset.py")
    lines.append(f" Waktu : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f" Mode  : {'EXECUTE' if args.execute else 'DRY-RUN (tidak ada perubahan)'}")
    lines.append("=" * 70)
    lines.append("")
    prob = plan["prob"]
    lines.append("RINGKASAN MASALAH (dari EDA):")
    lines.append(f"  Grup exact-duplicate     : {len(prob.get('exact_duplicates', []))}")
    lines.append(f"  Sumber video near-dup    : {len(prob.get('near_duplicates_by_video', {}))}")
    lines.append(f"  Corrupt label            : {len(prob.get('corrupt_labels', []))}")
    lines.append(f"  Gambar tanpa label       : {len(prob.get('images_without_labels', []))}")
    lines.append(f"  Orphan label             : {len(prob.get('orphan_labels', []))}")
    lines.append(f"  Gambar gelap             : {len(prob.get('dark_images', []))}")
    lines.append(f"  Gambar blur              : {len(prob.get('blurry_images', []))}")
    lines.append("")
    lines.append("RENCANA CLEANING:")
    lines.append(f"  Gambar dihapus           : {len(plan['delete_images'])}")
    lines.append(f"  Label dihapus            : {len(plan['delete_labels'])}")
    lines.append(f"  Label ditulis ulang      : {len(plan['rewrite_labels'])}")
    lines.append("")
    lines.append("FRAME THINNING per sumber video (total -> keep / removed):")
    for source, total, keep, removed in sorted(plan["thinning_summary"],
                                               key=lambda x: x[3], reverse=True):
        lines.append(f"  {source[:46]:46s} {total:4d} -> keep {keep:3d} / removed {removed:4d}")
    lines.append("")
    if stats is not None:
        lines.append("HASIL EKSEKUSI:")
        lines.append(f"  Gambar terhapus          : {stats['images_deleted']}")
        lines.append(f"  Label terhapus           : {stats['labels_deleted']}")
        lines.append(f"  Label ditulis ulang      : {stats['labels_rewritten']}")
        lines.append(f"  Baris label dibuang      : {stats['lines_removed']}")
        if stats["errors"]:
            lines.append(f"  ERROR ({len(stats['errors'])}):")
            for e in stats["errors"][:20]:
                lines.append(f"    - {e}")
    else:
        lines.append("(DRY-RUN: tidak ada perubahan yang dilakukan)")
    lines.append("")
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Cleaning dataset YOLOv8 berdasarkan EDA.")
    parser.add_argument("--execute", action="store_true",
                        help="Eksekusi penghapusan sungguhan (default: dry-run).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Hanya tampilkan rencana (default jika --execute tidak diberikan).")
    parser.add_argument("--remove-dark", action="store_true",
                        help="Ikut hapus gambar gelap (default: tidak).")
    parser.add_argument("--remove-blurry", action="store_true",
                        help="Ikut hapus gambar blur (default: tidak).")
    parser.add_argument("--yes", action="store_true",
                        help="Lewati prompt konfirmasi (untuk non-interaktif).")
    args = parser.parse_args()

    print("=" * 70)
    print(" CLEANING DATASET DETEKSI BARANG (YOLOv8)")
    print("=" * 70)
    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"Mode          : {mode}")
    print(f"remove_dark   : {args.remove_dark} | remove_blurry : {args.remove_blurry}\n")

    plan = build_plan(args)

    print(f"Rencana: hapus {len(plan['delete_images'])} gambar, "
          f"{len(plan['delete_labels'])} label, "
          f"tulis-ulang {len(plan['rewrite_labels'])} label.\n")

    # Manifest SELALU ditulis dulu, sebelum menyentuh file apa pun.
    manifest_path = write_manifest(plan, args)
    print(f"Deletion manifest ditulis: {rel(manifest_path)}")
    print("  (Tinjau file ini sebelum eksekusi.)\n")

    if not args.execute:
        report_path = write_cleaning_report(plan, args, stats=None)
        print("DRY-RUN selesai. Tidak ada file yang dihapus.")
        print(f"Laporan: {rel(report_path)}")
        print("\nUntuk eksekusi sungguhan: python clean_dataset.py --execute")
        return

    # --- Mode EXECUTE: minta konfirmasi eksplisit ---
    total_ops = (len(plan["delete_images"]) + len(plan["delete_labels"])
                 + len(plan["rewrite_labels"]))
    if total_ops == 0:
        print("Tidak ada yang perlu dibersihkan. Selesai.")
        return

    if not args.yes:
        print("PERINGATAN: operasi ini akan MENGHAPUS file secara permanen.")
        print(f"Total operasi: {total_ops}. Sudah tinjau manifest di atas?")
        try:
            answer = input("Ketik 'yes' untuk lanjut, lainnya untuk batal: ").strip().lower()
        except EOFError:
            answer = ""
        if answer != "yes":
            print("Dibatalkan. Tidak ada file yang dihapus.")
            return

    print("\nMengeksekusi cleaning ...")
    stats = execute_plan(plan)
    report_path = write_cleaning_report(plan, args, stats=stats)
    print("\nSELESAI.")
    print(f"  Gambar terhapus      : {stats['images_deleted']}")
    print(f"  Label terhapus       : {stats['labels_deleted']}")
    print(f"  Label ditulis ulang  : {stats['labels_rewritten']} ({stats['lines_removed']} baris dibuang)")
    if stats["errors"]:
        print(f"  ADA {len(stats['errors'])} ERROR — lihat laporan.")
    print(f"  Laporan: {rel(report_path)}")
    print("\nLangkah berikutnya: python verify_cleaning.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[DIBATALKAN] Cleaning dihentikan oleh user.")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Cleaning gagal: {e}")
        traceback.print_exc()
        sys.exit(1)
