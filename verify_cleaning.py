"""
verify_cleaning.py
------------------
Verifikasi dataset masih valid & sehat SETELAH clean_dataset.py dijalankan.

  - Hitung ulang distribusi class (kondisi sekarang / "after").
  - Bandingkan dengan kondisi "before" dari runs/eda/eda_report.txt
    (baris penanda: "CLASS_TOTAL\t<nama>\t<jumlah>").
  - WARNING bila ada class turun > 50% atau menjadi 0.
  - Tabel perbandingan before vs after.
  - Konfirmasi semua label masih valid (tidak ada corrupt tersisa).
  - Cek tidak ada gambar tanpa label / orphan label tersisa.
  - Simpan ke runs/cleaning/verification_report.txt

Jalankan: python verify_cleaning.py
"""

from pathlib import Path
from collections import Counter
import sys

# === KONFIGURASI PATH DASAR ===
def _resolve_base_dir() -> Path:
    """Deteksi BASE_DIR secara dinamis: coba cwd dulu, fallback ke lokasi script."""
    cwd = Path.cwd()
    if (cwd / "datasets" / "data.yaml").exists():
        return cwd
    return Path(__file__).resolve().parent

BASE_DIR = _resolve_base_dir()
DATASET_DIR = BASE_DIR / "datasets"
DATA_YAML = DATASET_DIR / "data.yaml"
EDA_REPORT = BASE_DIR / "runs" / "eda" / "eda_report.txt"
CLEANING_DIR = BASE_DIR / "runs" / "cleaning"

SPLITS = {"train": "train", "valid": "valid", "test": "test"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DROP_WARN_PCT = 50.0  # turun lebih dari ini -> warning


def load_class_names():
    import yaml
    with open(DATA_YAML, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    return list(names)


def parse_before_from_eda():
    """Ambil jumlah instance per class 'before' dari eda_report.txt."""
    before = {}
    total_images_before = None
    if not EDA_REPORT.exists():
        print(f"[WARNING] {EDA_REPORT} tidak ada; perbandingan before/after dilewati.")
        return before, total_images_before
    for line in EDA_REPORT.read_text().splitlines():
        s = line.strip()
        if s.startswith("CLASS_TOTAL"):
            parts = s.split("\t")
            if len(parts) >= 3:
                name, cnt = parts[1], parts[2]
                try:
                    before[name] = int(cnt)
                except ValueError:
                    pass
        elif s.startswith("TOTAL_IMAGES"):
            parts = s.split("\t")
            if len(parts) >= 2:
                try:
                    total_images_before = int(parts[1])
                except ValueError:
                    pass
    return before, total_images_before


def recompute_after(nc):
    """Hitung ulang distribusi class & integritas dari kondisi disk sekarang."""
    from tqdm import tqdm

    class_total = Counter()
    issues = {"corrupt_labels": [], "images_without_labels": [], "orphan_labels": []}
    total_images = 0

    for folder in SPLITS.values():
        images_dir = DATASET_DIR / folder / "images"
        labels_dir = DATASET_DIR / folder / "labels"

        img_stems, lbl_stems = {}, {}
        if images_dir.is_dir():
            for p in images_dir.iterdir():
                if p.suffix.lower() in IMG_EXTS:
                    img_stems[p.stem] = p
        if labels_dir.is_dir():
            for p in labels_dir.iterdir():
                if p.suffix.lower() == ".txt":
                    lbl_stems[p.stem] = p

        total_images += len(img_stems)

        # Gambar tanpa label
        for stem in img_stems:
            if stem not in lbl_stems:
                issues["images_without_labels"].append(str(img_stems[stem].relative_to(BASE_DIR)))
        # Orphan label
        for stem in lbl_stems:
            if stem not in img_stems:
                issues["orphan_labels"].append(str(lbl_stems[stem].relative_to(BASE_DIR)))

        # Validasi isi label
        for stem, lbl in tqdm(lbl_stems.items(), desc=f"Verifikasi [{folder}]", unit="file"):
            try:
                lines = lbl.read_text().splitlines()
            except Exception as e:
                issues["corrupt_labels"].append(f"{lbl.relative_to(BASE_DIR)}: gagal baca ({e})")
                continue
            for i, line in enumerate(lines, 1):
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                bad = None
                if len(parts) != 5:
                    bad = f"baris {i}: kolom={len(parts)}"
                else:
                    try:
                        cls = int(float(parts[0]))
                        xc, yc, w, h = map(float, parts[1:])
                        if not (0 <= cls < nc):
                            bad = f"baris {i}: class_id {cls} invalid"
                        elif not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                            bad = f"baris {i}: koordinat invalid"
                        else:
                            class_total[cls] += 1
                    except ValueError:
                        bad = f"baris {i}: non-numerik"
                if bad:
                    issues["corrupt_labels"].append(f"{lbl.relative_to(BASE_DIR)}: {bad}")
    return class_total, issues, total_images


def run_verify():
    print("=" * 70)
    print(" VERIFIKASI DATASET SETELAH CLEANING")
    print("=" * 70)

    class_names = load_class_names()
    nc = len(class_names)
    before, total_images_before = parse_before_from_eda()
    class_total_after, issues, total_images_after = recompute_after(nc)

    report = []
    def log(line=""):
        print(line)
        report.append(line)

    # --- Tabel perbandingan before vs after ---
    log("=" * 70)
    log(" PERBANDINGAN CLASS: BEFORE vs AFTER")
    log("=" * 70)
    if total_images_before is not None:
        log(f"  Total gambar : {total_images_before} -> {total_images_after} "
            f"(selisih {total_images_after - total_images_before})")
    else:
        log(f"  Total gambar sekarang : {total_images_after}")
    log("")
    log(f"  {'class':18s} {'before':>8s} {'after':>8s} {'delta':>8s} {'%change':>9s}  status")
    log("  " + "-" * 66)

    warnings = []
    has_before = bool(before)
    for cls in range(nc):
        name = class_names[cls]
        aft = class_total_after.get(cls, 0)
        bef = before.get(name) if has_before else None
        if bef is None:
            log(f"  {name:18s} {'?':>8s} {aft:8d} {'?':>8s} {'?':>9s}  (no-before)")
            continue
        delta = aft - bef
        pct = (delta / bef * 100) if bef > 0 else (0.0 if aft == 0 else 100.0)
        status = "ok"
        if aft == 0 and bef > 0:
            status = "JADI NOL!"
            warnings.append(f"Class '{name}' menjadi 0 (sebelumnya {bef}).")
        elif bef > 0 and (-pct) > DROP_WARN_PCT:
            status = "TURUN >50%"
            warnings.append(f"Class '{name}' turun {(-pct):.0f}% ({bef} -> {aft}).")
        log(f"  {name:18s} {bef:8d} {aft:8d} {delta:+8d} {pct:+8.1f}%  {status}")
    log("")

    # --- Integritas label setelah cleaning ---
    log("=" * 70)
    log(" INTEGRITAS LABEL (setelah cleaning)")
    log("=" * 70)
    log(f"  Corrupt label tersisa    : {len(issues['corrupt_labels'])}")
    log(f"  Gambar tanpa label       : {len(issues['images_without_labels'])}")
    log(f"  Orphan label             : {len(issues['orphan_labels'])}")
    for key, label in (("corrupt_labels", "Corrupt"),
                       ("images_without_labels", "Gambar tanpa label"),
                       ("orphan_labels", "Orphan label")):
        if issues[key]:
            log(f"    {label} (maks 10):")
            for x in issues[key][:10]:
                log(f"      {x}")
    log("")

    # --- Kesimpulan ---
    log("=" * 70)
    log(" KESIMPULAN")
    log("=" * 70)
    integrity_ok = (not issues["corrupt_labels"]
                    and not issues["images_without_labels"]
                    and not issues["orphan_labels"])
    if warnings:
        log(f"  [{len(warnings)} WARNING DISTRIBUSI]")
        for w in warnings:
            log(f"    - {w}")
    else:
        log("  Distribusi class: tidak ada penurunan drastis (>50%) atau class hilang.")

    if integrity_ok:
        log("  Integritas label: LULUS (semua label valid, tidak ada orphan/tanpa-label).")
    else:
        log("  Integritas label: ADA MASALAH (lihat detail di atas).")

    verdict_ok = integrity_ok and not warnings
    log("")
    log(f"  STATUS AKHIR: {'OK - dataset siap dipakai' if verdict_ok else 'PERLU DITINJAU'}")

    # --- Simpan laporan ---
    CLEANING_DIR.mkdir(parents=True, exist_ok=True)
    report_path = CLEANING_DIR / "verification_report.txt"
    report_path.write_text("\n".join(report) + "\n")
    print(f"\nLaporan verifikasi: {report_path.relative_to(BASE_DIR)}")

    return verdict_ok


if __name__ == "__main__":
    try:
        ok = run_verify()
        sys.exit(0 if ok else 2)
    except KeyboardInterrupt:
        print("\n[DIBATALKAN] Verifikasi dihentikan oleh user.")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Verifikasi gagal: {e}")
        traceback.print_exc()
        sys.exit(1)
