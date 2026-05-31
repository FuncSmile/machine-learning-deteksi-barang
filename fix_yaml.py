"""
fix_yaml.py
-----------
Memperbaiki data.yaml agar menggunakan ABSOLUTE PATH yang benar.

Masalah: data.yaml hasil export Roboflow memakai relative path seperti
`../train/images`. Path relatif ini bergantung pada lokasi terminal saat
script dijalankan, sehingga sering menyebabkan error "dataset not found".

Solusi: tulis ulang field `train`, `val`, dan `test` menjadi absolute path
yang menunjuk ke folder datasets/ di dalam working directory ini.

Script ini idempotent: aman dijalankan berkali-kali.

Deteksi BASE_DIR (otomatis, tanpa flag):
  1. Jika datasets/data.yaml ditemukan di current working directory → pakai cwd.
  2. Jika tidak, gunakan direktori tempat file script ini berada sebagai fallback.
"""

from pathlib import Path
import yaml


def _resolve_base_dir() -> Path:
    """Tentukan BASE_DIR secara dinamis berdasarkan lokasi pemanggilan script."""
    cwd = Path.cwd()
    if (cwd / "datasets" / "data.yaml").exists():
        return cwd
    # Fallback: direktori tempat fix_yaml.py berada
    return Path(__file__).resolve().parent


# === KONFIGURASI PATH DASAR ===
BASE_DIR = _resolve_base_dir()
DATASETS_DIR = BASE_DIR / "datasets"
DATA_YAML = DATASETS_DIR / "data.yaml"


def fix_yaml() -> bool:
    """Baca data.yaml, ganti path train/val/test menjadi absolute, lalu simpan.

    Returns:
        True jika berhasil di-update, False jika gagal.
    """
    print("=" * 70)
    print("FIX DATA.YAML  ->  mengubah path relatif menjadi absolute")
    print("=" * 70)
    print(f"[INFO] BASE_DIR terdeteksi: {BASE_DIR}")

    # --- Validasi keberadaan file & folder ---
    if not DATA_YAML.exists():
        print(f"[ERROR] data.yaml tidak ditemukan di: {DATA_YAML}")
        return False

    # Mapping split -> folder images yang seharusnya ada
    split_dirs = {
        "train": DATASETS_DIR / "train" / "images",
        "val": DATASETS_DIR / "valid" / "images",
        "test": DATASETS_DIR / "test" / "images",
    }

    try:
        # --- Baca konfigurasi lama ---
        with DATA_YAML.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            print("[ERROR] data.yaml kosong / tidak valid.")
            return False

        print("\n[INFO] Path lama:")
        for split in ("train", "val", "test"):
            print(f"   {split:5s}: {data.get(split, '(tidak ada)')}")

        # --- Set absolute path & beri warning jika foldernya tidak ada ---
        for split, img_dir in split_dirs.items():
            if not img_dir.exists():
                print(f"[WARNING] Folder untuk '{split}' tidak ditemukan: {img_dir}")
            # Tetap tulis path absolute meskipun folder belum ada,
            # agar konfigurasi konsisten.
            data[split] = str(img_dir)

        # Pastikan field penting tetap ada (jaga-jaga jika file rusak)
        if "nc" not in data or "names" not in data:
            print("[WARNING] field 'nc' atau 'names' tidak lengkap di data.yaml.")

        # --- Tulis kembali ---
        with DATA_YAML.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False,
                           allow_unicode=True)

        print("\n[INFO] Path baru (absolute):")
        for split in ("train", "val", "test"):
            print(f"   {split:5s}: {data[split]}")

        print(f"\n[OK] data.yaml berhasil diperbarui: {DATA_YAML}")
        return True

    except yaml.YAMLError as e:
        print(f"[ERROR] Gagal parsing YAML: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan tak terduga: {e}")
        return False


if __name__ == "__main__":
    success = fix_yaml()
    if not success:
        print("\n[GAGAL] data.yaml TIDAK berhasil diperbaiki. Periksa pesan di atas.")
        raise SystemExit(1)
