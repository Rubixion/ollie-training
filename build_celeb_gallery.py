"""One pic per celebrity, real name filenames. Run once after downloading VGGFace2."""
import csv, os, shutil, sys

VGGFACE2_ROOT = "vggface2_train"  # folder of id_folders, e.g. n000001/
META_CSV = "identity_meta.csv"    # Class_ID,Name,Sample_Num,Flag,Gender
OUT_DIR = "celeb_gallery"

os.makedirs(OUT_DIR, exist_ok=True)

with open(META_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        class_id, name = row["Class_ID"].strip(), row["Name"].strip().strip('"')
        src_dir = os.path.join(VGGFACE2_ROOT, class_id)
        if not os.path.isdir(src_dir):
            continue
        images = sorted(os.listdir(src_dir))
        if not images:
            continue
        safe_name = name.replace(" ", "_")
        ext = os.path.splitext(images[0])[1]
        shutil.copy(os.path.join(src_dir, images[0]), os.path.join(OUT_DIR, f"{safe_name}{ext}"))

print(f"done: {len(os.listdir(OUT_DIR))} celebrities in {OUT_DIR}/")
