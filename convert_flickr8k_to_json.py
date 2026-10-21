import json
import csv
from pathlib import Path

input_file = Path("data/captions.txt")   # CSV file
output_file = Path("data/captions.json")

records = []

with open(input_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)  # uses header: image,caption
    for row in reader:
        image = row.get("image", "").strip()
        caption = row.get("caption", "").strip()

        if not image or not caption:
            continue

        records.append({
            "image": image,
            "caption": caption
        })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)

print(f"✅ Converted {len(records)} captions")