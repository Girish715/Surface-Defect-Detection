import zipfile
import os

zip_path = "archive (3).zip"
extract_path = "/content/neu_dataset"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# List files to verify
for root, dirs, files in os.walk(extract_path):
    print("Root:", root)
    print("Dirs:", dirs)
    print("Files:", files[:5])  # print just a few filenames
    break
