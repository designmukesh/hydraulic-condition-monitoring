"""Download the UCI hydraulic-system condition-monitoring dataset.

Source: Helwig et al., UCI Machine Learning Repository (2018), donated 2018.
~500 MB total. Run from the repo root:  python data/download_data.py
"""
import os
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "hydraulic.zip")
    print("Downloading UCI hydraulic-system dataset (~500 MB) ...")
    urllib.request.urlretrieve(URL, zip_path)
    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    os.remove(zip_path)
    n = len([f for f in os.listdir(DATA_DIR) if f.endswith(".txt")])
    print(f"Done. {n} .txt files in {DATA_DIR}")


if __name__ == "__main__":
    main()
