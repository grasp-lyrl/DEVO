import os
import argparse
import requests
import zipfile

urls = [
    "http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_forward_3_davis_with_gt.zip",
    "http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_forward_5_davis_with_gt.zip",
    "http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_forward_6_davis_with_gt.zip",
    "http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_forward_7_davis_with_gt.zip",
    "http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_forward_9_davis_with_gt.zip",
    "http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_forward_10_davis_with_gt.zip"
]

def download_file(url, dest_folder):
    local_filename = os.path.join(dest_folder, url.split('/')[-1])
    print(f"Downloading {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def unzip_file(zip_path, extract_to):
    print(f"Unzipping {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def main():
    parser = argparse.ArgumentParser(description="Download UZH FPV dataset and extract them")
    parser.add_argument('--dir', required=True, dest='download_dir', help='Directory to download and extract the dataset into')
    args = parser.parse_args()

    download_dir = args.download_dir
    os.makedirs(download_dir, exist_ok=True)

    zip_files = []
    # Download files
    for url in urls:
        try:
            zip_path = download_file(url, download_dir)
            zip_files.append(zip_path)
        except Exception as e:
            print(f"Failed to download {url}: {e}")

    # Unzip files
    for zip_path in zip_files:
        try:
            folder_name = os.path.splitext(os.path.basename(zip_path))[0]
            unzip_file(zip_path, os.path.join(download_dir, folder_name))
        except Exception as e:
            print(f"Failed to unzip {zip_path}: {e}")

    print("All done!")

if __name__ == "__main__":
    main()