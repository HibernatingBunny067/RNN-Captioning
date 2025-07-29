import os
import zipfile
import urllib.request
import shutil
from tqdm import tqdm

'''
download_and_extract( ): used to download the MS COCO dataset to a desired folder 
location 
- used to download MS COCO in the Colab VM Storage during preprocessing 
'''


class CocoDownloader:
    COCO_URLS = {
        'train': 'http://images.cocodataset.org/zips/train2014.zip',
        'val': 'http://images.cocodataset.org/zips/val2014.zip'
    }

    def __init__(self, dest_path='coco_images', parts=['train', 'val']):
        self.dest_path = dest_path
        self.parts = parts
        os.makedirs(self.dest_path, exist_ok=True)

    def download_and_extract(self, name, url):
        zip_path = os.path.join(self.dest_path, f'{name}.zip')
        target_dir = os.path.join(self.dest_path, f'{name}2014')

        if os.path.exists(target_dir) and any(fname.endswith('.jpg') for fname in os.listdir(target_dir)):
            print(f'{target_dir} already exists and contains images, skipping download.')
            return

        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=f'Downloading {name}2014') as bar:
            def reporthook(block_num, block_size, total_size):
                bar.total = total_size
                bar.update(block_size)

            print(f'Downloading {name}...')
            urllib.request.urlretrieve(url, zip_path, reporthook)

        print(f'Extracting {name}...')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.dest_path)
        os.remove(zip_path)
        print(f'{name} downloaded and extracted.')

    def run(self):
        for part in self.parts:
            if part in CocoDownloader.COCO_URLS:
                self.download_and_extract(part, CocoDownloader.COCO_URLS[part])
            else:
                print(f'Unknown part: {part}')
