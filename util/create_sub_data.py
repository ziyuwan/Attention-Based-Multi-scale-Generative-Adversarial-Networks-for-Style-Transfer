import os
import random
import shutil

from tqdm import tqdm


def create_sub_data(folder, dataset):
    sub_dataset = dataset + '_sub'
    dataset_path = os.path.join(folder, dataset)
    sub_dataset_path = os.path.join(folder, sub_dataset)
    if not os.path.exists(sub_dataset_path):
        os.mkdir(sub_dataset_path)
    files = os.listdir(dataset_path)
    sub_files = random.sample(files, len(files) // 5)
    sub_files = [os.path.join(dataset_path, file) for file in sub_files]
    for file in tqdm(sub_files):
        shutil.copy(file, sub_dataset_path)


def test_image(folder, dataset):
    from PIL import Image
    dataset_path = os.path.join(folder, dataset)
    files = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path)]
    for file in tqdm(files):
        try:
            Image.open(file)
        except Exception:
            print(os.path.basename(file))


if __name__ == '__main__':
    test_image('/NAS2020/Workspaces/MLGroup/chenxianyu/Dataset/AMSGAN', 'train2014_sub')
