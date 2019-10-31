from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image


def get_train_test_split(train_test_split_path):
    """Reads train_test_split.txt in CUB_20 dataset and returns lists with train/test image id"""
    train_image_id_list = []
    test_image_id_list = []
    with open(train_test_split_path, "r") as infile:
        for line in infile:
            img_id, is_train_image = line.strip().split()
            if is_train_image:
                train_image_id_list.append(img_id)
            else:
                test_image_id_list.append(img_id)
    return train_image_id_list, test_image_id_list


def get_id_to_image_path_dict(images_file_path):
    """Reads images.txt in CUB_20 dataset and returns a dictionary of key=image_id, value=image_path"""
    id_to_path_dict = {}
    with open(images_file_path, "r") as infile:
        for line in infile:
            img_id, img_path = line.strip().split()
            id_to_path_dict[img_id] = img_path
    return id_to_path_dict


def get_image_to_label_dict(img_class_labels_path):
    """Reads image_class_labels.txt in CUB_20 dataset and returns a dictionary of key=image_is, value=label"""
    id_to_label = {}
    with open(img_class_labels_path, "r") as infile:
        for line in infile:
            img_id, label = line.strip().split()
            id_to_label[img_id] = label
    return id_to_label


class CUBDataset(Dataset):
    def __init__(self, root_directory, set_, transforms=None):
        self.train_set = True if set_ == "train" else False
        self.transforms = transforms

        root_directory = Path(root_directory).expanduser().resolve()
        train_test_split = root_directory / "train_test_split.txt"
        train_image_id_list, test_image_id_list = get_train_test_split(train_test_split)

        images_file_path = root_directory / "images.txt"
        id_to_path_dict = get_id_to_image_path_dict(images_file_path)

        img_class_labels_path = root_directory / "image_class_labels.txt"
        id_to_label_dict = get_image_to_label_dict(img_class_labels_path)
        self.number_classes = len(set(id_to_label_dict.values()))

        self.image_dir = root_directory/"images"
        if self.train_set:
            self.img_list = [(id_to_path_dict[x], id_to_label_dict[x]) for x in train_image_id_list]
        else:
            self.img_list = [(id_to_path_dict[x], id_to_label_dict[x]) for x in test_image_id_list]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        path, lbl = self.img_list[idx]
        full_path = self.image_dir/path
        img = Image.open(full_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return img, int(lbl) - 1
