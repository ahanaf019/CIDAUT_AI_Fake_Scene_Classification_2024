
from torch.utils.data import  Dataset
import cv2
from torchvision import transforms


def read_image(path, size):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size))
    return image


class ImageDataset(Dataset):
    def __init__(self, image_paths, size, transforms):
        super().__init__()
        self.image_paths = image_paths
        self.size = size
        self.labels = self.__get_labels()
        self.transforms = transforms
    
    def get_class_list(self):
        classnames = [path.split('/')[-2] for path in self.image_paths]
        return sorted(list(set(classnames)))

    def get_class_dict(self):
        class_list = self.get_class_list()
        return {classname: index for index, classname in enumerate(class_list)}
    
    def __get_labels(self):
        labels = [path.split('/')[-2] for path in self.image_paths]
        class_dict = self.get_class_dict()
        return [class_dict[label] for label in labels]


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = read_image(self.image_paths[index], self.size)
        label = self.labels[index]

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label


class ImageDatasetV2(Dataset):
    def __init__(self, base_dir, image_paths, labels, size, transforms):
        super().__init__()
        self.image_paths = image_paths
        self.labels_str = labels
        self.image_paths = [f'{base_dir}/{x}' for x in self.image_paths]
        self.size = size
        self.labels = self.__get_labels()
        # print(self.get_class_dict())
        self.transforms = transforms
    
    def get_class_list(self):
        classnames = self.labels_str
        return sorted(list(set(classnames)))

    def get_class_dict(self):
        class_list = self.get_class_list()
        return {classname: index for index, classname in enumerate(class_list)}
    
    def __get_labels(self):
        labels = self.labels_str
        class_dict = self.get_class_dict()
        return [class_dict[label] for label in labels]


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = read_image(self.image_paths[index], self.size)
        label = self.labels[index]

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('dataset/train.csv')
    image_paths = list(df['image'])
    labels = list(df['label'])
    DB_BASE_DIR = 'dataset/Train/'
    dataset = ImageDatasetV2(DB_BASE_DIR, image_paths, labels, 256, None)
    x, y = next(iter(dataset))
    print(x.shape, y)