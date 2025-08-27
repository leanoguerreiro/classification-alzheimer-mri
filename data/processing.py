import os
import numpy as np
import torch
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from config.settings import Config
from collections import Counter
import matplotlib.pyplot as plt


class DataProcessor:
    def __init__(self):
        self.base_transform = transforms.Compose([
            transforms.Resize(Config.IMG_SIZE),
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(Config.NORMALIZE_MEAN, Config.NORMALIZE_STD)
        ])
        # Transformação extra para classes minoritárias
        self.aug_transform = transforms.Compose([
            transforms.Resize(Config.IMG_SIZE),
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(Config.NORMALIZE_MEAN, Config.NORMALIZE_STD)
        ])
        # Defina aqui as classes minoritárias (exemplo: 1 para Moderate_Demented)
        self.minority_classes = [1]  # Ajuste conforme necessário

    def load_and_split_data(self):
        dataset = load_dataset(Config.DATASET_NAME)
        return dataset["train"].train_test_split(
            test_size=Config.TEST_SIZE,
            stratify_by_column="label",
            seed=Config.SPLIT_SEED
        )

    def preprocess_function(self, examples):
        processed_images = []
        for img, label in zip(examples["image"], examples["label"]):
            if label in self.minority_classes:
                processed_images.append(self.aug_transform(img))
            else:
                processed_images.append(self.base_transform(img))
        return {"image": processed_images, "label": examples["label"]}

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([torch.as_tensor(item['image']) for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        return {'image': images, 'label': labels}

    def create_data_loaders(self, dataset):
        loaders = {}
        for split in ["train", "validation", "test"]:
            kwargs = {
                "batch_size": Config.TRAIN_BATCH_SIZE if split == "train" else Config.VAL_TEST_BATCH_SIZE,
                "shuffle": (split == "train"),
                "num_workers": Config.NUM_WORKERS,
                "pin_memory": Config.PIN_MEMORY,
                "persistent_workers": Config.PERSISTENT_WORKERS
            }
            if Config.NUM_WORKERS > 0:
                kwargs["prefetch_factor"] = Config.PREFETCH_FACTOR
            loader = DataLoader(dataset[split], collate_fn=self.collate_fn, **kwargs)
            loaders[split] = loader
        return loaders

    def get_class_weights(self, dataset):
        labels = dataset['label']
        counts = Counter(labels)
        total = sum(counts.values())
        weights = [total / counts[i] if counts[i] > 0 else 0.0 for i in range(len(counts))]
        weights = torch.tensor(weights, dtype=torch.float)
        return weights


def plot_class_examples(dataset, n=4):
    classes = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
    plt.figure(figsize=(12, 8))
    for class_idx in range(4):
        idx = next(i for i, l in enumerate(dataset['label']) if l == class_idx)
        img = dataset['image'][idx]
        if torch.is_tensor(img):
            img = img.numpy()
        if img.shape[0] == 1:
            img = img.squeeze(0)
        elif img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        plt.subplot(1, 4, class_idx+1)
        plt.imshow(img, cmap='gray')
        plt.title(classes[class_idx])
        plt.axis('off')
    plt.suptitle('Exemplo de cada classe')
    plt.show()


def setup_environment():
    os.environ["OMP_NUM_THREADS"] = Config.OMP_THREADS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 50)
    print(f"Dispositivo em uso: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 50 + "\n")

    x = torch.rand(10).to("cuda")
    print(x)

    return device