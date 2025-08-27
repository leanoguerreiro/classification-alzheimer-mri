import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import Counter
import torch
import numpy as np

from config.settings import Config

def plot_class_distribution(labels, class_names):
    counts = Counter(labels)
    plt.figure(figsize=(8, 5))
    plt.bar(class_names, [counts[i] for i in range(len(class_names))], color='skyblue')
    plt.xlabel('Classe')
    plt.ylabel('Quantidade')
    plt.title('Distribuição das Classes')
    plt.savefig('class_distribution_bar.png')
    plt.show()
    plt.figure(figsize=(6, 6))
    plt.pie([counts[i] for i in range(len(class_names))], labels=class_names, autopct='%1.1f%%', startangle=90)
    plt.title('Proporção das Classes')
    plt.savefig('class_distribution.png')
    plt.show()

def show_class_examples(dataset, class_names, n=3):
    plt.figure(figsize=(12, 3 * len(class_names)))
    for class_idx, class_name in enumerate(class_names):
        idxs = [i for i, l in enumerate(dataset['label']) if l == class_idx][:n]
        for j, idx in enumerate(idxs):
            img = dataset['image'][idx]

            img = np.array(img)

            if torch.is_tensor(img):
                img = img.numpy()

            if len(img.shape) == 2 or img.shape[-1] == 1:
                cmap = 'gray'
            else:
                cmap = None

            plt.subplot(len(class_names), n, class_idx * n + j + 1)
            plt.imshow(img, cmap=cmap)
            plt.title(f'{class_name}')
            plt.axis('off')
    plt.suptitle('Exemplos de cada classe')
    plt.tight_layout()
    plt.savefig('class_examples.png')
    plt.show()

def main():
    print('Carregando dataset...')
    dataset = load_dataset(Config.DATASET_NAME)['train']
    labels = dataset['label']
    class_names = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
    print('Contagem por classe:')
    for i, name in enumerate(class_names):
        print(f'{name}: {labels.count(i)}')
    plot_class_distribution(labels, class_names)
    show_class_examples(dataset, class_names, n=3)
    print('\nSugestão: Classes com menor quantidade podem ser balanceadas via oversampling, data augmentation ou técnicas como SMOTE.')

if __name__ == "__main__":
    main()

