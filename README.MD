 # Projeto: Classificação de Alzheimer em Imagens de Ressonância (MRI)

Modelo de visão computacional para classificação de estágios de Alzheimer usando imagens de ressonância magnética.
Suporta pré-processamento com augmentations dirigidas a classes minoritárias, treinamento com pesos de classe,
registro no TensorBoard e exportação de métricas detalhadas (globais e por classe).

## Classes
1. Mild_Demented
2. Moderate_Demented
3. Non_Demented
4. Very_Mild_Demented

## Principais Recursos
- Download automático do dataset (HuggingFace: Falah/Alzheimer_MRI)
- Split estratificado e cache em disco (train / validation / test)
- Augmentation apenas para classes minoritárias configuráveis
- Cálculo automático de pesos das classes para CrossEntropy
- Suporte a modelos: EfficientNet-B0, MobileNet V3 Small, ResNet18
- Early Stopping + ReduceLROnPlateau
- Logs no TensorBoard (imagens, losses, accuracy, métricas por classe, matriz de confusão)
- Exportação de métricas:
  - metrics/validation_per_class_epoch.csv (por época / por classe)
  - metrics/test_per_class_metrics.csv (teste final por classe)
  - metrics/test_classification_report.txt (relatório completo)
  - metrics/confusion_matrix_test.png (matriz de confusão absoluta)
- Salvamento do melhor modelo: best_model.pth

## Estrutura Simplificada
```
config/settings.py        # Configurações centrais
main.py                   # Pipeline principal de treinamento e avaliação
data/processing.py        # Pré-processamento e DataLoaders
models/trainer.py         # Lógica de treino, validação e métricas
metrics/                  # Saída de métricas geradas
alzheimer_preprocessado/  # Cache dos splits HuggingFace após processamento
runs/                     # Logs TensorBoard
```

## Requisitos
Python 3.10+ (recomendado)

Pacotes principais:
- torch / torchvision
- datasets
- scikit-learn
- matplotlib
- seaborn
- tensorboard
- numpy

### Instalação (Exemplo)
Crie e ative um ambiente virtual e instale dependências:
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # ajuste conforme GPU/CPU
pip install datasets scikit-learn matplotlib seaborn tensorboard numpy
```

## Execução
### Treinamento + Avaliação
```
python main.py
```
Primeira execução fará o download, split, preprocessing e cache.

### Visualizar TensorBoard
```
tensorboard --logdir runs
```
Acesse em: http://localhost:6006

## Configuração
Edite config/settings.py para ajustar:
- DATASET_NAME: nome no HuggingFace
- MODEL_NAME: "efficientnet_b0" | "mobilenet_v3_small" | "resnet18"
- NUM_EPOCHS, LEARNING_RATE, OPTIMIZER
- TEST_SIZE, IMG_SIZE
- NUM_WORKERS (ajuste conforme CPU; em Windows pode reduzir se houver travamentos)

## Métricas por Classe
Durante validação (cada época): precision, recall, f1 e suporte são anexados ao arquivo:
metrics/validation_per_class_epoch.csv

Após teste final:
- metrics/test_per_class_metrics.csv
- metrics/test_classification_report.txt
- metrics/confusion_matrix_test.png

Isso permite análise temporal (overfitting, impacto de pesos e augmentations).

## Ajuste de Classes Minoritárias
Em data/processing.py:
```
self.minority_classes = [1]  # IDs das classes que recebem augmentations extras
```
Adapte a lista conforme distribuição de dados.

## Pesos de Classe
Calculados em tempo de execução na base train e aplicados em CrossEntropy.
Podem ser desativados passando class_weights=None ao Trainer.

## Substituindo o Modelo
Basta alterar MODEL_NAME em config/settings.py. Camada final é adaptada para NUM_CLASSES.

## Melhorias Futuras (Sugestões)
- Scheduler alternativo (OneCycleLR / CosineAnnealing)
- Mixed Precision (torch.cuda.amp) para acelerar em GPU
- Exportação ONNX / TorchScript do modelo final
- Implementar Grad-CAM para interpretabilidade
- A/B test de estratégias de augmentation

## Troubleshooting
- CUDA OOM: Reduza TRAIN_BATCH_SIZE / VAL_TEST_BATCH_SIZE.
- DataLoader lento: Ajuste NUM_WORKERS e PREFETCH_FACTOR.
- TensorBoard vazio: Verifique se a pasta runs/ foi gerada após pelo menos 1 época.

## Referências
Dataset: https://huggingface.co/datasets/Falah/Alzheimer_MRI
Modelos: TorchVision Model Zoo
Sklearn Metrics: https://scikit-learn.org/

