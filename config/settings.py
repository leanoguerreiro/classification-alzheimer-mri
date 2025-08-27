class Config:
    # Parâmetros de dataset
    DATASET_NAME = "Falah/Alzheimer_MRI"
    SPLIT_SEED = 42
    TEST_SIZE = 0.2

    # Pré-processamento
    IMG_SIZE = (224, 224)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

    # DataLoader
    TRAIN_BATCH_SIZE = 64
    VAL_TEST_BATCH_SIZE = 32
    NUM_WORKERS = 14
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    PREFETCH_FACTOR = 2
    OMP_THREADS = "1"

    # Hiperparâmetros do modelo
    MODEL_NAME = "efficientnet_b0"  # Nome do modelo. Opções: "mobilenet_v3_small", "resnet18", "efficientnet_b0"
    NUM_CLASSES = 4

    # Hiperparâmetros de treino
    NUM_EPOCHS = 10  # Número de épocas
    LEARNING_RATE = 1e-4  # Taxa de aprendizado
    OPTIMIZER = "adam"  # Tipo de otimizador
    SCHEDULER_MODE = "max"  # Modo do scheduler (min/max)
    SCHEDULER_PATIENCE = 2  # Patience do ReduceLROnPlateau
    EARLY_STOP_PATIENCE = 5  # Early stopping patience
    LOSS_FN = "cross_entropy"  # Função de loss
