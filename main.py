from datasets import load_dataset, DatasetDict, load_from_disk
from data.processing import DataProcessor, setup_environment, plot_class_examples
from models.trainer import Trainer
from config.settings import Config
import os

def main():
    # Configuração inicial
    device = setup_environment()
    processor = DataProcessor()

    # Diretório de dados
    base_dir = "alzheimer_preprocessado"
    paths = {
        "train": os.path.join(base_dir, "train"),
        "validation": os.path.join(base_dir, "validation"),
        "test": os.path.join(base_dir, "test")
    }

    # Verifica se os dados já estão salvos
    if all(os.path.exists(p) for p in paths.values()):
        print("\nCarregando datasets pré-processados...")
        final_dataset = DatasetDict({
            "train": load_from_disk(paths["train"]),
            "validation": load_from_disk(paths["validation"]),
            "test": load_from_disk(paths["test"])
        })
    else:
        print("\nPré-processando dados...")
        split_dataset = processor.load_and_split_data()

        final_dataset = DatasetDict({
            "train": split_dataset["train"],
            "validation": split_dataset["test"],  # valid será o antigo test do split
            "test": load_dataset(Config.DATASET_NAME)["test"]
        })

        final_dataset = final_dataset.map(
            processor.preprocess_function,
            batched=True,
            batch_size=Config.TRAIN_BATCH_SIZE,
            num_proc=Config.NUM_WORKERS,
            writer_batch_size=Config.TRAIN_BATCH_SIZE
        )

        # Salvar os splits separadamente
        for split in final_dataset:
            os.makedirs(paths[split], exist_ok=True)
            final_dataset[split].save_to_disk(paths[split])

    # Criar DataLoaders
    loaders = processor.create_data_loaders(final_dataset)

    # Calcular pesos das classes automaticamente
    class_weights = processor.get_class_weights(final_dataset['train'])
    print(f"Pesos das classes calculados: {class_weights}")

    # Iniciar treino com pesos
    trainer = Trainer(loaders, device, class_weights=class_weights)
    trainer.train(Config.NUM_EPOCHS)
    trainer.evaluate()

    # Plotar exemplos de cada classe
    plot_class_examples(final_dataset['train'])

if __name__ == "__main__":
    main()
    print("Processo concluído com sucesso!")
