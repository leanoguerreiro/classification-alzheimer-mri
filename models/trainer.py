import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
from config.settings import Config
from torch.utils.tensorboard import SummaryWriter
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
import os  # novo
import csv  # novo


class Trainer:
    def __init__(self, loaders, device, class_weights=None):
        self.loaders = loaders
        self.device = device
        self.model = self._initialize_model()
        self.criterion = self._get_loss_fn(class_weights)
        self.optimizer = self._get_optimizer()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, Config.SCHEDULER_MODE, patience=Config.SCHEDULER_PATIENCE)
        self.best_acc = 0.0
        self.early_stop_patience = Config.EARLY_STOP_PATIENCE
        self.no_improve_epochs = 0
        self.writer = SummaryWriter(log_dir="runs/alzheimer")
        self.class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']  # nomes canônicos
        os.makedirs('metrics', exist_ok=True)

    def _initialize_model(self):
        if Config.MODEL_NAME == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights='DEFAULT')
            model.classifier[3] = nn.Linear(1024, Config.NUM_CLASSES)
        elif Config.MODEL_NAME == "resnet18":
            model = models.resnet18(weights='DEFAULT')
            model.fc = nn.Linear(model.fc.in_features, Config.NUM_CLASSES)
        elif Config.MODEL_NAME == "efficientnet_b0":
            model = models.efficientnet_b0(weights='DEFAULT')
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, Config.NUM_CLASSES)
        else:
            raise NotImplementedError(f"Modelo {Config.MODEL_NAME} não implementado.")
        return model.to(self.device)

    def _get_loss_fn(self, class_weights=None):
        if Config.LOSS_FN == "cross_entropy":
            if class_weights is not None:
                return nn.CrossEntropyLoss(weight=class_weights.to(self.device))
            else:
                return nn.CrossEntropyLoss()
        elif Config.LOSS_FN == "mse":
            return nn.MSELoss()
        elif Config.LOSS_FN == "bce":
            return nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"Loss {Config.LOSS_FN} não implementada.")

    def _get_optimizer(self):
        if Config.OPTIMIZER == "adam":
            return optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        elif Config.OPTIMIZER == "sgd":
            return optim.SGD(self.model.parameters(), lr=Config.LEARNING_RATE, momentum=0.9)
        elif Config.OPTIMIZER == "rmsprop":
            return optim.RMSprop(self.model.parameters(), lr=Config.LEARNING_RATE)
        else:
            raise NotImplementedError(f"Otimizador {Config.OPTIMIZER} não implementado.")

    def _epoch_step(self, phase, dataloader, epoch=None):
        self.model.train() if phase == 'train' else self.model.eval()
        running_loss = 0.0
        corrects = 0
        total = 0
        first_batch_logged = False
        all_preds = []
        all_labels = []

        with torch.set_grad_enabled(phase == 'train'):
            for batch in tqdm(dataloader, desc=f'{phase.capitalize()} Epoch'):
                inputs = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

                if phase == 'validation':
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                # Log de imagens de exemplo na primeira época do treino
                if phase == 'train' and not first_batch_logged:
                    img_grid = torchvision.utils.make_grid(inputs[:8].cpu())
                    self.writer.add_image('Exemplos Treino', img_grid)
                    first_batch_logged = True

                del inputs, labels, outputs, loss
                torch.cuda.empty_cache()

        epoch_loss = running_loss / total
        epoch_acc = corrects.double() / total

        # Logs extras para validação
        if phase == 'validation' and epoch is not None:
            # Métricas adicionais globais
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            self.writer.add_scalar('Val/Precision', precision, epoch)
            self.writer.add_scalar('Val/Recall', recall, epoch)
            self.writer.add_scalar('Val/F1', f1, epoch)

            # Métricas por classe
            report_dict = classification_report(all_labels, all_preds, target_names=self.class_names, zero_division=0, output_dict=True)
            for cname in self.class_names:
                class_metrics = report_dict.get(cname, {})
                self.writer.add_scalar(f'Val/Class/{cname}_precision', class_metrics.get('precision', 0), epoch)
                self.writer.add_scalar(f'Val/Class/{cname}_recall', class_metrics.get('recall', 0), epoch)
                self.writer.add_scalar(f'Val/Class/{cname}_f1', class_metrics.get('f1-score', 0), epoch)

            # Salvar CSV incremental (por época)
            csv_path = os.path.join('metrics', 'validation_per_class_epoch.csv')
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    header = ['epoch', 'class', 'precision', 'recall', 'f1', 'support']
                    writer.writerow(header)
                for cname in self.class_names:
                    m = report_dict.get(cname, {})
                    writer.writerow([epoch, cname, m.get('precision', 0), m.get('recall', 0), m.get('f1-score', 0), m.get('support', 0)])

            # Imagens de validação (primeira época)
            if epoch == 0:
                val_batch = next(iter(dataloader))
                img_grid = torchvision.utils.make_grid(val_batch['image'][:8].cpu())
                self.writer.add_image('Exemplos Validação', img_grid, epoch)

            # Matriz de confusão
            cm = confusion_matrix(all_labels, all_preds, normalize='true')
            fig, ax = plt.subplots(figsize=(6,6))
            sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                        xticklabels=['Mild', 'Moderate', 'Non', 'Very Mild'],
                        yticklabels=['Mild', 'Moderate', 'Non', 'Very Mild'])
            plt.xlabel('Predito')
            plt.ylabel('Real')
            plt.title('Matriz de Confusão Normalizada')
            plt.tight_layout()
            self.writer.add_figure('Val/Confusion_Matrix', fig, epoch)
            plt.close(fig)

        return epoch_loss, epoch_acc.cpu().numpy()

    def train(self, num_epochs=None):
        if num_epochs is None:
            num_epochs = Config.NUM_EPOCHS
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')

            train_loss, train_acc = self._epoch_step('train', self.loaders['train'], epoch)
            val_loss, val_acc = self._epoch_step('validation', self.loaders['validation'], epoch)

            self.scheduler.step(val_acc)

            print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

            # Logs do TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)

            # Histogramas dos pesos
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(name, param, epoch)

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.no_improve_epochs = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
                print('Modelo melhorado! Salvando...')
            else:
                self.no_improve_epochs += 1
                print(f'Sem melhora há {self.no_improve_epochs} épocas')

            # Limpeza de memória após cada época
            del train_loss, train_acc, val_loss, val_acc
            torch.cuda.empty_cache()

            if self.no_improve_epochs >= self.early_stop_patience:
                print('\nEarly stopping!')
                break
        self.writer.close()

    def evaluate(self):
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.to(self.device)
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.loaders['test'], desc='Teste'):
                inputs = batch['image'].to(self.device)
                labels = batch['label'].cpu().numpy()

                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

        label_names = self.class_names
        print('\nRelatório de Classificação:')
        report_text = classification_report(all_labels, all_preds, target_names=label_names, zero_division=0)
        print(report_text)

        # Salvar relatório completo em TXT
        with open(os.path.join('metrics', 'test_classification_report.txt'), 'w') as f:
            f.write(report_text)

        # Salvar métricas por classe em CSV
        report_dict = classification_report(all_labels, all_preds, target_names=label_names, zero_division=0, output_dict=True)
        csv_path = os.path.join('metrics', 'test_per_class_metrics.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['class', 'precision', 'recall', 'f1', 'support'])
            for cname in label_names:
                m = report_dict.get(cname, {})
                writer.writerow([cname, m.get('precision', 0), m.get('recall', 0), m.get('f1-score', 0), m.get('support', 0)])

        # Matriz de confusão absoluta salva em arquivo
        cm = confusion_matrix(all_labels, all_preds, normalize=None)
        fig, ax = plt.subplots(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=['Mild', 'Moderate', 'Non', 'Very Mild'],
                    yticklabels=['Mild', 'Moderate', 'Non', 'Very Mild'])
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.title('Matriz de Confusão (Teste)')
        plt.tight_layout()
        fig_path = os.path.join('metrics', 'confusion_matrix_test.png')
        plt.savefig(fig_path)
        plt.close(fig)