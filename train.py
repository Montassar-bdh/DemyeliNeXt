import json
import os.path
from copy import deepcopy
from pprint import pprint
from statistics import mean
from typing import Optional, Tuple

import torch
from easyfsl.methods import FewShotClassifier
from matplotlib import pyplot as plt
from monai.utils.type_conversion import convert_to_tensor
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, ConfusionMatrix
from torchmetrics.classification import (BinaryF1Score, BinaryPrecision,
                                         BinaryRecall, BinarySpecificity)
from tqdm import tqdm


def run_epoch(model, criterion, optimizer, scheduler, phase, dataloaders, dataset_sizes, device, use_amp=True, ):
    """Runs a single epoch of training or validation on the given model.

    Args:
        model (torch.nn.Module): The neural network model.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for updating the model's parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        phase (str): Either "train" or "val" to indicate whether to train or validate the model.
        dataloaders (dict[str, torch.utils.data.DataLoader]): A dictionary containing the data loaders
            for the training, validation, and test sets.
        dataset_sizes (dict[str, int]): A dictionary containing the sizes of the training, validation,
            and test sets.
        device (torch.device): The device on which to perform the computation.

    Returns:
        Tuple[float, float]: The loss and accuracy for the epoch.
    """
    total_loss = 0.0
    correct_preds = 0
    targets = []
    all_predictions = []
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    # Set the model to train
    model.train()

    # Iterate over the training data loader
    for inputs, labels in tqdm(dataloaders):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            # Set gradients to zero in training phase only
            with torch.set_grad_enabled(phase == "train"):
                # Forward pass to get model predictions
                outputs = model(inputs)
                assert outputs.dtype is torch.float16
                # Get the predicted class for each input
                _, preds = torch.max(outputs, 1)
                # Compute the batch loss between the predicted and true labels
                batch_loss = criterion(outputs, labels)

            # Backpropagate the loss and update the model parameters
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Compute the total loss for the epoch
        total_loss += batch_loss.item() * inputs.shape[0]

        # Compute the number of correct predictions
        correct_preds += torch.sum(preds == labels.data)
        all_predictions.append(preds)
        targets.append(labels)
    # Update the learning rate if in training phase
    if phase == "train":
        scheduler.step()

    # Compute the average loss and accuracy for the epoch
    epoch_loss = total_loss / dataset_sizes[phase]
    epoch_acc = correct_preds.double() / dataset_sizes[phase]

    # Print the loss and accuracy for the epoch
    print(f"{phase} loss: {epoch_loss:.4f} Acc: {epoch_acc * 100:.2f}%")
    return epoch_loss, epoch_acc * 100, targets, all_predictions


def train_model(backbone, few_shot_classifier, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes,
                writer, device='cuda', validation_frequency=10,
                metrics_path="DemyeliNeXt/results/classic_training"):
    # initialize the best model variables
    best_acc = 0.0
    best_epoch = 0

    if not backbone.use_fc:
        backbone.set_use_fc(True)
    best_backbone = backbone
    phases = ["train", "valid"]
    # loop over the number of epochs
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print("*" * 4)

        # TODO: calculate and save training metrics

        epoch_loss, epoch_acc, targets, all_predictions = run_epoch(backbone, criterion, optimizer, scheduler,
                                                                    phases[0],
                                                                    dataloaders[phases[0]],
                                                                    dataset_sizes, device, )

        predictions_tensor = convert_to_tensor(torch.cat(all_predictions))
        predictions = torch.tensor(predictions_tensor).to(device)
        targets_tensor = torch.cat(targets).to(device)

        (train_accuracy, train_binary_precision, train_binary_specificity, train_binary_recall, train_binary_f1_score,
         train_confusion_matrix) = get_metrics(predictions, targets_tensor)

        conf = ConfusionMatrix(task='binary', num_classes=2)

        fig_, ax_, = conf.plot(val=train_confusion_matrix, labels=["NO MS", "MS"])
        path = os.path.join(metrics_path, "train")
        if not os.path.exists(path):
            os.makedirs(path)

        metrics = {"train_accuracy": round(train_accuracy * 100, 2),
                   "train_binary_precision": round(train_binary_precision, 2),
                   "train_binary_specificity": round(train_binary_specificity, 2),
                   "train_binary_recall": round(train_binary_recall, 2),
                   "train_binary_f1_score": round(train_binary_f1_score, 2),
                   "train_confusion_matrix": train_confusion_matrix.tolist(),
                   "train_loss": round(epoch_loss, 2)
                   }

        with open(os.path.join(path, f"train_metrics_epoch_{epoch}.txt"), "w") as file:
            json.dump(metrics, file, indent=4)

        fig_.savefig(os.path.join(path, f"train_confusion_matrix_epoch_{epoch}.png"))

        pprint(metrics)

        writer.add_scalar("Train/loss", epoch_loss, epoch)
        writer.add_scalar("Train/acc", train_accuracy * 100, epoch)

        if epoch % validation_frequency == validation_frequency - 1:

            # We use this very convenient method from EasyFSL's model to specify
            # that the backbone shouldn't use its last fully connected layer during validation.
            backbone.set_use_fc(False)
            few_shot_classifier.backbone = backbone

            (validation_accuracy, validation_binary_precision, validation_binary_specificity, validation_binary_recall,
             validation_binary_f1_score, val_loss) = evaluate_model(
                few_shot_classifier, dataloaders[phases[1]], device=device, tqdm_prefix="Validation",
                phase="validation",
                metrics_path=metrics_path, criterion=criterion, epoch=epoch)

            backbone.set_use_fc(True)
            # early stopping: check if the current validation accuracy is better than the best accuracy so far
            if validation_accuracy > best_acc:
                best_acc = validation_accuracy
                best_state = deepcopy(few_shot_classifier.state_dict())
                best_epoch = epoch + 1
                torch.save(best_state,
                           f"{metrics_path}/Densenet_classic_train_epoch{best_epoch}.pth")
                print("Ding ding ding! We found a new best model!")

        print("-" * 10)

    # print the best validation accuracy and the corresponding epoch number
    print(f"Best validation Acc: {best_acc * 100:.2f}% (epoch {best_epoch})")

    return best_backbone, best_epoch

def episodic_epoch(few_shot_classifier, data_loader, optimizer, criterion, device):
    all_loss = []
    predictions = []
    targets = []
    few_shot_classifier.backbone.train()
    with tqdm(enumerate(data_loader), total=len(data_loader), desc="Training") as tqdm_train:
        for episode_index, (support_images, support_labels, query_images, query_labels, _,) in tqdm_train:
            with torch.set_grad_enabled(True):
                optimizer.zero_grad()

                classification_scores = predict(few_shot_classifier,
                                                support_images.to(device).type(torch.cuda.FloatTensor),
                                                support_labels.to(device),
                                                query_images.to(device).type(torch.cuda.FloatTensor),
                                                enable_detach=False)

                predicted_classes = get_classes(classification_scores)

                loss = criterion(classification_scores, query_labels.to(device))
                loss.backward()
                optimizer.step()

            predictions.append(predicted_classes)
            targets.append(query_labels)

            all_loss.append(loss.item())

            tqdm_train.set_postfix(loss=mean(all_loss))

    return mean(all_loss), predictions, targets


def episodic_train(few_shot_classifier, n_epochs, train_loader, val_loader, device, optimizer, criterion, scheduler,
                   tb_writer, validation_frequency=1,
                   metrics_path="DemyeliNeXt/results/"):

    best_state = few_shot_classifier.state_dict()
    best_validation_accuracy = 0.0
    best_epoch = 0
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        average_loss, predictions, targets = episodic_epoch(few_shot_classifier, train_loader, optimizer, criterion,
                                                            device)

        scheduler.step()

        predictions_tensor = convert_to_tensor(torch.cat(predictions))
        predictions = torch.tensor(predictions_tensor).to(device)
        targets_tensor = torch.cat(targets).to(device)

        (train_accuracy, train_binary_precision, train_binary_specificity, train_binary_recall, train_binary_f1_score,
         train_confusion_matrix) = get_metrics(predictions, targets_tensor)

        conf = ConfusionMatrix(task='binary', num_classes=2)

        fig_, ax_, = conf.plot(val=train_confusion_matrix, labels=["NO MS", "MS"])
        path = os.path.join(metrics_path, "train")
        if not os.path.exists(path):
            os.makedirs(path)

        metrics = {"train_accuracy": round(train_accuracy * 100, 2),
                   "train_binary_precision": round(train_binary_precision, 2),
                   "train_binary_specificity": round(train_binary_specificity, 2),
                   "train_binary_recall": round(train_binary_recall, 2),
                   "train_binary_f1_score": round(train_binary_f1_score, 2),
                   "train_confusion_matrix": train_confusion_matrix.tolist(),
                   "train_loss": round(average_loss, 2)
                   }

        with open(os.path.join(path, f"train_metrics_epoch_{epoch}.txt"), "w") as file:
            json.dump(metrics, file, indent=4)

        fig_.savefig(os.path.join(path, f"train_confusion_matrix_epoch_{epoch}.png"))

        pprint(metrics)

        tb_writer.add_scalar("Train/loss", average_loss, epoch)
        tb_writer.add_scalar("Train/acc", train_accuracy * 100, epoch)
        if epoch % validation_frequency == validation_frequency - 1:

            (validation_accuracy, validation_binary_precision, validation_binary_specificity, validation_binary_recall,
             validation_binary_f1_score, val_loss) = evaluate_model(
                few_shot_classifier, val_loader, device=device, tqdm_prefix="Validation", phase="validation",
                metrics_path=metrics_path, criterion=criterion, epoch=epoch)

            tb_writer.add_scalar("Val/acc", validation_accuracy * 100, epoch)
            tb_writer.add_scalar("Val/loss", val_loss, epoch)
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_state = deepcopy(few_shot_classifier.state_dict())
                best_epoch = epoch
                # state_dict() returns a reference to the still evolving model's state so we deepcopy
                # https://pytorch.org/tutorials/beginner/saving_loading_models
                print("Ding ding ding! We found a new best model!")

    print(f'best model at epoch n°{best_epoch} with val accuracy {best_validation_accuracy}')
    return best_state, best_epoch


def evaluate_on_one_task(
        model: FewShotClassifier,
        support_images: Tensor,
        support_labels: Tensor,
        query_images: Tensor,
        query_labels: Tensor,
) -> Tuple[int, int]:
    """
    Returns the number of correct predictions of query labels, and the total number of
    predictions.
    """
    predictions = predict(model, support_images, support_labels, query_images, enable_detach=True)
    number_of_correct_predictions = (
        (torch.max(predictions, 1)[1] == query_labels).sum().item()
    )
    return number_of_correct_predictions, len(query_labels)


class Metrics:
    def __init__(self, preds, target, device):
        self.preds = preds
        self.target = target
        self.device = device

    def get_confusion_matrix(self, task="binary", num_classes=2):
        confmat = ConfusionMatrix(task=task, num_classes=num_classes).to(self.device)
        return confmat(self.preds, self.target)

    def get_binary_precision(self):
        bin_precision = BinaryPrecision().to(self.device)
        return bin_precision(self.preds, self.target).item()

    def get_accuracy(self, task="binary", num_classes=2):
        accuracy = Accuracy(task=task, num_classes=num_classes).to(self.device)
        return accuracy(self.preds, self.target).item()

    def get_binary_specificity(self):
        bin_specificity = BinarySpecificity().to(self.device)
        return bin_specificity(self.preds, self.target).item()

    def get_binary_recall(self):
        bin_recall = BinaryRecall().to(self.device)
        return bin_recall(self.preds, self.target).item()

    def get_binary_f1_score(self):
        bin_f1_score = BinaryF1Score().to(self.device)
        return bin_f1_score(self.preds, self.target).item()

    def get_roc_curve(self):
        pass


def get_classes(scores):
    # get labels in shape of (# samples)
    predicted_classes = torch.max(scores.data, 1)[1]
    return predicted_classes


def get_metrics(predictions, query_labels):
    metrics = Metrics(preds=predictions, target=query_labels, device=torch.device('cuda'))
    accuracy = metrics.get_accuracy()
    confusion_matrix = metrics.get_confusion_matrix()
    binary_precision = metrics.get_binary_precision()
    binary_specificity = metrics.get_binary_specificity()
    binary_recall = metrics.get_binary_recall()
    binary_f1_score = metrics.get_binary_f1_score()

    return accuracy, binary_precision, binary_specificity, binary_recall, binary_f1_score, confusion_matrix


def predict(
        model: FewShotClassifier,
        support_images: Tensor,
        support_labels: Tensor,
        query_images: Tensor,
        enable_detach
):
    model.process_support_set(support_images, support_labels)
    predictions = model(query_images).detach().data if enable_detach else model(query_images)
    return predictions


def evaluate_model(
        model: FewShotClassifier,
        data_loader: DataLoader,
        criterion,
        epoch,
        device: str = "cuda",
        use_tqdm: bool = True,
        tqdm_prefix: Optional[str] = None,
        metrics_path="DemyeliNeXt/results/",
        phase="validation",

) -> float:
    """
    Evaluate the model on few-shot classification tasks
    Args:
        metrics_path: path to save metrics
        model: a few-shot classifier
        data_loader: loads data in the shape of few-shot classification tasks*
        device: where to cast data tensors.
            Must be the same as the device hosting the model's parameters.
        use_tqdm: whether to display the evaluation's progress bar
        tqdm_prefix: prefix of the tqdm bar
    Returns:
        average classification accuracy
    """

    predictions = []
    targets = []
    all_loss = []
    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph
    model.backbone.set_use_fc(False)
    model.eval()
    with (torch.no_grad()):
        # We use a tqdm context to show a progress bar in the logs
        with tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                disable=not use_tqdm,
                desc=tqdm_prefix,
        ) as tqdm_eval:
            for _, (
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                    _,
            ) in tqdm_eval:
                batch_scores = predict(model, support_images.to(device), support_labels.to(device),
                                       query_images.to(device), enable_detach=True)

                loss = criterion(batch_scores, query_labels.to(device))
                all_loss.append(loss.item())

                batch_predictions = get_classes(batch_scores)

                predictions.append(batch_predictions)
                targets.append(query_labels)
                tqdm_eval.set_postfix(loss=mean(all_loss))

    predictions_tensor = convert_to_tensor(torch.cat(predictions))
    predictions = torch.tensor(predictions_tensor).to(device)
    targets_tensor = torch.cat(targets).to(device)

    accuracy, binary_precision, binary_specificity, binary_recall, binary_f1_score, confusion_matrix = get_metrics(
        predictions,
        targets_tensor
    )

    conf = ConfusionMatrix(task='binary', num_classes=2)

    fig_, ax_, = conf.plot(val=confusion_matrix, labels=["NO MS", "MS"])
    path = os.path.join(metrics_path, phase)
    if not os.path.exists(path):
        os.makedirs(path)

    metrics = {f"{phase}_accuracy": round(accuracy * 100, 2),
               f"{phase}_binary_precision": round(binary_precision, 2),
               f"{phase}_binary_specificity": round(binary_specificity, 2),
               f"{phase}_binary_recall": round(binary_recall, 2),
               f"{phase}_binary_f1_score": round(binary_f1_score, 2),
               f"{phase}_confusion_matrix": confusion_matrix.tolist(),
               f"{phase}_loss": mean(all_loss)
               }

    with open(os.path.join(path, f"{phase}_metrics_{epoch}.txt"), "w") as file:
        json.dump(metrics, file, indent=4)

    fig_.savefig(os.path.join(path, f"confusion_matrix_{epoch}.png"))

    pprint(metrics)

    return accuracy, binary_precision, binary_specificity, binary_recall, binary_f1_score, mean(all_loss)


def evaluate(
        model: FewShotClassifier,
        data_loader: DataLoader,
        device: str = "cuda",
        use_tqdm: bool = True,
        tqdm_prefix: Optional[str] = None,
) -> float:
    """
    Evaluate the model on few-shot classification tasks
    Args:
        model: a few-shot classifier
        data_loader: loads data in the shape of few-shot classification tasks*
        device: where to cast data tensors.
            Must be the same as the device hosting the model's parameters.
        use_tqdm: whether to display the evaluation's progress bar
        tqdm_prefix: prefix of the tqdm bar
    Returns:
        average classification accuracy
    """
    # We'll count everything and compute the ratio at the end
    total_predictions = 0
    correct_predictions = 0

    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph
    model.eval()
    with torch.no_grad():
        # We use a tqdm context to show a progress bar in the logs
        with tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                disable=not use_tqdm,
                desc=tqdm_prefix,
        ) as tqdm_eval:
            for _, (
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                    _,
            ) in tqdm_eval:
                correct, total = evaluate_on_one_task(
                    model,
                    support_images.to(device),
                    support_labels.to(device),
                    query_images.to(device),
                    query_labels.to(device),
                )

                total_predictions += total
                correct_predictions += correct

                # Log accuracy in real time
                tqdm_eval.set_postfix(accuracy=correct_predictions / total_predictions)

    return correct_predictions / total_predictions
