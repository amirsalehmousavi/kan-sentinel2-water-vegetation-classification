from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import csv

def train(model, device, train_loader, optimizer, epoch, criterion):
    """
    Train the model for one epoch

    Args:
        model: the neural network model
        device: cuda or cpu
        train_loader: DataLoader for training data
        optimizer: the optimizer to use (e.g. SGD)
        epoch: the current epoch
        criterion: the loss function (e.g. CrossEntropy)

    Returns:
        avg_loss: the average loss over the training set
    """

    model.to(device)
    model.train()
    train_loss = 0
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        # Recall that GPU is optimized for the operations we are dealing with
        data, target = data.to(device), target.to(device)

        # Reset the optimizer
        optimizer.zero_grad()

        # Push the data forward through the model layers
        output = model(data)

        # Get the loss
        loss = criterion(output, target)

        # Keep a running total
        train_loss += loss.item()

        # Backpropagate
        loss.backward()
        optimizer.step()

    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx + 1)
    # print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss


def test(model, device, test_loader, criterion):
    """
    Test the model

    Args:
        model: the neural network model
        device: cuda or cpu
        test_loader: DataLoader for test data
        criterion: the loss function (e.g. CrossEntropy)

    Returns:
        test_loss: the average loss over the test set
        accuracy: the accuracy of the model on the test set
        precision: the precision of the model on the test set
        recall: the recall of the model on the test set
        f1: the f1 score of the model on the test set
    """

    model.eval()
    test_loss = 0
    correct = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Get the predicted classes for this batch
            output = model(data)

            # Calculate the loss for this batch
            test_loss += criterion(output, target).item()

            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += (target == predicted).sum().item()

            # Collect all targets and predictions for metric calculations
            all_targets.extend(target.view_as(predicted).cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate overall metrics
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')

    # Normalize test loss
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Precision: {:.2f}, Recall: {:.2f}, F1 Score: {:.2f}\n'.format(
    #     test_loss, correct, len(test_loader.dataset), accuracy, precision, recall, f1))

    return test_loss, accuracy, precision, recall, f1

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_and_save_metrics(epochs, train_loss, test_loss, accuracy, name, total_params = None):
    """
    Create and save plots for accuracy vs epochs and test/train loss vs epochs

    Args:
        epochs: list of epoch numbers
        train_loss: list of training losses
        test_loss: list of test losses
        accuracy: list of accuracies
    """
    plt.rcParams.update({'font.size': 16})  # Increase base font size

    # Plot accuracy vs epochs
    plt.figure(figsize=(12, 8))
    accuracy_percentage = [acc * 100 for acc in accuracy]
    plt.plot(epochs, accuracy_percentage, 'b-', linewidth=2)
    plt.title('Accuracy vs Epochs', fontsize=24)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Accuracy (%)', fontsize=20)

    # Set y-axis limits to focus on the relevant range
    y_min = max(99.0, min(accuracy_percentage) - 0.01)  # Lower bound: 99% or slightly below min accuracy
    y_max = min(100.0, max(accuracy_percentage) + 0.01)  # Upper bound: 100% or slightly above max accuracy
    plt.ylim(y_min, y_max)

    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig(f'{name}_accuracy_vs_epochs.png', dpi=600)
    plt.close()

    # Plot train loss vs epochs
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_loss, 'r-', linewidth=2)
    plt.title('Train Loss vs Epochs', fontsize=24)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Train Loss', fontsize=20)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig(f'{name}_train_loss_vs_epochs.png', dpi=600)
    plt.close()

    # Plot test loss vs epochs
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, test_loss, 'g-', linewidth=2)
    plt.title('Test Loss vs Epochs', fontsize=24)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Test Loss', fontsize=20)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig(f'{name}_test_loss_vs_epochs.png', dpi=600)
    plt.close()

    # Save metrics information to CSV
    csv_filename = f'{name}_metrics.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if total_params is not None:
            csv_writer.writerow(['Total Parameters', total_params])
            csv_writer.writerow([])  # Empty row for separation
        csv_writer.writerow(['Epoch', 'Accuracy (%)', 'Train Loss', 'Test Loss'])  # Header
        for epoch, acc, train_l, test_l in zip(epochs, accuracy_percentage, train_loss, test_loss):
            csv_writer.writerow([epoch, acc, train_l, test_l])

    print(f"Metrics data saved to {csv_filename}")


def train_and_test_models(model, device, train_loader, test_loader, optimizer, criterion, epochs,
                          scheduler, model_type, name = 'general'):
    """
    Train and test the model

    Args:
        model: the neural network model
        device: cuda or cpu
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        optimizer: the optimizer to use (e.g. SGD)
        criterion: the loss function (e.g. CrossEntropy)
        epochs: the number of epochs to train
        scheduler: the learning rate scheduler

    Returns:
        all_train_loss: a list of the average training loss for each epoch
        all_test_loss: a list of the average test loss for each epoch
        all_test_accuracy: a list of the accuracy for each epoch
        all_test_precision: a list of the precision for each epoch
        all_test_recall: a list of the recall for each epoch
        all_test_f1: a list of the f1 score for each epoch
    """
    # Track metrics
    all_train_loss = []
    all_test_loss = []
    all_test_accuracy = []
    all_test_precision = []
    all_test_recall = []
    all_test_f1 = []

    for epoch in range(1, epochs + 1):
        # Train the model
        train_loss = train(model, device, train_loader, optimizer, epoch, criterion)
        all_train_loss.append(train_loss)

        # Test the model
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = test(model, device, test_loader, criterion)
        all_test_loss.append(test_loss)
        all_test_accuracy.append(test_accuracy)
        all_test_precision.append(test_precision)
        all_test_recall.append(test_recall)
        all_test_f1.append(test_f1)

        print(
            f'End of Epoch {epoch}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.8f}, Accuracy: {test_accuracy:.4%}')
        scheduler.step()

    model.all_test_accuracy = all_test_accuracy
    model.all_test_precision = all_test_precision
    model.all_test_f1 = all_test_f1
    model.all_test_recall = all_test_recall

    # Plot and save metrics
    total_params = count_parameters(model_type)
    print(f"Total trainable parameters for {name} model: {total_params:,}")
    plot_and_save_metrics(list(range(1, epochs + 1)), all_train_loss, all_test_loss, all_test_accuracy,
                          name, total_params = total_params)

    return all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1