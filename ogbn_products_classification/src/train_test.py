import time
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

torch.set_printoptions(threshold=torch.inf)


def test(model, test_loader, mode='no-backend'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch
            test_logits = model(batch['PRODUCT'].features, batch[('PRODUCT', 'LINK', 'PRODUCT')].edge_index)
            test_labels = batch['PRODUCT'].label

            _, predicted = torch.max(test_logits, 1)
            correct += (predicted == test_labels).sum().item()
            total += batch.size()[0]

    test_accuracy = correct / total
    print("Final test accuracy:", test_accuracy)


def train(model, train_loader, test_loader, params, mode='no-backend'):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

    losses = []
    test_accuracies = []

    for epoch in range(params["num_epochs"]):
        print("#"*40 + f" Epoch {epoch}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        total_loss = 0
        total_predictions = 0
        for batch_num, batch in enumerate(train_loader):
            print("#" * 20 + f" Batch {batch_num}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            start_time = time.time()
            batch = batch
            optimizer.zero_grad()

            logits = model(batch['PRODUCT'].features, batch[('PRODUCT', 'LINK', 'PRODUCT')].edge_index)
            y = batch['PRODUCT'].label.long()
                
            print(f"Total number of values: {y.numel()}")
            valid_mask = y != -9223372036854775808  # Mask for valid labels
            print(f"Size of non-NaN values: {valid_mask.sum().item()}")
        
            # Only use valid labels for loss calculation
            if valid_mask.sum() == 0:
                print("No valid elements, skipping iteration")
                continue

            logits = logits[valid_mask]
            y = y[valid_mask]

            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.numel()
            print(f"Loss {loss.item()}")
            total_predictions += y.numel()
            
            batch_time = time.time() - start_time
            print(f"Single Batch Time: {batch_time:.4f} seconds")

        avg_loss = total_loss / total_predictions
        losses.append(avg_loss)

        if epoch % 1 == 0:
            test_accuracy = test(model, test_loader, mode)
            test_accuracies.append(test_accuracy)

            print(f'Epoch [{epoch+1}/{params["num_epochs"]}] - Loss: {str(avg_loss)}, '
                  f'Test Accuracy: {str(test_accuracy)}')
            model.train()

    return losses, test_accuracies
