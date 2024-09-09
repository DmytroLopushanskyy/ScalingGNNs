import time
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import device

torch.set_printoptions(threshold=torch.inf)


def test(model, test_loader, mode='no-backend'):
    start_time = time.time()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)  # Ensure batch is on the correct device
            if mode == 'kuzu':
                test_logits = model(batch['paper'].x, batch[('paper', 'cites', 'paper')].edge_index)
                test_labels = batch['paper'].y
            elif mode == 'neo4j':
                test_logits = model(batch['Paper'].features, batch[('Paper', 'CITES', 'Paper')].edge_index)
                test_labels = batch['Paper'].label
            else:
                logits = model(batch.x, batch.edge_index)
                test_logits = logits[batch.test_mask]
                test_labels = batch.y[batch.test_mask]

            _, predicted = torch.max(test_logits, 1)
            correct += (predicted == test_labels).sum().item()

            if mode == 'kuzu':
                total += batch.size()[0]
            elif mode == 'neo4j':
                total += batch.size()[0]
            else:
                total += batch.test_mask.sum().item()

    test_accuracy = correct / total
    print(f"Test duration: {time.time() - start_time:.2f} seconds")
    print(f"Test accuracy: {test_accuracy}")


def train(model, train_loader, test_loader, params, mode='no-backend'):
    start_time = time.time()
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

    losses = []
    test_accuracies = []

    for epoch in range(params["num_epochs"]):
        print("#"*40 + f" Epoch {epoch}")
        total_loss = 0
        for batch_num, batch in enumerate(train_loader):
            print("#" * 20 + f" Batch {batch_num}")
            batch = batch.to(device)  # Ensure batch is on the correct device
            optimizer.zero_grad()

            if mode == 'kuzu':
                logits = model(batch['paper'].x, batch[('paper', 'cites', 'paper')].edge_index)
                loss = loss_fn(logits, batch['paper'].y)
            elif mode == 'neo4j':
                logits = model(batch['Paper'].features, batch[('Paper', 'CITES', 'Paper')].edge_index)
                loss = loss_fn(logits, batch['Paper'].label)
            else:
                logits = model(batch.x, batch.edge_index)
                loss = loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_nodes

        avg_loss = total_loss / len(train_loader.dataset)
        losses.append(avg_loss)

        if epoch % 10 == 0:
            test_accuracy = test(model, test_loader, mode)
            test_accuracies.append(test_accuracy)
        
            print(f'Epoch [{epoch+1}/{params["num_epochs"]}] - Loss: {avg_loss:.4f}, '
                  f'Test Accuracy: {test_accuracy:.4f}')
            model.train()

    print(f"Train duration: {time.time() - start_time:.2f} seconds")
    return losses, test_accuracies
