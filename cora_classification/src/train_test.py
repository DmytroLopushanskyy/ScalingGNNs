import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import device
# from src.utils import train_mask

torch.set_printoptions(threshold=torch.inf)


def test(model, test_loader, mode='no-backend'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)  # Ensure batch is on the correct device
            if mode == 'kuzu':
                # print(batch['paper'].x.shape, batch[('paper', 'cites', 'paper')].edge_index.shape)
                test_logits = model(batch['paper'].x, batch[('paper', 'cites', 'paper')].edge_index)
                test_labels = batch['paper'].y
            else:
                # print(batch.x.shape, batch.edge_index.shape)
                logits = model(batch.x, batch.edge_index)
                test_logits = logits[batch.test_mask]
                test_labels = batch.y[batch.test_mask]

            _, predicted = torch.max(test_logits, 1)
            correct += (predicted == test_labels).sum().item()

            if mode == 'kuzu':
                total += batch.size()[0]
            else:
                total += batch.test_mask.sum().item()

    test_accuracy = correct / total
    return test_accuracy


def train(model, train_loader, test_loader, params, mode='no-backend'):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

    losses = []
    test_accuracies = []

    for epoch in range(params["num_epochs"]):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)  # Ensure batch is on the correct device
            optimizer.zero_grad()

            if mode == 'kuzu':
                # print(batch['paper'].x.shape, batch[('paper', 'cites', 'paper')].edge_index.shape)
                # print(batch[('paper', 'cites', 'paper')].edge_index)
                # for row in batch['paper'].x:
                #     lst = (row != 0).nonzero(as_tuple=False).squeeze().tolist()
                #     print(lst)
                #     if isinstance(lst, list) and len(lst) > 1000:
                #         raise ValueError()

                logits = model(batch['paper'].x, batch[('paper', 'cites', 'paper')].edge_index)
                loss = loss_fn(logits, batch['paper'].y)
            else:
                # print(batch.x.shape, batch.edge_index.shape)
                # print(batch.edge_index)
                # for row in batch.x:
                #     lst = (row != 0).nonzero(as_tuple=False).squeeze().tolist()
                #     print(lst)
                #     if isinstance(lst, list) and len(lst) > 1000:
                #         raise ValueError()

                logits = model(batch.x, batch.edge_index)
                loss = loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_nodes

        avg_loss = total_loss / len(train_loader.dataset)
        losses.append(avg_loss)

        if epoch % 8 == 0:
            test_accuracy = test(model, test_loader, mode)
            test_accuracies.append(test_accuracy)

            print(f'Epoch [{epoch+1}/{params["num_epochs"]}] - Loss: {avg_loss:.4f}, '
                  f'Test Accuracy: {test_accuracy:.4f}')
            model.train()

    return losses, test_accuracies
