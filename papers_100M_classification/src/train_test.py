import time
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import device
from datetime import datetime
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
            elif mode == 'neo4j':
                test_logits = model(batch['PAPER'].features, batch[('PAPER', 'CITES', 'PAPER')].edge_index)
                test_labels = batch['PAPER'].label
            else:
                test_logits = model(batch['paper'].x, batch[('paper', 'cites', 'paper')].edge_index)
                test_labels = batch['paper'].y.squeeze().long()

            _, predicted = torch.max(test_logits, 1)
            correct += (predicted == test_labels).sum().item()

            if mode == 'kuzu':
                total += batch.size()[0]
            elif mode == 'neo4j':
                total += batch.size()[0]
            else:
                total += batch.size()[0]

    test_accuracy = correct / total
    print("Final test accuracy:", test_accuracy)


def train(model, train_loader, test_loader, params, mode='no-backend'):
    print("train")
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
            batch = batch.to(device)  # Ensure batch is on the correct device
            optimizer.zero_grad()
            transfer_time = time.time() - start_time
            print(f"Time to transfer batch to Device: {transfer_time:.4f} seconds")

            if mode == 'kuzu':
                # print(batch['paper'].x.shape, batch[('paper', 'cites', 'paper')].edge_index.shape)
                # print(batch[('paper', 'cites', 'paper')].edge_index)
                # for row in batch['paper'].x:
                #     lst = (row != 0).nonzero(as_tuple=False).squeeze().tolist()
                #     print(lst)
                #     if isinstance(lst, list) and len(lst) > 1000:
                #         raise ValueError()
                # print(batch['paper'].x.shape)
                # print(batch[('paper', 'cites', 'paper')].edge_index.shape)
                logits = model(batch['paper'].x, batch[('paper', 'cites', 'paper')].edge_index)
                y = batch['paper'].y.long()
                num_edges = batch[('paper', 'cites', 'paper')].edge_index.shape[1]
                print(f"Number of edges sampled in this batch: {num_edges}")
            elif mode == 'neo4j':
                # print(batch['Paper'].features)
                # print(batch[('Paper', 'CITES', 'Paper')].edge_index)
                # print(batch['Paper'].features.shape)
                # print(batch[('Paper', 'CITES', 'Paper')].edge_index.shape)
                # print("batch", batch)
                # print("features", batch['PAPER'].features)
                # print("batch[('PAPER', 'CITES', 'PAPER')].edge_index", batch[('PAPER', 'CITES', 'PAPER')].edge_index)
                # print("before logits")
                logits = model(batch['PAPER'].features, batch[('PAPER', 'CITES', 'PAPER')].edge_index)
                # print("label", batch['PAPER'].label)
                # print("labels changed")
                # batch['PAPER'].label = torch.full((1024,), 80, dtype=torch.long)
                y = batch['PAPER'].label.long()
                print(f"Edges shape: {batch[('PAPER', 'CITES', 'PAPER')].edge_index.shape}")
            else:
                # print(batch.x.shape, batch.edge_index.shape)
                # print(batch.edge_index)
                # for row in batch.x:
                #     lst = (row != 0).nonzero(as_tuple=False).squeeze().tolist()
                #     print(lst)
                #     if isinstance(lst, list) and len(lst) > 1000:
                #         raise ValueError()
                # print(batch['paper'].x.shape, batch[('paper', 'cites', 'paper')].edge_index.shape)
                # print(batch['paper'].y.shape, batch['paper'].y.squeeze().shape)
                # print(batch['paper'].y)
                # print(batch['paper'].y.squeeze())
                logits = model(batch['paper'].x, batch[('paper', 'cites', 'paper')].edge_index)
                y = batch['paper'].y.squeeze().long()
                print(f"Edges shape: {batch[('paper', 'cites', 'paper')].edge_index.shape}")
                
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

            if mode == 'kuzu' and torch.isnan(loss):
                print("loss is nan")
                # print("batch['paper'].x", batch['paper'].x)
                # print("batch[('paper', 'cites', 'paper')].edge_index", batch[('paper', 'cites', 'paper')].edge_index)
                # print("y valid", y)
                # print("logits", logits)
                # print("y.numel()", y.numel())

            total_loss += loss.item() * y.numel()
            print(f"Loss {loss.item()}")
            total_predictions += y.numel()
            
            batch_time = time.time() - start_time
            print(f"Single Batch Time: {batch_time:.4f} seconds")

            if batch_num >= 3: # params['num_batches']:
                losses.append(total_loss / total_predictions)
                return losses, []

        avg_loss = total_loss / total_predictions
        losses.append(avg_loss)

        # if epoch % 1 == 0:
        #     test_accuracy = test(model, test_loader, mode)
        #     test_accuracies.append(test_accuracy)

        #     print(f'Epoch [{epoch+1}/{params["num_epochs"]}] - Loss: {str(avg_loss)}, '
        #           f'Test Accuracy: {str(test_accuracy)}')
        #     model.train()

    return losses, test_accuracies
