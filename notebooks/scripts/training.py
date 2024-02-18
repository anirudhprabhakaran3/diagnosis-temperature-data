import torch
from datetime import datetime

from scripts.constants import DISEASE_CLASSES

from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import ConfusionMatrixDisplay

NUM_EPOCHS = 100

def train_one_epoch(model, training_loader, optimizer, loss_fn, epoch_index, tb_writer):
    last_loss = 0.

    all_reshaped_labels = []
    all_outputs = []
    running_labels = []
    running_outputs = []

    for i, data in enumerate(training_loader):
        inputs, labels = data
        reshaped_labels = labels.reshape(-1)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, reshaped_labels)
        loss.backward()

        optimizer.step()

        last_loss = loss.item()

        print('  batch {} loss: {}'.format(i + 1, last_loss))

        tb_x = epoch_index * len(training_loader) + i + 1

        running_labels.extend(reshaped_labels)
        running_outputs.extend(torch.argmax(outputs.detach(), dim=1))

        if tb_x % 20 == 1:
            running_labels = torch.tensor(running_labels)
            running_outputs = torch.tensor(running_outputs)

            accuracy = (running_outputs == running_labels).float().mean()
            precision = precision_score(running_labels, running_outputs, average='weighted')
            f1 = f1_score(running_labels, running_outputs, labels=DISEASE_CLASSES, average='weighted')
            recall = recall_score(running_labels, running_outputs, average='weighted', labels=DISEASE_CLASSES)
            
            all_reshaped_labels.extend(reshaped_labels)
            all_outputs.extend(torch.argmax(outputs.detach(), dim=1))

            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            tb_writer.add_scalar('Precision/train', precision, tb_x)
            tb_writer.add_scalar('F1/train', f1, tb_x)
            tb_writer.add_scalar('Recall/train', recall, tb_x)
            tb_writer.add_scalar('Accuracy/train', accuracy, tb_x)

            running_outputs = []
            running_labels = []
    

    cf = ConfusionMatrixDisplay.from_predictions(all_reshaped_labels, all_outputs, labels=[0, 1, 2, 3])
    tb_writer.add_figure('Confusion Matrix/train', cf.figure_, epoch_index)
    acc = (torch.tensor(all_reshaped_labels) == torch.tensor(all_outputs)).float().mean()
    
    return last_loss, acc

def train_test_loop(model_type, model, writer, training_loader, testing_loader, optimizer, loss_fn):
    best_vloss = float('inf')

    for epoch in range(NUM_EPOCHS):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        print('EPOCH {}:'.format(epoch))

        model.train(True)
        avg_loss, train_acc = train_one_epoch(model, training_loader, optimizer, loss_fn, epoch, writer)


        running_vloss = 0.0
        model.eval()

        all_reshaped_labels = []
        all_outputs = []
        running_labels = []
        running_outputs = []

        with torch.no_grad():
            
            for i, vdata in enumerate(testing_loader):
                vinputs, vlabels = vdata
                reshaped_vlabels = vlabels.reshape(-1)

                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, reshaped_vlabels)
                running_vloss += vloss

                tb_x = epoch * len(testing_loader) + i + 1
                running_labels.extend(reshaped_vlabels)
                running_outputs.extend(torch.argmax(voutputs.detach(), dim=1))

                if tb_x % 20 == 1:
                    running_labels = torch.tensor(running_labels)
                    running_outputs = torch.tensor(running_outputs)

                    accuracy = (running_outputs == running_labels).float().mean()
                    precision = precision_score(running_labels, running_outputs, average='weighted')
                    f1 = f1_score(running_labels, running_outputs, labels=DISEASE_CLASSES, average='weighted')
                    recall = recall_score(running_labels, running_outputs, average='weighted', labels=DISEASE_CLASSES)
                    
                    all_reshaped_labels.extend(reshaped_vlabels)
                    all_outputs.extend(torch.argmax(voutputs.detach(), dim=1))

                    writer.add_scalar('Loss/test', vloss.item(), tb_x)
                    writer.add_scalar('Precision/test', precision, tb_x)
                    writer.add_scalar('F1/test', f1, tb_x)
                    writer.add_scalar('Recall/test', recall, tb_x)
                    writer.add_scalar('Accuracy/test', accuracy, tb_x)

                    running_outputs = []
                    running_labels = []

        cf = ConfusionMatrixDisplay.from_predictions(all_reshaped_labels, all_outputs, labels=[0, 1, 2, 3])
        writer.add_figure('Confusion Matrix/test', cf.figure_, epoch)

        test_acc = (torch.tensor(all_reshaped_labels) == torch.tensor(all_outputs)).float().mean()

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print('ACCURACY train: {} valid {}'.format(train_acc, test_acc))

        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'saved_models/model_{}_{}_{}'.format(model_type, timestamp, epoch)
            torch.save(model.state_dict(), model_path)
