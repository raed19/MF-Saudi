import torch
import numpy as np
from tqdm import tqdm
from utils import setlr
from feature_extraction import extract_features

def train_our_model(AlignmentModel, sound_model, loss_fn, train_loader, valid_loader, test_loader, epochs, optimizer, train_losses, valid_losses, test_losses, tokenizer,text_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def lr_decay(optimizer, epoch):
        if epoch % 20 == 0:
            new_lr = learning_rate / (10 ** (epoch // 20))
            optimizer = setlr(optimizer, new_lr)
            print(f'Changed learning rate to {new_lr}')
        return optimizer

    train_accuracies = []
    valid_accuracies = []
    test_accuracies = []
    
    # print(type(text_model))
    for epoch in tqdm(range(1, epochs + 1)):
        sound_model.train()
        batch_losses = []

        if lr_decay:
            optimizer = lr_decay(optimizer, epoch)

        for i, data in enumerate(train_loader):
            audio, y, text = data

            optimizer.zero_grad()

            audio = audio.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)

            b_input_ids = text['input_ids'].to(device)
            b_input_mask = text['attention_mask'].to(device)
            labels = text['label'].to(device)
            ProcessedText = text['ProcessedText']
            features = [extract_features(text,tokenizer,text_model) for text in ProcessedText]
            features = torch.cat(features)
            features_reshaped = features.reshape((16, -1))

            # Forward pass for audio model
            output = sound_model(audio, features_reshaped)

            audio_emd, text_emd, alignment_loss = AlignmentModel(audio, features_reshaped, epoch)
            loss_audio = loss_fn(output, y)

            loss = loss_audio + alignment_loss

            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_losses.append(batch_losses)
        print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}')

        sound_model.eval()
        batch_losses = []
        trace_y = []
        trace_yhat = []

        for i, data in enumerate(valid_loader):
            audio, y, text = data

            audio = audio.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)

            b_input_ids = text['input_ids'].to(device)
            b_input_mask = text['attention_mask'].to(device)
            labels = text['label'].to(device)
            ProcessedText = text['ProcessedText']
            features = [extract_features(text,tokenizer,text_model) for text in ProcessedText]
            features = torch.cat(features)
            features_reshaped = features.reshape((16, -1))

            # Forward pass for audio model
            output = sound_model(audio, features_reshaped)
            audio_emd, text_emd, alignment_loss = AlignmentModel(audio, features_reshaped, epoch)

            loss_audio = loss_fn(output, y)

            loss = loss_audio*0.1 + alignment_loss

            batch_losses.append(loss.item())

            trace_y.append(y.cpu().detach().numpy())
            trace_yhat.append(output.cpu().detach().numpy())

        valid_losses.append(batch_losses)

        trace_y = np.concatenate(trace_y)
        trace_yhat = np.concatenate(trace_yhat)
        accuracy = np.mean(trace_yhat.argmax(axis=1) == trace_y)
        valid_accuracies.append(accuracy)

        print(f'Epoch - {epoch} Valid-Loss : {np.mean(valid_losses[-1])} Valid-Accuracy : {accuracy}')

        # Testing
        batch_losses = []
        trace_y_t = []
        trace_yhat_t = []

        sound_model.eval()
        for i, data in enumerate(test_loader):
            audio, y, text = data

            audio = audio.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)

            b_input_ids = text['input_ids'].to(device)
            b_input_mask = text['attention_mask'].to(device)
            labels = text['label'].to(device)
            ProcessedText = text['ProcessedText']
            features = [extract_features(text, tokenizer,text_model) for text in ProcessedText]
            features = torch.cat(features)

            features_reshaped = features.reshape((16, -1))

            # Forward pass for audio model
            output = sound_model(audio, features_reshaped)

            audio_emd, text_emd, alignment_loss = AlignmentModel(audio, features_reshaped, epoch)
            loss_audio = loss_fn(output, y)

            loss = loss_audio + alignment_loss

            batch_losses.append(loss.item())

            trace_y_t.append(y.cpu().detach().numpy())
            trace_yhat_t.append(output.cpu().detach().numpy())

        test_losses.append(batch_losses)

        trace_y_t = np.concatenate(trace_y_t)
        trace_yhat_t = np.concatenate(trace_yhat_t)
        accuracy = np.mean(trace_yhat_t.argmax(axis=1) == trace_y_t)
        test_accuracies.append(accuracy)

        print(f'Epoch - {epoch} Test-Loss : {np.mean(test_losses[-1])} Test-Accuracy : {accuracy}')

    return train_losses, valid_losses, test_losses, valid_accuracies, test_accuracies
