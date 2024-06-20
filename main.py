import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from tqdm import tqdm_notebook as tqdm
import os
import librosa
import librosa.display
from utils import spec_to_image
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from data_processing import load_and_preprocess_data
from utils import download_pretrained_model, spec_to_image, setlr, lr_decay
from model_training import train_our_model
from plotting import plot_metrics
from feature_extraction import get_melspectrogram_db, Audio_text_dataset
from models import Sound_Model, ModalityAlignmentModel_2
import pickle
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertAdam, BertForSequenceClassification
from transformers import BertForSequenceClassification, AdamW, BertConfig


class audio_text_dataset(Dataset):
    def __init__(self, base, df, in_col, out_col, tokenizer):
        """
        Custom dataset for audio and text data.

        Args:
            base (str): The base path of the data.
            df (pd.DataFrame): The dataframe containing the data.
            in_col (str): The column name for audio file paths.
            out_col (str): The column name for labels (Speaker Dialect).
            tokenizer: The tokenizer for processing text data.
        """
        self.df = df
        self.text = []
        self.labels = []
        self.audio = []
        self.c2i = {}
        self.i2c = {}

        # Convert ProcessedText to string
        df['ProcessedText'] = df['ProcessedText'].astype(str)

        # Categories (unique Speaker Dialects)
        self.categories = sorted(df[out_col].unique())

        # Mapping categories to indices and vice versa
        for i, category in enumerate(self.categories):
            self.c2i[category] = i
            self.i2c[i] = category

        # Loop through dataframe rows
        for ind in tqdm(range(len(df))):
            row = df.iloc[ind]
            file_path = os.path.join(base, row[in_col])

            try:
                # Get spectrogram from audio file
                spectrogram = get_melspectrogram_db(file_path)
                self.audio.append(spec_to_image(spectrogram)[np.newaxis, ...])
                self.labels.append(self.c2i[row['SpeakerDialect']])

                # Tokenize text
                encoded_dict = tokenizer.encode_plus(
                        row['ProcessedText'],
                        add_special_tokens=True,
                        max_length=63,
                        pad_to_max_length=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                )

                # Store tokenized text and associated information
                self.text.append({
                    'input_ids': encoded_dict['input_ids'],
                    'attention_mask': encoded_dict['attention_mask'],
                    'label': self.c2i[row['SpeakerDialect']],
                    'ProcessedText': row['ProcessedText'],
                })

            except FileNotFoundError:
                print(f"File not found: {file_path}. Skipping...")
                continue

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        return self.audio[idx], self.labels[idx], self.text[idx]
        
def main(args):
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Preprocess data and get train, valid, and test sets
    train, valid, test, all_data = load_and_preprocess_data(args.train_path, args.valid_path, args.remove_values, args.base)

    # # Download the tokenizer and text model
    tokenizer, t = download_pretrained_model(args.pretrained_model_name)

    # Initialize models
    sound_model = Sound_Model(input_shape=(1, 128, 431), num_cats=4).to(device)
    AlignmentModel = ModalityAlignmentModel_2(args.audio_dim, args.text_dim).to(device)

    # Create datasets 
    # train_data = Audio_text_dataset(args.audio_base, train, 'FileName', 'SpeakerDialect', tokenizer)
    # valid_data = Audio_text_dataset(args.audio_base, valid, 'FileName', 'SpeakerDialect', tokenizer)
    # test_data = Audio_text_dataset(args.audio_base, test, 'FileName', 'SpeakerDialect', tokenizer)

    # Load the dataset objects 
    with open('/content/drive/MyDrive/journal_SADA/train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    with open('/content/drive/MyDrive/journal_SADA/valid_data.pkl', 'rb') as f:
        valid_data = pickle.load(f)
    
    with open('/content/drive/MyDrive/journal_SADA/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)


    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Define text model for classification
    text_model = BertForSequenceClassification.from_pretrained(
        args.pretrained_model_name,
        num_labels=4,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=False
    )

    text_model.cuda()
    # print(type(text_model))

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(sound_model.parameters()) + list(AlignmentModel.parameters()), lr=args.learning_rate)

    # Initialize lists to store losses and accuracies
    train_losses, valid_losses, test_losses = [], [], []
    valid_accuracies, test_accuracies = [], []

    # Train the model
    train_losses, valid_losses, test_losses, valid_accuracies, test_accuracies = train_our_model(
        AlignmentModel, sound_model, loss_fn, train_loader, valid_loader, test_loader, args.epochs, optimizer, train_losses, valid_losses, test_losses, tokenizer, text_model
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train audio-text alignment model")

    parser.add_argument("--train_path", type=str, default="/content/drive/MyDrive/journal_SADA/train.csv", help="Path to the training data CSV")
    parser.add_argument("--valid_path", type=str, default="/content/drive/MyDrive/journal_SADA/valid.csv", help="Path to the validation data CSV")
    parser.add_argument("--remove_values", nargs='+', default=[
        "More than 1 speaker اكثر من متحدث",
        "Unknown",
        "Notapplicable",
        "Levantine",
        "Egyptian",
        "Maghrebi",
        "Janubi",
        "Shamali"
    ], help="List of values to remove from the dataset")
    parser.add_argument("--base", type=str, default='/content/drive/MyDrive/journal_SADA/batch1/batch_1', help="Base path for the data files")
    parser.add_argument("--pretrained_model_name", type=str, default='aubmindlab/bert-base-arabertv01', help="Name of the pretrained model")
    parser.add_argument("--audio_dim", type=int, default=2000, help="Dimension of audio features")
    parser.add_argument("--text_dim", type=int, default=500, help="Dimension of text features")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--audio_base", type=str, default='/content/drive/MyDrive/journal_SADA/batch1')

    args = parser.parse_args()
    main(args)
