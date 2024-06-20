import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from tqdm import tqdm_notebook as tqdm
import os
import librosa
import librosa.display
from utils import spec_to_image
import numpy as np

class Audio_text_dataset(Dataset):
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
        
def extract_features(text,tokenizer, text_model):
    """
    Extract features from the text using a pre-trained model.

    Args:
        text (str): The input text.

    Returns:
        torch.Tensor: Extracted features.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Tokenize the text
    input_ids = torch.tensor(tokenizer.encode(
        text,
        add_special_tokens=True,
        max_length=63,
        pad_to_max_length=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )).to(device=device)

    # Get the hidden states for each token
    # print(type(text_model))
    with torch.no_grad():
        outputs = text_model(input_ids)
        hidden_states = outputs[1]  # 13

    # Concatenate the last 4 hidden states
    token_vecs = []
    for layer in range(-4, 0):
        token_vecs.append(hidden_states[layer][0])

    # Calculate the mean of the last 4 hidden states
    features = []
    for token in token_vecs:
        features.append(torch.mean(token, dim=0))

    # Return the features as a tensor
    return torch.stack(features)

def get_melspectrogram_db(file_path, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
    # Load the audio file
    wav, sr = librosa.load(file_path, sr=sr)

    # Ensure the audio is at least 5 seconds long
    if wav.shape[0] < 5 * sr:
        wav = np.pad(wav, int(np.ceil((5 * sr - wav.shape[0]) / 2)), mode='reflect')
    else:
        wav = wav[:5 * sr]

    # Compute the mel spectrogram
    spec = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)

    # Convert to decibels (log scale)
    spec_db = librosa.power_to_db(spec, top_db=top_db)
    return spec_db
