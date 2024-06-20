from torch import nn, optim
import torch.nn.functional as F
import torch

class Sound_Model(nn.Module):
  def __init__(self, input_shape, batch_size=64, num_cats=500):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 32, kernel_size = 3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.bn3 = nn.BatchNorm2d(64)
    self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.bn4 = nn.BatchNorm2d(64)
    self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.bn5 = nn.BatchNorm2d(128)
    self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.bn6 = nn.BatchNorm2d(128)
    self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.bn7 = nn.BatchNorm2d(256)
    self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.bn8 = nn.BatchNorm2d(256)
    self.dense1 = nn.Linear(77824,500)
    self.dropout = nn.Dropout(0.5)
    self.dense2 = nn.Linear(3572, num_cats)

  def forward(self, x, text_features):
    # print(x.shape)
    x = self.conv1(x)
    x = F.relu(self.bn1(x))
    x = self.conv2(x)
    x = F.relu(self.bn2(x))
    x = F.max_pool2d(x, kernel_size=2)
    x = self.conv3(x)
    x = F.relu(self.bn3(x))
    x = self.conv4(x)
    x = F.relu(self.bn4(x))
    x = F.max_pool2d(x, kernel_size=2)
    x = self.conv5(x)
    x = F.relu(self.bn5(x))
    x = self.conv6(x)
    x = F.relu(self.bn6(x))
    x = F.max_pool2d(x, kernel_size=2)
    x = self.conv7(x)
    x = F.relu(self.bn7(x))
    x = self.conv8(x)
    x = F.relu(self.bn8(x))
    x = x.view(x.size(0),-1)
    x = F.relu(self.dense1(x))
    x = self.dropout(x)

     # Apply Textual Attention Integration (TAI) to text_features only
    text_attention_weights = F.softmax(torch.matmul(text_features, text_features.t()), dim=1)
    weighted_text_features = torch.matmul(text_attention_weights, text_features)

    combined_features = torch.cat((x, weighted_text_features), dim=1)
    combined_features = self.dense2(combined_features)


    return combined_features

class ModalityAlignmentModel_2(nn.Module):
    def __init__(self, audio_input_shape, text_input_shape,):
        """
        Model for aligning audio and text modalities.

        Args:
            audio_input_shape (tuple): Shape of the audio input data (channels, height, width).
            text_input_shape (tuple): Shape of the text input data.
            batch_size (int): Batch size for training. Default is 1500.
        """
        super().__init__()

        # Define layers
        self.audio_conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.dense_audio = nn.Linear(321536, audio_input_shape)
        self.dense_text = nn.Linear(3072, text_input_shape)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, text_features, epoch):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input audio data.
            text_features (torch.Tensor): Tokenized text features.

        Returns:
            torch.Tensor: Audio embedding, text embedding, InfoNCE loss.
        """
        # Apply convolutional layers to audio
        x = F.relu(self.audio_conv1(x))
        x = self.dropout(x)
        x = x.view(16, -1)
        audio_embedding = self.dense_audio(x)

        # Divide audio embeddings into sub-embeddings
        num_sub_embeddings = 4  # Adjust this based on how many sub-embeddings you want
        embedding_dim = audio_embedding.shape[1]
        sub_embedding_size = embedding_dim // num_sub_embeddings

        sub_embeddings = []

        for i in range(num_sub_embeddings):
            start_idx = i * sub_embedding_size
            end_idx = (i + 1) * sub_embedding_size

            sub_embedding = audio_embedding[:, start_idx:end_idx]
            sub_embeddings.append(sub_embedding)

        # Project text features into the shared embedding space
        text_embedding = self.dense_text(text_features)

        # Compute alignment scores
        alignment_scores = []

        for sub_embedding in sub_embeddings:
            alignment_scores.append(torch.matmul(sub_embedding, text_embedding.T))

        # print('alignment_scores',alignment_scores)
         # Define positive samples (top-k scores)
        # Define positive samples (top-k scores)



         # Apply softmax to alignment scores
        alignment_probs = [F.softmax(score, dim=1) for score in alignment_scores]



        k = 5  # Define the number of top positive samples you want to select
        top_k_positive_samples = []

        for score in alignment_probs:
            top_k_positive_samples.extend(torch.topk(score.view(-1), k)[0])

        top_k_positive_samples = torch.cat([x.unsqueeze(0) for x in top_k_positive_samples], dim=0)



        # Define negative samples (lowest scores)
        negative_samples = []

        for score in alignment_probs:
            bottom_k_negative_samples = torch.topk(score.view(-1), k, largest=False)[0]
            negative_samples.extend(bottom_k_negative_samples)

        negative_samples = torch.cat([x.unsqueeze(0) for x in negative_samples], dim=0)

        # Define labels for contrastive loss (1 for positive, 0 for negative)
        labels = torch.cat([torch.ones_like(top_k_positive_samples), torch.zeros_like(negative_samples)], dim=0)

        # Concatenate positive and negative samples
        samples = torch.cat([top_k_positive_samples, negative_samples], dim=0)

        # Compute contrastive loss
        contrastive_loss = F.binary_cross_entropy_with_logits(samples, labels)

        return audio_embedding, text_embedding, contrastive_loss
