# MF-Saudi: A Multimodal Framework for Bridging the Gap Between Audio and Textual Data for Saudi Dialect Detection
This is the source code for the paper: MF-Saudi: A Multimodal Framework for Bridging the Gap Between Audio and Textual Data for Saudi Dialect Detection by Raed Alharbi, proceeding in Journal of King Saud University - Computer and Information Sciences - 2024 

## Files

- `data_processing.py`: Contains functions for loading and preprocessing data.
- `models.py`: Contains the main models in the paper.
- `model_training.py`: Contains the main training loop for the model.
- `plotting.py`: Contains functions for plotting training metrics.
- `feature_extraction.py`: Contains functions for extracting features from text and audio.
- `utils.py`: Contains utility functions such as learning rate scheduling and downloading pre-trained models.
- `example.ipynb`: example file to run the training script from jupyter notebook.

## Requirements

In order to run the code, will need:

1. tensorflow_io
2. pytorch-pretrained-bert
3. transformers==3.5.1
4. sentencepiece
5. keras-preprocessing
7. librosa==0.9.2
8. torchaudio

You can install the required packages using:
 ```
pip install -r requirements.txt
 ```
--------------------------------------------------
To train the model, you can use the following script:
 ```
python /content/drive/MyDrive/journal_SADA/github_code/main.py --train_path /content/drive/MyDrive/journal_SADA/train.csv --valid_path /content/drive/MyDrive/journal_SADA/valid.csv --base /content/drive/MyDrive/journal_SADA/batch1/batch_1
 ```

or, if you prefer to use jupyter notebook, check the file: example.ipynb

--------------------------------------------------
**Dataset**  You can download the dataset used in the expeirment from: (https://www.kaggle.com/datasets/sdaiancai/sada2022)


