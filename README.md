# Chess Piece Identifier

A deep learning application that identifies chess pieces from images using PyTorch and Streamlit.

![Chess Piece Identifier](UsageExample.png)

## Overview

This project uses a Convolutional Neural Network (CNN) to classify chess pieces into six categories:
- Pawn
- Rook
- Knight
- Bishop
- Queen
- King

The model is built with PyTorch and deployed through a user-friendly Streamlit web interface.

## Features

- Upload your own chess piece images
- Real-time piece identification
- Confidence scores visualization for each prediction
- Simple and intuitive user interface

## Dataset and Pre-trained Model

Due to size constraints, the dataset and pre-trained model are not included in this repository.

### Download Links:
- [Pre-trained model (TheTrainedModel.pth)](https://drive.google.com/file/d/1-BX5j2DeKPEILSsk8PnK1m6xNexP2_wT/view?usp=drive_link)
- [Chess piece dataset](https://drive.google.com/drive/folders/1-YmFt46roRVnEhXTih1MpIzXsK9r1YK6?usp=drive_link)

### Alternative: Train Your Own Model
If you prefer to train your own model, organize your chess piece images according to the dataset structure described above, and run:

```python
from chess_piece_identifier import train_model
model, device, class_names = train_model('path/to/your/dataset')
## Installation

1. Clone this repository:
```
git clone https://github.com/MostafaShams5/chess-piece-identifier.git
cd chess-piece-identifier
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Download the pre-trained model or train your own:
   - To use the pre-trained model, ensure `TheTrainedModel.pth` is in the root directory
   - To train your own model, prepare a dataset and run the training script

## Dataset Structure

To train the model, organize your chess piece images in the following structure:
```
Chess/
├── Bishop/
│   ├── bishop1.jpg
│   ├── bishop2.jpg
│   └── ...
├── King/
│   ├── king1.jpg
│   └── ...
├── Knight/
│   └── ...
├── Pawn/
│   └── ...
├── Queen/
│   └── ...
└── Rook/
    └── ...
```

## Usage

### Training the Model

To train the model on your own dataset:

```python
from chess_piece_identifier import load_or_train_model

# Path to your dataset
dataset_path = 'path/to/Chess'
model, device, class_names = load_or_train_model(dataset_path)
```

### Running the Streamlit App

Run the following command to start the Streamlit interface:

```
streamlit run app.py
```

The app will open in your default web browser. Upload an image of a chess piece, and the model will predict which piece it is, along with confidence scores.

## Model Architecture

The chess piece classifier uses a CNN with the following architecture:
- Three convolutional blocks with batch normalization and max pooling
- Fully connected layers with dropout to prevent overfitting
- Input images are resized to 224×224 pixels and converted to 3-channel grayscale


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

