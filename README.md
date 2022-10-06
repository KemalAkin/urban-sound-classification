# Urban Sound Classification w/ CNN

As the final project of [Global AI Hub DL Bootcamp](https://globalaihub.com/activity/), we used Convolutional Neural Network (CNN) to classify urban sounds.

The project includes the following steps:
- Creating Mel spectrograms from sound files
- Preprocessing the spectrograms for training (grayscaling - resizing - normalization)
- Splitting the data into train, test, and validation sets
- Model building and training 

## Table of Contents
1. [Dataset](#Dataset)
2. [Mel Spectrogram](#Mel-Spectrogram)
3. [Preprocessing](#Preprocessing)
4. [Training the Model](#Training-the-Model)

## Dataset
The dataset contains 8732 labeled sound excerpts of urban sounds from 10 classes and it is publicly available [here](https://urbansounddataset.weebly.com/urbansound8k.html).

A random sample from the metadata is given below:

|      | slice_file_name   |   fsID |    start |     end |   salience |   fold |   classID | class            |
|-----:|:------------------|-------:|---------:|--------:|-----------:|-------:|----------:|:-----------------|
|  892 | 118278-4-0-7.wav  | 118278 |  3.5     |  7.5    |          2 |     10 |         4 | drilling         |
| 6352 | 39967-9-0-0.wav   |  39967 | 19.3035  | 23.3035 |          1 |      8 |         9 | street_music     |
| 1024 | 123688-8-0-0.wav  | 123688 | 14.1756  | 18.1756 |          2 |      2 |         8 | siren            |
| 2625 | 159738-8-0-12.wav | 159738 |  6.74699 | 10.747  |          2 |      1 |         8 | siren            |
| 1581 | 138465-1-0-0.wav  | 138465 | 19.2123  | 22.9625 |          2 |      8 |         1 | car_horn         |
| 5766 | 24347-8-0-20.wav  |  24347 | 14.1204  | 18.1204 |          2 |      4 |         8 | siren            |
| 7888 | 75743-0-0-22.wav  |  75743 | 11       | 15      |          2 |      9 |         0 | air_conditioner  |
| 8293 | 85249-2-0-68.wav  |  85249 | 34       | 38      |          1 |      6 |         2 | children_playing |
| 1347 | 13230-0-0-7.wav   |  13230 |  3.5     |  7.5    |          1 |      3 |         0 | air_conditioner  |
| 2734 | 159751-8-0-11.wav | 159751 |  6.6505  | 10.6505 |          2 |      4 |         8 | siren            |

## Mel Spectrogram
A spectrogram is a visualization of the frequency spectrum of a signal, where the frequency spectrum of a signal is the frequency range that is contained by the signal. Mel spectrogram is basically a spectrogram that is represented in Mel scale.

We used the library ```Librosa``` in order to create Mel Spectrograms. In order to save them according to their class ID, we used metadata to match the filename with the class ID. The relevant Jupyter notebook is available [here](https://github.com/KemalAkin/urban-sound-classification/blob/main/spectogram.ipynb). 

[Mel Spectrogram](TestFiles/mspect_test.png)

## Preprocessing  
This steps consists of grayscaling, resizing and normalization of the spectrograms using ```OpenCV```. Further, the data is splitted to train, test, and validation sets. The relevant Jupyter notebook is available [here](https://github.com/KemalAkin/urban-sound-classification/blob/main/preprocessing.ipynb).

[Processed Image](TestFiles/mspect_test_gray.png)

## Training the Model
Finally, CNN model is used for training. The model summary is given below:
| Layer (type)                   | Output Shape       | Param # |
|--------------------------------|--------------------|---------|
| conv2d (Conv2D)                | (None, 64, 64, 32) | 320     |
| max_pooling2d (MaxPooling2D)   | (None, 32, 32, 32) | 0       |
| conv2d_1 (Conv2D)              | (None, 32, 32, 64) | 18496   |
| max_pooling2d_1 (MaxPooling2D) | (None, 16, 16, 64) | 0       |
| conv2d_2 (Conv2D)              | (None, 16, 16, 64) | 36928   |
| flatten (Flatten)              | (None, 16384)      | 0       |
| dense (Dense)                  | (None, 64)         | 1048640 |
| dropout (Dropout)              | (None, 64)         | 0       |
| dense_1 (Dense)                | (None, 64)         | 4160    |
| dropout_1 (Dropout)            | (None, 64)         | 0       |
| dense_2 (Dense)                | (None, 10)         | 650     |

Relevant Jupyter notebook is available [here](https://github.com/KemalAkin/urban-sound-classification/blob/main/cnn_model_trained.ipynb).

