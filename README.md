# Urban Sound Classification w/ CNN

As the final project of [Global AI Hub DL Bootcamp](https://globalaihub.com/activity/), we used Convolutional Neural Network (CNN) to classify urban sounds. 

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

## Preprocessing with OpenCV

## Training the Model