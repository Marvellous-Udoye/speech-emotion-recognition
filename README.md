# Speech Emotion Recognition - Sound Classification

This project implements a speech emotion recognition pipeline on the
TESS dataset. It builds MFCC features from raw WAV audio, trains an LSTM-based
classifier to predict seven emotions, and visualizes both the audio signals
and training curves. 

## Setup (Windows, Python 3.12)

Recommended: use a virtual environment to keep dependencies isolated.

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install ipykernel
```

If TensorFlow fails to import on Windows, install the Microsoft Visual C++
Redistributable 2015-2022 (x64), then restart the notebook kernel.

TensorFlow 2.16 on Windows requires `numpy<2.0`. This is pinned in
`requirements.txt`.

## Notebook usage

1) Open `Speech Emotion Recognition - Sound Classification.ipynb`
2) Ensure the kernel is the project venv
3) Update the dataset path in the data-loading cell (see below)
4) Use TensorFlow Keras imports (`from tensorflow.keras ...`)
5) Run cells top to bottom

### Dataset path (local machine)

The tutorial code uses `/kaggle/input` which only exists on Kaggle. On your
machine, point to the local dataset folder.

Example:

```python
for dirname, _, filenames in os.walk(
    r"C:\Users\Marvel\Desktop\codes\collabs\speech-emotion-recognition\TESS Toronto emotional speech set data"
):
    ...
```

## Expected results

The baseline LSTM model in the notebook typically reaches around 65-72%
validation accuracy depending on random seed and environment.


## Dataset Information

There are a set of 200 target words were spoken in the carrier phrase "Say the word _' by two actresses (aged 26 and 64 years) and recordings were made of the set portraying each of seven emotions (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are 2800 data points (audio files) in total.
The dataset is organised such that each of the two female actor and their emotions are contain within its own folder. And within that, all 200 target words audio file can be found. The format of the audio file is a WAV format.

## Output Attributes

* anger
* disgust
* fear
* happiness
* pleasant surprise
* sadness
* neutral

## Download Links

Download link: [https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess](https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess)
More Datasets: [https://www.kaggle.com/dmitrybabko/speech-emotion-recognition-en](https://www.kaggle.com/dmitrybabko/speech-emotion-recognition-en)

## Libraries

* pandas
* matplotlib
* tensorflow
* librosa

## Neural Network

LSTM Network
Validation accuracy: ~65-72% (baseline)
