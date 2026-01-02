# LSTM-Based Speech Emotion Recognition for Student Public Speaking Training

This project builds an emotion-aware feedback system for student public speaking
training. It extracts MFCC features from speech audio, trains an LSTM-based
classifier to recognize seven emotions, and delivers live emotion feedback in a
web interface to support delivery stability and emotional control.

## Research scope

The system targets student public speaking practice sessions and provides
emotion cues (e.g., fear, sadness, anger, neutrality) as a proxy for delivery
stability and confidence trends during rehearsal.

## Research objective

Design and evaluate an LSTM-based speech emotion recognition pipeline that
provides real-time emotion feedback for student public speaking training, with
quantitative evaluation of classification performance and qualitative analysis
of delivery stability cues.

## Research contributions

- A reproducible MFCC + LSTM pipeline for speech emotion recognition on TESS.
- A live web application that captures microphone audio and displays emotion
  confidence scores for rehearsal feedback.
- An evaluation workflow that produces publication-ready metrics tables and
  visualizations (confusion matrix, ROC, PR curves, and accuracy/loss plots).

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

## Live demo (Web app)

Train the model (required once):

```bash
python train_model.py --data-dir "C:\Users\...\speech-emotion-recognition\TESS Toronto emotional speech set data"
```

Or auto-detect the dataset:

```bash
python train_model_auto.py --root "C:\Users\...\speech-emotion-recognition"
```

Run the web app:

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` and allow microphone access. Turn on "Live
streaming" for continuous updates while recording.

## Notebook usage

1) Open `Speech Emotion Recognition - Sound Classification.ipynb`
2) Ensure the kernel is the project venv
3) Update the dataset path in the data-loading cell (see below)
4) Use TensorFlow Keras imports (`from tensorflow.keras ...`)
5) Run cells top to bottom

### Dataset location

The TESS dataset is downloaded and stored in the project root. The expected
folder name is `TESS Toronto emotional speech set data`.

## Expected results

The baseline LSTM model in the notebook typically reaches around 65-72%
validation accuracy depending on random seed and environment.

## Evaluation (figures + tables)

Generate paper-ready plots and metrics:

```bash
python evaluate_model.py --data-dir "C:\Users\...\speech-emotion-recognition\TESS Toronto emotional speech set data"
```

Artifacts are saved under `results/`.

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
