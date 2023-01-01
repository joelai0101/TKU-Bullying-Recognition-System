import sys
import pandas as pd
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
# ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

df = pd.read_csv("./emotion speech recognition/features.csv")
X = df.iloc[:, 1:].values
Y = df['labels'].values

scaler = StandardScaler()
encoder = OneHotEncoder()

X = scaler.fit_transform(X)
Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()

def predict(data):
    df = pd.DataFrame(columns=['features'])
    df.features = get_features(data)
    feature_df = pd.DataFrame(df['features'].values.tolist()).T

    feature = feature_df.iloc[:, :].values
    feature = scaler.transform(feature)
    feature = np.expand_dims(feature, axis=2)

    model = load_model(
        "./emotion speech recognition/models/Emotion_Model.h5")
    pred_test = model.predict(feature)

    Y_pred = encoder.inverse_transform(pred_test)

    # print(type(Y_pred.item(0)))
    
    return Y_pred, Y_pred.item(0)

def get_features(path):

    X, sample_rate = librosa.load(
        path, res_type='kaiser_fast', duration=10.0, sr=None)

    print(sample_rate)
    stft = np.abs(librosa.stft(X))

    # fmin 和 fmax 對應於人類語音的最小最大基本頻率
    pitches, magnitudes = librosa.piptrack(
        X, sr=sample_rate, S=stft, fmin=70, fmax=400)
    pitch = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, 1].argmax()
        pitch.append(pitches[index, i])

    pitch_tuning_offset = librosa.pitch_tuning(pitches)
    pitchmean = np.mean(pitch)
    pitchstd = np.std(pitch)
    pitchmax = np.max(pitch)
    pitchmin = np.min(pitch)

    # 頻譜質心
    cent = librosa.feature.spectral_centroid(y=X, sr=sample_rate)
    cent = cent / np.sum(cent)
    meancent = np.mean(cent)
    stdcent = np.std(cent)
    maxcent = np.max(cent)

    # 譜平面
    flatness = np.mean(librosa.feature.spectral_flatness(y=X))

    # 使用系數為13的MFCC特徵
    mfccs = np.mean(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=13).T, axis=0)
    mfccsstd = np.std(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=13).T, axis=0)
    mfccmax = np.max(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=13).T, axis=0)

    # 色譜圖
    chroma = np.mean(librosa.feature.chroma_stft(
        S=stft, sr=sample_rate).T, axis=0)

    # 梅爾頻率
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)

    # ottava對比
    contrast = np.mean(librosa.feature.spectral_contrast(
        S=stft, sr=sample_rate).T, axis=0)

    # 過零率
    zerocr = np.mean(librosa.feature.zero_crossing_rate(X))

    S, phase = librosa.magphase(stft)
    meanMagnitude = np.mean(S)
    stdMagnitude = np.std(S)
    maxMagnitude = np.max(S)

    # 均方根能量
    rms = librosa.feature.rms(S=S)[0]
    meanrms = np.mean(rms)
    stdrms = np.std(rms)
    maxrms = np.max(rms)

    ext_features = np.array([
        flatness, zerocr, meanMagnitude, maxMagnitude, meancent, stdcent,
        maxcent, stdMagnitude, pitchmean, pitchmax, pitchstd,
        pitch_tuning_offset, meanrms, maxrms, stdrms
    ])

    ext_features = np.concatenate(
        (ext_features, mfccs, mfccsstd, mfccmax, chroma, mel, contrast))

    return ext_features
