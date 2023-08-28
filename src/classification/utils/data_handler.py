import glob
import os
import numpy as np
import opensmile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from mat4py import loadmat
from collections import Counter

RAVDESS_speech_corpus_path = "src/classification/data/RAVDESS/Audio_Speech_Actors_01-24/*/*.wav"
JL_corpus_path = "src/classification/data/JL Corpus/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/*.wav"
Sindhi_corpus_path = "src/classification/data/Feats - Sindhi - 1/ComParE funcs/Matfiles/*.mat"

def extract_feature(file_name):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    result = smile.process_file(file_name).iloc[0].to_list()
    return result

def load_RAVDESS_speech_corpus_with_random_testset(test_size=0.25):
    X, y = [], []
    for file in glob.glob(RAVDESS_speech_corpus_path):
        basename = os.path.basename(file)
        emotion = basename.split("-")[2]
        # print(emotion)
        features = extract_feature(file)
        X.append(features)
        y.append(emotion)
    encoder = OrdinalEncoder()
    # # print(set(y))
    y = encoder.fit_transform(np.array(y).reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(np.array(X), y, test_size=test_size, random_state=7)
    return X_train, X_test, y_train, y_test, encoder.categories_

def load_jl_corpus_with_random_testset(test_size=0.25):
    X, y = [], []
    for file in glob.glob(JL_corpus_path):
        basename = os.path.basename(file)
        emotion = basename.split("_")[1]
        if emotion in ['happy', 'sad', 'angry', 'neutral', 'excited']: 
            features = extract_feature(file)
            X.append(features)
            y.append(emotion)
    encoder = OrdinalEncoder()
    # print(set(y))
    y = encoder.fit_transform(np.array(y).reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(np.array(X), y, test_size=test_size, random_state=7)
    return X_train, X_test, y_train, y_test, encoder.categories_

def load_jl_corpus_with_speaker_based_testset():
    X_train, X_test, y_train, y_test = [], [], [], []
    for file in glob.glob(JL_corpus_path):
        basename = os.path.basename(file)
        speaker = basename.split("_")[0]
        emotion = basename.split("_")[1]
        if emotion in ['happy', 'sad', 'angry', 'neutral', 'excited']: 
            features = extract_feature(file)
            if speaker == "male1":
                X_test.append(features)
                y_test.append(emotion)
            else:
                X_train.append(features)
                y_train.append(emotion)
    encoder = OrdinalEncoder()
    y_train = encoder.fit_transform(np.array(y_train).reshape(-1, 1))
    y_test = encoder.transform(np.array(y_test).reshape(-1, 1))
    return X_train, X_test, y_train, y_test, encoder.categories_


def get_jl_corpus_statistics():
    emotion_list = []
    speaker_list = []
    for file in glob.glob(JL_corpus_path):
        basename = os.path.basename(file)
        speaker = basename.split("_")[0]
        emotion = basename.split("_")[1]
        speaker_list.append(speaker)
        emotion_list.append(emotion)
    print(Counter(speaker_list).keys())
    print(Counter(speaker_list).values())
    print(Counter(emotion_list).keys())
    print(Counter(emotion_list).values())

def get_sindhi_corpus_statistics(Sindhi_corpus_path):
    emotion_list = []
    for file in glob.glob():
        basename = os.path.basename(file)
        emotion = basename.split("_")[0]
        emotion_list.append(emotion)
    print(Counter(emotion_list).keys())
    print(Counter(emotion_list).values())    

def load_Sindhi_data(test_size=0.25):
    X, y = [], []
    for file in glob.glob(Sindhi_corpus_path):
        basename = os.path.basename(file)
        emotion = basename.split("_")[0]
        features = loadmat(file)["Feats"]
        X.append(features)
        y.append(emotion)
    encoder = OrdinalEncoder()
    y = encoder.fit_transform(np.array(y).reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(np.array(X), y, test_size=test_size, random_state=7)
    return X_train, X_test, y_train, y_test, encoder.categories_