import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def przewidz_emocje(model, sciezka_pliku, srednia, odchylenie_standardowe):
    # Ekstrakcja cech z nowego pliku dźwiękowego
    cechy = ekstrahuj_cechy(sciezka_pliku)

    # Normalizacja danych na podstawie wcześniej obliczonych średnich i odchyleń standardowych
    znormalizowane_cechy = (cechy - srednia) / odchylenie_standardowe

    # Przewidywanie emocji
    prognozy = model.predict(np.expand_dims(znormalizowane_cechy, axis=0))

    # Konwersja wyników do czytelnej formy
    etykiety_emocji = label_encoder.classes_
    przewidziana_emocja = etykiety_emocji[np.argmax(prognozy)]

    return przewidziana_emocja

def ekstrahuj_cechy(sciezka_pliku, mfcc=True, chroma=True, mel=True):
    y, sr = librosa.load(sciezka_pliku, mono=True, duration=1.1)
    cechy = []

    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        cechy.extend(mfccs)

    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        cechy.extend(chroma)

    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)
        cechy.extend(mel)

    return cechy

def wczytaj_dane(sciezka_danych):
    dane = []
    etykiety = []

    emocje = os.listdir(sciezka_danych)

    for emocja in emocje:
        sciezka_emocji = os.path.join(sciezka_danych, emocja)
        
        if os.path.isdir(sciezka_emocji):  # Sprawdzamy, czy to katalog
            for nazwa_pliku in os.listdir(sciezka_emocji):
                if nazwa_pliku.endswith(".wav"):  # Sprawdzamy, czy plik ma rozszerzenie .wav
                    sciezka_pliku = os.path.join(sciezka_emocji, nazwa_pliku)
                    
                    # Ekstrakcja cech z każdego pliku dźwiękowego
                    cechy = ekstrahuj_cechy(sciezka_pliku)
                    
                    # Dodaj cechy i odpowiadającą etykietę emocji
                    dane.append(cechy)
                    etykiety.append(emocja)

    return np.array(dane), np.array(etykiety)


def zwizualizuj_emocje(model, srednia, odchylenie_standardowe):
    # Lista emocji
    emocje = ['happy', 'angry', 'fearful', 'calm']

    plt.figure(figsize=(15, 10))

    for i, emocja in enumerate(emocje, 1):
        sciezka_danych = "/Users/gracjansadkowski/Desktop/data1"
        sciezka_emocji = os.path.join(sciezka_danych, emocja)

        # Wybierz jeden plik z danej emocji
        plik_emocji = os.path.join(sciezka_emocji, os.listdir(sciezka_emocji)[0])

        # Wczytaj dane audio
        y, sr = librosa.load(plik_emocji, mono=True, duration=5)

def historia_uczenia(model_fit):
    # Wykres dokładności w czasie uczenia
    dokladnosc = model_fit.history['accuracy']
    strata = model_fit.history['loss']
    epoki = range(1, len(dokladnosc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoki, dokladnosc, 'bo', label='Dokładność trenowania')
    plt.title('Dokładność trenowania w czasie')
    plt.xlabel('Epoki')
    plt.ylabel('Dokładność trenowania')
    plt.legend()

    # Wykres straty w czasie uczenia
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 2)
    plt.plot(epoki, strata, 'r', label='Strata trenowania')
    plt.title('Strata trenowania w czasie')
    plt.xlabel('Epoki')
    plt.ylabel('Strata trenowania')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    sciezka_danych = "/Users/gracjansadkowski/Desktop/data1"
    
    # Wczytaj dane i etykiety
    X, y = wczytaj_dane(sciezka_danych)

    # Konwersja etykiet na formę numeryczną
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = to_categorical(y)

    # Podział danych na zbiór treningowy, walidacyjny i testowy
    X_treningowe, X_testowe, y_treningowe, y_testowe = train_test_split(X, y, test_size=0.2, random_state=42)
    X_treningowe, X_walidacyjne, y_treningowe, y_walidacyjne = train_test_split(X_treningowe, y_treningowe, test_size=0.1, random_state=42)

    # Normalizacja danych
    srednia = np.mean(X_treningowe, axis=0)
    odchylenie_standardowe = np.std(X_treningowe, axis=0)
    X_treningowe = (X_treningowe - srednia) / odchylenie_standardowe
    X_walidacyjne = (X_walidacyjne - srednia) / odchylenie_standardowe
    X_testowe = (X_testowe - srednia) / odchylenie_standardowe

    # Budowa modelu sieci neuronowej przy uzyciu bibl Keras
    model = Sequential()
    # Warstwa wejściowa 
    model.add(Dense(256, input_shape=(X_treningowe.shape[1],), activation='relu'))
    # Warstwa Dropout (pozwala na unikniecie przeuczenia)
    model.add(Dropout(0.5))
    # Warstwa ukryta (model uczy sie bardziej złożonych danych)
    model.add(Dense(128, activation='relu'))
    # Warstwa Dropout
    model.add(Dropout(0.5))
    # Warstwa ukryta (64 neurony)
    model.add(Dense(64, activation='relu'))
    # Warstwa wyjsciowa 
    model.add(Dense(y.shape[1], activation='softmax'))

    # Kompilacja modelu
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Trenowanie modelu z danymi walidacyjnymi
    historia_uczenia_modelu = model.fit(X_treningowe, y_treningowe, epochs=100, batch_size=32, validation_data=(X_walidacyjne, y_walidacyjne))

    # Zwizualizuj fale dźwiękowe dla różnych emocji
    zwizualizuj_emocje(model, srednia, odchylenie_standardowe)

    # Wykres dokładności w czasie uczenia
    historia_uczenia(historia_uczenia_modelu)

    # Ocena modelu na danych testowych
    strata, dokladnosc = model.evaluate(X_testowe, y_testowe)
    print(f'Dokładność na danych testowych: {dokladnosc * 100:.2f}%')

    # Przykład użycia dla nowego pliku
    sciezka_nowego_pliku = "/Users/gracjansadkowski/Desktop/emotions/go-safe-fearful.wav"
    przewidziana_emocja = przewidz_emocje(model, sciezka_nowego_pliku, srednia, odchylenie_standardowe)
    print(f"Przewidziana Emocja: {przewidziana_emocja}")
    
    # Wczytaj dane audio dla nowego pliku
    y_nowego_pliku, _ = librosa.load(sciezka_nowego_pliku, mono=True, duration=1.1)

    # Zwizualizuj fale dźwiękowe dla nowego pliku
    plt.figure(figsize=(15, 5))
    plt.plot(librosa.times_like(y_nowego_pliku), y_nowego_pliku)
    plt.title(f'Emocja dla nowego pliku: {przewidziana_emocja}')
    plt.xlabel('Czas (s)')
    plt.ylabel('Amplituda')
    plt.grid()
    plt.show()





