import numpy as np
import pandas as pd

class SiecNeuronowa:
    def __init__(self, uczace, liczba_neuronow_w_warstwie_ukrytej=4):
        np.random.seed(11)
        wejscie = pd.read_csv(uczace)

        zbior_uczacy = self.przygotowanie_danych(wejscie)
        liczba_kolumn = len(zbior_uczacy.columns)
        liczba_wierszy = len(zbior_uczacy.index)
        self.X = zbior_uczacy.iloc[:, 0:(liczba_kolumn - 3)].values.reshape(liczba_wierszy, liczba_kolumn - 3)
        self.y = zbior_uczacy.iloc[:, (liczba_kolumn - 3):].values.reshape(liczba_wierszy, 3)

        liczba_neuronow_w_warstwie_wejsciowej = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            rozmiar_warstwy_wyjsciowej = 1
        else:
            rozmiar_warstwy_wyjsciowej = len(self.y[0])

        # Przypisanie losowych wag
        self.wagi_wejsciowa_do_ukrytej = 2 * np.random.random((liczba_neuronow_w_warstwie_wejsciowej, liczba_neuronow_w_warstwie_ukrytej)) - 1
        self.dane_wejsciowe_do_ukrytej = self.X
        self.delta_wejsciowa_do_ukrytej = np.zeros((liczba_neuronow_w_warstwie_wejsciowej, liczba_neuronow_w_warstwie_ukrytej))
        self.wagi_ukryta_do_wyjsciowej = 2 * np.random.random((liczba_neuronow_w_warstwie_ukrytej, rozmiar_warstwy_wyjsciowej)) - 1
        self.dane_ukryta_do_wyjsciowej = np.zeros((len(self.X), liczba_neuronow_w_warstwie_ukrytej))
        self.delta_ukryta_do_wyjsciowej = np.zeros((liczba_neuronow_w_warstwie_ukrytej, rozmiar_warstwy_wyjsciowej))
        self.delta_wyjscie = np.zeros((rozmiar_warstwy_wyjsciowej, 1))

    def sigmoid(self, x):
        """Funkcja licząca funkcję sigmoidalną"""
        return 1 / (1 + np.exp(-x))

    def pochodna_sigmoid(self, x):
        """Pochodna funkcji sigmoidalnej- indykuje pewnośc wobec obecnej wagi"""
        return x * (1 - x)

    def przygotowanie_danych(self, dane):

        # 3 neurony wyjściowe ponieważ mamy 3 gatunki kwiatów
        dane["Y1"] = (dane["Gatunek"] == "Iris-setosa") * 1
        dane["Y2"] = (dane["Gatunek"] == "Iris-versicolor") * 1
        dane["Y3"] = (dane["Gatunek"] == "Iris-virginica") * 1
        dane = dane.drop("Gatunek", 1)  # Usuwanie kolumny z gatunkami
        # Standaryzacja/ normalizacja
        for kolumna in dane.columns[0:4]:
            srednia = sum(dane[kolumna]) / dane.shape[0]
            dane[kolumna] = dane[kolumna] - srednia
            dane[kolumna] = dane[kolumna] / dane[kolumna].max()
        return dane

    # Trenowanie
    def trenuj(self, liczba_iteracji, wspolczynnik_uczenia=0.1):
        blad = 0
        for _ in range(liczba_iteracji):
            wyjscie = self.przekanazanie_danych_do_nastepnej_warstwy()
            blad = 0.5 * np.power((wyjscie - self.y), 2)
            self.propagacja_wsteczna(wyjscie)
            aktualizacja_warstwy_ukrytej = wspolczynnik_uczenia * self.dane_ukryta_do_wyjsciowej.T.dot(self.delta_wyjscie)
            aktualizajca_warstwy_wejsciowej = wspolczynnik_uczenia * self.dane_wejsciowe_do_ukrytej.T.dot(self.delta_ukryta_do_wyjsciowej)

            self.wagi_ukryta_do_wyjsciowej += aktualizacja_warstwy_ukrytej
            self.wagi_wejsciowa_do_ukrytej += aktualizajca_warstwy_wejsciowej

        print(f"Błąd po {liczba_iteracji} iteracjach wynosi {np.sum(blad)}")
        print("Finalne wagi wektorów w wejściowej do ukrytej")
        print(self.wagi_wejsciowa_do_ukrytej)
        print("Finalne wagi wektorów w ukrytej do wyjsciowej")
        print(self.wagi_ukryta_do_wyjsciowej)

    def przekanazanie_danych_do_nastepnej_warstwy(self):
        wejscie1 = np.dot(self.X, self.wagi_wejsciowa_do_ukrytej)
        self.dane_ukryta_do_wyjsciowej = self.sigmoid(wejscie1)
        wejscie2 = np.dot(self.dane_ukryta_do_wyjsciowej, self.wagi_ukryta_do_wyjsciowej)
        wyjscie = self.sigmoid(wejscie2)
        return wyjscie

    def propagacja_wsteczna(self, wyjscie):
        self.delta_warstwy_wyjsciowej(wyjscie)
        self.delta_warstwy_ukrytej()


    def delta_warstwy_wyjsciowej(self, out):
        self.delta_wyjscie = (self.y - out) * (self.pochodna_sigmoid(out))

    def delta_warstwy_ukrytej(self):
        self.delta_ukryta_do_wyjsciowej = (self.delta_wyjscie.dot(self.wagi_ukryta_do_wyjsciowej.T)) * (self.pochodna_sigmoid(self.dane_ukryta_do_wyjsciowej))

    def delta_warstwy_wejsciowej(self):
        self.delta_wejsciowa_do_ukrytej = np.multiply(self.pochodna_sigmoid(self.dane_wejsciowe_do_ukrytej), self.delta_wejsciowa_do_ukrytej.dot(self.wagi_wejsciowa_do_ukrytej.T))

    def predykcja(self, test):
        zbior_testujacy = pd.read_csv(test)
        zbior_testujacy = self.przygotowanie_danych(zbior_testujacy)
        liczba_kolumn = len(zbior_testujacy.columns)
        liczba_wierszy = len(zbior_testujacy.index)
        self.X = zbior_testujacy.iloc[:, 0:(liczba_kolumn - 3)].values.reshape(liczba_wierszy, liczba_kolumn - 3)
        self.y = zbior_testujacy.iloc[:, (liczba_kolumn - 3):].values.reshape(liczba_wierszy, 3)
        wyjscie = self.przekanazanie_danych_do_nastepnej_warstwy()
        blad = np.power((wyjscie - self.y), 2)
        blad_testowania = 0.5 * np.sum(blad)
        print(f"\nDokładność wynosi {round(100 - blad_testowania,2)}%")
        return blad_testowania


siec_neuronowa = SiecNeuronowa("uczace.csv")
siec_neuronowa.trenuj(liczba_iteracji=360, wspolczynnik_uczenia=0.01)
blad_testowy = siec_neuronowa.predykcja("testujace.csv")
print(f"Klasyfikacja została błędnie wykonana w {round(blad_testowy, 2)}% przypadków.")
