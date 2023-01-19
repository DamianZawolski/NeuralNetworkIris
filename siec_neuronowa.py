import numpy as np
import pandas as pd


class SiecNeuronowa:
    def __init__(self, uczace, seed_generatora_losowego, liczba_neuronow_w_warstwie_ukrytej=4):
        np.random.seed(seed_generatora_losowego)
        wejscie = pd.read_csv(uczace)

        zbior_uczacy = self.przygotowanie_danych(wejscie)

        liczba_kolumn = len(zbior_uczacy.columns)
        liczba_wierszy = len(zbior_uczacy.index)
        "[x1,x2,x3,x4]"
        self.x = zbior_uczacy.iloc[:, 0:(liczba_kolumn - 3)].values.reshape(liczba_wierszy, liczba_kolumn - 3)
        "[d1,d2,d3]"
        self.d = zbior_uczacy.iloc[:, (liczba_kolumn - 3):].values.reshape(liczba_wierszy, 3)

        liczba_neuronow_w_warstwie_wejsciowej = len(self.x[0])
        rozmiar_warstwy_wyjsciowej = len(self.d[0])

        # Przypisanie losowych wag
        "wagi pomiędzy warstwą wejściową, a ukrytą, macierz 4x4"
        self.w1 = 2 * np.random.random(
            (liczba_neuronow_w_warstwie_wejsciowej, liczba_neuronow_w_warstwie_ukrytej)) - 1

        "początkowo zera, później różnica względem poprzednich x1, wektor 4x1"
        self.delta_x = np.zeros(
            (liczba_neuronow_w_warstwie_wejsciowej, liczba_neuronow_w_warstwie_ukrytej))
        "wagi pomiędzy warstwą ukrytą, a wyjściową losowe wartości pomiędzy -1 a 1, 3 kolumny i 4 wiersze"
        self.w2 = 2 * np.random.random(
            (liczba_neuronow_w_warstwie_ukrytej, rozmiar_warstwy_wyjsciowej)) - 1
        "początkowo zera, później wartości w warstwie ukrytej, wektor 4x1"
        self.v = np.zeros((len(self.x), liczba_neuronow_w_warstwie_ukrytej))
        "początkowo zera, później różnica względem poprzednich v, wektor 4x1"
        self.delta_v = np.zeros((liczba_neuronow_w_warstwie_ukrytej, rozmiar_warstwy_wyjsciowej))
        "początkowo zera, później różnica względem d a y"
        self.delta_y = np.zeros((rozmiar_warstwy_wyjsciowej, 1))

    def sigmoid(self, s):
        """Funkcja licząca funkcję sigmoidalną"""
        return 1 / (1 + np.exp(-s))

    def pochodna_sigmoid(self, s):
        """Pochodna funkcji sigmoidalnej- wskazuje pewnośc wobec obecnej wagi"""
        return s * (1 - s)

    def przygotowanie_danych(self, dane):

        # Standaryzacja/ normalizacja
        for kolumna in dane.columns[0:4]:
            srednia = sum(dane[kolumna]) / dane.shape[0]
            dane[kolumna] = dane[kolumna] - srednia
            dane[kolumna] = dane[kolumna] / dane[kolumna].max()

        return dane

    # Trenowanie
    def uczenie(self, liczba_iteracji, wspolczynnik_uczenia):
        for _ in range(liczba_iteracji):
            y = self.przekanazanie_danych_do_warstwy_wyjsciowej()
            self.propagacja_wsteczna(y)
            aktualizacja_warstwy_ukrytej = wspolczynnik_uczenia * self.v.T.dot(self.delta_y)
            aktualizajca_warstwy_wejsciowej = wspolczynnik_uczenia * self.x.T.dot(self.delta_v)

            self.w2 += aktualizacja_warstwy_ukrytej
            self.w1 += aktualizajca_warstwy_wejsciowej

        # print("Finalne wagi wektorów pomiedzy wejściową a ukrytą")
        # print(self.w1)
        # print("Finalne wagi wektorów pomiedzy ukrytą a wyjściową")
        # print(self.w2)
        return self.w1, self.w2

    def przekanazanie_danych_do_warstwy_wyjsciowej(self):
        "pomnożony wektor wartości warstwy wejściowej przez macierz wag"
        warstwa_wejsciowa = np.dot(self.x, self.w1)
        "przypisanie wartościom warstwy ukrytej powyższej macierzy przepuszczonej przez funkcję sigmoidalną"
        self.v = self.sigmoid(warstwa_wejsciowa)
        "pomnożony wektor wartości warstwy ukrytej przez macierz wag"
        warstwa_ukryta = np.dot(self.v, self.w2)
        "przypisanie wartościom warstwy wyjściowej powyższej macierzy przepuszczonej przez funkcję sigmoidalną"
        y = self.sigmoid(warstwa_ukryta)
        return y

    def propagacja_wsteczna(self, y):
        self.delta_warstwy_wyjsciowej(y)
        self.delta_warstwy_ukrytej()

    def delta_warstwy_wyjsciowej(self, y):
        "aktualizacja zmiennej delta_y"
        self.delta_y = (self.d - y) * (self.pochodna_sigmoid(y))

    def delta_warstwy_ukrytej(self):
        "aktualizacja zmiennej delta_v"
        self.delta_v = (self.delta_y.dot(self.w2.T)) * (self.pochodna_sigmoid(self.v))

    def predykcja(self, test):
        "odczytanie danych z pliku"
        zbior_testujacy = pd.read_csv(test)
        "przygotowanie danych do pracy"
        zbior_testujacy = self.przygotowanie_danych(zbior_testujacy)
        liczba_kolumn = len(zbior_testujacy.columns)
        liczba_wierszy = len(zbior_testujacy.index)
        "odczytanie wartości wejściowych ze zbioru"
        self.x = zbior_testujacy.iloc[:, 0:(liczba_kolumn - 3)].values.reshape(liczba_wierszy, liczba_kolumn - 3)
        "odczytanie wartości oczekiwanych ze zbioru"
        self.d = zbior_testujacy.iloc[:, (liczba_kolumn - 3):].values.reshape(liczba_wierszy, 3)
        "wyniki działania sieci"
        y = self.przekanazanie_danych_do_warstwy_wyjsciowej()
        prawidlowe = 0
        for i in range(len(y)):
            if (y[i][0] > y[i][1] and y[i][0] > y[i][2] and self.d[i][0] == 1) or \
                    (y[i][1] > y[i][0] and y[i][1] > y[i][2] and self.d[i][1] == 1) or \
                    (y[i][2] > y[i][0] and y[i][2] > y[i][1] and self.d[i][2] == 1):
                prawidlowe += 1
            # else:
            # print(f"Błędna klasyfikacja: prawidłowe ({self.d[i]}), uzyskane ({y[i]})")
        procent_prawidlowej_klasyfikacji = round(prawidlowe / len(y) * 100, 2)
        # print(f"Prawidłowo zidentyfikowano {prawidlowe}/{len(y)} kwiatów ({procent_prawidlowej_klasyfikacji}%)")

        "Kwadrat różnicy"
        epsilon = np.power((y - self.d), 2)
        "Średni błąd kwadratowy"
        e = 0.5 * np.sum(epsilon)
        e = e / len(y)
        # print(f"\nDokładność wynosi {round(100 - e, 2)}%")
        return e, procent_prawidlowej_klasyfikacji
