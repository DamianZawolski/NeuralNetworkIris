import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pdf import *


class SiecNeuronowa:
    def __init__(self, uczace, seed_generatora_losowego, liczba_neuronow_w_warstwie_ukrytej=4):
        np.random.seed(seed_generatora_losowego)
        wejscie = pd.read_csv(uczace)

        zbior_uczacy = self.przygotowanie_danych(wejscie)
        liczba_kolumn = len(zbior_uczacy.columns)
        liczba_wierszy = len(zbior_uczacy.index)
        self.x = zbior_uczacy.iloc[:, 0:(liczba_kolumn - 3)].values.reshape(liczba_wierszy, liczba_kolumn - 3)
        self.d = zbior_uczacy.iloc[:, (liczba_kolumn - 3):].values.reshape(liczba_wierszy, 3)

        liczba_neuronow_w_warstwie_wejsciowej = len(self.x[0])
        rozmiar_warstwy_wyjsciowej = len(self.d[0])

        # Przypisanie losowych wag
        self.w1 = 2 * np.random.random(
            (liczba_neuronow_w_warstwie_wejsciowej, liczba_neuronow_w_warstwie_ukrytej)) - 1
        self.x1 = self.x
        self.delta_x1 = np.zeros(
            (liczba_neuronow_w_warstwie_wejsciowej, liczba_neuronow_w_warstwie_ukrytej))
        self.w2 = 2 * np.random.random(
            (liczba_neuronow_w_warstwie_ukrytej, rozmiar_warstwy_wyjsciowej)) - 1
        self.v = np.zeros((len(self.x), liczba_neuronow_w_warstwie_ukrytej))
        self.delta_v = np.zeros((liczba_neuronow_w_warstwie_ukrytej, rozmiar_warstwy_wyjsciowej))
        self.delta_y = np.zeros((rozmiar_warstwy_wyjsciowej, 1))

    def sigmoid(self, s):
        """Funkcja licząca funkcję sigmoidalną"""
        return 1 / (1 + np.exp(-s))

    def pochodna_sigmoid(self, s):
        """Pochodna funkcji sigmoidalnej- wskazuje pewnośc wobec obecnej wagi"""
        return s * (1 - s)

    def przygotowanie_danych(self, dane):

        # 3 neurony wyjściowe ponieważ mamy 3 gatunki kwiatów
        dane["J1"] = (dane["Gatunek"] == "Iris-setosa") * 1
        dane["J2"] = (dane["Gatunek"] == "Iris-versicolor") * 1
        dane["J3"] = (dane["Gatunek"] == "Iris-virginica") * 1
        dane = dane.drop("Gatunek", 1)  # Usuwanie kolumny z gatunkami
        # Standaryzacja/ normalizacja
        for kolumna in dane.columns[0:4]:
            srednia = sum(dane[kolumna]) / dane.shape[0]
            dane[kolumna] = dane[kolumna] - srednia
            dane[kolumna] = dane[kolumna] / dane[kolumna].max()
        return dane

    # Trenowanie
    def uczenie(self, liczba_iteracji, wspolczynnik_uczenia=0.1):
        e = 0
        for _ in range(liczba_iteracji):
            y = self.przekanazanie_danych_do_warstwy_wyjsciowej()
            e = 0.5 * np.power((y - self.d), 2)
            self.propagacja_wsteczna(y)
            aktualizacja_warstwy_ukrytej = wspolczynnik_uczenia * self.v.T.dot(self.delta_y)
            aktualizajca_warstwy_wejsciowej = wspolczynnik_uczenia * self.x1.T.dot(self.delta_v)

            self.w2 += aktualizacja_warstwy_ukrytej
            self.w1 += aktualizajca_warstwy_wejsciowej

        print(f"Błąd po {liczba_iteracji} iteracjach wynosi {np.sum(e)}")
        print("Finalne wagi wektorów pomiedzy wejściową a ukrytą")
        print(self.w1)
        print("Finalne wagi wektorów pomiedzy ukrytą a wyjściową")
        print(self.w2)
        return self.w1, self.w2

    def przekanazanie_danych_do_warstwy_wyjsciowej(self):
        warstwa_wejsciowa = np.dot(self.x, self.w1)
        self.v = self.sigmoid(warstwa_wejsciowa)
        warstwa_ukryta = np.dot(self.v, self.w2)
        warstwa_wyjsciowa = self.sigmoid(warstwa_ukryta)
        return warstwa_wyjsciowa

    def propagacja_wsteczna(self, wyjscie):
        self.delta_warstwy_wyjsciowej(wyjscie)
        self.delta_warstwy_ukrytej()

    def delta_warstwy_wyjsciowej(self, out):
        self.delta_y = (self.d - out) * (self.pochodna_sigmoid(out))

    def delta_warstwy_ukrytej(self):
        self.delta_v = (self.delta_y.dot(self.w2.T)) * (self.pochodna_sigmoid(self.v))

    def delta_warstwy_wejsciowej(self):
        self.delta_x1 = np.multiply(self.pochodna_sigmoid(self.x1), self.delta_x1.dot(self.w1.T))

    def predykcja(self, test):
        zbior_testujacy = pd.read_csv(test)
        zbior_testujacy = self.przygotowanie_danych(zbior_testujacy)
        liczba_kolumn = len(zbior_testujacy.columns)
        liczba_wierszy = len(zbior_testujacy.index)
        self.x = zbior_testujacy.iloc[:, 0:(liczba_kolumn - 3)].values.reshape(liczba_wierszy, liczba_kolumn - 3)
        self.d = zbior_testujacy.iloc[:, (liczba_kolumn - 3):].values.reshape(liczba_wierszy, 3)
        y = self.przekanazanie_danych_do_warstwy_wyjsciowej()
        prawidlowe = 0
        for i in range(len(y)):
            if (y[i][0] > y[i][1] and y[i][0] > y[i][2] and self.d[i][0] == 1) or \
                    (y[i][1] > y[i][0] and y[i][1] > y[i][2] and self.d[i][1] == 1) or \
                    (y[i][2] > y[i][0] and y[i][2] > y[i][1] and self.d[i][2] == 1):
                prawidlowe += 1
            else:
                print(f"Błędna klasyfikacja: prawidłowe ({self.d[i]}), uzyskane ({y[i]})")
        procent_prawidlowej_klasyfikacji = round(prawidlowe / len(y) * 100, 2)
        print(
            f"Prawidłowo zidentyfikowano {prawidlowe}/{len(y)} kwiatów ({procent_prawidlowej_klasyfikacji} procent)")

        "Kwadrat różnicy"
        epsilon = np.power((y - self.d), 2)
        "Średni błąd kwadratowy"
        e = 0.5 * np.sum(epsilon)
        e = e/len(y)
        print(f"\nDokładność wynosi {round(100 - e, 2)}%")
        return e, procent_prawidlowej_klasyfikacji

wspolczynnik_uczenia = 0.25
seed_generatora_losowego = 11
siec_neuronowa = SiecNeuronowa("uczace.csv", seed_generatora_losowego)
wagi_wejsciowa, wagi_ukryta = siec_neuronowa.uczenie(liczba_iteracji=3, wspolczynnik_uczenia=wspolczynnik_uczenia)
blad_testowy1, procent1 = siec_neuronowa.predykcja("testujace.csv")
print(f"Średni błąd kwadratowy wyniósł {round(blad_testowy1, 2)}")

x = []
lista_bledow = []
procenty = []
for i in range(100):
    x.append(i+1)
    siec_neuronowa = SiecNeuronowa("uczace.csv", seed_generatora_losowego)
    siec_neuronowa.uczenie(liczba_iteracji=i, wspolczynnik_uczenia=0.3)
    blad_testowy, procent = siec_neuronowa.predykcja("testujace.csv")
    lista_bledow.append(blad_testowy)
    procenty.append(procent)



_, ax = plt.subplots(1)
ax.scatter(x,lista_bledow, label='Błąd wyrażony w procentach')
fig = plt.gcf()
fig.set_size_inches(10, 8)
ax.set_xlabel('Liczba iteracji')
ax.set_ylabel('Średnie e dla zbioru testującego')
ax.legend()
ax.set_xlim(0, 100)
ax.set_title("Błąd testowy w zależności od ilości przejść przez cały zbiór danych uczących")
plt.savefig('Wykres błędów testowych.png')
plt.show()

_, ax = plt.subplots(1)
ax.scatter(x,procenty, label='Dokładność wyrażona w procentach')
fig = plt.gcf()
fig.set_size_inches(10, 8)
ax.set_xlabel('Liczba iteracji')
ax.set_ylabel('Procent prawidłowo sklasyfikowanych kwiatów')
ax.legend()
ax.set_xlim(0, 100)
ax.set_title("Prawidłowo sklasyfikowane kwiaty")
plt.savefig('Wykres prawidłowych klasyfikacji.png')
plt.show()

lista_bledow = []
procenty = []
x = []
for i in range(50):
    x.append(i*0.05 + 0.05)
    siec_neuronowa = SiecNeuronowa("uczace.csv", seed_generatora_losowego)
    siec_neuronowa.uczenie(liczba_iteracji=3, wspolczynnik_uczenia=0.05 + i*0.05)
    blad_testowy, procent = siec_neuronowa.predykcja("testujace.csv")
    lista_bledow.append(blad_testowy)
    procenty.append(procent)

_, ax = plt.subplots(1)
ax.scatter(x, lista_bledow, label='Błąd wyrażony w procentach')
fig = plt.gcf()
fig.set_size_inches(10, 8)
ax.set_xlabel('Współczynnik uczenia')
ax.set_ylabel('Średnie e dla zbioru testującego')
ax.legend()
ax.set_xlim(0, 2.6)
ax.set_title("3 przejścia przez cały zbiór w nauczaniu- błąd testowy w zależności od współczynnika uczenia")
plt.savefig('Współczynniki uczenia- wykres błędów testowych.png')
plt.show()


def create_pdf():
    pdf = PDF()
    pdf.add_page()
    pdf.lines()
    pdf.info("Wykres prawidłowych klasyfikacji.png", wagi_wejsciowa, wagi_ukryta, procent1, blad_testowy1, wspolczynnik_uczenia)
    pdf.set_author('Damian Zawolski')
    pdf.output(f"raport.pdf", 'F')


create_pdf()
