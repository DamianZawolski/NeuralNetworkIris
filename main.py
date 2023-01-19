from pdf import *
from graf_z_wagami import *
from siec_neuronowa import *
import warnings

warnings.filterwarnings("ignore")

wspolczynnik_uczenia = 0.25
seed_generatora_losowego = 11
siec_neuronowa = SiecNeuronowa("uczace.csv", seed_generatora_losowego)
wagi_wejsciowa, wagi_ukryta = siec_neuronowa.uczenie(liczba_iteracji=3, wspolczynnik_uczenia=wspolczynnik_uczenia)
blad_testowy1, procent1 = siec_neuronowa.predykcja("testujace.csv")
# print(f"Średni błąd kwadratowy wyniósł {round(blad_testowy1, 2)}")
wejsciowe_lista = []
ukryte_lista = []
for elem in wagi_wejsciowa:
    for elem2 in elem:
        wejsciowe_lista.append(elem2)

for elem in wagi_ukryta:
    for elem2 in elem:
        ukryte_lista.append(elem2)

rysuj_graf(wejsciowe_lista, ukryte_lista)
x = []
for i in range(100):
    x.append(i + 1)

lista_bledow_10 = []
procenty_10 = []
seedy = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for elem in seedy:
    lista_bledow = []
    procenty = []
    for i in range(100):
        siec_neuronowa = SiecNeuronowa("uczace.csv", seed_generatora_losowego=elem)
        siec_neuronowa.uczenie(liczba_iteracji=i, wspolczynnik_uczenia=0.3)
        blad_testowy, procent = siec_neuronowa.predykcja("testujace.csv")
        lista_bledow.append(blad_testowy)
        procenty.append(procent)
    procenty_10.append(procenty)
    lista_bledow_10.append(lista_bledow)

procenty = []
lista_bledow = []
for i in range(100):
    procenty_tymczasowe = 0
    bledy_tymczasowe = 0
    for j in range(len(lista_bledow_10)):
        procenty_tymczasowe += procenty_10[j][i]
        bledy_tymczasowe += lista_bledow_10[j][i]
    procenty.append(procenty_tymczasowe / len(lista_bledow_10))
    lista_bledow.append(bledy_tymczasowe / len(lista_bledow_10))

_, ax = plt.subplots(1)
ax.scatter(x, lista_bledow, label='Błąd wyrażony w procentach', color="darkviolet")
fig = plt.gcf()
fig.set_size_inches(10, 8)
ax.set_xlabel('Liczba iteracji')
ax.set_ylabel('Średnie e dla zbioru testującego')
ax.legend()
ax.set_xlim(0, 100)
ax.set_title("Średni błąd testowy w zależności od ilości przejść przez cały zbiór danych uczących")
plt.savefig('Wykres błędów testowych.png')
plt.show()

_, ax = plt.subplots(1)
ax.scatter(x, procenty, label='Dokładność wyrażona w procentach', color="darkviolet")
fig = plt.gcf()
fig.set_size_inches(10, 8)
ax.set_xlabel('Liczba iteracji')
ax.set_ylabel('Średni procent prawidłowo sklasyfikowanych kwiatów')
ax.legend()
ax.set_xlim(0, 100)
ax.set_title("Prawidłowo sklasyfikowane kwiaty")
plt.savefig('Wykres prawidłowych klasyfikacji.png')
plt.show()

lista_bledow = []
procenty = []
x = []
for i in range(50):
    x.append(i * 0.05 + 0.05)
lista_bledow_10 = []
procenty_10 = []
for elem in seedy:
    lista_bledow = []
    for i in range(50):
        siec_neuronowa = SiecNeuronowa("uczace.csv", seed_generatora_losowego=elem)
        siec_neuronowa.uczenie(liczba_iteracji=3, wspolczynnik_uczenia=0.05 + i * 0.05)
        blad_testowy, procent = siec_neuronowa.predykcja("testujace.csv")
        lista_bledow.append(blad_testowy)
    lista_bledow_10.append(lista_bledow)

lista_bledow = []
for i in range(50):
    bledy_tymczasowe = 0
    for j in range(len(lista_bledow_10)):
        bledy_tymczasowe += lista_bledow_10[j][i]
    lista_bledow.append(bledy_tymczasowe / len(lista_bledow_10))

_, ax = plt.subplots(1)
ax.scatter(x, lista_bledow, label='Błąd wyrażony w procentach', color="darkviolet")
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
    pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
    pdf.add_font('DejaVu-Bold', '', 'DejaVuSansCondensed-Bold.ttf', uni=True)

    pdf.set_font('DejaVu', '', 14)
    pdf.add_page()
    pdf.info(procent1, blad_testowy1, wspolczynnik_uczenia, lista_bledow)
    pdf.set_author('Damian Zawolski')
    pdf.output(f"raport.pdf", 'F')


create_pdf()
