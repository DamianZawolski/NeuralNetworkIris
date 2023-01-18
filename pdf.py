from fpdf import FPDF
from PIL import Image

class PDF(FPDF):
    def lines(self):
        self.rect(5.0, 5.0, 200.0, 287.0)

    def info(self, zdjecie, wagi_wejsciowa, wagi_ukryta, procent1, blad_testowy1, wspolczynnik_uczenia):
        im = Image.open("irys.jpg")
        width, height = im.size
        max_height = 45
        self.image("irys.jpg", link='', type='', w=width * (max_height / height), h=height * (max_height / height))
        self.set_xy(0.0, 0.0)
        self.set_font('Arial', 'B', 26)
        self.set_text_color(0, 0, 0)
        self.multi_cell(w=210.0, h=20.0, align='C', txt=f"\nWyniki klasyfikacji irysow", border=0)
        self.set_font('Arial', 'B', 12)
        self.multi_cell(w=190.0, h=12.0, align='C', txt=f"(wyniki po trzykrotnym przejsciu przez zbior uczacy)",
                        border=0)
        self.multi_cell(w=190.0, h=12.0, align='C',
                        txt=f"\nWagi pomiedzy warstwa wejsciowa a ukryta \n{wagi_wejsciowa}", border=0)
        self.multi_cell(w=190.0, h=12.0, align='C', txt=f"\nWagi pomiedzy warstwa ukryta a wyjsciowa \n{wagi_ukryta}",
                        border=0)
        self.multi_cell(w=190.0, h=12.0, align='C', txt=f"\nWspolczynnik uczenia wynosil {wspolczynnik_uczenia}",
                        border=0)
        self.multi_cell(w=190.0, h=12.0, align='C', txt=f"Prawidlowo zidentyfikowano ({procent1} % kwiatow)",
                        border=0)
        self.multi_cell(w=190.0, h=12.0, align='C', txt=f"Blad kwadratowy wyniosl {round(blad_testowy1, 2)}%.",
                        border=0)
        im = Image.open(zdjecie)
        width, height = im.size
        max_width = 100
        height = height * (max_width / width)

        self.set_xy(10.0, 1500)
        self.set_font('Arial', 'B', 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(w=190.0, h=10.0, align='C', txt=f"Wykresy dla seeda generatora liczb losowych równego 11", border=0)
        self.multi_cell(w=190.0, h=10.0, align='C', txt=f"\nPonizej znajduje sie wykres porownujacy sredni blad e dla sieci uczonych zbiorem testowym od 1 do 100 razy, wspolczynnik uczenia wynosi 0,3",
                        border=0)
        self.image(f"Wykres błędów testowych.png", link='', type='',w=185, h=125)
        self.set_xy(10.0, 1625)
        self.multi_cell(w=190.0, h=10.0, align='C',
                        txt=f"\nPonizej znajduje sie wykres pokazujacy procent prawidlowo sklasyfikowanych kwiatow ze zbioru testujacego do ilosci iteracji nauczania zbiorem uczacym (od 1 do 100 razy), wspolczynnik uczenia wynosi 0,3",
                        border=0)
        self.image(f"Wykres prawidłowych klasyfikacji.png", link='', type='',w=185, h=125)
        self.set_xy(10.0, 1625)
        self.multi_cell(w=190.0, h=10.0, align='C',
                        txt=f"\nPonizej znajduje sie wykres porownujacy sredni blad e dla sieci uczonych zbiorem testowym 3-krotnie, wspolczynnik uczenia wynosi miedzy 0,05 a 2,5 w odstepach co 0,05",
                        border=0)
        self.image(f"Współczynniki uczenia- wykres błędów testowych.png", link='', type='',w=185, h=125)

