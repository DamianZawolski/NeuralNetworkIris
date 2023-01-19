from fpdf import FPDF
from PIL import Image


class PDF(FPDF):
    def lines(self):
        self.rect(5.0, 5.0, 200.0, 287.0)

    def info(self, procent1, blad_testowy1, wspolczynnik_uczenia, lista_bledow):
        im = Image.open("irys.jpg")
        width, height = im.size
        max_height = 45
        self.image("irys.jpg", link='', type='', w=width * (max_height / height), h=height * (max_height / height))
        self.set_xy(0.0, 0.0)
        self.set_font('DejaVu-Bold', '', 26)
        self.set_text_color(0, 0, 0)
        self.multi_cell(w=210.0, h=20.0, align='C', txt=f"\nWyniki klasyfikacji irysów", border=0)
        self.set_font('DejaVu', '', 12)
        self.multi_cell(w=190.0, h=12.0, align='C', txt=f"(Wyniki po trzykrotnym przejściu przez zbiór uczący).",
                        border=0)
        self.set_font('DejaVu-Bold', '', 12)
        self.multi_cell(w=190.0, h=12.0, align='C', txt=f"Graf przedstawiający finalne wagi w strukturze sieci:",
                        border=0)
        self.set_font('DejaVu', '', 12)
        self.image(f"graf.png", link='', type='', w=200, h=200)
        self.multi_cell(w=190.0, h=12.0, align='C', txt=f"\nWspółczynnik uczenia wynosił {wspolczynnik_uczenia}.",
                        border=0)
        self.multi_cell(w=190.0, h=12.0, align='C', txt=f"Prawidłowo zidentyfikowano ({procent1}% kwiatów).",
                        border=0)
        self.multi_cell(w=190.0, h=12.0, align='C', txt=f"Średni błąd kwadratowy wyniósł {round(blad_testowy1, 2)}%.",
                        border=0)
        self.set_font('DejaVu-Bold', '', 14)
        self.set_text_color(0, 0, 0)
        self.multi_cell(w=190.0, h=10.0, align='C', txt=f"\nUśrednione wyniki z badań na 10 seedach (seedy od 1 do 10).",
                        border=0)
        self.set_font('DejaVu', '', 12)
        self.multi_cell(w=190.0, h=10.0, align='C',
                        txt=f"Poniżej znajduje się wykres porównujący średni błąd e dla sieci uczonych zbiorem "
                            f"testowym od 1 do 100 razy, współczynnik uczenia wynosi 0,3.",
                        border=0)
        self.image(f"Wykres błędów testowych.png", link='', type='', w=185, h=125)
        self.set_xy(10.0, 1625)
        self.multi_cell(w=190.0, h=10.0, align='C',
                        txt=f"\nPoniżej znajduje się wykres pokazujący procent prawidłowo sklasyfikowanych kwiatów ze "
                            f"zbioru testującego do ilości iteracji nauczania zbiorem uczącym (od 1 do 100 razy), "
                            f"współczynnik uczenia wynosi 0,3.",
                        border=0)
        self.image(f"Wykres prawidłowych klasyfikacji.png", link='', type='', w=185, h=125)
        self.set_xy(10.0, 1625)
        self.multi_cell(w=190.0, h=10.0, align='C',
                        txt=f"\nPoniżej znajduje się wykres porównujący średni błąd e dla sieci uczonych zbiorem "
                            f"testowym 3-krotnie, współczynnik uczenia wynosi między 0,05 a 2,5 w odstępach co 0,05.",
                        border=0)
        self.image(f"Współczynniki uczenia- wykres błędów testowych.png", link='', type='', w=185, h=125)
        id_najmniejszego = lista_bledow.index(min(lista_bledow))
        wspolczynnik_najmniejszego = (1 + id_najmniejszego) * 0.05
        self.multi_cell(w=190.0, h=10.0, align='C',
                        txt=f"\nNajmniejszy średni błąd wystąpił dla współczynnika uczenia "
                            f"{round(wspolczynnik_najmniejszego, 2)} i wynosił {round(min(lista_bledow), 2)}.",
                        border=0)
