# Wstępne porównanie modeli dla TSP

## Cel eksperymentu

Sprawdzono, jak liczba dostępnych przykładów tras wpływa na przewidywanie
długości trasy przez trzy modele: TabPFN, drzewo decyzyjne i regresję liniową.
Nie zmieniano liczby miast ani ich położeń.

Dla 7 miast istnieje 360 różnych tras po ustaleniu miasta startowego i
potraktowaniu trasy oraz jej odwrócenia jako tej samej. W eksperymencie
wykorzystano próbki 40, 100, 200 i 360 tras. Próbki były zagnieżdżone:
mniejsza próbka stanowiła część większej. Każdą próbkę podzielono w tej samej
proporcji - 75% na trening i 25% na test. Wszystkie modele oceniano na tej
samej części testowej w obrębie danego przebiegu.

## Wyniki

Wyniki poniżej pochodzą z jednego uruchomienia z ziarnem losowym 42. MAE jest
średnim błędem bezwzględnym i niższa wartość oznacza mniejszy błąd. Wyższe R²
oznacza lepsze dopasowanie; wynik poniżej zera oznacza rezultat gorszy niż
przewidywanie średniego kosztu.

| Wylosowane trasy | Trening | Test | Model | MAE | R² |
|---:|---:|---:|---|---:|---:|
| 40 | 30 | 10 | TabPFN | 0,1414 | 0,8625 |
| 40 | 30 | 10 | Drzewo decyzyjne | 0,3003 | 0,5741 |
| 40 | 30 | 10 | Regresja liniowa | 0,0000 | 1,0000 |
| 100 | 75 | 25 | TabPFN | 0,0239 | 0,9944 |
| 100 | 75 | 25 | Drzewo decyzyjne | 0,3614 | -0,1613 |
| 100 | 75 | 25 | Regresja liniowa | 0,0000 | 1,0000 |
| 200 | 150 | 50 | TabPFN | 0,0099 | 0,9991 |
| 200 | 150 | 50 | Drzewo decyzyjne | 0,2149 | 0,5359 |
| 200 | 150 | 50 | Regresja liniowa | 0,0000 | 1,0000 |
| 360 | 270 | 90 | TabPFN | 0,0068 | 0,9995 |
| 360 | 270 | 90 | Drzewo decyzyjne | 0,1568 | 0,7104 |
| 360 | 270 | 90 | Regresja liniowa | 0,0000 | 1,0000 |

## Interpretacja

- **TabPFN:** w tym przebiegu błąd malał wraz ze wzrostem próbki - MAE spadło
  z 0,1414 przy 40 trasach do 0,0068 przy 360. Więcej przykładów daje modelowi
  więcej informacji o zależności między kolejnością przejazdów a długością
  trasy. 
- **Drzewo decyzyjne:** w tej konfiguracji nie odtworzyło tak dobrze kosztu
  jako sumy wielu odcinków. Drzewo dzieli dane kolejnymi progami i może wymagać
  wielu liści, by opisać kombinacje przejazdów. Przy małej próbce część
  kombinacji w teście nie jest dobrze reprezentowana w treningu. Wynik dla
  100 tras (R² = -0,1613) pokazuje też, że rezultat nie poprawia się
  koniecznie monotonicznie - wpływ ma konkretny podział trening/test.
- **Regresja liniowa:** w tym zadaniu jest wyjątkowo dobrze dopasowana do
  sposobu tworzenia etykiety. One-hot kodowanie każdej kolumny pozwala modelowi
  przypisać współczynnik możliwemu przejazdowi z danego miasta, a predykcja
  liniowa sumuje wkłady przejazdów. Rzeczywisty koszt trasy jest właśnie sumą
  odległości kolejnych przejazdów. Dlatego regresja uzyskała w tych
  eksperymentach MAE zaokrąglone do 0,0000 i R² zaokrąglone do 1,0000.


## Ograniczenia wniosków

To porównanie dotyczy **jednego układu 7 miast, jednego ziarna i jednego
podziału trening/test**. Nie wystarcza do stwierdzenia, że jeden model ogólnie
przewyższa drugi. Warto powtórzyć eksperyment z kilkoma ziarnami, a także na
różnych losowych układach miast. Ponadto testowane są nowe kolejności tras dla
tych samych miast, a nie uogólnianie na nieznane układy miast.
