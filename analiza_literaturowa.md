# Analiza Literaturowa


## 1. TabPFN

- **[Centralne repozytorium PriorLabs](https://github.com/PriorLabs/tabpfn)**  
  Na stronie centralnego repozytorium znajdują się 4 artykuły, które dokumentują rozwój rodziny modeli TabPFN. Kod i wagi modeli są dostępne na [Hugging Face firmy](https://huggingface.co/Prior-Labs).

### Accurate predictions on small data with a tabular foundation model

**Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister i Frank Hutter, _Nature_, tom 637, 2025.**
[Artykuł](https://www.nature.com/articles/s41586-024-08328-6) | [DOI: 10.1038/s41586-024-08328-6](https://doi.org/10.1038/s41586-024-08328-6)

### Cel i metoda

Artykuł przedstawia TabPFN (Tabular Prior-data Fitted Network) - model bazowy dla danych tabelarycznych. Autorzy zwracają uwagę, że w odróżnieniu od obrazów i tekstu, w analizie tabel nadal często bardzo dobrze sprawdzają się metody oparte na drzewach decyzyjnych, takie jak CatBoost, XGBoost czy Random Forest. Jednym z powodów jest różnorodność tabel: mogą mieć różną liczbę i typ cech, braki danych, wartości odstające oraz odmienne zależności między kolumnami.

TabPFN jest transformatorem wykorzystującym **uczenie kontekstowe** (in-context learning, ICL). Podczas wstępnego treningu model widzi miliony syntetycznych zbiorów danych wygenerowanych z rozmaitymi zależnościami między cechami i etykietami. Przy użyciu dostaje przykłady treningowe i testowe jako kontekst, a następnie przewiduje etykiety przykładów testowych w jednym przebiegu — bez trenowania od zera osobnego modelu dla każdego zadania.

Artykuł opisuje zastosowanie TabPFN do klasyfikacji i regresji; omawia też jego potencjalne możliwości generatywne, tworzenie reprezentacji danych oraz dostrajanie. Model jest przeznaczony przede wszystkim do **małych i średnich zbiorów**. W eksperymentach autorzy skupiają się na danych o maksymalnie 10 000 próbek, 500 cechach i — w klasyfikacji — 10 klasach. TabPFN automatycznie obsługuje m.in. wartości brakujące i cechy kategoryczne, choć odpowiednie przygotowanie danych i wiedza dziedzinowa nadal mogą poprawić wyniki.

### Ewaluacja i wyniki

Autorzy porównali TabPFN z metodami takimi jak CatBoost, XGBoost, LightGBM, Random Forest, regresja logistyczna, SVM i MLP. Główna ewaluacja obejmowała 29 zbiorów klasyfikacyjnych i 28 regresyjnych z benchmarków AutoML Benchmark oraz OpenML-CTR23. Dla każdego zbioru wykonano 10 powtórzeń z różnymi podziałami na część treningową i testową; stosowano podział 90/10. Oceniano m.in. ROC AUC i accuracy w klasyfikacji oraz RMSE i R² w regresji.

W głównym porównaniu z CatBoost TabPFN osiągnął wyższe znormalizowane wyniki:

- **Klasyfikacja, ustawienia domyślne:** ROC AUC 0,939 dla TabPFN i 0,752 dla CatBoost.
- **Klasyfikacja, po dostrajaniu:** ROC AUC 0,952 dla TabPFN i 0,822 dla CatBoost.
- **Regresja, ustawienia domyślne:** znormalizowane RMSE 0,923 dla TabPFN i 0,872 dla CatBoost.
- **Regresja, po dostrajaniu:** znormalizowane RMSE 0,968 dla TabPFN i 0,875 dla CatBoost.

Wartości są znormalizowanymi wynikami benchmarkowymi, a nie bezpośrednimi wartościami ROC AUC lub RMSE na jednym konkretnym zbiorze. TabPFN wygrywał na większości, ale nie na wszystkich porównywanych zbiorach.

Autorzy podają też, że domyślny TabPFN uzyskiwał mocne wyniki w średnio 2,8 sekundy dla klasyfikacji i 4,8 sekundy dla regresji. W ich eksperymencie przewyższał metody bazowe dostrajane nawet przez cztery godziny.

### Zalety i ograniczenia

- Dobre wyniki na małych i średnich danych tabelarycznych bez rozbudowanego dostrajania.
- Możliwość zwracania rozkładu predykcji, co pozwala modelować niepewność.
- Zakres głównej ewaluacji jest ograniczony do maksymalnie 10 000 próbek i 500 cech; skalowanie na większe zbiory wymaga dalszych badań.
- Zużycie pamięci rośnie wraz z rozmiarem danych, a wymagania obliczeniowe architektury rosną szybko wraz z liczbą próbek i cech.
- Predykcja może być wolniejsza niż w mocno zoptymalizowanych metodach drzewiastych. W przykładzie autorów predykcja dla 10 000 wierszy i 10 kolumn trwała 0,2 s na GPU (0,6 s bez GPU), podczas gdy CatBoost potrzebował około 0,0002 s.
- Artykuł podkreśla, że nadal istotne są czyszczenie danych, inżynieria cech i wiedza dziedzinowa.

### Znaczenie dla naszego projektu

Projekt dotyczy analizy jakości i sposobu działania modeli In-Context Learning, takich jak TabPFN, gdy na wejściu otrzymują permutacje. Artykuł stanowi punkt odniesienia, ponieważ opisuje architekturę TabPFN, mechanizm ICL oraz zastosowanie modelu do klasyfikacji i regresji na danych tabelarycznych. Nie bada jednak bezpośrednio permutacji jako wejścia, dlatego jego wyniki nie przesądzają o skuteczności modelu w naszym zadaniu.

- **[[2511.08667] TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models](https://arxiv.org/abs/2511.08667)**  
   

- **[[2605.13986] TabPFN-3: Technical Report](https://arxiv.org/abs/2605.13986)**  

- **[[2609.17895] TabPFN-3.5: Technical Report](https://arxiv.org/abs/2609.17895)**  



## 2. Rodzina modeli TabICL

- **[Centralne repozytorium TabICL](https://github.com/soda-inria/tabicl)**  
 
- **[TabICL: A Tabular Foundation Model for In-Context Learning on Large Data (Qu et al., ICML 2025)](https://arxiv.org/abs/2502.05564)**  
    

- **[TabICLv2: A better, faster, scalable, and open tabular foundation model (Qu et al., ICML 2026)](https://github.com/soda-inria/tabicl)**  