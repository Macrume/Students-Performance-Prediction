# Students-Performance-Prediction

## Opis

**Students-Performance-Prediction** to aplikacja webowa wykorzystująca modele uczenia maszynowego do przewidywania wyników w nauce uczniów, na podstawie danych demograficznych i edukacyjnych a także umożliwia analizę czynników wpływających na osiągnięcia edukacyjne.

## Cele projektu

- **Opracowanie modeli predykcyjnych**: Przewidywanie wyników uczniów na podstawie dostępnych danych oraz identyfikacja najlepszego modelu.
- **Analiza czynników wpływających na wyniki**: Zbadanie wpływu czynników demograficznych i edukacyjnych, takich jak wykształcenie rodziców, płeć czy dostęp do dodatkowych zasobów edukacyjnych.
- **Stworzenie narzędzia analitycznego**: Wizualizacja danych i wyników, wspierająca nauczycieli w dostosowywaniu strategii nauczania.

## Znaczenie projektu

Projekt umożliwia:

- **Identyfikację kluczowych czynników** wpływających na wyniki uczniów.
- **Promowanie zrównoważonego rozwoju edukacji** poprzez wykorzystanie technologii wspierającej równość i efektywność systemu nauczania.

## Użyte dane i metody

**Podsumowanie użytych danych i metod**
W projekcie wykorzystano dane dotyczące osiągnięć edukacyjnych uczniów z dwóch portugalskich szkół średnich. Zbiór danych zawiera szczegółowe informacje o uczniach, takie jak dane demograficzne, społeczne i związane ze szkołą, zebrane za pomocą raportów szkolnych oraz ankiet.

**Opis danych:**
Liczba rekordów: 649 uczniów.
Liczba atrybutów: 30 cech (plus zmienna docelowa).
Zmienna docelowa: G3 (końcowa ocena w skali 0–20).
Charakterystyka zbioru: Multidyscyplinarny, z cechami numerycznymi i kategorycznymi, umożliwiający zadania klasyfikacji i regresji.
Brakujące dane: Zbiór danych jest kompletny, nie zawiera brakujących wartości.

**Dane obejmują między innymi:**
Demografię: wiek, płeć, typ zamieszkania (miejski/wiejski).
Czynniki społeczne: wielkość rodziny, wykształcenie rodziców, ich zawód, jakość relacji rodzinnych.
Czynniki szkolne: czas dojazdu do szkoły, tygodniowy czas nauki, uczestnictwo w zajęciach dodatkowych.
Styl życia: korzystanie z Internetu, spożycie alkoholu, ilość wolnego czasu.
Historia edukacyjna: oceny z dwóch wcześniejszych okresów (G1, G2), liczba nieobecności oraz wcześniejsze porażki.

**Metody analizy i modelowania:**
W projekcie wykorzystano różnorodne metody uczenia maszynowego, w tym klasyfikację i regresję. Modele zostały przetestowane pod kątem przewidywania oceny końcowej (G3) oraz przeanalizowano ich wydajność za pomocą kluczowych metryk, takich jak:
Dokładność (Accuracy): Ocenia poprawność klasyfikacji.
Błąd średniokwadratowy (RMSE): Mierzy różnice między wartościami przewidywanymi a rzeczywistymi.
Średni błąd absolutny (MAE): Pokazuje średnią wartość błędu predykcji.

**Dodatkowe informacje:**
Warto zauważyć, że ocena końcowa G3 jest silnie skorelowana z wcześniejszymi ocenami (G1, G2), co stanowi ważny aspekt przy budowie modeli.
Dane są dobrze ustrukturyzowane i pozwalają na eksplorację czynników wpływających na wyniki edukacyjne uczniów.
Projekt ten stanowi znaczący krok w wykorzystaniu technologii do analizy i wspierania procesów edukacyjnych. Pozwala na identyfikację kluczowych czynników wpływających na wyniki uczniów, co może być pomocne dla nauczycieli i administratorów w podejmowaniu decyzji opartej na danych.

## Instalacja

Aby uruchomić aplikację, postępuj zgodnie z poniższymi krokami:

1. **Sklonuj repozytorium:**

   ```bash
   git clone https://github.com/Macrume/Students-Performance-Prediction/tree/main
   cd prognozator-wynikow-uczniow
   ```
2. **Zainstaluj zależności za pomocą Poetry**
    ```bash
    poetry install
    ```
3. **Aktywuj środowisko Poetry**
    ```bash
    poetry shell
    ```

## Uruchomienie aplikacji

Aplikację można uruchomić za pomocą Streamlit. W terminalu wpisz:

```bash
streamlit run interface_streamlit.py --server.port 8081
```

Po uruchomieniu, aplikacja będzie dostępna pod adresem http://localhost:8081.

## Metryki modeli

Najlepszy model na podstawie F1 Score:

| Model               | Accuracy | F1 Score | ROC AUC  |
|---------------------|----------|----------|----------|
| Logistic Regression | 0.596639 | 0.739130 | 0.693043 |
| KNN                 | 0.563025 | 0.717391 | 0.663913 |
| SVM                 | 0.596639 | 0.739130 | 0.724928 |

