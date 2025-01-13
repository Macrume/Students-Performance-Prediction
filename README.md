# Students-Performance-Prediction

## Opis

**Students-Performance-Prediction** to aplikacja webowa wykorzystująca modele uczenia maszynowego do przewidywania wyników testów uczniów z matematyki, czytania i pisania na podstawie danych demograficznych i edukacyjnych. Narzędzie to wspiera nauczycieli i administratorów w podejmowaniu decyzji dotyczących dostosowywania programów nauczania do indywidualnych potrzeb uczniów, a także umożliwia analizę czynników wpływających na osiągnięcia edukacyjne.

## Cele projektu

- **Opracowanie modeli predykcyjnych**: Przewidywanie wyników uczniów na podstawie dostępnych danych oraz identyfikacja najlepszego modelu.
- **Analiza czynników wpływających na wyniki**: Zbadanie wpływu czynników demograficznych i edukacyjnych, takich jak wykształcenie rodziców, płeć czy dostęp do dodatkowych zasobów edukacyjnych.
- **Stworzenie narzędzia analitycznego**: Wizualizacja danych i wyników, wspierająca nauczycieli w dostosowywaniu strategii nauczania.

## Znaczenie projektu

Projekt umożliwia:

- **Identyfikację kluczowych czynników** wpływających na wyniki uczniów.
- **Wsparcie decyzji edukacyjnych** na podstawie danych, co prowadzi do bardziej spersonalizowanego nauczania.
- **Zmniejszenie nierówności edukacyjnych** poprzez identyfikację grup uczniów wymagających szczególnego wsparcia.
- **Promowanie zrównoważonego rozwoju edukacji** poprzez wykorzystanie technologii wspierającej równość i efektywność systemu nauczania.

## Użyte dane i metody

Dane wykorzystane w projekcie pochodzą z trzech amerykańskich szkół średnich i obejmują:

- Płeć uczniów, grupa etniczna, poziom wykształcenia rodziców, status korzystania z dofinansowania obiadów oraz ukończenie kursu przygotowawczego.
- Wyniki testów z matematyki, czytania i pisania.

Do analizy danych zastosowano techniki uczenia maszynowego takie jak regresja, klasyfikacja oraz analiza skupień. Modele były trenowane i oceniane przy użyciu metryk takich jak dokładność (Accuracy), błąd średniokwadratowy (RMSE) i średni błąd absolutny (MAE).

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

