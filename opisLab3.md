# Zastosowanie systemu rozmytego do klasteryzacji

- Cmeans:
    - W celu implementacji tej metody została wykorzystana biblioteka skfuzzy.
    - Na potrzeby zadania dokonano predykcji punktów z pliku s1.txt . Wszystkie algorytmy działają tak samo jak w przypadku breast.txt, różni się jedynie funkcja zczytująca wartości z pliku. 
    - Funkcje do wytrenowania zbioru oraz generacji predykcji dostępne są na stronie biblioteki [tutaj](https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_cmeans.html)
    - Dokonano porównania wpływu wielkości kryterium stopu na ostateczy wynik predykcji. 
    ![Kryterium stopu](predykcjaerror.png)
    - W celu sprawdzenia dokładnośći predykcji zaimplementowano algorytm, przekształcający centroidy pochodzące z wytrenowanego systemu rozmytego do postaci początkowych centroidów. Tym samym można dokonać porównania jak dobrze na tle początkowych klasteryzacji wypadają te dokonane przez system rozmyty.
    



