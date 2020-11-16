# Zastosowanie ANFIS do predykcji przynależności do klastra

- ANFIS:
    - W celu implementacji tej metody została wykorzystana klasa z GitHub znajdująca się pod tym adresem [tutaj](https://github.com/tiagoCuervo/TensorANFIS).
    - Dokonano porównania jakości wyznaczania przynależności w zależności od metody.
    - 
    - Dokonano porównania wpływu wielkości kryterium stopu na ostateczy wynik predykcji. 
    ![Kryterium stopu](predykcjaerror.png)
    - W celu sprawdzenia dokładnośći predykcji zaimplementowano algorytm, przekształcający centroidy pochodzące z wytrenowanego systemu rozmytego do postaci początkowych centroidów. Tym samym można dokonać porównania jak dobrze na tle początkowych klasteryzacji wypadają te dokonane przez system rozmyty.
    



