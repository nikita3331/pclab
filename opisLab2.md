# Różnice pomiędzy algorytmami

- K-średnich
    - Algorytm k-średnich wyznacza w każdej iteracji średnią odległość danego punktu od wszystkich centrów klastrów.
    - W tych wyznaczonych grupach jest wyznaczany nowy środek (nowy klaster)
    - W następnej iteracji znowu sprawdzane są odległości wszystkich punktów, tym razem od nowo utworzonych klastrów.
    - Algorytm trwa do momentu, kiedy wszystkie punkty przestaną zmieniać swój klaster.
- Sieć neuronowa
    - Sieć utworzona została za pomocą sklearn. 
    - Uczona jest za pomocą danych uczących, które stanowią 80% całości zbioru.
    - Działa ona na zasadzie najlepszego dopasowania wielomianu do poszczególnych inputów. W przypadku s1, input jest dwuwymiarowy. Czyli wielomian utworzony za pomocą optymalizacji metodą gradientu stochastycznego próbuje zminimalizować błąd pomiędzy wszystkimi inputami, a outputami. W istocie by przybliżyć zasadę działania sieci neuronowej, można powiedzieć o prostej linii najlepiej dopasowującej się do pewnego zbioru punktów. W wypadku skwantowanych outputów, będą to schodki, a nie prosta linia. A wielomian w ten sposób utworzony nie będzie linią prostą, lecz skomplikowaną funkcją zależną od wielkości ukrytych warstw sieci neuronowej.
    - Znając w ten sposób utworzony wielomian na zbiorze uczącym, wystarczy podstawić dane testowe i przeprowadzić analizę wyników.
    - Badając różne nastawy sieci neuronowej oraz stosując normalizację danych, otrzymany rezultat predykcji był na poziomie 75%. Zważywszy na to, że jest 15 klastrów, a 3 z 4 odpowiedzi były poprawne, można uznać to za dobry wynik.
- SOM
    - Mapa samoorganizująca została utworzona przy wykorzystaniu biblioteki minisom.
    - W celu implementacji tego algorytmu, dokonano analizy oraz lekkiej modyfikacji przykładowego kodu dostępnego na repozytorium biblioteki pod tym  [adresem](https://github.com/JustGlowing/minisom/blob/master/examples/Clustering.ipynb)
    - Zasada działania algorytmu przypomina uczenie nienadzorowanej sieci neuronowej. Dostosowuje ona swoje wagi w ten sposób, że w każdej iteracji sprawdzane jest który neuron wygrywa. Czyli jesy najbardziej podobny do danego wektora wejściowego. Następnie wagi wygrywającego neuronu oraz neuronów w pobliżu są dostosowywane by był on jeszcze bardziej do niego podobny. Dzięki temu będzie on miał mocniejszą odpowiedź na pobudzenie podobne do danego wektora wejściowego. W pobliżu czyli gdy mają podobne parametry. Wyznaczane jest to za pomocą różnych funkcji sąsiedzstwa. W przypadku danego projektu zastosowano funkcję Gaussa. Dostosowywanie wag opiera się o wzór dostępny pod tym [adresem](https://glowingpython.blogspot.com/2013/09/self-organizing-maps.html).
    - W celu uzyskania lepszych wyników przeprowadzono analizę wpływu iteracji uczenia sieci, na ostateczny błąd kwantyzacji. Testowano iteracje z zakresu od 1000 do 50000. Najlepszy wynik wyszedł dla 38000 iteracji. 
    - Dokonano porównania wyników klasteryzacji dokonanych przez k-średnich oraz SOM. Drugi algorytm miał skuteczność na poziomie 98% co jest bardzo dobrym wynikiem.


