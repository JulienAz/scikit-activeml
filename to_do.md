# Ideen
- Clusterensemble
  - proba prediction ensemble base classifier + cluster classifier
    - Nur cluster classifier?
  - Changedetection auf accuracy der cluster classifier
- Dynamische Clusteranzahl
  - trennen bei bestimmten punkten, vergleichen um changes zu erkennen
- Clusterclassifier
  - 1. z.B. Prediction ist meist vorkommende Klasse des Clusters wo reinfällt
  - Oder meist vorkommende der C nähsten CLuster
  - 2. Statistiken von Cluster in gelernten Featurevektor integrieren
    - Distanz zu Cluster
- Change Detector
  - Auf prediction error
  - Auf clusterstatistiken

- Ausprobieren Ensemble aber classifier in cluster nur majority vote

- ADWIN2 probieren mit labeled samples (mit features)

- Gewichtung der Ensemble nach Accuracy

- Naive Bayes in den Clustern

# Daily To-do
## General Todos:
- Testscript anpassen
  - ~~Args mit append~~
  - ~~Anzahl Cluster experimente~~
  - Approaches struktur verbessern (Erweiterbarer)
- Refactor to Dataset class
  - ~~Logging von Datensatz~~
  - mehrere Datensätze parallel
  - Farben in plots einheitlich
  - Datensätze übersicht in Overleaf schreiben
  - Logger Überklasse
## Bugs
- ~~Overflow von timestamps~~ 
- Wieso ClusterIDs Nan bei budget=0.01?
- Split Error
- Overflow Euler BEta
## 25.05

- [x] Random Strategy einbauen als Baseline

- [X] Time Window Clustering
  - Unterschiedliche time windows testen
  - Label Frequency plots betrachten
    - Auswirkung auf accuracy?
    - Sample verteilung Cluster

## 26.05
- [X] Neue Datensätze ausprobieren
  - [X] Einbauen (Git subproject)
    - Data generator anpassen

## 27.05
- [X] Neue Datensätze ausprobieren
  - [X] Analyse auf neuen Datensätzen
    - ~~Electricity~~
    - Ggf. Airline
    - ~~Artifical streams~~
      - ~~Rausfinden wie die aufgebaut sind~~

## 28.05
- [X] Weitere Analysen
  - [ ] verschieden CluStream Windows ausprobieren/vergleichen
- [X] Weitere Datensätze betrachten
  - Ggf. Airline

## Meetiningpunkte
- Electricity CLusterverteilung bei kleinen Budget
  - ggf. 1 CLuster und kleines Timewindow
- Batch Classifier auf 100000 setzen == Inkrementell?
- CluStream mit Batch classifier
- Clustering Splitten
  - AUf clusterdaten fitten wenn cluster gelöscht wird
  - Schematisch aufzeichen
- Chess board clustering angucken

## ToDo vor Meeting
- [X] Clustream auf window fitten
- [X] Clustream auf cluster daten fitten
- [ ] Clustering betrachten
- [X] CluStream Classifier refitten wenn cluster gelöscht

## 02.06
- [X] Batch CluStream Classifier
- [X] CluStream Cluster fitting
- [ ] Clusterverteilung Experimente

## Idee
- Clustering abhängig von Classifier machen
  - z.B. Cluster löschen wenn accuracy auf samples die in das Cluster fallen schlechter werden

## Meeting 07.06
- Wird bei Runtime error und budget 1 trotzdem gelabelt?
  - Kleinen wert aufaddieren oder abfangen
- RBF anzahl Cluster auf Anzahl Klassen
- Vergleich
  - 3 Clustering ansätze
    - ~gleiche time windows (Batch 300, timewindow 150...)
- Classifier approximieren mit Clusterstatistiken
- Utility mit aggregierten statistiken
- Aggregierte statistiken loggen
  - Schauen wie sich verändert über zeit
  - Changes erkennbar?

## 12.06
- [X] 3 Cluster based vergleich
- [X] Clusterstatistiken analysieren
- [X] Entropy einbauen

# Meeting 13.06
## Vorher
- Overflow error fix
- Vergleich Batch, Incrementell, Refit
  - Zu lange gedauert
- Cluster statistiken (Nach identifikatoren für Change gesucht)
  - logging sehr große dateien
  - Plotten hat RAM überlastet auf pc
  - Dann nur noch experimente mit 1 Rep da eh nur für eins Plotbar
- Geloggte Statistiken
  - Radius
  - N_samples
  - N_labeled_samples
  - Bessere Möglichkeiten arrays zu loggen?
- Clustering visualisiert
- Entropy logging
  - Changes erkennbar?

# Meetingpunkte
- Paquet
- GGf.
  - Random Sampling um zu Validieren ob N_sample plot und N-labelde korrespondieren
- Experioment RBF (Reaktion auf Change)
  - 3 Klassen
  - RBF Generator (https://scikit-multiflow.readthedocs.io/en/stable/api/generated/skmultiflow.data.RandomRBFGenerator.html)
  - 3000 Zeitschritte Change wo 2 sich abwechseln und 3 Gleich bleibt
  - In den beiden Clustern wo change is daten wegwerfen
    - Wie viele gelabelt wurden loggen, nicht wie viele drin sind
- Herausfinden ob change erkennen überhaupt was bringt
- Changedetector für Radius
  - ADWIN?
- Überwachen von Radius/ Entropy

-Bei SEA subsamplen (Alle 5 Datenpunkte nehmen)

# Mögliche Probleme:
  - Merges verändern statistiken
    - Radi steigt an und sinkt dann wieder
    - N Springt
    - Entropy womöglich veränderunt
  - Entropy bei binären Classification
    - Oszilliert stark zwischen 0-1 mit Merges
    - Relative Entropy abhängig von Anzahl an Samples?

# Meeting 21.06
## Punkte
- Wie Cluster updaten?
  - Erst mal löschen und warten bis punkt nicht in anderes passt dann mit neuem aufmachen
  - 2 Clusterstatistiken pro cluster halten
  - "Exponential bucketing"/"Pyramidical Timeframe"
    - Pro Cluster statistiken die unterschiedliche Zeithorizonte betrachten

## 03.07
- [X] Ensemble Clf implementieren
- [ ] Change detection auf cluster prediction
- [ ] Thesis struktur machen

# Meeting 21.06
## Punkte
- Bei Merge, wie umgehen?
  - Bei merge ist ja gar kein Change
  - Ggf einfach changedetector neu initialisieren, ausprobieren!
- Performance von Refit auf Chessboard über viele Reps
  - Unterschiedliche Parameters
- Change Detector auf Accuracy
  - Muss nicht unbedingt auf Ensemble
- Zliobate implementieren

## 06.07
-[X] Change Detector refactoring
-[X] Performance ergebnisse auswerten
-[X] Zliobate ansatz implementieren
  -[X] Implementierung
  -[X] Vergleiche mit Clustering- Refit/Ensemble

## 07.07
-[X] Paper recherche

Meeting: punkte

- Sensitivity Study mit Parametern
- Chessboard über Budget plotten
- Electricity:
  - Vor Clustering Kernel PCA

## 13.07
- [X] Auswahl prediction error/entropy implementieren
- [ ] KDA implementieren und testen
- [ ] Abstract schreiben

## 15.07
- [X] KDA implementieren und testen
- [ ] Abstract schreiben

# Meeting 18.07
## Punkte
- Finden die bei Covertype überhaupt noch einen change?
- Airline datensatz
  - Zu viele changes erkannt?
- ADWIN ausprobieren
- Andere Change detectoren testen?
- Mehr Reps testen
- Sensitivity Study
  - z.B. Accuracy über threshold, col = budget
  - z.B. Accuracy über n_cluster, col = budget
  - Erst Study für anzahl Cluster, dann zu schlechte rauswerfen

# Meeting 25.07
## Punkte
- Adaptive Classifier mit random sampling
- Liste mit Baselines


# Meeting 01.08
## Zeigen:
- Auf river implementierung von detectorn gewechselt
- Überlegt wie ich die Sensitivity Plots gerne machen würde
  - Plot über thresholds und N_cluster und Budget zu viel
  - Was ist mit Datasets?
  - Welches Budget?
  
- Aufgefallen dass VarUncertainty das budget nicht ausnutzt

- ADWIN vielleicht mit anderen statistiken?

## Punkte
- Plot über alle budgets mitteln
  - Anonsten über 3 werte z.b. klein-mittel-groß
  - Quantile filter + Stratgien 