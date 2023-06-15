# Daily To-do
## General Todos:
- Testscript anpassen
  - ~~Args mit append~~
  - Approaches struktur verbessern (Erweiterbarer)
- Refactor to Dataset class
  - ~~Logging von Datensatz~~
  - mehrere Datensätze parallel
  - Farben in plots einheitlich
  - Datensätze übersicht in Overleaf schreiben
  - Anzahl Cluster experimente
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

# 12.06
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

# Idee
- Clusterensemble
  - Bei change in Cluster muss nur der eine Classifier refittet werden

# Mögliche Probleme:
  - Merges verändern statistiken
    - Radi steigt an und sinkt dann wieder
    - N Springt
    - Entropy womöglich veränderunt
  - Entropy bei binären Classification
    - Oszilliert stark zwischen 0-1 mit Merges
    - Relative Entropy abhängig von Anzahl an Samples?