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
- [ ] Clustream auf cluster daten fitten
- [ ] Clustering betrachten
- [ ] CluStream Classifier refitten wenn cluster gelöscht

## 02.06
- [X] Batch CluStream Classifier
- [ ] CluStream Cluster fitting
- [ ] Clusterverteilung Experimente

## Idee
- Clustering abhängig von Classifier machen
  - z.B. Cluster löschen wenn accuracy auf samples die in das Cluster fallen schlechter werden
