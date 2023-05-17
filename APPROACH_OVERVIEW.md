# Overview

## Training/Evaluation
Im Trainingsskript (z.B. stream/clustering/tests/parallel_test.py) werden der Stream, die Ansätze, Hyperparameter initialisiert und über den 

stream/clustering/tests/stream_runner.py

der stream evaluiert:

1. Initialisierung von pre trainings daten und sliding training_window
2. `clf.fit(pre_trainings_daten)` pretraining
3. Iteration über Stream (x_t, y_t)
   1. Prediction für `x_t`
   2. `al_label = query(x_t)`Labelstrategy Entscheidung
   3. `(x_t, al_label)` zu Trainingswindow hinzufügen
   4. Classifier fitten
      5. Inkrementell: das aktuelle Tupel
      6. Batch: Auf Trainingswindow

## Ansätze
Ansätze bestehen generell aus:
1. Query Strategy
2. Classifier

und werde im jeweiligen Trainingsscript in einem dictionary als tupel
`{Ansatzname: (Query_strategy, Classifier)}`
definiert

Bei den Ansätzen verwende ich als Query strategie für alle Ansätze: StreamProbabilisticAL (OPAL)

Für die Query Entscheidung(Schritt `3.ii.`) nimmt OPAL den Classifier als Eingabe um die Utilities für den `BalancedIncrementalQuantileFilter` zu berechnen. 

`utility = query_strategy.query(X, Classifier, ...)`

Die Ansätze unterscheiden sich vor Allem in der Berechnung dieser Utility und der art des Trainings

### 1. TraditionalBatch
Classifier: ParzenWindwowClassifier

**Query()**
- Eingabe:
  - x_t
- `predict_frequency` methode von PWC wird aufgerufen
  - Kernel berechnung mit in Classifier trainierten Instanzen
  - One-Hot-Encoding der Labels der Instanzen
  - `Utility = KernelDensity * One-Hot-Encoding`
  - (Eigentlich einfach die Wahrscheinlichkeitspredictions des Classifiers für die Klassen)

**Training()**
- `clf.fit(trainingswindow)`
- Classifier wird auf Trainingswindow trainiert (Batch)

### 2. Traditional Incremental
Classifier: NaiveBayes oder HoeffdingTree

**Query()**
- Eingabe:
  - x_t
  - Trainingswindow
- NaiveBayes macht `probability_prediction`  für x_t 
- Neuer ParzenWindowCLassifier wird auf dem sliding Trainingswindow trainiert für FrequencyPrediction
- `Utility = probability * frequency`
- 
**Training()**
- Classifier wird inkrementell trainiert falls es ein label gibt
- `clf.partial_fit((x_t, al_label))`
### 3. ClusteringBased
Classifier: CluStreamClassifier
- Eigentlich nur Wrapper klasse um Clustering und Classifier zu verbinden
- hat intern eigenen NaiveBayes oder HoeffdingTree classifier `clf`

**Query()**
- Eingabe:
  - x_t
- `predict_frequency` methode von CluStreamClassifier wird aufgerufen
  - `clf` macht `probability_prediction`  für x_t 
  - Neuer ParzenWindowCLassifier wird auf in Cluster gespeicherten, gelabelten Daten trainiert für FrequencyPrediction
- `Utility = probability * frequency`
- 
**Training()**
- `partial_fit(x_t, al_label)` wird aufgerufen auch wenn kein Label gibt
  - Clustering wird mit Sample trainiert
  - Interner Classifier wird nur inkrementell trainiert wenn es label gibt


