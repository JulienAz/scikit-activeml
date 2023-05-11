REP = "rep"
ROUND = "round"
APPROACH = "Approach"
CLASSIFIER = "Classifier"
TIME = "time"
BUDGET = "Budget"
ACCURACY = "Accuracy"
DECISION = "query_decision"
DRIFT_WARNING = "drift_warning"
DRIFT_CHANGE = "drift_change"
DECISION_avg = "decision_avg"
TRAINING_WINDOW = "training_window"
PREDICTION_DEVIATION = "prediction_deviation"

# Add column names to list
all_ids = [
    REP,
    ROUND,
    APPROACH,
    CLASSIFIER,
    TIME,
    BUDGET,
    ACCURACY,
    DECISION,
    DRIFT_WARNING,
    DRIFT_CHANGE,
    DECISION_avg,
    TRAINING_WINDOW,
    PREDICTION_DEVIATION
]

class ExperimentLogger:

    # Nothing to do here
    def __init__(self):
        self._columns = all_ids
        self._data = []
        self._current_row = [np.nan for _ in range(len(self._columns))]
        self._column_indices = {key: i for (i, key) in enumerate(self._columns)}

    # Add own functions for tracking different metrics
    def track_rep(self, value: int):
        self._track_value(value, REP)

    def track_approach(self, value: str):
        self._track_value(value, APPROACH)

    def track_classifier(self, value: str):
        self._track_value(value, CLASSIFIER)

    def track_round(self, value: int):
        self._track_value(value, ROUND)

    def track_budget(self, value: int):
        self._track_value(value, BUDGET)

    def track_accuracy(self, value: float):
        self._track_value(value, ACCURACY)

    def track_decision(self, value: bool):
        self._track_value(value, DECISION)

    def track_decision_avg(self, value: float):
        self._track_value(value, DECISION_avg)

    def track_drift_warning(self, value: bool):
        self._track_value(value, DRIFT_WARNING)

    def track_drift_change(self, value: bool):
        self._track_value(value, DRIFT_CHANGE)

    def finalize_round(self):
        self._data.append(self._current_row)
        self._current_row = [np.nan for _ in range(len(self._columns))]

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._data, columns=self._columns)
        self._reset()
        return df

    def track_window_type(self, value: str):
        self._track_value(value, TRAINING_WINDOW)

    def track_prediction_deviation(self, value: float):
        self._track_value(value, PREDICTION_DEVIATION)

    # Noting to do here
    def track_time(self):
        current_time = time.perf_counter_ns()
        self._track_value(current_time, "time")

    def _track_value(self, newval, id):
        self._current_row[self._index_of(id)] = newval

    def _index_of(self, id):
        return self._column_indices[id]

    def _reset(self):
        self._current_row = [np.nan for _ in range(len(self._columns))]
        self._data = []

    @property
    def data(self):
        return self._data
