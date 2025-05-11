from queue import Queue
import neurokit2 as nk
import heartpy as hp


def pick_features(all_measures):
    features = ['bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2', 's', 'sd1/sd2',
                'breathingrate', 'vlf', 'lf', 'hf', 'lf/hf', 'p_total', 'vlf_perc', 'lf_perc', 'hf_perc']
    ppg_features = []
    for i in range(len(all_measures['bpm'])):
        row = []
        for j in features:
            value = all_measures[j][i]
            row.append(value)
        ppg_features.append(row)
    return ppg_features


def processPPGSignal(signal):
    ppg_cleaned = nk.ppg_clean(signal, sampling_rate=100)
    wd, measures = hp.process(ppg_cleaned, sample_rate=100, calc_freq=True)
    return pick_features(measures)


class FiveMinAnalysis:
    def __init__(self):
        self.a = []
        self.b = []
        self.aFirst = True

    def addData(self, records):
        if len(self.a) < 15000:
            for r in records:
                self.a.append(r.red_signal)
        else:
            for r in records:
                self.b.append(r.red_signal)

    def analysisThread(self):
        signal = []
        while True:
            # 100Hz 5min -> 30 000 samples
            if len(self.a) >= 15000 and len(self.b) >= 15000:
                if self.aFirst:
                    signal = self.a + self.b
                else:
                    signal = self.b + self.a
                ppg_features = processPPGSignal(signal)
                print(ppg_features)


