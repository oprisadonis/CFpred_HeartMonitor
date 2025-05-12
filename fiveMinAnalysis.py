import neurokit2 as nk
import heartpy as hp
from data_models import PPGFeatures


def processPPGSignal(signal):
    ppg_cleaned = nk.ppg_clean(signal, sampling_rate=100)
    wd, measures = hp.process(ppg_cleaned, sample_rate=100, calc_freq=True)
    return measures


class FiveMinAnalysis:
    def __init__(self, user_id, db):
        self.user_id = user_id
        self.a = []
        self.b = []
        self.aFirst = True
        self.firstA = 0
        self.firstB = 0
        self.lastA = 0
        self.lastB = 0
        self.finishID = 0
        self.db = db

    def addData(self, records):
        print(f'Add data a:{len(self.a)}   b:{len(self.b)}\n')
        if len(self.a) == 0:
            self.firstA = records[0].timestamp
        if len(self.b) == 0:
            self.firstB = records[0].timestamp
        if len(self.a) < 15000:
            for r in records:
                self.a.append(r.red_signal)
            self.lastA = records[-1].timestamp
        else:
            for r in records:
                self.b.append(r.red_signal)
            self.lastB = records[-1].timestamp

        # 100Hz 5min -> 30 000 samples
        if len(self.a) >= 15000 and len(self.b) >= 15000:
            self.PPGAnalysis()

    def PPGAnalysis(self):
        print("PPG analysis started\n")
        if self.aFirst:
            signal = self.a + self.b
            self.aFirst = False
            self.a = []
        else:
            signal = self.b + self.a
            self.aFirst = True
            self.b = []
        ppg_features = processPPGSignal(signal)
        print(ppg_features)

        # inverse logic because self.aFirst was reset
        if not self.aFirst:
            self.savePPGFeatures(ppg_features, self.firstA, self.lastB)
        else:
            self.savePPGFeatures(ppg_features, self.firstB, self.lastA)
        print('Saved features')


    def savePPGFeatures(self, ppg_features, first, last):
        features = PPGFeatures(
            user_id=self.user_id,
            start_time=first,
            finish_time=last,
            bpm=ppg_features['bpm'],
            ibi=ppg_features['ibi'],
            sdnn=ppg_features['sdnn'],
            sdsd=ppg_features['sdsd'],
            rmssd=ppg_features['rmssd'],
            pnn20=ppg_features['pnn20'],
            pnn50=ppg_features['pnn50'],
            hr_mad=ppg_features['hr_mad'],
            sd1=ppg_features['sd1'],
            sd2=ppg_features['sd2'],
            s=ppg_features['s'],
            sd1_sd2=ppg_features['sd1/sd2'],
            breathingrate=ppg_features['breathingrate'],
            vlf=ppg_features['vlf'],
            lf=ppg_features['lf'],
            hf=ppg_features['hf'],
            lf_hf=ppg_features['lf/hf'],
            p_total=ppg_features['p_total'],
            vlf_perc=ppg_features['vlf_perc'],
            lf_perc=ppg_features['lf_perc'],
            hf_perc=ppg_features['hf_perc']
        )
        self.db.session.add(features)
        self.db.session.commit()
        print("Saved PPG features")
