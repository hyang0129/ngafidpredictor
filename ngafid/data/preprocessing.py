import pandas as pd
from tqdm.autonotebook import tqdm
import numpy as np
from sklearn import preprocessing
import pickle


class PreProcessor():

    def __init__(self, path_to_scaler):
        self.path_to_scaler = path_to_scaler
        self.input_columns = ['volt1', 'volt2', 'amp1', 'amp2', 'FQtyL', 'FQtyR', 'E1 FFlow',
       'E1 OilT', 'E1 OilP', 'E1 RPM', 'E1 CHT1', 'E1 CHT2', 'E1 CHT3',
       'E1 CHT4', 'E1 EGT1', 'E1 EGT2', 'E1 EGT3', 'E1 EGT4']

    def load_scaler(self):
        self.scaler = pickle.load(open(self.path_to_scaler, 'rb'))

    def scale_dataframe(self, df):

        qt = pickle.load(open(self.path_to_scaler, 'rb'))

        df.loc[:, self.input_columns] = qt.transform(df.loc[:, self.input_columns].values)

        return df

    def prepare_data_for_prediction(self, filenames):
        df, sources = extract_engine_data_all(filenames)

        df = self.scale_dataframe(df)
        df = df.dropna()
        df['source'] = df.id.apply(lambda x: sources[x])

        return df, sources

def extract_engine_data(filename, id=None):
    df = pd.read_csv(filename, skiprows=2, low_memory=False)

    df = df.iloc[:, 19:37]
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    isreal = df.iloc[:, :-1].applymap(np.isreal)

    if not isreal.all().all():
        print('non real values in ', filename)

    df['id'] = id

    return df

def extract_engine_data_all(filenames):

    # the sources dataframe references the filename, to avoid having the main dataframe
    # repeat the filename each line

    min_seconds = 1800
    results = []
    sources = {}
    for id, f in tqdm(enumerate(filenames), total=len(filenames)):
        try:
            data = extract_engine_data(f, id)
            if len(data) > min_seconds:
                results.append(data)
                sources[id] = f
            else:
                print('File was shorter than %i seconds, see %s' % (i, f))
        except:
            print('File could not be processed, see %s' % f)

    df = pd.concat(results)
    df.columns = [col.strip() for col in df.columns]



    return df, sources

