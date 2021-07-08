
import numpy as np
import random
import tensorflow as tf
from tqdm.autonotebook import tqdm


def get_dataset(df, has_y = True, relevant_columns = None, sources = None):
    # converts a dataframe into a dataset

    ids = df.id.unique()

    print(ids)

    sensor_datas = []
    afters = []

    for id in tqdm(ids):
        sensor_data = df[df.id == id].iloc[-8192:].loc[:, relevant_columns].values

        sensor_data = np.pad(sensor_data, [[0, 8192- len(sensor_data)], [0,0]])

        if has_y:
            after = df[df.id == id]['after'].iloc[0]
        else:
            after = 0

        sensor_datas.append(sensor_data)
        afters.append(after)

    sensor_datas = np.stack(sensor_datas)
    afters = np.stack(afters)

    ds = tf.data.Dataset.from_tensor_slices( (sensor_datas, afters))

    return ds


def prepare_for_training(ds, shuffle = False, repeat = False, predict = True, batch_size = 1):
    # final transforms before dataset is usable

    if not predict:
        ds = ds.map(lambda x, y : (x, x) )
    else:
        # ds = ds.map(lambda x, y : (x, tf.one_hot(y, 4)) )
        ds = ds.map(lambda x, y : (x, y) )

    ds = ds.shuffle(256) if shuffle else ds
    ds = ds.repeat() if repeat else ds
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds