from ngafid.data.dataset import get_dataset, prepare_for_training
from ngafid.data.preprocessing import PreProcessor
import glob
import tensorflow as tf
from ngafid.model.autoencoder import vae_conv
from ngafid.model.predictor import get_pred_model
import argparse
import pandas as pd

parser = argparse.ArgumentParser("Hard Negative Miner")
parser.add_argument(
    "--inputdirectory",
    type=str,
    default="example_flights",
    help="location of csv containing data to import",
)

if __name__ == "__main__":
    args = parser.parse_args()

    directory = args.inputdirectory + '/*'

    pp = PreProcessor('/content/ngafidpredictor/scaler.pkl')

    filenames = glob.glob(directory)

    df, sources = pp.prepare_data_for_prediction(filenames)
    df = df.dropna()

    ds = get_dataset(df, has_y=False, relevant_columns=pp.input_columns)
    ds = prepare_for_training(ds, shuffle = False, repeat = False, predict = True, batch_size = 1)

    strategy = tf.distribute.get_strategy()
    vae = vae_conv(shape = (8192, 18), strategy = strategy, verbose = False)
    pred_model = get_pred_model(vae, strategy, verbose = False)

    pred_model.load_weights('/content/ngafidpredictor/predictor_model.h5')

    result = pred_model.predict(ds, verbose = True)
    result


    res_df = pd.DataFrame( {'source': pd.Series(sources), 'prediction' : pd.Series(result[:, 0])})
    res_df['target'] = res_df.source.apply( lambda x : 1 if 'before' in x else 0 )
    res_df.to_csv('results.csv')
    res_df