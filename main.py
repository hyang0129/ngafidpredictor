from ngafid.data.dataset import get_dataset, prepare_for_training
from ngafid.data.preprocessing import PreProcessor
import glob
import tensorflow as tf
from ngafid.model.autoencoder import vae_conv
from ngafid.model.predictor import get_pred_model
import argparse
import pandas as pd
from loguru import logger

logger.add("{time}.log")

parser = argparse.ArgumentParser("Hard Negative Miner")
parser.add_argument(
    "--inputdirectory",
    type=str,
    default="example_flights",
    help="location of csv containing data to import",
)

if __name__ == "__main__":
    args = parser.parse_args()

    logger.info('Looking for csvs in %s' % args.inputdirectory)

    directory = args.inputdirectory + '/*C172*.csv'

    filenames = glob.glob(directory)

    logger.info('Found %i csvs' % len(filenames))

    pp = PreProcessor('scaler.pkl')

    logger.info('Loaded Preprocessor')

    df, sources = pp.prepare_data_for_prediction(filenames)

    print(len(df))
    df = df.dropna()
    print(len(df))

    ds = get_dataset(df, has_y=False, relevant_columns=pp.input_columns, sources=sources)
    ds = prepare_for_training(ds, shuffle = False, repeat = False, predict = True, batch_size = 1)

    logger.info('Prepared Dataset for Prediction')

    strategy = tf.distribute.get_strategy()
    vae = vae_conv(shape = (8192, 18), strategy = strategy, verbose = False)
    pred_model = get_pred_model(vae, strategy, verbose = False)

    pred_model.load_weights('predictor_model.h5')

    logger.info('Loaded Model')

    result = pred_model.predict(ds, verbose = True)

    logger.info('Predicted on All Data')

    res_df = pd.DataFrame({'source': list(sources.values()), 'prediction': list(result[:, 0])})

    # res_df = pd.DataFrame( {'source': pd.Series(listsources.values()), 'prediction' : pd.Series(result[:, 0])})
    # res_df['target'] = res_df.source.apply( lambda x : 1 if 'before' in x else 0 )
    res_df.to_csv('results.csv')

    logger.info('Results saved to results.csv')
