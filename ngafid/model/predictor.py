import tensorflow as tf
tfk = tf.keras
tfkl = tf.keras.layers

def get_pred_model(vae, strategy, verbose = True):
    with strategy.scope():
        encoded_size = 256
        predictor = tfk.Sequential([
            tfkl.InputLayer(input_shape=(64, 4, encoded_size)),
            tfkl.Reshape((256, encoded_size)),
            tfkl.Bidirectional(tfkl.GRU(128, return_sequences=True)),
            tfkl.Bidirectional(tfkl.GRU(128, return_sequences=True)),
            tfkl.Bidirectional(tfkl.GRU(128, return_sequences=False)),

            tfkl.Dropout(0.5),
            tfkl.Dense(1, activation='sigmoid'),

        ], name='predictor')

        pred_model = tfk.Model(inputs=vae.encoder.inputs, outputs=predictor(vae.encoder.outputs[0]))

        pred_model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=1e-4),
            metrics=['accuracy', tf.keras.metrics.AUC()],
            loss=[tfk.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)])

        if verbose: pred_model.summary()

    return pred_model