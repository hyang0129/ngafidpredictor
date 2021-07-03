import tensorflow as tf
tfk = tf.keras
tfkl = tf.keras.layers


def vae_conv(shape, strategy):
    SHAPE = shape
    with strategy.scope():
        encoded_size = 256

        encoder = tfk.Sequential([
            tfkl.InputLayer(input_shape=SHAPE, name='input'),
            tfkl.Reshape((64, 128, 18)),
            tfkl.Conv1D(128, 7, strides=2, padding='same', activation='relu'),
            tfkl.Conv1D(256, 7, strides=2, padding='same', activation='relu'),
            tfkl.Conv1D(256, 7, strides=2, padding='same', activation='relu'),
            tfkl.Conv1D(256, 7, strides=2, padding='same', activation='relu'),
            tfkl.Conv1D(512, 7, strides=2, padding='same', activation='relu'),
            tfkl.Dense(encoded_size, activation='tanh'),

        ], name='encoder')


        sub_decoder = tfk.Sequential([
            tfkl.InputLayer(input_shape=(4, encoded_size)),
            tfkl.Conv1DTranspose(512, 7, strides=2, padding='same', activation='relu'),
            tfkl.Conv1DTranspose(256, 7, strides=2, padding='same', activation='relu'),
            tfkl.Conv1DTranspose(256, 7, strides=2, padding='same', activation='relu'),
            tfkl.Conv1DTranspose(256, 7, strides=2, padding='same', activation='relu'),
            tfkl.Conv1DTranspose(128, 7, strides=2, padding='same', activation='relu'),
            tfkl.TimeDistributed(tfkl.Dense(SHAPE[-1], activation=None)),

        ], name='sub_decoder')

        decoder = tfk.Sequential([
            tfkl.InputLayer(input_shape=(64, 4, encoded_size)),
            tfkl.TimeDistributed(sub_decoder),
            tfkl.Reshape((8192, 18)),
        ], name='decoder')



        vae = tfk.Model(inputs=encoder.inputs,
                        outputs=decoder(encoder.outputs[0]))

        vae.encoder = encoder

        vae.compile(
            optimizer=tf.optimizers.Adam(learning_rate=1e-4),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
            loss=tfk.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

        )

    vae.summary()

    return vae