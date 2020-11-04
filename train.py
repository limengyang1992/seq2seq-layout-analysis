import os
import time
import glob
import anyconfig
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from dataset import gen_seq2seq_model as gen
from model import Encoder, Decoder, BahdanauAttention

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


config = anyconfig.load(open("config.yaml", 'rb'))
BATCH_SIZE = config["trainer"]["batch_size"]
SEQ_LENGTH = config["trainer"]["sequnce_length"]

# datasets
paths = glob.glob(config["datasets"]["train_path"]+"/*.json")
trainloader = gen(paths, batch_size=BATCH_SIZE, seq_length=SEQ_LENGTH)

# encoder
encoder = Encoder(config["trainer"]["gru_units"], BATCH_SIZE)
sample_hidden = encoder.initialize_hidden_state()
# attention
attention_layer = BahdanauAttention(config["trainer"]["attention_units"])
# decoder
decoder = Decoder(config["trainer"]["label_length"],
                  config["trainer"]["gru_units"], BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


checkpoint_dir = config["trainer"]["checkpoint_dir"]
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


# @tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        for t in range(0, targ.shape[1]):
            dec_input = tf.expand_dims(inp[:, t], 1)
            predictions, dec_hidden, _ = decoder(
                dec_input, dec_hidden, enc_output)
            logits = tf.nn.softmax(predictions)
            loss += tf.keras.losses.categorical_crossentropy(
                y_true=targ[:, t], y_pred=logits)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


for epoch in range(BATCH_SIZE):
    start = time.time()
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0
    for batch in range(100):
        inp, targ = next(trainloader)
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss
        if batch % 10 == 0:
            logger.info('Epoch {} Batch {} Loss {}'.format(epoch + 1,
                                                           batch,
                                                           np.mean(batch_loss)))

    checkpoint.save(file_prefix=checkpoint_prefix)
    logger.info('Epoch {} Loss {}'.format(epoch + 1, total_loss))
    logger.info('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
