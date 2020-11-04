import os
import anyconfig
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import dataset
from model import Encoder, Decoder, BahdanauAttention


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    config = anyconfig.load(open("config.yaml", 'rb'))
    BATCH_SIZE = 1
    SEQ_LENGTH = config["trainer"]["sequnce_length"]
    # encoder
    encoder = Encoder(config["trainer"]["gru_units"], BATCH_SIZE)
    sample_hidden = encoder.initialize_hidden_state()
    # attention
    attention_layer = BahdanauAttention(config["trainer"]["attention_units"])
    # decoder
    decoder = Decoder(config["trainer"]["label_length"],
                      config["trainer"]["gru_units"], BATCH_SIZE)
    # reloder
    optimizer = tf.keras.optimizers.Adam()
    checkpoint_dir = config["trainer"]["checkpoint_dir"]
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # test
    inp, targ = dataset.get_json_test("datasets/test/160230555682988.json")
    enc_hidden = encoder.initialize_hidden_state()
    enc_output, enc_hidden = encoder(inp, enc_hidden)
    dec_hidden = enc_hidden
    for t in range(100):
        dec_input = tf.expand_dims(inp[:, t], 1)
        predictions, dec_hidden, _ = decoder(
            dec_input, dec_hidden, enc_output)
        logits = tf.nn.softmax(predictions)
        # 是否正确
        correct = tf.equal(tf.argmax(targ[:, t], 1), tf.argmax(logits, 1))
        print(t, tf.argmax(logits, 1), correct)

