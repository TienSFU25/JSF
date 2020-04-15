import argparse
import numpy as np
import pdb
import os
import torch

from data import get_snips, get_snips_raw
from constants import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='./data/snips/train', help='path of training data, to load correct word dictionaries')
    parser.add_argument('--test_dir', type=str, default='./data/snips/test', help='path of test data')
    parser.add_argument('--model_dir', type=str, default='./models/snips/', help='where to load trained weights')
    parser.add_argument('--attention', type=int, default=1, help='Use attention or no?')

    # Model parameters
    parser.add_argument('--max_length', type=int, default=60, help='max sequence length')
    parser.add_argument('--embedding_size', type=int, default=64, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=64, help='dimension of lstm hidden states')

    config = parser.parse_args()

    _, word2index,tag2index,intent2index = get_snips(config.train_dir, config.max_length)

    if config.attention:
        from gru_model import Encoder, Decoder
        from test import test
        encoder_weights_path = Model_Encoder_Dir
        decoder_weights_path = Model_Decoder_Dir
    else:
        from no_attention_model import Encoder, Decoder
        from test_no_attention import test
        encoder_weights_path = Model_Encoder_NA_Dir
        decoder_weights_path = Model_Decoder_NA_Dir

    encoder = Encoder(len(word2index), config.embedding_size, config.hidden_size)
    decoder = Decoder(len(tag2index), len(intent2index), len(tag2index)//3, config.hidden_size*2)

    ee = torch.load(os.path.join(config.model_dir, encoder_weights_path))
    dd = torch.load(os.path.join(config.model_dir, decoder_weights_path))

    encoder.load_state_dict(ee)
    decoder.load_state_dict(dd)

    test_data = get_snips_raw(config.test_dir, length=60, padding=False)[:3]

    # basically transpose, so we don't have to rewrite test code
    rearranged = []
    for i, _ in enumerate(test_data[0]):
        rearranged.append((test_data[0][i], test_data[1][i], test_data[2][i]))

    test(config, (word2index, tag2index, intent2index), encoder, decoder, rearranged)