import argparse
import numpy as np
import pdb

from data import get_snips

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./data/snips/train', help='path of train data')
    parser.add_argument('--model_dir', type=str, default='./models/snips', help='path for saving trained models')
    parser.add_argument('--attention', type=bool, default=True, help='Use attention or no?')

    # Model parameters
    parser.add_argument('--max_length', type=int, default=60, help='max sequence length')
    parser.add_argument('--embedding_size', type=int, default=64, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=64, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    config = parser.parse_args()

    train_data, word2index, tag2index, intent2index = get_snips(config.file_path, config.max_length)
        
    print("Training on", len(train_data), "samples")

    if config.attention:
        from gru_model import Encoder, Decoder
        from train import train
    else:
        from no_attention_model import Encoder, Decoder
        from train_no_attention import train

    encoder = Encoder(len(word2index), config.embedding_size, config.hidden_size)
    decoder = Decoder(len(tag2index), len(intent2index), len(tag2index)//3, config.hidden_size*2)

    encoder.init_weights()
    decoder.init_weights()

    train(config, train_data, encoder, decoder)