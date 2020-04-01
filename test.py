import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import pdb

from data_tienboy import *
from model import Encoder, Decoder

def test(config, stuff, some_encoder, some_decoder):
    # word2index, tag2index, intent2index = stuff
    word_enc, tag_enc, intent_enc = stuff

    # index2tag = {v:k for k,v in tag2index.items()}
    # index2intent = {v:k for k,v in intent2index.items()}

    test_data = get_raw("./data/atis-2.dev.w-intent.iob")

    total = 0
    correct = 0

    for i in range(100):
        index = random.choice(range(len(test_data)))
        test_raw = test_data[index][0]
        test_in = prepare_sequence(test_raw, word_enc)
        pdb.set_trace()
        test_mask = Variable(torch.BoolTensor(tuple(map(lambda s: s ==0, test_in)))).view(1,-1)
        start_decode = Variable(torch.LongTensor([[0]])).transpose(1,0)

        output, hidden_c = encoder(test_in.unsqueeze(0),test_mask.unsqueeze(0))
        tag_score, intent_score = decoder(start_decode,hidden_c,output,test_mask)

        v, predicted = torch.max(tag_score,1)
        truth = prepare_sequence(test_data[index][1], tag_enc)
        
        corrects = torch.sum(truth == predicted).item()
        correct += corrects
        total += truth.size(0)

        # print("Input Sentence : ", *test_data[index][0])
        # print("Truth        : ", *truth)
        # print("Prediction : ", *predicted)

    # pdb.set_trace()

    # v,i = torch.max(intent_score,1)
    # print("Truth        : ",test_data[index][2])
    # print("Prediction : ",index2intent[i.data.tolist()[0]])

    print("Total", total, "correct", correct, "accuracy", float(correct/total))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./data/atis-2.train.w-intent.iob', help='path of train data')
    parser.add_argument('--model_dir', type=str, default='./models/', help='path for saving trained models')

    # Model parameters
    parser.add_argument('--max_length', type=int, default=60, help='max sequence length')
    parser.add_argument('--embedding_size', type=int, default=64, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=64, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    config = parser.parse_args()

    _, word_enc, tag_enc, intent_enc = preprocessing('./data/atis-2.train.w-intent.iob',60)

    encoder = Encoder(len(word_enc.classes_), config.embedding_size, config.hidden_size)
    decoder = Decoder(len(tag_enc.classes_), len(intent_enc.classes_), len(tag_enc.classes_)//3, config.hidden_size*2)

    encoder.init_weights()
    decoder.init_weights()

    ee = torch.load(os.path.join(config.model_dir, 'jointnlu-encoder.pkl'))
    dd = torch.load(os.path.join(config.model_dir, 'jointnlu-decoder.pkl'))

    encoder.load_state_dict(ee)
    decoder.load_state_dict(dd)
    encoder.eval()
    decoder.eval()
    
    test(config, (word_enc, tag_enc, intent_enc), encoder, decoder)
