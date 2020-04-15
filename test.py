import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import pdb
import tqdm

from data import *
# from gru_model import Encoder, Decoder
from no_attention_model import Encoder, Decoder
from constants import *

def test(config, stuff, encoder, decoder, test_data, save=False):
    word2index, tag2index, intent2index = stuff

    total_tag = 0
    correct_tag = 0
    total_intent = 0
    correct_intent = 0

    truths = []
    predicteds = []

    for index in tqdm.tqdm(range(len(test_data))):
        test_item = test_data[index]
        test_raw, tag_raw, intent_raw = test_item
        test_in = prepare_sequence(test_raw,word2index)
        test_mask = Variable(torch.BoolTensor(tuple(map(lambda s: s ==0, test_in)))).view(1,-1)
        start_decode = Variable(torch.LongTensor([[0]*1])).transpose(1,0)

        output, hidden_c = encoder(test_in.unsqueeze(0),test_mask.unsqueeze(0))
        tag_score, intent_score = decoder(start_decode,hidden_c,output,test_mask)

        _, predicted = torch.max(tag_score, dim=1)
        truth = prepare_sequence(tag_raw, tag2index)
        
        truths.append(" ".join([str(t.item()) for t in truth]) + '\n')
        predicteds.append(" ".join([str(t.item()) for t in predicted]) + '\n')

        corrects = torch.sum(truth == predicted).item()
        correct_tag += corrects
        total_tag += truth.size(0)

        _, predicted_intent = torch.max(intent_score, dim=1)
        # pdb.set_trace()

        intent_raw = intent_raw if intent_raw in intent2index else UNK
        true_intent = intent2index[intent_raw]
        if true_intent == predicted_intent.item():
            correct_intent += 1

        total_intent += 1

    if save:
        with open('./results/pred.txt', 'w') as f1:
            f1.writelines(predicteds)

        with open('./results/truth.txt', 'w') as f2:
            f2.writelines(truths)

    print("N =", len(test_data))
    print("Total tag", total_tag, "correct", correct_tag, "accuracy", float(correct_tag/total_tag))
    print("Total intent", total_intent, "correct", correct_intent, "accuracy", float(correct_intent/total_intent))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default=Train_Data_Dir, help='path of training data, to load correct word dictionaries')
    parser.add_argument('--model_dir', type=str, default='./models/', help='where to load trained weights')

    # Model parameters
    parser.add_argument('--max_length', type=int, default=60, help='max sequence length')
    parser.add_argument('--embedding_size', type=int, default=64, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=64, help='dimension of lstm hidden states')

    config = parser.parse_args()

    _, word2index,tag2index,intent2index = preprocessing(config.train_dir, config.max_length)

    encoder = Encoder(len(word2index), config.embedding_size, config.hidden_size)
    decoder = Decoder(len(tag2index), len(intent2index), len(tag2index)//3, config.hidden_size*2)

    ee = torch.load(os.path.join(config.model_dir, Model_Encoder_Dir))
    dd = torch.load(os.path.join(config.model_dir, Model_Decoder_Dir))

    encoder.load_state_dict(ee)
    decoder.load_state_dict(dd)

    test_data = []
    [test_data.extend(get_raw(t)) for t in Test_Data_Dirs]

    test(config, (word2index, tag2index, intent2index), encoder, decoder, test_data)