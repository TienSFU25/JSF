import torch
import pickle
import random
import os
import pdb
from torch.autograd import Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

USE_CUDA = torch.cuda.is_available()
flatten = lambda l: [item for sublist in l for item in sublist]

EOS = '<EOS>'
UNK = '<UNK>'
PAD = '<PAD>'
SOS = '<SOS>'

def pad(some_sequence, length=60):
    if len(some_sequence) < length:
        while len(some_sequence) < length:
            some_sequence.append(PAD)
    else:
        some_sequence = some_sequence[:length]
        some_sequence[-1] = EOS

    return some_sequence

def prepare_sequence(seq, to_ix):
    idxs = list(map(lambda w: to_ix[w] if w in to_ix.keys() else to_ix[UNK], seq))
    
    tensor = Variable(torch.LongTensor(idxs))

    # tensor = tensor.view(1, -1)

    return tensor

def labeler_to_ix(le):
    return dict(zip(le.classes_, le.transform(le.classes_)))  

def get_raw(file_path):
    train = open(file_path, "r").readlines()
    train = [t[:-1] for t in train]
    train = [[t.split("\t")[0].split(" "), t.split("\t")[1].split(" ")[:-1], t.split("\t")[1].split(" ")[-1]] for t in train]
    train = [[t[0][1:-1], t[1][1:], t[2]] for t in train]

    return train

def preprocessing(file_path, length):
    """
    atis-2.train.w-intent.iob
    """
    processed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/")
    print("processed_data_path : %s" % processed_path)

    if os.path.exists(os.path.join(processed_path, "processed_train_data.pkl")):
        train_data, word2index, tag2index, intent2index = pickle.load(open(os.path.join(processed_path,
                                                                                        "processed_train_data.pkl"),
                                                                           "rb"))
        return train_data, word2index, tag2index, intent2index
                                  
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    
    try:
        train = open(file_path, "r").readlines()
        print("Successfully load data. # of set : %d " % len(train))
    except:
        print("No such file!")
        return None, None, None, None

    try:
        vocab = set()
        slot_tag = set()
        intent_tag = set()

        padded_sentences = []
        padded_slots = []
        all_intents = []

        for i in range(len(train)):
            # BOS what's restriction ap68 EOS	O O O B-restriction_code O \n
            bunch_of_crap = train[i]

            # what's restriction ap68 EOS	O O O B-restriction_code O
            line = bunch_of_crap[:-1]

            pdb.set_trace()

            tab_split = line.split("\t")
            words_in_sentence = tab_split[0].split(" ")[1:-1]
            slots_in_sentence = tab_split[1].split(" ")[:-1][1:]
            intent = tab_split[1].split(" ")[-1]

            assert len(words_in_sentence) == len(slots_in_sentence)
            assert len(intent) > 0

            vocab.update(words_in_sentence)
            slot_tag.update(slots_in_sentence)

            intent_tag.update([intent])

            if len(words_in_sentence) < length:
                words_in_sentence.append(EOS)
            
            pdb.set_trace()
            padded_sentences.append(pad(words_in_sentence))
            padded_slots.append(pad(slots_in_sentence))
            all_intents.append(intent)

        print("# of vocab : {vocab}, # of slot_tag : {slot_tag}, # of intent_tag : {intent_tag}"
              .format(vocab=len(vocab), slot_tag=len(slot_tag), intent_tag=len(intent_tag)))

        vocab_labenc = LabelEncoder()
        slot_labenc = LabelEncoder()
        intent_labenc = LabelEncoder()
        vocab_labenc.fit([*vocab, PAD, UNK, SOS, EOS])
        slot_labenc.fit([*slot_tag, PAD, UNK])
        intent_labenc.fit([*intent_tag, UNK])

        train = list(zip(padded_sentences, padded_slots, all_intents))
                
        train_data = []
        word2index = labeler_to_ix(vocab_labenc)
        tag2index = labeler_to_ix(slot_labenc)
        intent2index = labeler_to_ix(intent_labenc)

        for tr in train:
            temp = prepare_sequence(tr[0], word2index)
            temp = temp.view(1, -1)

            temp2 = prepare_sequence(tr[1], tag2index)
            temp2 = temp2.view(1, -1)

            as_idx = intent_labenc.transform([tr[2]])
            temp3 = Variable(torch.LongTensor(as_idx))
            # pdb.set_trace()

            train_data.append((temp, temp2, temp3))

        pickle.dump((train_data,word2index,tag2index,intent2index),open(os.path.join(processed_path, "processed_train_data.pkl"), "wb"))
        # pickle
        print("Preprocessing complete!")
        return train_data, word2index, tag2index, intent2index
        # return train_data, vocab_labenc, slot_labenc, intent_labenc

    except Exception as e:
        print(e)
        return None, None, None, None
              
def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex+batch_size
        sindex = temp
        
        yield batch

def get_snips_raw(file_path, length=60, padding=True):
    seq_in = open(file_path + "/seq.in", "r").readlines()
    seq_out = open(file_path + "/seq.out", "r").readlines()
    intents = open(file_path + "/label", "r").readlines()

    p_fn = pad if padding else lambda p: p

    assert(len(seq_out) == len(seq_in))

    vocab = set()
    slot_tag = set()
    intent_tag = set()

    padded_sentences = []
    padded_slots = []
    all_intents = []

    for i, _ in enumerate(seq_in):
        sentence = seq_in[i][:-1].rstrip(' ')
        slots = seq_out[i][:-1].rstrip(' ')
        intent = intents[i][:-1].rstrip(' ')

        # get rid of annoying white spaces
        words_in_sentence = [word for word in sentence.split(' ') if len(word) > 0 and word != ' ']
        slots_in_sentence = [slot for slot in slots.split(' ') if len(slot) > 0 and slot != ' ']

        if len(words_in_sentence) != len(slots_in_sentence) or len(intent) == 0:
            pdb.set_trace()

        assert len(words_in_sentence) == len(slots_in_sentence)
        assert len(intent) > 0

        vocab.update(words_in_sentence)
        slot_tag.update(slots_in_sentence)

        intent_tag.update([intent])

        if len(words_in_sentence) < length and padding:
            words_in_sentence.append(EOS)
        
        padded_sentences.append(p_fn(words_in_sentence))
        padded_slots.append(p_fn(slots_in_sentence))
        all_intents.append(intent)

    return padded_sentences, padded_slots, all_intents, vocab, slot_tag, intent_tag

def get_snips(file_path, length):
    processed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/")
    path_to_save = os.path.join(processed_path, "snips_train_data.pkl")

    if os.path.exists(path_to_save):
        train_data, word2index, tag2index, intent2index = pickle.load(open(path_to_save, "rb"))
        return train_data, word2index, tag2index, intent2index
                                  
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    try:
        vocab = set()
        slot_tag = set()
        intent_tag = set()

        padded_sentences = []
        padded_slots = []
        all_intents = []

        padded_sentences, padded_slots, all_intents, vocab, slot_tag, intent_tag = get_snips_raw(file_path, length)

        print("# of vocab : {vocab}, # of slot_tag : {slot_tag}, # of intent_tag : {intent_tag}"
              .format(vocab=len(vocab), slot_tag=len(slot_tag), intent_tag=len(intent_tag)))

        vocab_labenc = LabelEncoder()
        slot_labenc = LabelEncoder()
        intent_labenc = LabelEncoder()
        vocab_labenc.fit([*vocab, PAD, UNK, SOS, EOS])
        slot_labenc.fit([*slot_tag, PAD, UNK])
        intent_labenc.fit([*intent_tag, UNK])

        train = list(zip(padded_sentences, padded_slots, all_intents))
                
        train_data = []
        word2index = labeler_to_ix(vocab_labenc)
        tag2index = labeler_to_ix(slot_labenc)
        intent2index = labeler_to_ix(intent_labenc)

        for tr in train:
            temp = prepare_sequence(tr[0], word2index)
            temp = temp.view(1, -1)

            temp2 = prepare_sequence(tr[1], tag2index)
            temp2 = temp2.view(1, -1)

            as_idx = intent_labenc.transform([tr[2]])
            temp3 = Variable(torch.LongTensor(as_idx))
            # pdb.set_trace()

            train_data.append((temp, temp2, temp3))

        pickle.dump((train_data,word2index,tag2index,intent2index),open(path_to_save, "wb"))
        print("Preprocessing complete!")
        return train_data, word2index, tag2index, intent2index

    except Exception as e:
        print(e)
        return None, None, None, None              