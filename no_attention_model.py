import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

USE_CUDA = torch.cuda.is_available()

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, batch_size=16, n_layers=1):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, batch_first=True, bidirectional=True)

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def init_hidden(self, input):
        hidden = Variable(torch.zeros(self.n_layers*2, input.size(0), self.hidden_size))

        if USE_CUDA:
            hidden = hidden.cuda()

        return hidden

    def forward(self, input, input_masking):
        """
        input : B,T
        input_masking : B,T
        output : B,T,D  B,1,D
        """
        hidden = self.init_hidden(input)
        # B,T,D
        embedded = self.embedding(input)
        # B,T,D
        output, _ = self.gru(embedded, hidden)

        return output

class Decoder(nn.Module):
    def __init__(self, slot_size, intent_size, embedding_size, hidden_size, batch_size=16, n_layers=1, dropout_p=0.1):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.slot_size = slot_size
        self.intent_size = intent_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        # Define the layers
        self.embedding = nn.Embedding(self.slot_size, self.embedding_size)

        self.gru = nn.GRU(self.embedding_size+self.hidden_size, self.hidden_size, self.n_layers, batch_first=True)
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        self.slot_out = nn.Linear(self.hidden_size, self.slot_size)
        self.intent_out = nn.Linear(self.hidden_size, self.intent_size)
    
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def init_hidden(self, input):
        hidden = Variable(torch.zeros(self.n_layers, input.size(0), self.hidden_size))

        if USE_CUDA:
            hidden = hidden.cuda()

        return hidden
    
    def forward(self, input, encoder_outputs, encoder_maskings, training=True):
        """
        input : B,1
        encoder_outputs : B,T,D
        output: B*T,slot_size  B,D
        """

        # B,1 -> B,1,D
        embedded = self.embedding(input)
        hidden = self.init_hidden(input)
        decode = []
        # B,T,D -> T,B,D
        aligns = encoder_outputs.transpose(0, 1)
        # T
        length = encoder_outputs.size(1)

        for i in range(length):
            # B,D -> B,1,D
            aligned = aligns[i].unsqueeze(1)
            _, hidden = self.gru(torch.cat((embedded, aligned), 2), hidden)

            # for Intent Detection
            if i == 0:
                # 1,B,D
                intent_hidden = hidden.clone()
                intent_score = self.intent_out(intent_hidden.squeeze(0))

            # B,slot_size
            score = self.slot_out(hidden.squeeze(0))

            softmaxed = F.log_softmax(score, dim=1)
            decode.append(softmaxed)
            # B
            _, input = torch.max(softmaxed, 1)

            # B,1 -> B,1,D
            embedded = self.embedding(input.unsqueeze(1))

        # B,slot_size*T
        slot_scores = torch.cat(decode, 1)

        # B*T,slot_size  B,D
        return slot_scores.view(input.size(0)*length, -1), intent_score