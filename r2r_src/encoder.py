import torch
import torch.nn as nn
from torch.autograd import Variable
import transformers as ppb
import numpy as np
import en_core_web_lg
from param import args
from transformers import LxmertTokenizer, LxmertModel
from utils import glove_embedding

class CustomRNN(nn.Module):
    """
    A module that runs multiple steps of RNN cell
    With this module, you can use mask for variable-length input
    """
    def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0, **kwargs):
        super(CustomRNN, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = cell_class(input_size=layer_input_size,
                              hidden_size=hidden_size,
                              **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, mask, hx):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            h_next, c_next = cell(input_[time], hx=hx)
            mask_ = mask[time].unsqueeze(1).expand_as(h_next)
            h_next = h_next*mask_ + hx[0]*(1 - mask_)
            c_next = c_next*mask_ + hx[1]*(1 - mask_)
            hx_next = (h_next, c_next)
            output.append(h_next)
            hx = hx_next
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, mask, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
            mask = mask.transpose(0, 1)
        max_time, batch_size, _ = input_.size()

        if hx is None:
            hx = input_.new(batch_size, self.hidden_size).zero_()
            hx = (hx, hx)
        h_n = []
        c_n = []
        layer_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            layer_output, (layer_h_n, layer_c_n) = CustomRNN._forward_rnn(
                cell=cell, input_=input_, mask=mask, hx=hx)
            input_ = self.dropout_layer(layer_output)
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)
        output = layer_output
        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)
        return output, (h_n, c_n)


class EncoderBERT(nn.Module):
    """ Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. """

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                            dropout_ratio, bidirectional=False, num_layers=1):
        super(EncoderBERT, self).__init__()
        model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.bert_model = model_class.from_pretrained(pretrained_weights)
        self.embedding_size = embedding_size
        hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.bidirectional = bidirectional          
        self.hidden_size = hidden_size
        self.rnn_kwargs = {
            'cell_class': nn.LSTMCell,
            'input_size': embedding_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'batch_first': True,
            'dropout': 0,
        }
        self.rnn = CustomRNN(**self.rnn_kwargs)
        self.sf = SoftAttention(dimension=embedding_size)
        self.glove_indexer, self.glove_vector = glove_embedding("/VL/space/zhan1624/obj-vln/r2r_src/MAF/data/glove/glove.6B.300d.txt")

    def create_mask(self, batchsize, max_length, length):
        """Given the length create a mask given a padded tensor"""
        tensor_mask = torch.zeros(batchsize, max_length)
        for idx, row in enumerate(tensor_mask):
            row[:length[idx]] = 1
        return tensor_mask.to(self.device)

    def flip(self, x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]
    
    def bert_sentence_embedding(self, inputs, seq_len, new_size=False, landmark_input=None):
        tokenized_dict = self.bert_tokenizer.batch_encode_plus(inputs, add_special_tokens=True, return_attention_mask=True, return_tensors='pt', pad_to_max_length=True, max_length=80)
        split_index_list = []
        landmark_id_list = []
        for count_id1, each_token_id in enumerate(tokenized_dict['input_ids']):
            tmp_landmark_id_list = {}
            if landmark_input:
                if landmark_input[count_id1]:
                    for count_id2, each_landmark_token in enumerate(self.bert_tokenizer.convert_ids_to_tokens(each_token_id)):
                        if each_landmark_token in landmark_input[count_id1]:
                            tmp_landmark_id_list[count_id2] = self.glove_vector[self.glove_indexer.add_and_get_index(each_landmark_token)]
            landmark_id_list.append(tmp_landmark_id_list)
            tmp_split_index = list(np.where(each_token_id.numpy()==24110)[0])
            split_index_list.append(tmp_split_index)
        padded = tokenized_dict['input_ids'].to(self.bert_model.device)
        attention_mask = tokenized_dict['attention_mask'].to(self.bert_model.device)
        with torch.no_grad():
            if new_size:
                self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))
            last_hidden_states = self.bert_model(padded, attention_mask=attention_mask)
        if landmark_input:
            return last_hidden_states[0], attention_mask, split_index_list, landmark_id_list
        else:
            return last_hidden_states[0], attention_mask, split_index_list
    
    def original_bert(self, inputs, seq_len):
        tokenized_dict = self.bert_tokenizer.batch_encode_plus(inputs, add_special_tokens=True, return_attention_mask=True, return_tensors='pt', pad_to_max_length=True, max_length=seq_len)
        split_index_list = []
        padded = tokenized_dict['input_ids'].to(self.bert_model.device)
        attention_mask = tokenized_dict['attention_mask'].to(self.bert_model.device,dtype=torch.uint8)
        with torch.no_grad():
            last_hidden_states = self.bert_model(padded, attention_mask=attention_mask)
        return last_hidden_states[0], attention_mask
    
    def init_state(self, batch_size, max_config_num, config_mask):
        """ Initial state of model
        a_0: batch x max_config_num
        a_0: batch x 2
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        """
        a0 = Variable(torch.zeros( 
            batch_size, 
            max_config_num, 
            device=self.bert_model.device
            ), requires_grad=False)
        # a0[:,:2] = config_mask[:,:2]/config_mask[:,:2].sum(dim=1, keepdim=True)


        # tmp_a0 = config_mask[:,:2].sum(dim=1, keepdim=True)
        # a0[:,:2] = torch.where(tmp_a0 == 1, torch.tensor([1.,0.], device=self.bert_model.device), torch.tensor([0.50,0.50], device=self.bert_model.device))
 
        a0[:,0] = 1
        r0 = Variable(torch.zeros(
            batch_size, 
            2, 
            device=self.bert_model.device
            ), requires_grad=False)
        r0[:,0] = 1
        
        return a0, r0

    def forward(self, inputs, seq_len=0 , new_size=False, landmark_input=None):
        """
        Expects input vocab indices as (batch, seq_len). Also requires a list of lengths for dynamic batching.
        """
        
        if args.configuration:
            if landmark_input:
                original_embeds, embeds_mask, split_index_list, landmark_id_list = self.bert_sentence_embedding(inputs, seq_len, new_size, landmark_input)
            else:
                original_embeds, embeds_mask, split_index_list = self.bert_sentence_embedding(inputs, seq_len, new_size, landmark_input)
        else:
            original_embeds, embeds_mask = self.original_bert(inputs, seq_len)
        embeds = self.drop(original_embeds)
       
        if self.bidirectional:
            output_1, (ht_1, ct_1) = self.rnn(embeds, mask=embeds_mask)
            output_2, (ht_2, ct_2) = self.rnn(self.flip(embeds, 1), mask=self.flip(embeds_mask, 1))
            output = torch.cat((output_1, self.flip(output_2, 0)), 2)
            ht = torch.cat((ht_1, ht_2), 2)
            ct = torch.cat((ct_1, ct_2), 2)
        else:
            output, (ht, ct) = self.rnn(embeds, mask=embeds_mask)

        if args.configuration:
            if landmark_input:
                return  output.transpose(0, 1), ht.squeeze(dim=0), ct.squeeze(dim=0), embeds_mask, split_index_list, original_embeds, landmark_id_list
            else:
                return  output.transpose(0, 1), ht.squeeze(dim=0), ct.squeeze(dim=0), embeds_mask, split_index_list



class SoftAttention(nn.Module):
    """Soft-Attention without learnable parameters
    """

    def __init__(self, dimension):
        super(SoftAttention, self).__init__()
        self.softmax = nn.Softmax(dim=2)
        self.conf_linear = nn.Linear(512, 512)
        self.conf_linear1 = nn.Linear(300, 512)

    def forward(self, cls_input, cls_mask, token_input, token_mask):
        """Propagate h through the network.
        cls_input: batch x 10 x 768
        cls_mask: batch x 10
        cls_input: batch x 10 x max_token_len x 768
        token_mask: batch x 10 x 13
        """
        # Get attention
        if cls_input.shape[-1] == 512:
            cls_input = self.conf_linear(cls_input)
        else:
            cls_input = self.conf_linear1(cls_input)
        global new_cls_input
        new_cls_input = cls_input

        attn = torch.matmul(cls_input.unsqueeze(dim=2), token_input.transpose(2,3)).squeeze(2)  # batch x 10 x 13

        if token_mask is not None:
            attn = torch.where(token_mask == 1, attn, torch.tensor(-float('inf'), device=attn.device))
           # attn.data.masked_fill_((token_mask == 0).data, -float('inf')) #batch x 10 x 13

        attn = self.softmax(attn)
        new_attn = torch.where(attn != attn, torch.zeros_like(attn), attn)
        weighted_token_input = torch.matmul(new_attn.unsqueeze(dim=2), token_input).squeeze(2) # batch x 10 x 768

        return  weighted_token_input, new_attn



