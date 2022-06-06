import logging

import gensim
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SentAttNet(nn.Module):
    def __init__(self, pretrained_vector_path, pretrained_vector_freeze,
                 embedding_len, embedding_size,
                 sent_attn_hidden_size, sent_rnn_num_layers, sent_rnn_bidirectional,
                 pad_token_id):
        super(SentAttNet, self).__init__()
        self.num_rnn_directions = 2 if sent_rnn_bidirectional else 1
        if pretrained_vector_path:
            model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_vector_path, binary=True)
            logger.warning("Note that index of words in pretrained_vector should be same in current model")
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(model.vectors),
                                                          freeze=pretrained_vector_freeze,
                                                          padding_idx=pad_token_id)
            embedding_size = model.vector_size
        else:
            self.embedding = nn.Embedding(num_embeddings=embedding_len, embedding_dim=embedding_size,
                                          padding_idx=pad_token_id)

        self.rnn = nn.GRU(embedding_size, sent_attn_hidden_size, bidirectional=sent_rnn_bidirectional,
                          num_layers=sent_rnn_num_layers,
                          batch_first=True)

        self.linear = nn.Linear(self.num_rnn_directions * sent_attn_hidden_size,
                                self.num_rnn_directions * sent_attn_hidden_size, bias=True)
        self.context_weight = nn.Parameter(torch.Tensor(self.num_rnn_directions * sent_attn_hidden_size),
                                           requires_grad=True)
        self._init_weights(mean=0.0, std=0.05)

    def _init_weights(self, mean=0.0, std=0.05):
        self.context_weight.data.normal_(mean, std)

    def forward(self, batch):
        embedding = self.embedding(batch)
        self.rnn.flatten_parameters()
        rnn_out, final_hidden_state = self.rnn(embedding)

        output = torch.tanh(self.linear(rnn_out))
        output = torch.matmul(output, self.context_weight)  # -> B x timesteps
        output = torch.softmax(output, dim=1)  # -> B x timesteps
        output = output.unsqueeze(2).expand(-1, -1, rnn_out.size(2)) * rnn_out
        # -> B x timesteps x num_directions*sent_attn_hidden_size
        return output.sum(1), final_hidden_state  # -> B x num_directions*sent_attn_hidden_size


class SentCNNAttNet(nn.Module):
    """ TODO: Add conv layer https://www.mdpi.com/1999-5903/11/12/255
    """
    pass
