import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DocAttNet(nn.Module):
    def __init__(self, num_sent_rnn_directions, doc_rnn_size, sent_att_hidden_size,
                 doc_rnn_num_layers, doc_rnn_bidirectional):
        super(DocAttNet, self).__init__()
        self.rnn = nn.GRU(num_sent_rnn_directions * sent_att_hidden_size, doc_rnn_size,
                          bidirectional=doc_rnn_bidirectional,
                          num_layers=doc_rnn_num_layers,
                          batch_first=True)

        # Attention
        self.linear = nn.Linear(num_sent_rnn_directions * doc_rnn_size,
                                num_sent_rnn_directions * doc_rnn_size, bias=True)
        self.context_weight = nn.Parameter(torch.Tensor(num_sent_rnn_directions * doc_rnn_size),
                                           requires_grad=True)

        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.context_weight.data.normal_(mean, std)

    def forward(self, batch, lengths):
        # Handle variable length sequences
        sent_embedding = pack_padded_sequence(batch, batch_first=True,
                                              lengths=lengths, enforce_sorted=False)
        self.rnn.flatten_parameters()  # To bring all parameters to the same device

        rnn_out, final_hidden_state = self.rnn(sent_embedding)
        # B x timesteps x sent_attn_hidden_size, num_directions x B x sent_attn_hidden_size
        rnn_out, lengths = pad_packed_sequence(rnn_out, batch_first=True)

        # Attention
        output = torch.tanh(self.linear(rnn_out))
        output = torch.matmul(output, self.context_weight)  # -> B x timesteps
        output = torch.softmax(output, dim=1)  # -> B x timesteps
        output = output.unsqueeze(2).expand(-1, -1, rnn_out.size(2)) * rnn_out
        # -> B x timesteps x num_directions*doc_rnn_size
        return output.sum(1)  # -> B x num_directions*doc_rnn_size
