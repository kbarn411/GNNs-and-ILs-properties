import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, MFConv, GATConv, CGConv, GraphConv, GINConv
from torch_geometric.nn import TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class GCN(torch.nn.Module):
    def __init__(self, conv_function, input_channels, embedding_size, linear_size, add_params_num=0):
        # Init parent
        super(GCN, self).__init__()
        self.crafted_add_params_num = add_params_num

        # GCN layers
        self.conv1 = conv_function(input_channels, embedding_size[0])
        self.conv2 = conv_function(embedding_size[0], embedding_size[1])
        self.conv3 = conv_function(embedding_size[1], embedding_size[2])
        self.conv4 = conv_function(embedding_size[2], embedding_size[3])

        # Dropout
        self.dropout1 = torch.nn.Dropout(0.2)

        # Linear layers
        self.linear1 = Linear(embedding_size[-1]+add_params_num, linear_size[0])
        self.linear2 = Linear(linear_size[0],linear_size[1])

        # Dropout 2
        self.dropout2 = torch.nn.Dropout(0.3)

        # batch normalization
        self.bnf = torch.nn.BatchNorm1d(linear_size[-1])

        # Output layer
        self.out = Linear(linear_size[-1], 1)


    def forward(self, x, edge_index, batch_index, cond=None):
        # Conv layers
        hidden = self.conv1(x, edge_index).relu()
        hidden = self.dropout1(hidden)
        hidden = self.conv2(hidden, edge_index).relu()
        hidden = self.dropout1(hidden)
        hidden = self.conv3(hidden, edge_index).relu()
        hidden = self.dropout1(hidden)
        hidden = self.conv4(hidden, edge_index).relu()
        hidden = self.dropout1(hidden)

        # Pooling
        hidden = gap(hidden, batch_index)

        # adding pressure and temperature info
        if self.crafted_add_params_num != 0:
            hidden = torch.cat([hidden, cond], dim=1)

        # Apply a final (linear) classifier.
        hidden = self.linear1(hidden)
        hidden = self.linear2(hidden)
        hidden = self.dropout2(hidden)
        hidden = self.bnf(hidden)
        hidden = torch.nn.functional.relu(hidden)
        out = self.out(hidden)

        return out, hidden
    
# ADDED Architecture for two netoworks
class GCN2(torch.nn.Module):
    def __init__(self, conv_function, input_channels, embedding_size, linear_size, add_params_num=0):
        # Init parent
        super(GCN2, self).__init__()
        self.crafted_add_params_num = add_params_num

        # GCN layers
        self.conv1 = conv_function(input_channels, embedding_size[0])
        self.conv2 = conv_function(embedding_size[0], embedding_size[1])
        self.conv3 = conv_function(embedding_size[1], embedding_size[2])
        self.conv4 = conv_function(embedding_size[2], embedding_size[3])

        # Dropout
        self.dropout1 = torch.nn.Dropout(0.2)

        # Linear layers
        self.linear1 = Linear(embedding_size[-1]*2+add_params_num, linear_size[0])
        self.linear2 = Linear(linear_size[0],linear_size[1])

        # Dropout 2
        self.dropout2 = torch.nn.Dropout(0.3)

        # batch normalization
        self.bnf = torch.nn.BatchNorm1d(linear_size[-1])

        # Output layer
        self.out = Linear(linear_size[-1], 1)


    def forward(self, x_c, edge_index_c, x_a, edge_index_a, batch_index_c, batch_index_a, cond=None):
        # Conv layers
        hidden1 = self.conv1(x_c, edge_index_c).relu()
        hidden1 = self.dropout1(hidden1)
        hidden1 = self.conv2(hidden1, edge_index_c).relu()
        hidden1 = self.dropout1(hidden1)
        hidden1 = self.conv3(hidden1, edge_index_c).relu()
        hidden1 = self.dropout1(hidden1)
        hidden1 = self.conv4(hidden1, edge_index_c).relu()
        hidden1 = self.dropout1(hidden1)
        hidden1 = gap(hidden1, batch_index_c)

        hidden2 = self.conv1(x_a, edge_index_a).relu()
        hidden2 = self.dropout1(hidden2)
        hidden2 = self.conv2(hidden2, edge_index_a).relu()
        hidden2 = self.dropout1(hidden2)
        hidden2 = self.conv3(hidden2, edge_index_a).relu()
        hidden2 = self.dropout1(hidden2)
        hidden2 = self.conv4(hidden2, edge_index_a).relu()
        hidden2 = self.dropout1(hidden2)
        hidden2 = gap(hidden2, batch_index_a)

        # adding pressure and temperature info
        if self.crafted_add_params_num != 0:
            hidden = torch.cat([hidden1, hidden2, cond], dim=1)
        else:
            hidden = torch.cat([hidden1, hidden2], dim=1)

        # Apply a final (linear) classifier.
        hidden = self.linear1(hidden)
        hidden = self.linear2(hidden)
        hidden = self.dropout2(hidden)
        hidden = self.bnf(hidden)
        hidden = torch.nn.functional.relu(hidden)
        out = self.out(hidden)

        return out, hidden