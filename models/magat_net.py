"""
Create a model based on MAGAT implementation, as described in:
Q. Li, W. Lin, Z. Liu and A. Prorok,
"Message-Aware Graph Attention Networks for Large-Scale Multi-Robot Path Planning,"
in IEEE Robotics and Automation Letters, vol. 6, no. 3, pp. 5533-5540, July 2021, doi: 10.1109/LRA.2021.3077863.

Model structure:
- CNN + Optional MLP
    Extract and encode input features

- Graph Attention Neural Network
    Get CNN features and embed it with neighbours information

- MLP
    Perceptron layer to derive the action policy
    Receives both output of GAT and CNN (through a skip connection)
"""

import torch
import torch.nn as nn
import logging

from models.basic.res_net import ResNet
from models.basic.gat_gso_net import GatGSO
from models.basic.mlp import MLP


class MAGATNet(nn.Module):
    """
    MAGAT Model implementation
    Check yaml_configs\\magat.yaml for further model and parameter specification
    GSO needs to be externally set with method 'set_gso' before calling 'forward'

    Input Tensor shape = (B, N, C, W, H)
        B = batch size
        N = agents number
        C = channels of the input image
        W, H = width, height of the input image

    Output Tensor shape = (B, 5)
        B = batch size
        5 = actions

    CNN:
        Residual Network with 3 layers, extract features from input
        Optional MLP to compress extracted feature vector
    GAT:
        Graph Convolution Attentional Neural Network, shares extracted features between agents
        Optional skip connection to join agent's own extracted feature from before the GNN with its output,
        who aggregates neighbours information as well
    MLP:
        Final Multi Layer Perceptron Network, to map features into actions
    """
    def __init__(self, config):
        # initialize parent
        super().__init__()

        # config Namespace and logging
        self.config = config
        self.logger = logging.getLogger('MAGAT_Net')

        #######################
        #         CNN         #
        #######################
        '''CNN for extracting features'''
        # type of cnn network
        self.cnn_model = self.config.cnn_model
        # number of input channels
        self.cnn_in_channels = self.config.cnn_in_channels
        # number of 'extracted' features by the CNN
        self.cnn_out_features = self.config.cnn_out_features
        # number of channels for cnn blocks
        self.cnn_blocks_size = self.config.cnn_blocks_size
        # number of blocks that compose each layer of the cnn
        self.cnn_depths = self.config.cnn_depths

        if self.cnn_model.lower() == 'res':
            self.cnn = \
                ResNet(
                    in_channels=self.cnn_in_channels,
                    out_features=self.cnn_out_features,
                    blocks_size=self.cnn_blocks_size,
                    depths=self.cnn_depths
                )
        else:
            self.logger.error('No CNN model was specified')

        '''Optional MLP layers for further feature compression'''
        # if True, an additional MLP layer is placed at the end of the CNN
        self.feature_compression = self.config.feature_compression
        # output from features compression. These coincide with skip connection features
        self.feature_compr_out = self.config.feature_compr_out
        # number of features of the hidden layers in the MLP
        self.feature_compr_hidden_feat = self.config.feature_compr_hidden_feat
        # if True, add dropout to MLP
        self.feature_compr_dropout = self.config.feature_compr_dropout
        # ratio of dropout, if used
        self.feature_compr_dropout_rate = self.config.feature_compr_dropout_rate

        if self.feature_compression:
            self.feature_compressor = \
                MLP(
                    in_features=self.cnn_out_features,
                    out_features=self.feature_compr_out,
                    hidden_features=self.feature_compr_hidden_feat,
                    use_dropout=self.feature_compr_dropout,
                    dropout_rate=self.feature_compr_dropout_rate,
                    learn_bias=True
                )
            # if feature compression is used, residual size = output of feature compression
            self.residual_size = self.feature_compr_out
        else:
            # no feature compression, size of residual = output of CNN
            self.residual_size = self.cnn_out_features

        # if True, a skip connection is added between before and after the GNN
        self.skip_connection = self.config.skip_connection
        if not self.skip_connection:
            self.residual_size = 0

        #######################
        #         GNN         #
        #######################
        '''GNN for sharing features amongst agents (communication)'''
        # type of graph convolution attentional network
        self.gnn_model = self.config.gnn_model
        # tuple of int, vector of INPUT features of hidden layers
        # len = number of desired graph filtering layers - 1
        self.gnn_hidden_features = self.config.gnn_hidden_features
        # tuple of int, 'power' of the GSO. Vector with number of filter taps for each layer
        # len = number of graph filtering layers
        self.graph_filter_taps = self.config.graph_filter_taps
        # tuple of int, vector with number of attention heads for each layer
        # len = number of graph filtering layers
        self.attention_heads = self.config.attention_heads
        # if True, concatenate the output of the attention heads. False, average the output
        self.attention_concat = self.config.attention_concat

        # input features of GNN
        # if feature compression is active, use output of previous MLP
        # if not, use output of CNN
        if self.feature_compression:
            self.gnn_features = self.feature_compr_out
        else:
            self.gnn_features = self.cnn_out_features

        if self.gnn_model.lower() == 'gat_gso':
            self.gnn = \
                GatGSO(
                    in_features=self.gnn_features,
                    out_features=self.gnn_features,       # output features == input features
                    hidden_features=self.gnn_hidden_features,
                    graph_filter_taps=self.graph_filter_taps,
                    attention_heads=self.attention_heads,
                    attention_concat=self.attention_concat
                )
        else:
            self.logger.error('No GNN model was specified')

        #######################
        #         MLP         #
        #######################
        '''MLP for mapping features coming from GNN into actions feature vector'''
        # tuple of int, vector of INPUT features of hidden layers
        # len = number of layers - 1
        self.mlp_hidden_features = self.config.mlp_hidden_features
        # if True, add dropout to MLP
        self.mlp_dropout = self.config.mlp_dropout
        # ratio of dropout, if used
        self.mlp_dropout_rate = self.config.mlp_dropout_rate

        # input of action-mapper depends on residual size and attention heads (number and concat mode)
        # always add residual connection size, it's 0 if the skip connection is not used
        # if concatenation = True -> add (attention heads * gnn output)
        # if concatenation = False -> add (gnn output)
        if self.attention_concat:
            self.mlp_in_features = self.residual_size \
                                   + int(self.attention_heads[-1] * self.gnn_features)
        else:
            self.mlp_in_features = self.residual_size + self.gnn_features

        self.mlp = \
            MLP(
                in_features=self.mlp_in_features,
                out_features=5,             # 5 actions available
                hidden_features=self.mlp_hidden_features,
                use_dropout=self.mlp_dropout,
                dropout_rate=self.mlp_dropout_rate,
                learn_bias=True
            )

    def set_gso(self, S):
        self.gnn.set_gso(S)

    # noinspection PyUnboundLocalVariable
    def forward(self, input_tensor):

        # B = batch size
        # N = agents number
        # C = input channels
        # W, H = width, height of the input
        B, N, C, W, H = input_tensor.shape
        # reshape for current agent
        input_current_agent = input_tensor.reshape(B * N, C, W, H).to(self.config.device)

        # extract feature through cnn,
        # B*N x F (cnn_out_feature)
        extracted_features = self.cnn(input_current_agent).to(self.config.device)

        # additional reduction of features
        if self.feature_compression:
            # B*N x F (compr_out_feature)
            extracted_features = self.feature_compressor(extracted_features)

        # add skip connection
        if self.skip_connection:
            # B*N x F
            residual = extracted_features

        # first, B*N x F -> B x N x F, with F that can be either cnn_out_feature or compr_out_feature
        # second, reshape B x N x F -> B x F x N, gnn input ordering
        # using view to guarantee copy, to not affect residual
        extracted_features = extracted_features.view(B, N, -1).permute([0, 2, 1]).to(self.config.device)

        # pass through gnn to get information from other agents
        # B x G (gnn_features) x N
        shared_features = self.gnn(extracted_features)

        # reshape to allow concatenation with skip connection
        # B x G x N -> B*N x G
        shared_features = shared_features.permute([0, 2, 1]).reshape(B*N, -1).to(self.config.device)

        # if residual was set, add it
        if self.skip_connection:
            # concat B*N x G + B*N x F on dimension 1
            # B*N x G+F
            shared_features = torch.cat((shared_features, residual), dim=1)

        # pass through mlp to map features to action
        # B*N x 5
        action_vector = self.mlp(shared_features)

        return action_vector
