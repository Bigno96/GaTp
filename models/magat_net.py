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

import logging
import torch

import torch.nn as nn
import models.basic.gat_gso_net as gat_gso
import models.basic.mlp as mlp
import models.basic.res_net as res_net

from easydict import EasyDict

ACTION_NUMBER = 5


class MAGATNet(nn.Module):
    """
    MAGAT Model implementation
    Check yaml_configs\\magat.yaml for further model and parameter specification
    GSO needs to be externally set with method 'set_gso' before calling 'forward'

    Input Tensor shape = (B, N, C, H, W)
        B = batch size
        N = agents number
        C = channels of the input image
        H, W = height, width of the input image

    Output Tensor shape = (B*N, 5)
        B*N = batch size * agents number
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
    def __init__(self,
                 config: EasyDict):
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
        # feature expansion factor in the intermediate layer of res-net decoder
        self.cnn_decoder_expansion = self.config.cnn_decoder_expansion
        # bool to activate down sampling
        self.use_down_sampling = self.config.use_down_sampling
        # if True, add dropout to res net decoder
        self.cnn_dropout = self.config.cnn_dropout
        # ratio of dropout, if used
        self.cnn_dropout_rate = self.config.cnn_dropout_rate

        if self.cnn_model.lower() == 'res':
            self.cnn = \
                res_net.ResNet(
                    in_channels=self.cnn_in_channels,
                    out_features=self.cnn_out_features,
                    blocks_size=self.cnn_blocks_size,
                    depths=self.cnn_depths,
                    expansion=self.cnn_decoder_expansion,
                    use_down_sampling=self.use_down_sampling,
                    use_dropout=self.cnn_dropout,
                    dropout_rate=self.cnn_dropout_rate
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
        # if True, learn bias in the MLP
        self.feature_compr_learn_bias = self.config.feature_compr_learn_bias

        if self.feature_compression:
            self.feature_compressor = \
                mlp.MLP(
                    in_features=self.cnn_out_features,
                    out_features=self.feature_compr_out,
                    hidden_features=self.feature_compr_hidden_feat,
                    use_dropout=self.feature_compr_dropout,
                    dropout_rate=self.feature_compr_dropout_rate,
                    learn_bias=self.feature_compr_learn_bias
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
                gat_gso.GatGSO(
                    in_features=self.gnn_features,
                    out_features=self.gnn_features,     # output features == input features
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
        # if True, learn bias in the MLP
        self.mlp_learn_bias = self.config.mlp_learn_bias

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
            mlp.MLP(
                in_features=self.mlp_in_features,
                out_features=ACTION_NUMBER,     # 5 actions available
                hidden_features=self.mlp_hidden_features,
                use_dropout=self.mlp_dropout,
                dropout_rate=self.mlp_dropout_rate,
                learn_bias=self.mlp_learn_bias,
                append_relu=False   # last layer here will use multinorm, not relu
            )

    def set_gso(self,
                S: torch.Tensor) -> None:
        """
        Set given GSO to the graph neural network
        """
        self.gnn.set_gso(S)

    # noinspection PyUnboundLocalVariable
    def forward(self,
                input_tensor: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass
        """
        # B = batch size
        # N = agents number
        # C = input channels
        # H, W = height, width of the input
        B, N, C, H, W = input_tensor.shape
        # reshape for current agent
        x = input_tensor.reshape(B * N, C, H, W)

        # extract feature through cnn,
        # B*N x F (cnn_out_feature)
        x = self.cnn(x)

        # additional reduction of features
        if self.feature_compression:
            # B*N x F (compr_out_feature)
            x = self.feature_compressor(x)

        # add skip connection
        # B*N x F
        residual = x.clone()

        # first, B*N x F -> B x N x F, with F that can be either cnn_out_feature or compr_out_feature
        # second, reshape B x N x F -> B x F x N, gnn input ordering
        x = x.view(B, N, -1).permute([0, 2, 1])

        # pass through gnn to get information from other agents
        # B x G (gnn_features) x N
        x = self.gnn(x)

        # reshape to allow concatenation with skip connection
        # B x G x N -> B*N x G
        x = x.permute([0, 2, 1]).reshape(B*N, -1)

        # if residual was set, add it
        if self.skip_connection:
            # concat B*N x G + B*N x F on dimension 1
            # B*N x G+F
            x = torch.cat((x, residual), dim=1)

        # pass through mlp to map features to action
        # B*N x 5
        return self.mlp(x)
