from __future__ import print_function, division
import torch
from .layers import layers
from .layers import cspn
from .layers import attention
from collections import OrderedDict


class MultiConvResidualBlock(torch.nn.Module):
    def __init__(self, res_depth, filter_size, channels,
                 use_sparse_conv, activation, keep_prob):
        super(MultiConvResidualBlock, self).__init__()
        self.conv_res_list = torch.nn.ModuleList()
        self.use_sparse_conv = use_sparse_conv
        self.res_depth = res_depth
        self.keep_prob = keep_prob
        self.activation = None
        if activation is not None:
            self.activation = activation()
        for a_res in range(0, res_depth):
            if a_res < res_depth-1:
                self.conv_res_list.append(
                    layers.Conv2dBnLrnDrop(
                        [filter_size, filter_size, channels, channels],
                        activation=activation,
                        use_sparse_conv=self.use_sparse_conv,
                        keep_prob=self.keep_prob
                    )
                )
            else:
                self.conv_res_list.append(
                    layers.Conv2dBnLrnDrop(
                        [filter_size, filter_size, channels, channels],
                        activation=None,
                        use_sparse_conv=self.use_sparse_conv,
                        keep_prob=self.keep_prob
                    )
                )
        self.relu = torch.nn.ReLU()

    def forward(self, x, binary_mask=None):
        orig_x = x
        x = self.relu(x)
        for a_res in range(self.res_depth):
            if self.use_sparse_conv:
                x, binary_mask = self.conv_res_list[a_res](x, binary_mask)
            else:
                x = self.conv_res_list[a_res](x, binary_mask)
        x += orig_x
        if self.activation is not None:
            x = self.activation(x)
        if self.use_sparse_conv:
            return x, binary_mask
        return x
class DownSamplingBlock(torch.nn.Module):
    '''
    DownSamplingBlock may be used for both original and coupled U-Net

    Args:
        use_residual: whether to use the multiple conv blocks with residual connections
        channels: num inp channels
        scale_space_num: number of downsamplings
        res_depth: the residual multi conv blocks depth
        filter_size: size of each filter (int)
        pool_size: pooling size (int)
        activation: the activation function
        use_sparse_conv: whether to use the sparse convolution op
        use_prev_coupled: is not the first block in the coupled UNet?
        rate: for dilation convultuion
    '''
    def __init__(self, use_residual, channels, scale_space_num,
                 res_depth, feat_root, filter_size, pool_size,
                 activation, use_sparse_conv=False,
                 use_prev_coupled=False, keep_prob=1.0,rate=0):
        super(DownSamplingBlock, self).__init__()
        self.input_channels = channels
        self.ksize_pool = [pool_size, pool_size]
        self.stride_pool = self.ksize_pool
        self.use_sparse_conv = False
        self.scale_space_num = scale_space_num
        self.use_residual = use_residual
        self.res_depth = res_depth
        self.keep_prob = keep_prob
#         self.conv_res_list = torch.nn.ModuleList()
        self.act_feat_num = feat_root
#         self.conv1s = torch.nn.ModuleList()
        self.relu = torch.nn.ReLU()
#         self.conv1_1s = torch.nn.ModuleList()
        self.use_prev_coupled = use_prev_coupled
#         self.layer_attentions = torch.nn.ModuleList()
        self.custom_pad = lambda inp: layers.pad_2d(inp, 'SAME',
            kind='pool2d',
            k_h=self.ksize_pool[0],
            k_w=self.ksize_pool[1],
            s_h=self.stride_pool[0],
            s_w=self.stride_pool[1])
        self.max_pool = torch.nn.MaxPool2d(
                self.ksize_pool, self.stride_pool)
        self.activation = activation
        self.last_feat_num = channels
        self.act_feat_num = feat_root
        #TODO
        self.rate= rate
        if self.use_residual:
            self.dilconv1s=layers.DilConv2dBnLrnDrop(
                [filter_size, filter_size,
                 self.last_feat_num, self.act_feat_num],
                rate=2 ** self.rate, padding='SAME',
                activation=None, keep_prob=self.keep_prob)
            self.conv_res_list=MultiConvResidualBlock(
                res_depth, filter_size, self.act_feat_num,
                self.use_sparse_conv, self.activation,self.keep_prob)
            if use_prev_coupled:
                self.conv1_1s=layers.Conv2dBnLrnDrop(
                    [1, 1, 2*self.act_feat_num, self.act_feat_num],
                    activation=activation,keep_prob=self.keep_prob)
                
            self.layer_attentions = attention.SAWrapperBlock(
                self.act_feat_num
                )
        else:
            #FOR NORMAL CNN
            self.conv1s.append(layers.Conv2dBnLrnDrop(
                [filter_size, filter_size, channels, feat_root],
                activation=activation, keep_prob=self.keep_prob))
            self.conv1_1s.append(layers.Conv2dBnLrnDrop(
                [filter_size, filter_size,
                 self.act_feat_num, self.act_feat_num],
                activation=activation, keep_prob=self.keep_prob))
        #import ipdb; ipdb.set_trace()
        # need to checl what to do with these
#         self.last_feat_num = self.act_feat_num
#         self.act_feat_num *= pool_size
                            
    def forward(self, unet_inp, prev_dw_h_convs=None,
                binary_mask=None):
        '''
        :param prev_dw_h_convs: previous down-sampling tower's outputs
                                (used for coupling connection)
        '''
        dw_h_convs = OrderedDict()
        if self.use_residual:
            x = self.dilconv1s(unet_inp)
            if self.use_sparse_conv:
                x, binary_mask = self.conv_res_list(x, binary_mask)
            else:
                x = self.conv_res_list(x)
            if self.use_prev_coupled:
                assert(prev_dw_h_convs is not None),\
                    "ERROR: Second Unet block not fed with previous data"
                prev_dw_h_conv = prev_dw_h_convs
                x = torch.cat([prev_dw_h_conv, x], dim=1)
                x = self.conv1_1s(x)
            next_dw_block = self.custom_pad(x)
            next_dw_block = self.max_pool(next_dw_block)
            up_block=self.layer_attentions(x)

            
            
            
        else:
            #TODO
            pass
#             conv1 = self.conv1s(unet_inp)
#             dw_h_convs[layer] = self.conv1_1s(conv1)
#             x = dw_h_convs
        # x is for next U-NET block
        #  next_dw_block next down block
        #  up_block skip connection
        return next_dw_block, up_block, x

class UpSamplingBlock(torch.nn.Module):
    '''
    UpSamplingBlock, may be used for both original and coupled U-Net
    '''
    def __init__(self, use_residual, channels, scale_space_num,
                 res_depth, feat_root, filter_size, pool_size,
                 activation, last_feat_num, act_feat_num,
                 use_prev_coupled=False,keep_prob=1.0):
        super(UpSamplingBlock, self).__init__()
        self.input_channels = channels
        self.ksize_pool = [pool_size, pool_size]
        self.stride_pool = self.ksize_pool
        self.scale_space_num = scale_space_num
        self.use_residual = use_residual
        self.res_depth = res_depth
        self.keep_prob = keep_prob
#         self.conv_res_list = torch.nn.ModuleList()
        self.act_feat_num = feat_root
#         self.conv1s = torch.nn.ModuleList()
        self.relu = torch.nn.ReLU()
        self.res_depth = res_depth
#         self.conv1s = torch.nn.ModuleList([None for i in range(self.scale_space_num - 2, -1, -1)])
#         self.conv1_1s = torch.nn.ModuleList([None for i in range(self.scale_space_num - 2, -1, -1)])
#         self.deconvs = torch.nn.ModuleList([None for i in range(self.scale_space_num - 2, -1, -1)])
        self.use_prev_coupled = use_prev_coupled
#         self.layer_attentions = torch.nn.ModuleList()
        self.channels = channels
#         self.conv_res_list = torch.nn.ModuleList([None for i in range(self.scale_space_num - 2, -1, -1)])
        self.activation = activation
        self.act_feat_num = act_feat_num
        self.last_feat_num = last_feat_num
        
        self.deconvs = layers.Deconv2DBnLrnDrop(
            [filter_size, filter_size,
             self.act_feat_num, self.last_feat_num],
            activation=None, keep_prob=self.keep_prob)
        if self.use_residual:
            #why this is here
            self.conv1s = layers.Conv2dBnLrnDrop(
                    [filter_size, filter_size,
                     pool_size*self.act_feat_num,
                     self.act_feat_num], activation=None, keep_prob=self.keep_prob)
            self.conv_res_list = MultiConvResidualBlock(
                res_depth, filter_size, self.act_feat_num, False,
                self.activation, self.keep_prob)
            if self.use_prev_coupled:
                self.conv1_1s = layers.Conv2dBnLrnDrop(
                    [1, 1, 2 * self.act_feat_num, self.act_feat_num],
                    activation=activation, keep_prob=self.keep_prob)
        else:
            #pass take care later
            self.conv1s[layer] = layers.Conv2dBnLrnDrop(
                [filter_size, filter_size, pool_size * self.act_feat_num,
                 self.act_feat_num], activation=self.activation,keep_prob=self.keep_prob)
            self.conv1_1s[layer] = layers.Conv2dBnLrnDrop(
                [filter_size, filter_size, self.act_feat_num,
                 self.act_feat_num], activation=self.activation, keep_prob=self.keep_prob)
#             self.last_feat_num = self.act_feat_num
#             self.act_feat_num //= pool_size

    def forward(self, dw_h_conv, down_sampled_out, prev_up_h_conv=None):
        """
        dw_h_convs: skip connections
        down_sampled_out: output below of upSampling
        prev_up_h_convs: from previous U-NET
        """
        up_dw_h_convs = OrderedDict()
        skip_conn = dw_h_conv
        # Need to pad
        #print("skip_conn",skip_conn.shape)
        #print("Down sampled out size: ", down_sampled_out.size())
        deconv = self.deconvs(down_sampled_out, output_size=skip_conn.size()[2:]) #upsampling
        # #print("Target size: ", dw_h_conv.size())
        #print("Deconv out size: ", deconv.size())

        conc = torch.cat([skip_conn, deconv], dim=1) # half from skip conection and..
        #print("after concat",conc.shape)
        if self.use_residual:
            x = self.conv1s(conc)
            x = self.conv_res_list(x)
            if self.use_prev_coupled:
                assert prev_up_h_convs is not None,\
                    "ERROR: Use coupled but no data provided in \
                    upsampling block"
                prev_up_dw_h_conv = prev_up_h_conv
                x = torch.cat([prev_up_dw_h_conv, x], dim=1)
                x = self.conv1_1s(x)
            next_up_convs = x

        else:
            conv1 = self.conv1s[layer](conc)
            down_sampled_out = self.conv2s(conv1)
        # x for next unet up block
        # next_up_convs for next up block
        return next_up_convs, x

class UNetBlock(torch.nn.Module):
    """
    UNetBlock according to the model
    :param input: input image
    :param useResidual: use residual connection (ResNet)
    :param use_lstm: run a separable LSTM horizontally then
                    vertically across input features
    :param useSPN: use Spatial Propagation Network
    :param channels: number of input channels
    :param scale_space_num: number of down-sampling / up-sampling blocks
    :param res_depth: number of convolution layers in a down-sampling block
    :param featRoot: number of features in the first layers
    :param filter_size: convolution kernel size
    :param pool_size: pooling size
    :param activation: activation function
    :param use_prev_coupled: is this the second in the coupled block?

    :return:

    """
    def __init__(self, use_residual, use_lstm, use_spn, channels,
                 scale_space_num, res_depth, feat_root, filter_size, pool_size,
                 activation, use_sparse_conv=False,
                 use_prev_coupled=False,keep_prob=1.0):
        super(UNetBlock, self).__init__()
        self.input_channels = channels
        self.pool_size = pool_size
        self.ksize_pool = [pool_size, pool_size]
        self.stride_pool = self.ksize_pool
        self.use_sparse_conv = use_sparse_conv
        self.scale_space_num = scale_space_num
        self.use_residual = use_residual
        self.use_spn = use_spn
        self.res_depth = res_depth
        self.channels = channels
        self.act_feat_num = feat_root
        self.activation = activation
        self.use_prev_coupled = use_prev_coupled
        self.keep_prob = keep_prob
        self.last_feat_num =channels
        #print("Using sparse conv: ", use_sparse_conv)
        #print("Use prev coupled: ", use_prev_coupled)
        #print("Dropout: ", round((1.0 - keep_prob),2))
        self.down_blocks = torch.nn.ModuleList()
        self.up_blocks= torch.nn.ModuleList()
        for i in range(0,4):
            self.downsampling = DownSamplingBlock(
                use_residual, self.last_feat_num, scale_space_num,
                res_depth, self.act_feat_num, filter_size,
                pool_size, activation, use_sparse_conv,
                use_prev_coupled, self.keep_prob,i)
            
            self.down_blocks.append(self.downsampling)
            self.last_feat_num = self.act_feat_num
            self.act_feat_num *= pool_size
            
        self.act_feat_num = self.last_feat_num // pool_size
        self.last_feat_num = self.last_feat_num 
#         self.bottle_neck = layers.Conv2dBnLrnDrop(
#             [filter_size, filter_size, channels, feat_root],
#             activation=activation, keep_prob=self.keep_prob)

        self.lstm = None
        self.use_lstm = use_lstm
        if self.use_lstm:
            self.lstm = layers.SeparableRNNBlock(self.act_feat_num,
                                                 self.last_feat_num,
                                                 cell_type='LSTM')
        if self.use_spn:
            self.downsample_resnet = layers.DownSampleResNet(
                self.act_feat_num, self.act_feat_num,
                filter_size, res_depth, self.ksize_pool, activation)
            #self.cspn = layers.cspn.Affinity_Propagate()
            self.cspn = cspn.Affinity_Propagate()
        for j in range(0,3):
            self.upsampling = UpSamplingBlock(
                self.use_residual, self.channels, self.scale_space_num,
                res_depth, self.act_feat_num, filter_size, self.pool_size,
                self.activation, self.last_feat_num, self.act_feat_num,
                self.use_prev_coupled, self.keep_prob)
        
            self.up_blocks.append(self.upsampling)
            self.last_feat_num = self.act_feat_num
            self.act_feat_num //= pool_size

        #self.last_feat_num = self.act_feat_num
        #self.act_feat_num = self.last_feat_num
        self.conv1_1s = layers.Conv2dBnLrnDrop(
            [1, 1, 2 * self.act_feat_num, 2*self.act_feat_num],
            activation=activation, keep_prob=self.keep_prob)
    def forward(self, unet_inp, prev_dw_h_convs=None,
                prev_up_h_convs=None,
                binary_mask=None):
        next_dw_block0, up_block0, x0 = self.down_blocks[0](unet_inp,
                                               prev_dw_h_convs[0] if self.use_prev_coupled else None,
                                               binary_mask)
        #print('#output next_dw_block0',next_dw_block0.shape)
        next_dw_block1, up_block1, x1 = self.down_blocks[1](next_dw_block0,
                                               prev_dw_h_convs[1] if self.use_prev_coupled else None,
                                               binary_mask)
        #print('#output next_dw_block1',next_dw_block1.shape)
        next_dw_block2, up_block2, x2 = self.down_blocks[2](next_dw_block1,
                                               prev_dw_h_convs[2] if self.use_prev_coupled else None,
                                               binary_mask)
        #print('#output next_dw_block2',next_dw_block2.shape)
        next_dw_block3, up_block3, x3 = self.down_blocks[3](next_dw_block2,
                                       prev_dw_h_convs[3] if self.use_prev_coupled else None,
                                       binary_mask)
        #print('#output next_dw_block3',next_dw_block3.shape)
        #print('#output dw x3',x3.shape)
        dw_h_unet=[x0,x1,x2,x3]
        #TODO check if bottleneck required

        next_up_conv0, up_uet_conv0 = self.up_blocks[0](
            up_block2, x3, prev_up_h_conv=prev_up_h_conv[0] if self.use_prev_coupled else None)
        
        next_up_conv1, up_uet_conv1 = self.up_blocks[1](
            up_block1, next_up_conv0, prev_up_h_conv=prev_up_h_conv[1] if self.use_prev_coupled else None)
        
        next_up_convs2, up_uet_conv2 = self.up_blocks[2](
            up_block0, next_up_conv1, prev_up_h_conv=prev_up_h_conv[2] if self.use_prev_coupled else None)
        
#         next_up_conv3, up_uet_conv3 = self.up_blocks[3](
#             up_block0, next_up_convs2, prev_up_h_conv=prev_up_h_conv[3] if self.use_prev_coupled else None)
        
        up_h_unet=[up_uet_conv0, up_uet_conv1, up_uet_conv2]
        #TODO check the feature
        end_unet=self.conv1_1s(up_uet_conv2)
        # up_h_unet: for next up unet
        # dw_h_unet: for net down unet
        # output for next unet with 1*1 convolution
        
        return dw_h_unet, up_h_unet, end_unet

class MSAUNet(torch.nn.Module):
    def __init__(self, channels, n_class, scale_space_num, res_depth,
                 feat_root, filter_size, pool_size, activation, keep_prob,use_auxiliary_loss):
        super(MSAUNet, self).__init__()
        use_residual = True
        use_lstm = False
        self.use_spn = False
        self.use_auxiliary_loss=use_auxiliary_loss
        self.num_blocks = 1     # Number of Unet Blocks
        self.blocks = torch.nn.ModuleList()
        self.end_convs = torch.nn.ModuleList()
        for block_id in range(self.num_blocks):
            if block_id == 0:
                use_prev_coupled = False
                num_channels = channels
            else:
                num_channels = n_class
                use_prev_coupled = True
            if self.use_spn and block_id == self.num_blocks - 1:
                enable_spn = True
            else:
                enable_spn = False
            self.blocks.append(UNetBlock(use_residual, use_lstm, enable_spn,
                                         num_channels, scale_space_num,
                                         res_depth,
                                         feat_root, filter_size, pool_size,
                                         activation, use_sparse_conv=False,
                                         use_prev_coupled=use_prev_coupled,
                                         keep_prob=keep_prob))
            self.end_convs.append(layers.Conv2dBnLrnDrop(
                [4, 4, feat_root, n_class], activation=None, keep_prob=keep_prob))

    def forward(self, inp):
        inp_scale_map = OrderedDict()
        inp_scale_map[0] = inp
        binary_mask = None
        prev_dw_h_convs = None
        prev_up_h_convs = None
        logits_aux = None
        for block_id in range(self.num_blocks):
            prev_dw_h_convs, prev_up_h_convs, out =\
                self.blocks[block_id](inp, prev_dw_h_convs=prev_dw_h_convs,
                                      prev_up_h_convs=prev_up_h_convs,
                                      binary_mask=binary_mask)
            out = self.end_convs[block_id](out)
            inp = out
            if block_id == self.num_blocks - 2 and self.use_auxiliary_loss:
                logits_aux = out
        out_map = out
        logits = out_map
        return logits, logits_aux
class MSAUWrapper(torch.nn.Module):
    def __init__(self, channels=1, n_class=2, model_kwargs={}):
        super(MSAUWrapper, self).__init__()
        self.n_class = n_class
        self.channels = channels

        ### model hyper-parameters
        self.scale_space_num = model_kwargs.get("scale_space_num", 6)
        self.res_depth = model_kwargs.get("res_depth", 3)
        self.featRoot = model_kwargs.get("featRoot", 8)
        self.filter_size = model_kwargs.get("filter_size", 3)
        self.pool_size = model_kwargs.get("pool_size", 2)
        self.keep_prob = model_kwargs.get("keep_prob", 0.95)

        self.activation_name = model_kwargs.get("activation_name", "relu")
        if self.activation_name == "relu":
            self.activation = torch.nn.ReLU
        if self.activation_name == "elu":
            self.activation = torch.nn.ELU
        self.model = model_kwargs.get("model", "msau")
        self.num_scales = model_kwargs.get("num_scales", 3)
        self.final_act = model_kwargs.get("final_act", "sigmoid")
        self.use_auxiliary_loss = model_kwargs['use_auxiliary_loss']

        self.msau_net = MSAUNet(self.channels, self.n_class,
                                self.scale_space_num, self.res_depth,
                                self.featRoot, self.filter_size,
                                self.pool_size, self.activation, self.keep_prob,self.use_auxiliary_loss)

        if self.final_act == "softmax":
            self.predictor = torch.nn.Softmax(dim=1)
        elif self.final_act == "sigmoid":
            self.predictor = torch.nn.Sigmoid(dim=1)
        elif self.final_act == "identity":
            self.predictor = torch.nn.Sequential()

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, inp):
        logits, aux_logits = self.msau_net(inp)
        return self.predictor(logits), logits, aux_logits

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        pretrained_dict = torch.load(path)
        self.load_state_dict(pretrained_dict)

    def loss(self, out_grid, out_grid_aux, label_mask):
        '''
        Args:
            :param out_grid: 1xCxHxW
        '''
        # if out_grid_aux is None means not using aux loss
        # First, gather the point where label_mask != 0
        #import ipdb; ipdb.set_trace()
        label_mask_expanded = (label_mask != 0
                               ).unsqueeze(0).repeat(1, self.n_class, 1, 1)
        out_grid = out_grid[label_mask_expanded].view(1, self.n_class, -1)
        out_grid = torch.transpose(out_grid,2,1).squeeze()
        if out_grid_aux:
            out_grid_aux = out_grid_aux[label_mask_expanded].view(1, self.n_class,-1)
            out_grid_aux = torch.transpose(out_grid_aux,2,1).squeeze()
        
        label_mask = label_mask[label_mask != 0]
        loss_grid= self.criterion(out_grid, label_mask)
        if out_grid_aux:
            loss_grid_aux= self.criterion(out_grid_aux, label_mask)

        #label_mask = label_mask[label_mask != 0].view(1, -1)


        final_loss= loss_grid + 0.3*loss_grid_aux if out_grid_aux else loss_grid
        return final_loss

