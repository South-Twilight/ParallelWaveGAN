import numpy as np
import torch
import torch.nn.functional as F
import math

from parallel_wavegan.models.hifigan import DiscreteSymbolHiFiGANGenerator

class DiscreteSymbolF0Generator_MR(DiscreteSymbolHiFiGANGenerator):
    """Discrete Symbol HiFiGAN generator module with f0.
        To Extract Multi Resolution Feature (Stage 1)."""
    
    def __init__(
        self,
        in_channels=512,
        out_channels=1,
        channels=512,
        linear_channel=256,
        num_embs=100,
        num_spk_embs=128,
        spk_emb_dim=128,
        concat_spk_emb=False,
        kernel_size=7,
        upsample_scales=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        use_additional_convs=True,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        # discret token
        use_embedding_feats=False,
        use_weight_sum=False,
        layer_num=12,
        use_fix_weight=False,
        use_f0=False,
        # multi resoltuion
        rs_kernel=1,
        rs_dropout=0.1,
        rs_activation="GELU",
        use_multi_resolution=False,
        src_rs=20,
        add_rs=[40, 80],
        residual_scale=0.4,
    ):
        """Initialize HiFiGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            num_embs (int): Discrete symbol size
            num_spk_embs (int): Speaker numbers for sPkeaer ID-based embedding
            spk_emb_dim (int): Dimension of speaker embedding
            concat_spk_emb (bool): whether to concat speaker embedding to the input
            kernel_size (int): Kernel size of initial and final conv layer.
            upsample_scales (list): List of upsampling scales.
            upsample_kernel_sizes (list): List of kernel sizes for upsampling layers.
            resblock_kernel_sizes (list): List of kernel sizes for residual blocks.
            resblock_dilations (list): List of dilation list for residual blocks.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            src_rs(int): origin resoultion for input feature
            add_rs(list): resoulution will be added. (traget resolution = src_rs + add_rs)use_embedding_feats(bool): Whether to use continous embedding features from pre-trained model
            use_weight_sum(bool): Whether to use weighted sum for multi layer feats (multi layer)
            layer_num(int): Numbert of layers used (multi layer)
            use_fix_weight(bool): Whether to frozen the weight in use_weight_sum (multi_layer, Residual Cluster)
            use_f0(bool): Whether to add additioal f0
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            num_embs=num_embs,
            num_spk_embs=num_spk_embs,
            spk_emb_dim=spk_emb_dim,
            concat_spk_emb=concat_spk_emb,
            kernel_size=kernel_size,
            upsample_scales=upsample_scales,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilations=resblock_dilations,
            use_additional_convs=use_additional_convs,
            bias=bias,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            use_weight_norm=use_weight_norm,
        )
        
        self.use_f0 = use_f0
        if use_f0 is True:
            self.f0_embedding = torch.nn.Linear(
                in_features=1,
                out_features=linear_channel,
            )
        
        self.use_embedding_feats = use_embedding_feats
        
        self.use_weight_sum = use_weight_sum
        if use_weight_sum is True:
            self.layer_num = layer_num
            self.weights = torch.nn.Parameter(torch.ones(self.layer_num))
            self.use_fix_weight = use_fix_weight
            
            if use_fix_weight is True: # fix update
                self.weights = torch.nn.Parameter(torch.ones(self.layer_num), requires_grad=False)
            else: 
                self.weights = torch.nn.Parameter(torch.ones(self.layer_num))
                
            self.emb = torch.nn.ModuleList([
                torch.nn.Embedding(num_embeddings=num_embs, embedding_dim=in_channels) for _ in range(self.layer_num)
            ])
        
        self.use_multi_resolution = use_multi_resolution
        if use_multi_resolution is True:
            
            assert isinstance(add_rs, list), "add_rs is not a type of list."
            add_rs = sorted(add_rs)
            
            self.src_rs = src_rs
            self.add_rs = add_rs
        
            self.conv_encode = torch.nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size,
                1,
                padding=(kernel_size - 1) // 2,
            )
            self.downsample = torch.nn.ModuleList()
            self.upsample = torch.nn.ModuleList()
            for rs in add_rs:
                gcd_rs = math.gcd(rs, src_rs)
                self.downsample.append(
                    ConvAdapter(
                        k=rs_kernel,
                        label_rate=[src_rs // gcd_rs, rs // gcd_rs],
                        dropout=rs_dropout,
                        channels=in_channels,
                        activation=torch.nn.GELU() if rs_activation == "GELU" else torch.nn.ReLU(),
                    )
                )
                self.upsample.append(
                    ConvAdapter(
                        k=rs_kernel,
                        label_rate=[rs // gcd_rs, src_rs // gcd_rs],
                        dropout=rs_dropout,
                        channels=in_channels,
                        activation=torch.nn.GELU() if rs_activation == "GELU" else torch.nn.ReLU(),
                    )
                )
                src_rs = rs
            
            self.res_scl = math.sqrt(residual_scale)

        self.input_conv = torch.nn.Conv1d(
            in_channels + linear_channel if use_f0 is True else in_channels,
            channels,
            kernel_size,
            1,
            padding=(kernel_size - 1) // 2,
        )
        
    
    def forward(
        self, 
        c, 
        f0=None,
        store_feature=False,
    ):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor token: (B, 2, T) or (B, 1, T) or (B, L, T).
                        or for embedding feature: (B, T, C) or (B, L, T, C).
            f0 (Tensor): Input tensor (B, 1, T)
            sotre_feature: Store intermidiate embedding feature
        Returns:
            Tensor: Output tensor (B, out_channels, T').
        """
        # logging.info(f'feats({c.shape})')
        # convert idx to embedding
        if self.num_spk_embs > 0:
            assert c.size(1) == 2
            c_idx, g_idx = c.long().split(1, dim=1)
            c = self.emb(c_idx.squeeze(1)).transpose(1, 2)  # (B, C, T)
            g = self.spk_emb(g_idx[:, 0, 0])

            # integrate global embedding
            if not self.concat_spk_emb:
                c = c + g.unsqueeze(2)
            else:
                g = g.unsqueeze(1).expand(-1, c.size(1), -1)
                c = torch.cat([c, g], dim=-1)
                
        # NOTE(Yuxun): update for using pretrain model layer output as input
        if self.use_weight_sum:
            assert c.size(1) == self.layer_num # (B, L, T) or (B, L, T, C)
            if not self.use_embedding_feats:
                embedded = []
                for i, embedding_layer in enumerate(self.emb):
                # Apply the i-th embedding layer to the i-th layer of input
                    embedded.append(embedding_layer(c[:, i].long()))
                c = torch.stack(embedded, dim=1).transpose(-1, 1)
            else:
                c = c.transpose(-1, 1)
            # weights: [L,]
            if self.use_fix_weight:
                norm_weights = self.weights
            else:
                norm_weights = F.softmax(self.weights, dim=-1) 
            # logging.info(f'norm_weights({norm_weights.shape}): {norm_weights}')
            # c: (B, C, T, L) * (L,) -> (B, C, T)
            c = torch.matmul(c, norm_weights)
        elif self.use_embedding_feats is False and self.use_weight_sum is False:
            assert c.size(1) == 1
            c = self.emb(c.squeeze(1).long()).transpose(1, 2)  # (B, C, T)
            
        if self.use_multi_resolution:
            # input c as (B, C, T)
            c = self.conv_encode(c)
            x = c.transpose(-1, -2)
            # conv_adapter should input as [B, T, C]
            residual = []
            for i, down_conv in enumerate(self.downsample):
                residual.append(x)
                x = down_conv(x)[0]

            # for feat in residual:
            #     logging.info(f'residual: {feat.shape}')
            # logging.info(f'x: {x.shape}')
            
            gen_rs = self.add_rs[::-1]
            gen_rs.append(self.src_rs)
            store_dict = {}
            store_dict[gen_rs[0]] = x
            for i, up_conv in enumerate(self.upsample, start=1):
                residual_feat = residual[len(residual) - i]
                up_x = up_conv(x)[0]
                if up_x.shape[1] != residual_feat.shape[1]:
                    minlen = min(up_x.shape[1], residual_feat.shape[1])
                    up_x = up_x[:, : minlen]
                    residual_feat = residual_feat[:, : minlen]
                x = (up_x + residual_feat) * self.res_scl
                store_dict[gen_rs[i]] = x

            # for key, value in store_dict.items():
            #     logging.info(f'feat-{key}: {value.shape}')
            if store_feature:
                return store_dict
            
            c = x.transpose(-1, -2)
                
        # logging.info(f'f0({f0.shape}): {f0} ')        
        if f0 is not None and self.use_f0:
            f0 = self.f0_embedding(f0.transpose(1, 2)).transpose(1, 2)
            c = torch.cat((c, f0), dim=1)
        
        # c should input as (B, C, T)
        # logging.info(f'c: {c.shape}')
        c = self.input_conv(c)
        for i in range(self.num_upsamples):
            c = self.upsamples[i](c)
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](c)
            c = cs / self.num_blocks
        c = self.output_conv(c)
        return c
    
    
    def inference(self, c, f0=None, g=None, normalize_before=False, store_feature=False):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Input tensor token: (T, 2) or (T, 1) or (T, L).
                                        embedding feature: (T, L, C)
            f0 (Tensor): Input f0 (T,).
        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).

        """
        # logging.info(f'c: {c.shape}')
        assert not normalize_before, "No statistics are used."
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.long).to(next(self.parameters()).device)
        if g is not None:
            c = c[:, 0:1]
            c = torch.cat([c, c.new_zeros(*c.size()).fill_(g).to(c.device)], dim=1)
            
        if self.use_f0 and f0 is not None:
            f0 = f0.unsqueeze(0).unsqueeze(0)
            
        if not self.use_embedding_feats:
            if self.num_spk_embs <= 0 and not self.use_weight_sum:
                c = c[:, 0:1]
        
        c = self.forward(c.unsqueeze(0).transpose(1, 2), f0, store_feature)
        if store_feature:
            return c
        else:
            return c.squeeze(0).transpose(1, 0)


class DiscreteSymbolF0Generator_MRToken(DiscreteSymbolHiFiGANGenerator):
    """Discrete Symbol HiFiGAN generator module with f0.
        Input with multi-resolution discrete token (Stage 2)."""
    
    def __init__(
        self,
        in_channels=512,
        out_channels=1,
        channels=512,
        linear_channel=256,
        num_embs=100,
        num_spk_embs=128,
        spk_emb_dim=128,
        concat_spk_emb=False,
        kernel_size=7,
        upsample_scales=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        use_additional_convs=True,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        # discret token
        use_embedding_feats=False,
        use_weight_sum=False,
        layer_num=12,
        use_fix_weight=False,
        use_f0=False,
        # multi resoltuion
        rs_kernel=1,
        rs_dropout=0.1,
        rs_activation="GELU",
        use_multi_resolution=False,
        resolution=[20, 40, 80],
        residual_scale=0.4,
    ):
        """Initialize HiFiGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            num_embs (int): Discrete symbol size
            num_spk_embs (int): Speaker numbers for sPkeaer ID-based embedding
            spk_emb_dim (int): Dimension of speaker embedding
            concat_spk_emb (bool): whether to concat speaker embedding to the input
            kernel_size (int): Kernel size of initial and final conv layer.
            upsample_scales (list): List of upsampling scales.
            upsample_kernel_sizes (list): List of kernel sizes for upsampling layers.
            resblock_kernel_sizes (list): List of kernel sizes for residual blocks.
            resblock_dilations (list): List of dilation list for residual blocks.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            src_rs(int): origin resoultion for input feature
            add_rs(list): resoulution will be added. (traget resolution = src_rs + add_rs)
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            num_embs=num_embs,
            num_spk_embs=num_spk_embs,
            spk_emb_dim=spk_emb_dim,
            concat_spk_emb=concat_spk_emb,
            kernel_size=kernel_size,
            upsample_scales=upsample_scales,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilations=resblock_dilations,
            use_additional_convs=use_additional_convs,
            bias=bias,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            use_weight_norm=use_weight_norm,
        )
        
        self.use_f0 = use_f0
        if use_f0 is True:
            self.f0_embedding = torch.nn.Linear(
                in_features=1,
                out_features=linear_channel,
            )
        
        self.use_embedding_feats = use_embedding_feats
        
        self.use_weight_sum = use_weight_sum
        if use_weight_sum is True:
            self.layer_num = layer_num
            self.weights = torch.nn.Parameter(torch.ones(self.layer_num))
            self.use_fix_weight = use_fix_weight
            
            if use_fix_weight is True: # fix update
                self.weights = torch.nn.Parameter(torch.ones(self.layer_num), requires_grad=False)
            else: 
                self.weights = torch.nn.Parameter(torch.ones(self.layer_num))
                
            self.emb = torch.nn.ModuleList([
                torch.nn.Embedding(num_embeddings=num_embs, embedding_dim=in_channels) for _ in range(self.layer_num)
            ])
        
        self.use_multi_resolution = use_multi_resolution
        if use_multi_resolution is True:
            assert isinstance(resolution, list), "resolution is not a type of list."
            assert len(resolution) == layer_num, "layer_number is not equal to len(resolution)"
            resolution = sorted(resolution)
            
            self.resolution = resolution
            tgt_rs = resolution[0]
            
            self.resample = torch.nn.ModuleList()
            for rs in resolution:
                gcd_rs = math.gcd(rs, tgt_rs)
                self.resample.append(
                    ConvAdapter(
                        k=rs_kernel,
                        label_rate=[rs // gcd_rs, tgt_rs // gcd_rs],
                        dropout=rs_dropout,
                        channels=1,
                        activation=torch.nn.GELU() if rs_activation == "GELU" else torch.nn.ReLU(),
                    )
                )

        self.input_conv = torch.nn.Conv1d(
            in_channels + linear_channel if use_f0 is True else in_channels,
            channels,
            kernel_size,
            1,
            padding=(kernel_size - 1) // 2,
        )
        
    
    def forward(
        self, 
        c, 
        f0=None,
    ):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor token: (B, 2, T) or (B, 1, T) or (B, L, T).
                        or for embedding feature: (B, T, C) or (B, L, T, C).
                (dict): Input multi resolution token {rs (int): token (B, 1, T)}.
            f0 (Tensor): Input tensor (B, 1, T)
            sotre_feature: Store intermidiate embedding feature
        Returns:
            Tensor: Output tensor (B, out_channels, T').
        """
        # logging.info(f'feat: {c}')
        
        if self.use_multi_resolution:
            c_list = []
            for rs, resample in zip(self.resolution, self.resample):
                feat = c[rs].transpose(1, 2)
                # resample should input as [B, T, C]
                # logging.info(f'feat: {feat.dtype}')
                feat = resample(feat)[0]
                c_list.append(feat)
                # logging.info(f'{rs}: {feat.shape}')
            minlen = min(c.shape[1] for c in c_list)
            c_list = [c[:, :minlen] for c in c_list]
            c = torch.cat(c_list, dim=2).transpose(1, 2)
        
        # logging.info(f'feats({c.shape})')
        # convert idx to embedding
        if self.num_spk_embs > 0:
            assert c.size(1) == 2
            c_idx, g_idx = c.long().split(1, dim=1)
            c = self.emb(c_idx.squeeze(1)).transpose(1, 2)  # (B, C, T)
            g = self.spk_emb(g_idx[:, 0, 0])

            # integrate global embedding
            if not self.concat_spk_emb:
                c = c + g.unsqueeze(2)
            else:
                g = g.unsqueeze(1).expand(-1, c.size(1), -1)
                c = torch.cat([c, g], dim=-1)
                
        # NOTE(Yuxun): update for using pretrain model layer output as input
        if self.use_weight_sum:
            assert c.size(1) == self.layer_num # (B, L, T) or (B, L, T, C)
            if not self.use_embedding_feats:
                embedded = []
                for i, embedding_layer in enumerate(self.emb):
                # Apply the i-th embedding layer to the i-th layer of input
                    embedded.append(embedding_layer(c[:, i].long()))
                c = torch.stack(embedded, dim=1).transpose(-1, 1)
            else:
                c = c.transpose(-1, 1)
            # weights: [L,]
            if self.use_fix_weight:
                norm_weights = self.weights
            else:
                norm_weights = F.softmax(self.weights, dim=-1) 
            # logging.info(f'norm_weights({norm_weights.shape}): {norm_weights}')
            # c: (B, C, T, L) * (L,) -> (B, C, T)
            c = torch.matmul(c, norm_weights)
        elif self.use_embedding_feats is False and self.use_weight_sum is False:
            assert c.size(1) == 1
            c = self.emb(c.squeeze(1).long()).transpose(1, 2)  # (B, C, T)
                            
        # logging.info(f'f0({f0.shape}): {f0} ')        
        if f0 is not None and self.use_f0:
            f0 = self.f0_embedding(f0.transpose(1, 2)).transpose(1, 2)
            c = torch.cat((c, f0), dim=1)
        
        # c should input as (B, C, T)
        # logging.info(f'c: {c.shape}')
        c = self.input_conv(c)
        for i in range(self.num_upsamples):
            c = self.upsamples[i](c)
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](c)
            c = cs / self.num_blocks
        c = self.output_conv(c)
        return c
    
    
    def inference(self, c, f0=None, g=None, normalize_before=False,):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Input tensor token: (T, 2) or (T, 1) or (T, L).
                                        embedding feature: (T, L, C)
                                        dcit: {rs: (T, 1)}
            f0 (Tensor): Input f0 (T,).
        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).

        """
        # logging.info(f'c: {c}')
        assert not normalize_before, "No statistics are used."
        if not isinstance(c, torch.Tensor) and not isinstance(c, dict):
            c = torch.tensor(c, dtype=torch.long).to(next(self.parameters()).device)
        if g is not None:
            c = c[:, 0:1]
            c = torch.cat([c, c.new_zeros(*c.size()).fill_(g).to(c.device)], dim=1)
            
        if self.use_f0 and f0 is not None:
            f0 = f0.unsqueeze(0).unsqueeze(0)
            
        if not self.use_embedding_feats:
            if self.num_spk_embs <= 0 and not self.use_weight_sum and not self.use_multi_resolution:
                c = c[:, 0:1]
        
        if self.use_multi_resolution:
            for rs in self.resolution:
                c[rs] = c[rs].to(torch.float).unsqueeze(0).transpose(1, 2)
        else:
            c = c.unsqueeze(0).transpose(1, 2)
        c = self.forward(c, f0)
        return c.squeeze(0).transpose(1, 0)
