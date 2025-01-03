# -*- coding: utf-8 -*-

"""HiFi-GAN Modules.

This code is based on https://github.com/jik876/hifi-gan.

"""

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
import math

from parallel_wavegan.layers import CausalConv1d, CausalConvTranspose1d
from parallel_wavegan.layers import HiFiGANResidualBlock as ResidualBlock
from parallel_wavegan.layers.duration_predictor import DurationPredictor
from parallel_wavegan.layers.length_regulator import LengthRegulator
from parallel_wavegan.layers.conv_adapter import ConvAdapter
from parallel_wavegan.utils import read_hdf5


class HiFiGANGenerator(torch.nn.Module):
    """HiFiGAN generator module."""

    def __init__(
        self,
        in_channels=80,
        out_channels=1,
        channels=512,
        kernel_size=7,
        upsample_scales=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        use_additional_convs=True,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_causal_conv=False,
        use_weight_norm=True,
    ):
        """Initialize HiFiGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            upsample_scales (list): List of upsampling scales.
            upsample_kernel_sizes (list): List of kernel sizes for upsampling layers.
            resblock_kernel_sizes (list): List of kernel sizes for residual blocks.
            resblock_dilations (list): List of dilation list for residual blocks.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_causal_conv (bool): Whether to use causal structure.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)

        # define modules
        self.num_upsamples = len(upsample_kernel_sizes)
        self.num_blocks = len(resblock_kernel_sizes)
        self.use_causal_conv = use_causal_conv
        if not use_causal_conv:
            self.input_conv = torch.nn.Conv1d(
                in_channels,
                channels,
                kernel_size,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            )
        else:
            self.input_conv = CausalConv1d(
                in_channels,
                channels,
                kernel_size,
                bias=bias,
            )
        self.upsamples = torch.nn.ModuleList()
        self.blocks = torch.nn.ModuleList()
        for i in range(len(upsample_kernel_sizes)):
            assert upsample_kernel_sizes[i] == 2 * upsample_scales[i]
            if not use_causal_conv:
                self.upsamples += [
                    torch.nn.Sequential(
                        getattr(torch.nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                        torch.nn.ConvTranspose1d(
                            channels // (2**i),
                            channels // (2 ** (i + 1)),
                            upsample_kernel_sizes[i],
                            upsample_scales[i],
                            padding=upsample_scales[i] // 2 + upsample_scales[i] % 2,
                            output_padding=upsample_scales[i] % 2,
                            bias=bias,
                        ),
                    )
                ]
            else:
                self.upsamples += [
                    torch.nn.Sequential(
                        getattr(torch.nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                        CausalConvTranspose1d(
                            channels // (2**i),
                            channels // (2 ** (i + 1)),
                            upsample_kernel_sizes[i],
                            upsample_scales[i],
                            bias=bias,
                        ),
                    )
                ]
            for j in range(len(resblock_kernel_sizes)):
                self.blocks += [
                    ResidualBlock(
                        kernel_size=resblock_kernel_sizes[j],
                        channels=channels // (2 ** (i + 1)),
                        dilations=resblock_dilations[j],
                        bias=bias,
                        use_additional_convs=use_additional_convs,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        use_causal_conv=use_causal_conv,
                    )
                ]
        if not use_causal_conv:
            self.output_conv = torch.nn.Sequential(
                # NOTE(kan-bayashi): follow official implementation but why
                #   using different slope parameter here? (0.1 vs. 0.01)
                torch.nn.LeakyReLU(),
                torch.nn.Conv1d(
                    channels // (2 ** (i + 1)),
                    out_channels,
                    kernel_size,
                    bias=bias,
                    padding=(kernel_size - 1) // 2,
                ),
                torch.nn.Tanh(),
            )
        else:
            self.output_conv = torch.nn.Sequential(
                # NOTE(kan-bayashi): follow official implementation but why
                #   using different slope parameter here? (0.1 vs. 0.01)
                torch.nn.LeakyReLU(),
                CausalConv1d(
                    channels // (2 ** (i + 1)),
                    out_channels,
                    kernel_size,
                    bias=bias,
                ),
                torch.nn.Tanh(),
            )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, in_channels, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T).

        """
        c = self.input_conv(c)
        for i in range(self.num_upsamples):
            c = self.upsamples[i](c)
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](c)
            c = cs / self.num_blocks
        c = self.output_conv(c)

        return c

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows the official implementation manner.
        https://github.com/jik876/hifi-gan/blob/master/models.py

        """

        def _reset_parameters(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def register_stats(self, stats):
        """Register stats for de-normalization as buffer.

        Args:
            stats (str): Path of statistics file (".npy" or ".h5").

        """
        assert stats.endswith(".h5") or stats.endswith(".npy")
        if stats.endswith(".h5"):
            mean = read_hdf5(stats, "mean").reshape(-1)
            scale = read_hdf5(stats, "scale").reshape(-1)
        else:
            mean = np.load(stats)[0].reshape(-1)
            scale = np.load(stats)[1].reshape(-1)
        self.register_buffer("mean", torch.from_numpy(mean).float())
        self.register_buffer("scale", torch.from_numpy(scale).float())
        logging.info("Successfully registered stats as buffer.")

    def inference(self, c, normalize_before=False):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, in_channels).
            normalize_before (bool): Whether to perform normalization.

        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).

        """
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
        if normalize_before:
            c = (c - self.mean) / self.scale
        c = self.forward(c.transpose(1, 0).unsqueeze(0))
        return c.squeeze(0).transpose(1, 0)


class HiFiGANPeriodDiscriminator(torch.nn.Module):
    """HiFiGAN period discriminator module."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        period=3,
        kernel_sizes=[5, 3],
        channels=32,
        downsample_scales=[3, 3, 3, 3, 1],
        max_downsample_channels=1024,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        use_spectral_norm=False,
    ):
        """Initialize HiFiGANPeriodDiscriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            period (int): Period.
            kernel_sizes (list): Kernel sizes of initial conv layers and the final conv layer.
            channels (int): Number of initial channels.
            downsample_scales (list): List of downsampling scales.
            max_downsample_channels (int): Number of maximum downsampling channels.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_spectral_norm (bool): Whether to use spectral norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1, "Kernel size must be odd number."
        assert kernel_sizes[1] % 2 == 1, "Kernel size must be odd number."

        self.period = period
        self.convs = torch.nn.ModuleList()
        in_chs = in_channels
        out_chs = channels
        for downsample_scale in downsample_scales:
            self.convs += [
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_chs,
                        out_chs,
                        (kernel_sizes[0], 1),
                        (downsample_scale, 1),
                        padding=((kernel_sizes[0] - 1) // 2, 0),
                    ),
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                )
            ]
            in_chs = out_chs
            # NOTE(kan-bayashi): Use downsample_scale + 1?
            out_chs = min(out_chs * 4, max_downsample_channels)
        self.output_conv = torch.nn.Conv2d(
            out_chs,
            out_channels,
            (kernel_sizes[1] - 1, 1),
            1,
            padding=((kernel_sizes[1] - 1) // 2, 0),
        )

        if use_weight_norm and use_spectral_norm:
            raise ValueError("Either use use_weight_norm or use_spectral_norm.")

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # apply spectral norm
        if use_spectral_norm:
            self.apply_spectral_norm()

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, in_channels, T).

        Returns:
            list: List of each layer's tensors.

        """
        # transform 1d to 2d -> (B, C, T/P, P)
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)

        # forward conv
        outs = []
        for layer in self.convs:
            x = layer(x)
            outs += [x]
        x = self.output_conv(x)
        x = torch.flatten(x, 1, -1)
        outs += [x]

        return outs

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def apply_spectral_norm(self):
        """Apply spectral normalization module from all of the layers."""

        def _apply_spectral_norm(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.spectral_norm(m)
                logging.debug(f"Spectral norm is applied to {m}.")

        self.apply(_apply_spectral_norm)


class HiFiGANMultiPeriodDiscriminator(torch.nn.Module):
    """HiFiGAN multi-period discriminator module."""

    def __init__(
        self,
        periods=[2, 3, 5, 7, 11],
        discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 32,
            "downsample_scales": [3, 3, 3, 3, 1],
            "max_downsample_channels": 1024,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
    ):
        """Initialize HiFiGANMultiPeriodDiscriminator module.

        Args:
            periods (list): List of periods.
            discriminator_params (dict): Parameters for hifi-gan period discriminator module.
                The period parameter will be overwritten.

        """
        super().__init__()
        self.discriminators = torch.nn.ModuleList()
        for period in periods:
            params = copy.deepcopy(discriminator_params)
            params["period"] = period
            self.discriminators += [HiFiGANPeriodDiscriminator(**params)]

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.

        """
        outs = []
        for f in self.discriminators:
            outs += [f(x)]

        return outs


class HiFiGANScaleDiscriminator(torch.nn.Module):
    """HiFi-GAN scale discriminator module."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_sizes=[15, 41, 5, 3],
        channels=128,
        max_downsample_channels=1024,
        max_groups=16,
        bias=True,
        downsample_scales=[2, 2, 4, 4, 1],
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        use_spectral_norm=False,
    ):
        """Initilize HiFiGAN scale discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_sizes (list): List of four kernel sizes. The first will be used for the first conv layer,
                and the second is for downsampling part, and the remaining two are for output layers.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_spectral_norm (bool): Whether to use spectral norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()
        self.layers = torch.nn.ModuleList()

        # check kernel size is valid
        assert len(kernel_sizes) == 4
        for ks in kernel_sizes:
            assert ks % 2 == 1

        # add first layer
        self.layers += [
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels,
                    channels,
                    # NOTE(kan-bayashi): Use always the same kernel size
                    kernel_sizes[0],
                    bias=bias,
                    padding=(kernel_sizes[0] - 1) // 2,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]

        # add downsample layers
        in_chs = channels
        out_chs = channels
        # NOTE(kan-bayashi): Remove hard coding?
        groups = 4
        for downsample_scale in downsample_scales:
            self.layers += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chs,
                        out_chs,
                        kernel_size=kernel_sizes[1],
                        stride=downsample_scale,
                        padding=(kernel_sizes[1] - 1) // 2,
                        groups=groups,
                        bias=bias,
                    ),
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                )
            ]
            in_chs = out_chs
            # NOTE(kan-bayashi): Remove hard coding?
            out_chs = min(in_chs * 2, max_downsample_channels)
            # NOTE(kan-bayashi): Remove hard coding?
            groups = min(groups * 4, max_groups)

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_sizes[2],
                    stride=1,
                    padding=(kernel_sizes[2] - 1) // 2,
                    bias=bias,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]
        self.layers += [
            torch.nn.Conv1d(
                out_chs,
                out_channels,
                kernel_size=kernel_sizes[3],
                stride=1,
                padding=(kernel_sizes[3] - 1) // 2,
                bias=bias,
            ),
        ]

        if use_weight_norm and use_spectral_norm:
            raise ValueError("Either use use_weight_norm or use_spectral_norm.")

        # apply weight norm
        self.use_weight_norm = use_weight_norm
        if use_weight_norm:
            self.apply_weight_norm()

        # apply spectral norm
        self.use_spectral_norm = use_spectral_norm
        if use_spectral_norm:
            self.apply_spectral_norm()

        # backward compatibility
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of output tensors of each layer.

        """
        outs = []
        for f in self.layers:
            x = f(x)
            outs += [x]

        return outs

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def apply_spectral_norm(self):
        """Apply spectral normalization module from all of the layers."""

        def _apply_spectral_norm(m):
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.utils.spectral_norm(m)
                logging.debug(f"Spectral norm is applied to {m}.")

        self.apply(_apply_spectral_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def remove_spectral_norm(self):
        """Remove spectral normalization module from all of the layers."""

        def _remove_spectral_norm(m):
            try:
                logging.debug(f"Spectral norm is removed from {m}.")
                torch.nn.utils.remove_spectral_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_spectral_norm)

    def _load_state_dict_pre_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Fix the compatibility of weight / spectral normalization issue.

        Some pretrained models are trained with configs that use weight / spectral
        normalization, but actually, the norm is not applied. This causes the mismatch
        of the parameters with configs. To solve this issue, when parameter mismatch
        happens in loading pretrained model, we remove the norm from the current model.

        See also:
            - https://github.com/kan-bayashi/ParallelWaveGAN/pull/409
            - https://github.com/espnet/espnet/pull/5240

        """
        current_module_keys = [x for x in state_dict.keys() if x.startswith(prefix)]
        if self.use_weight_norm and not any(
            ["weight_g" in k for k in current_module_keys]
        ):
            logging.warning(
                "It seems weight norm is not applied in the pretrained model but the"
                " current model uses it. To keep the compatibility, we remove the norm"
                " from the current model. This may causes training error due to the the"
                " parameter mismatch when finetuning. To avoid this issue, please"
                " change the following parameters in config to false: \n"
                " - discriminator_params.follow_official_norm \n"
                " - discriminator_params.scale_discriminator_params.use_weight_norm \n"
                " - discriminator_params.scale_discriminator_params.use_spectral_norm \n"
                " See also: https://github.com/kan-bayashi/ParallelWaveGAN/pull/409"
            )
            self.remove_weight_norm()
            self.use_weight_norm = False

        if self.use_spectral_norm and not any(
            ["weight_u" in k for k in current_module_keys]
        ):
            logging.warning(
                "It seems spectral norm is not applied in the pretrained model but the"
                " current model uses it. To keep the compatibility, we remove the norm"
                " from the current model. This may causes training error due to the the"
                " parameter mismatch when finetuning. To avoid this issue, please"
                " change the following parameters in config to false: \n"
                " - discriminator_params.follow_official_norm \n"
                " - discriminator_params.scale_discriminator_params.use_weight_norm \n"
                " - discriminator_params.scale_discriminator_params.use_spectral_norm \n"
                " See also: https://github.com/kan-bayashi/ParallelWaveGAN/pull/409"
            )
            self.remove_spectral_norm()
            self.use_spectral_norm = False


class HiFiGANMultiScaleDiscriminator(torch.nn.Module):
    """HiFi-GAN multi-scale discriminator module."""

    def __init__(
        self,
        scales=3,
        downsample_pooling="AvgPool1d",
        # follow the official implementation setting
        downsample_pooling_params={
            "kernel_size": 4,
            "stride": 2,
            "padding": 2,
        },
        discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [15, 41, 5, 3],
            "channels": 128,
            "max_downsample_channels": 1024,
            "max_groups": 16,
            "bias": True,
            "downsample_scales": [2, 2, 4, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
        },
        follow_official_norm=False,
    ):
        """Initilize HiFiGAN multi-scale discriminator module.

        Args:
            scales (int): Number of multi-scales.
            downsample_pooling (str): Pooling module name for downsampling of the inputs.
            downsample_pooling_params (dict): Parameters for the above pooling module.
            discriminator_params (dict): Parameters for hifi-gan scale discriminator module.
            follow_official_norm (bool): Whether to follow the norm setting of the official
                implementaion. The first discriminator uses spectral norm and the other
                discriminators use weight norm.

        """
        super().__init__()
        self.discriminators = torch.nn.ModuleList()

        # add discriminators
        for i in range(scales):
            params = copy.deepcopy(discriminator_params)
            if follow_official_norm:
                if i == 0:
                    params["use_weight_norm"] = False
                    params["use_spectral_norm"] = True
                else:
                    params["use_weight_norm"] = True
                    params["use_spectral_norm"] = False
            self.discriminators += [HiFiGANScaleDiscriminator(**params)]
        self.pooling = getattr(torch.nn, downsample_pooling)(
            **downsample_pooling_params
        )

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.

        """
        outs = []
        for f in self.discriminators:
            outs += [f(x)]
            x = self.pooling(x)

        return outs


class HiFiGANMultiScaleMultiPeriodDiscriminator(torch.nn.Module):
    """HiFi-GAN multi-scale + multi-period discriminator module."""

    def __init__(
        self,
        # Multi-scale discriminator related
        scales=3,
        scale_downsample_pooling="AvgPool1d",
        scale_downsample_pooling_params={
            "kernel_size": 4,
            "stride": 2,
            "padding": 2,
        },
        scale_discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [15, 41, 5, 3],
            "channels": 128,
            "max_downsample_channels": 1024,
            "max_groups": 16,
            "bias": True,
            "downsample_scales": [2, 2, 4, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
        },
        follow_official_norm=True,
        # Multi-period discriminator related
        periods=[2, 3, 5, 7, 11],
        period_discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 32,
            "downsample_scales": [3, 3, 3, 3, 1],
            "max_downsample_channels": 1024,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
    ):
        """Initilize HiFiGAN multi-scale + multi-period discriminator module.

        Args:
            scales (int): Number of multi-scales.
            scale_downsample_pooling (str): Pooling module name for downsampling of the inputs.
            scale_downsample_pooling_params (dict): Parameters for the above pooling module.
            scale_discriminator_params (dict): Parameters for hifi-gan scale discriminator module.
            follow_official_norm (bool): Whether to follow the norm setting of the official
                implementaion. The first discriminator uses spectral norm and the other
                discriminators use weight norm.
            periods (list): List of periods.
            period_discriminator_params (dict): Parameters for hifi-gan period discriminator module.
                The period parameter will be overwritten.

        """
        super().__init__()
        self.msd = HiFiGANMultiScaleDiscriminator(
            scales=scales,
            downsample_pooling=scale_downsample_pooling,
            downsample_pooling_params=scale_downsample_pooling_params,
            discriminator_params=scale_discriminator_params,
            follow_official_norm=follow_official_norm,
        )
        self.mpd = HiFiGANMultiPeriodDiscriminator(
            periods=periods,
            discriminator_params=period_discriminator_params,
        )

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of list of each discriminator outputs,
                which consists of each layer output tensors.
                Multi scale and multi period ones are concatenated.

        """
        msd_outs = self.msd(x)
        mpd_outs = self.mpd(x)
        return msd_outs + mpd_outs


class DiscreteSymbolHiFiGANGenerator(torch.nn.Module):
    """Discrete Symbol HiFiGAN generator module."""

    def __init__(
        self,
        in_channels=512,
        out_channels=1,
        channels=512,
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

        """
        super().__init__()
        self.num_spk_embs = num_spk_embs

        # define id embedding
        self.emb = torch.nn.Embedding(
            num_embeddings=num_embs, embedding_dim=in_channels
        )
        if self.num_spk_embs > 0:
            self.spk_emb = torch.nn.Embedding(
                num_embeddings=num_spk_embs, embedding_dim=spk_emb_dim
            )
            self.concat_spk_emb = concat_spk_emb
            if not concat_spk_emb:
                assert in_channels == spk_emb_dim
            else:
                in_channels = in_channels + spk_emb_dim

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernal size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)

        # define modules
        self.num_upsamples = len(upsample_kernel_sizes)
        self.num_blocks = len(resblock_kernel_sizes)
        self.input_conv = torch.nn.Conv1d(
            in_channels,
            channels,
            kernel_size,
            1,
            padding=(kernel_size - 1) // 2,
        )
        self.upsamples = torch.nn.ModuleList()
        self.blocks = torch.nn.ModuleList()
        for i in range(len(upsample_kernel_sizes)):
            self.upsamples += [
                torch.nn.Sequential(
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                    torch.nn.ConvTranspose1d(
                        channels // (2**i),
                        channels // (2 ** (i + 1)),
                        upsample_kernel_sizes[i],
                        upsample_scales[i],
                        padding=(upsample_kernel_sizes[i] - upsample_scales[i]) // 2,
                    ),
                )
            ]
            for j in range(len(resblock_kernel_sizes)):
                self.blocks += [
                    ResidualBlock(
                        kernel_size=resblock_kernel_sizes[j],
                        channels=channels // (2 ** (i + 1)),
                        dilations=resblock_dilations[j],
                        bias=bias,
                        use_additional_convs=use_additional_convs,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                    )
                ]
        self.output_conv = torch.nn.Sequential(
            # NOTE(kan-bayashi): follow official implementation but why
            #   using different slope parameter here? (0.1 vs. 0.01)
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(
                channels // (2 ** (i + 1)),
                out_channels,
                kernel_size,
                1,
                padding=(kernel_size - 1) // 2,
            ),
            torch.nn.Tanh(),
        )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, 2, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T).

        """
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
        else:
            assert c.size(1) == 1
            c = self.emb(c.squeeze(1).long()).transpose(1, 2)  # (B, C, T)

        c = self.input_conv(c)
        for i in range(self.num_upsamples):
            c = self.upsamples[i](c)
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](c)
            c = cs / self.num_blocks
        c = self.output_conv(c)

        return c

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows the official implementation manner.
        https://github.com/jik876/hifi-gan/blob/master/models.py

        """

        def _reset_parameters(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def inference(self, c, g=None, normalize_before=False):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, 2).

        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).

        """
        assert not normalize_before, "No statistics are used."
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.long).to(next(self.parameters()).device)
        if g is not None:
            c = c[:, 0:1]
            c = torch.cat([c, c.new_zeros(*c.size()).fill_(g).to(c.device)], dim=1)
        if self.num_spk_embs <= 0:
            c = c[:, 0:1]
        c = self.forward(c.transpose(1, 0).unsqueeze(0))
        return c.squeeze(0).transpose(1, 0)


class DiscreteSymbolDurationGenerator(DiscreteSymbolHiFiGANGenerator):
    """Discrete Symbol HiFiGAN generator with duration predictor module."""

    def __init__(
        self,
        in_channels=512,
        out_channels=1,
        channels=512,
        num_embs=100,
        num_spk_embs=128,
        spk_emb_dim=128,
        concat_spk_emb=False,
        duration_layers=2,
        duration_chans=384,
        duration_kernel_size=3,
        duration_offset=1.0,
        duration_dropout_rate=0.5,
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
    ):
        """Initialize DiscreteSymbolDurationGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            num_embs (int): Discrete symbol size
            num_spk_embs (int): Speaker numbers for sPkeaer ID-based embedding
            spk_emb_dim (int): Dimension of speaker embedding
            concat_spk_emb (bool): whether to concat speaker embedding to the input
            duration_layers (int): number of duration predictor layers
            duration_chans (int): number of duration predictor channels
            duration_kernel_size (int): kernel size for the duration predictor
            duration_offset (float): duration predictor offset
            duration_dropout_rate (float): duration predictor dropout rate
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

        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            num_embs=num_embs + 1,  # for padding case
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

        if self.num_spk_embs > 0:
            in_channels = in_channels + spk_emb_dim

        self.duration_predictor = DurationPredictor(
            in_channels,
            n_layers=duration_layers,
            n_chans=duration_chans,
            kernel_size=duration_kernel_size,
            dropout_rate=duration_dropout_rate,
            offset=duration_offset,
        )

        self.length_regulator = LengthRegulator()

    def forward(self, c, ds):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, 2, T). or (B, 1, T)
            ds (Tensor): Input tensor (B, T)

        Returns:
            Tensor: Output tensor (B, out_channels, T').
        """
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
        else:
            assert c.size(1) == 1
            c = self.emb(c.squeeze(1).long()).transpose(1, 2)  # (B, C, T)

        ds_out = self.duration_predictor(c.transpose(1, 2))
        c = self.length_regulator(c.transpose(1, 2), ds).transpose(1, 2)

        c = self.input_conv(c)
        for i in range(self.num_upsamples):
            c = self.upsamples[i](c)
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](c)
            c = cs / self.num_blocks
        c = self.output_conv(c)

        return c, ds_out

    def inference(self, c, g=None, ds=None, normalize_before=False):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, 2).

        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).

        """
        assert not normalize_before, "No statistics are used."
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.long).to(next(self.parameters()).device)
        if g is not None:
            c = c[:, 0:1]
            c = torch.cat([c, c.new_zeros(*c.size()).fill_(g).to(c.device)], dim=1)
        if self.num_spk_embs <= 0:
            c = c[:, 0:1]

        if ds is None:
            c, _ = self.synthesis(c.transpose(1, 0).unsqueeze(0))
        else:
            c, _ = self.forward(c.transpose(1, 0).unsqueeze(0), ds.unsqueeze(0))
        return c.squeeze(0).transpose(1, 0)

    def synthesis(self, c):
        """Synthesis with duration prediction.

        Args:
            c (Tensor): Input tensor (B, 2, T) or (B, 1, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T').

        """
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
        else:
            assert c.size(1) == 1
            c = self.emb(c.squeeze(1).long()).transpose(1, 2)  # (B, C, T)

        ds_out = self.duration_predictor.inference(c.transpose(1, 2))
        c = self.length_regulator(c.transpose(1, 2), ds_out).transpose(1, 2)

        c = self.input_conv(c)
        for i in range(self.num_upsamples):
            c = self.upsamples[i](c)
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](c)
            c = cs / self.num_blocks
        c = self.output_conv(c)

        return c, ds_out


class DiscreteSymbolF0Generator(DiscreteSymbolHiFiGANGenerator):
    """Discrete Symbol HiFiGAN generator module with f0."""
    
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
        use_f0=True,
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
            use_embedding_feats(bool): Whether to use continous embedding features from pre-trained model
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

        self.input_conv = torch.nn.Conv1d(
            in_channels + linear_channel if use_f0 is True else in_channels,
            channels,
            kernel_size,
            1,
            padding=(kernel_size - 1) // 2,
        )

        self.use_embedding_feats = use_embedding_feats
        
    
    def forward(self, c, f0=None, store_feature=False):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor token: (B, 2, T). or (B, 1, T) or (B, L, T)
                        or for embedding feature: (B, T, C) or (B, L, T, C)
            f0 (Tensor): Input tensor (B, 1, T)
            store_feature (Boolean): Whether to store embedding feature 
        Returns:
            Tensor: Output tensor (B, out_channels, T').
        """
        # convert idx to embedding
        if self.num_spk_embs > 0:
            # NOTE(Yuxun): Only for single layer token
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
        else:
            # NOTE(Yuxun): update for using pretrain model layer output as input
            if self.use_weight_sum:
                assert c.size(1) == self.layer_num # (B, L, T) or (B, L, T, C)
                if not self.use_embedding_feats:
                    embedded = []
                    for i, embedding_layer in enumerate(self.emb):
                    # Apply the i-th embedding layer to the i-th layer of input
                        embedded.append(embedding_layer(c[:, i].long()))
                    c = torch.stack(embedded, dim=1)
                    c = c.transpose(-1, 1)
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
                
            elif self.use_embedding_feats is False:
                assert c.size(1) == 1
                c = self.emb(c.squeeze(1).long()).transpose(1, 2)  # (B, C, T)
        
        # NOTE(Yuxun): c shoulde reshape as (B, T, C)
        if store_feature:
            return c
            
        if f0 is not None and self.use_f0:
            f0 = self.f0_embedding(f0.transpose(1, 2)).transpose(1, 2)
            c = torch.cat((c, f0), dim=1)
        
        # c should input as (B, C, T)
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
            c (Union[Tensor, ndarray]): Input tensor token: (T, 2) or (T, 1) or (T, L). "frame" token in multi layer
                                        embedding feature: (T, L, C)
            f0 (Tensor): Input f0 (T,).
        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).

        """
        # logging.info(f'hifigan_infer: c({c.shape}): {c}')
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
    
    
class DiscreteMRSymbolF0Generator(DiscreteSymbolF0Generator):
    """Discrete Multi Resolution Symbol HiFiGAN generator module with f0."""
    
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
            use_embedding_feats(bool): Whether to use continous embedding features from pre-trained model
            use_weight_sum(bool): Whether to use weighted sum for multi layer feats (multi layer)
            layer_num(int): Numbert of layers used (multi layer)
            use_fix_weight(bool): Whether to frozen the weight in use_weight_sum (multi_layer, Residual Cluster)
            use_f0(bool): Whether to add additioal f0
            src_rs(int): Resolution for input token
            add_rs(list): Additional resolution for used token. (traget resolution = src_rs + add_rs)
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
            use_embedding_feats=use_embedding_feats,
            use_weight_sum=use_weight_sum,
            layer_num=layer_num,
            use_fix_weight=use_fix_weight,
            use_f0=use_f0,
        )
        
        self.use_multi_resolution = use_multi_resolution
        if use_multi_resolution is True:
            assert isinstance(add_rs, list), "add_rs is not a type of list."
            resolution = [src_rs]
            resolution.extend(add_rs)
            add_rs = sorted(add_rs)
            resolution = sorted(resolution)
            
            if use_embedding_feats is True:
                #########################################################################
                #         Input: Single resolution continous embedding features         # 
                #########################################################################
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
            else:
                ###########################################################
                #         Input: Multi resolution discrete tokens         # 
                ###########################################################
                assert len(resolution) == layer_num, "layer_number is not equal to len(resolution)"      
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
            """ Parameters of ConvAdapter in multi resolution referred as 
                https://github.com/facebookresearch/fairseq/blob/5aaabf69187c7c1e6913e53ed17a4c92f74b234c/fairseq/models/multires_hubert/multires_hubert.py#L934
            """

    
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
        if self.use_multi_resolution and not self.use_embedding_feats:
        ###########################################################
        #         Input: Multi resolution discrete tokens         # 
        ###########################################################
            c_list = []
            for rs, resample in zip(self.resolution, self.resample):
                feat = c[rs].transpose(1, 2)
                # resample should input as [B, T, C]
                feat = resample(feat)[0]
                c_list.append(feat)
            minlen = min(c.shape[1] for c in c_list)
            c_list = [c[:, :minlen] for c in c_list]
            c = torch.cat(c_list, dim=2).transpose(1, 2)
            
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
                
        if self.use_weight_sum:
        # weighted sum for multi layer continuous features / discrete tokens
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
            c = torch.matmul(c, norm_weights) # c: (B, C, T, L) * (L,) -> (B, C, T)
        elif self.use_embedding_feats is False and self.use_weight_sum is False:
            assert c.size(1) == 1
            c = self.emb(c.squeeze(1).long()).transpose(1, 2)  # (B, C, T)
        
        if self.use_multi_resolution and self.use_embedding_feats:
        #########################################################################
        #         Input: Single resolution continous embedding features         # 
        #########################################################################
            # input c as (B, C, T)
            c = self.conv_encode(c)
            x = c.transpose(-1, -2)
            # conv_adapter should input as [B, T, C]
            residual = []
            for i, down_conv in enumerate(self.downsample):
                residual.append(x)
                x = down_conv(x)[0]
                            
            gen_rs = self.add_rs[::-1]
            gen_rs.append(self.src_rs)
            store_dict = {}
            store_dict[gen_rs[0]] = x
            
            for i, up_conv in enumerate(self.upsample, start=1):
                residual_feat = residual[len(residual) - i]
                up_x = up_conv(x)[0]
                
                # align origin feats and resmpale feats
                if up_x.shape[1] != residual_feat.shape[1]:
                    minlen = min(up_x.shape[1], residual_feat.shape[1])
                    up_x = up_x[:, : minlen]
                    residual_feat = residual_feat[:, : minlen]
                    
                x = (up_x + residual_feat) * self.res_scl
                store_dict[gen_rs[i]] = x

            if store_feature:
                return store_dict
            
            c = x.transpose(-1, -2)

        # Follows are Unit HiFiGAN part
        
        if f0 is not None and self.use_f0:
            f0 = self.f0_embedding(f0.transpose(1, 2)).transpose(1, 2)
            c = torch.cat((c, f0), dim=1)
        
        # c should input as (B, C, T)
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
            store_feature (bool): Whether to store intermediate token 
        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).

        """
        assert not normalize_before, "No statistics are used."
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.long).to(next(self.parameters()).device)
        if g is not None:
            c = c[:, 0:1]
            c = torch.cat([c, c.new_zeros(*c.size()).fill_(g).to(c.device)], dim=1)
            
        if self.use_f0 and f0 is not None:
            f0 = f0.unsqueeze(0).unsqueeze(0)
            
        if not self.use_embedding_feats:
            if self.use_multi_resolution:
                for rs in self.resolution:
                    c[rs] = c[rs].to(torch.float).unsqueeze(0).transpose(1, 2)
            elif self.num_spk_embs <= 0 and not self.use_weight_sum:
                c = c[:, 0:1]
        
        if not (not self.use_embedding_feats and self.use_multi_resolution):
            c = c.unsqueeze(0).transpose(1, 2)
        
        c = self.forward(c, f0, store_feature)
        if store_feature:
            return c
        else:
            return c.squeeze(0).transpose(1, 0)