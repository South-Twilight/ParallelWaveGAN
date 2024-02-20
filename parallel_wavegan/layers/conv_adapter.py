# origin link: https://github.com/facebookresearch/fairseq/blob/5aaabf69187c7c1e6913e53ed17a4c92f74b234c/fairseq/models/multires_hubert/multires_hubert.py#L934
import torch
import torch.nn as nn


class ConvAdapter(nn.Module):
    """Conv adapter that combines two modules with different label rate with downsample or upsample.
    To allow different ratios than integer, two convs are utilized with first to upsample (numerator)
    and the second to downsample (denominator)"""

    def __init__(
        self,
        k,
        label_rate,
        dropout,
        channels,
        activation,
        log_compression=False,
        skip_connections=True,
        highway=True,
        residual_scale=0.4,
        non_affine_group_norm=False,
    ):
        super().__init__()

        def downsample_block(channel, k, stride):
            return nn.Sequential(
                # with padding (k - 1) // 2 to keep the same size
                nn.Conv1d(
                    channel,
                    channel,
                    k,
                    stride=stride,
                    bias=False,
                    padding=(k - 1) // 2,
                ),
                nn.Dropout(p=dropout),
                norm_block(
                    is_layer_norm=False, dim=channel, affine=not non_affine_group_norm
                ),
                activation,
            )

        def upsample_block(channel, k, stride):
            return nn.Sequential(
                # with padding (k - 1) // 2 to keep the same size
                nn.ConvTranspose1d(
                    channel,
                    channel,
                    k,
                    stride=stride,
                    bias=False,
                    padding=0,  # padding=(k - 1) // 2,
                    output_padding=(stride - 1),
                ),
                nn.Dropout(p=dropout),
                norm_block(
                    is_layer_norm=False, dim=channel, affine=not non_affine_group_norm
                ),
                activation,
            )

        assert len(label_rate) == 2, "label_rate should be sized two to apply fusion"
        # Lout =(Lin~H~R1)~Wstride~H~R2~Wpadding+dilation~W(kernel_size~H~R1)+output_padding+1
        self.upsample_conv = upsample_block(channels, k, label_rate[0])
        self.downsample_conv = downsample_block(channels, k, label_rate[1])

        self.upsample_rate, self.downsample_rate = label_rate
        self.log_compression = log_compression
        self.skip_connections = skip_connections
        self.highway = highway
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x, padding=None, mask_indices=None):
        # Assume x1 = (B, T, C) as input
        x = x.permute(0, 2, 1)
        residual_before_upsample = x
        x = self.upsample_conv(x)
        upsample_size = x.size(2)

        # conduct upsample
        if self.skip_connections:
            residual_upsample = torch.repeat_interleave(
                residual_before_upsample, self.upsample_rate, dim=2
            )
            upsample_size = min(upsample_size, residual_upsample.size(2))
            x = (
                x[..., :upsample_size] + residual_upsample[..., :upsample_size]
            ) * self.residual_scale

        residual_before_downsample = x
        x = self.downsample_conv(x)
        downsample_size = x.size(2)

        if self.skip_connections:
            residual_downsample = residual_before_downsample[
                ..., :: self.downsample_rate
            ]
            downsample_size = min(x.size(2), residual_downsample.size(2))
            x = (
                x[..., :downsample_size] + residual_downsample[..., :downsample_size]
            ) * self.residual_scale

        if self.highway:
            residual_after_sample = residual_upsample[..., :: self.downsample_rate]
            final_size = min(x.size(2), residual_after_sample.size(2))
            x = (
                x[..., :final_size] + residual_after_sample[..., :final_size]
            ) * self.residual_scale

        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()

        x = x.permute(0, 2, 1)

        # process padding
        if padding is not None:
            padding = torch.repeat_interleave(padding, self.upsample_rate, dim=1)
            padding = padding[..., :: self.downsample_rate]
            padding = padding[..., : x.size(1)]

        # process mask indices
        if mask_indices is not None:
            mask_indices = torch.repeat_interleave(
                mask_indices, self.upsample_rate, dim=1
            )
            mask_indices = mask_indices[..., :: self.downsample_rate]
            mask_indices = mask_indices[..., : x.size(1)]
        return x, padding, mask_indices

