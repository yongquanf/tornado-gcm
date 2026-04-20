"""Tower modules — wrappers that apply NN layers over spatial fields.

In the JAX version, these use `coordax.cmap` + `nnx.vmap` to vectorize
over spatial dimensions. In PyTorch, we reshape to (batch, features) and
apply the network, then reshape back.

Zone: Z3 (BF16) — neural network computation.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn

from pytorch_src.neural import standard_layers


class ForwardTower(nn.Module):
    """Apply a neural network to input fields, vectorized over spatial dims.

    Assumes input shape: (..., features, *spatial) where the network acts
    on the `features` dimension and spatial dims are preserved.
    """

    def __init__(
        self,
        neural_net: nn.Module,
        feature_axis: int = -3,
        final_activation: Callable | None = None,
    ):
        super().__init__()
        self.neural_net = neural_net
        self.feature_axis = feature_axis
        self.final_activation = final_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bring feature axis to last, flatten spatial, apply net, reshape
        fa = self.feature_axis
        if fa < 0:
            fa = x.ndim + fa
        # Move feature axis to last
        perm = list(range(x.ndim))
        perm.pop(fa)
        perm.append(fa)
        x_t = x.permute(*perm)  # (..., *spatial, features)

        orig_shape = x_t.shape
        features = orig_shape[-1]
        spatial = orig_shape[:-1]

        x_flat = x_t.reshape(-1, features)
        param = next(self.neural_net.parameters(), None)
        if param is not None and x_flat.dtype != param.dtype:
            x_flat = x_flat.to(param.dtype)
        out_flat = self.neural_net(x_flat)
        out_features = out_flat.shape[-1]

        out = out_flat.reshape(*spatial, out_features)

        # Move features back
        inv_perm = list(range(len(perm)))
        inv_perm.insert(fa, inv_perm.pop())
        out = out.permute(*inv_perm)

        if self.final_activation is not None:
            out = self.final_activation(out)
        return out


class ColumnTower(nn.Module):
    """Apply a neural network column-wise (along the level + feature axis).

    Input: (levels, features, lon, lat) → flatten (levels*features) →
    apply net → reshape back.
    """

    def __init__(self, neural_net: nn.Module):
        super().__init__()
        self.neural_net = neural_net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (levels, features, lon, lat)
        levels, features, lon, lat = x.shape
        # Reshape to (lon*lat, levels*features)
        x_t = x.permute(2, 3, 0, 1).reshape(lon * lat, levels * features)
        param = next(self.neural_net.parameters(), None)
        if param is not None and x_t.dtype != param.dtype:
            x_t = x_t.to(param.dtype)
        out = self.neural_net(x_t)
        out_feat = out.shape[-1] // levels
        out = out.reshape(lon, lat, levels, out_feat).permute(2, 3, 0, 1)
        return out


class ColumnTransformerTower(ColumnTower):
    """ColumnTower variant accepting optional latent/positional encoding args.

    The underlying neural_net should accept (inputs, latents, positional_encoding)
    so that transformer-style cross-attention can be used within a column.
    """

    def __init__(self, neural_net: nn.Module):
        super().__init__(neural_net=neural_net)

    def forward(
        self,
        x: torch.Tensor,
        latents: Optional[torch.Tensor] = None,
        positional_encoding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply column transformer to inputs.

        Args:
            x: (levels, features, lon, lat).
            latents: optional latent array for cross-attention.
            positional_encoding: optional positional encoding.
        """
        levels, features, lon, lat = x.shape
        x_t = x.permute(2, 3, 0, 1).reshape(lon * lat, levels * features)
        param = next(self.neural_net.parameters(), None)
        if param is not None and x_t.dtype != param.dtype:
            x_t = x_t.to(param.dtype)

        if hasattr(self.neural_net, 'forward') and latents is not None:
            out = self.neural_net(x_t, latents, positional_encoding)
        else:
            out = self.neural_net(x_t)

        out_feat = out.shape[-1] // levels
        out = out.reshape(lon, lat, levels, out_feat).permute(2, 3, 0, 1)
        return out


class VerticalConvTower(nn.Module):
    """Tower that stacks 1D vertical convolutions (ConvLevel layers).

    Input shape: (channels, levels, lon, lat)
    Output shape: (output_channels, levels, lon, lat)

    Convolutions act along the 'levels' dimension; lon/lat are treated as
    independent batch dimensions via reshape.

    Args:
        input_channels: number of input channels.
        output_channels: number of output channels in the final layer.
        hidden_channels: list of intermediate channel sizes.
        kernel_size: vertical convolution kernel size.
        activation: activation function between layers.
        activate_final: whether to apply activation after the last layer.
        use_bias: whether convolution layers use bias.
        checkpoint_tower: use gradient checkpointing for memory savings.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: Sequence[int],
        kernel_size: int = 3,
        activation: Callable = torch.nn.functional.relu,
        activate_final: bool = False,
        use_bias: bool = True,
        checkpoint_tower: bool = False,
    ):
        super().__init__()
        self.activation = activation
        self.activate_final = activate_final
        self.checkpoint_tower = checkpoint_tower

        all_channels = list(hidden_channels) + [output_channels]
        layers = []
        d_in = input_channels
        for d_out in all_channels:
            layers.append(standard_layers.ConvLevel(
                d_in, d_out, kernel_size=kernel_size, use_bias=use_bias,
            ))
            d_in = d_out
        self.layers = nn.ModuleList(layers)

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution stack: x is (batch, channels, levels)."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1 or self.activate_final:
                x = self.activation(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply vertical conv tower.

        Args:
            x: (channels, levels, lon, lat)

        Returns:
            (output_channels, levels, lon, lat)
        """
        channels, levels, lon, lat = x.shape
        # Reshape: treat (lon, lat) as batch → (lon*lat, channels, levels)
        x_r = x.permute(2, 3, 0, 1).reshape(lon * lat, channels, levels)
        param = next(self.layers.parameters(), None)
        if param is not None and x_r.dtype != param.dtype:
            x_r = x_r.to(param.dtype)

        if self.checkpoint_tower:
            out = torch.utils.checkpoint.checkpoint(self._net, x_r, use_reentrant=False)
        else:
            out = self._net(x_r)

        out_c = out.shape[1]
        return out.reshape(lon, lat, out_c, levels).permute(2, 3, 0, 1)


class Conv2DTower(nn.Module):
    """2D spatial convolution tower (lon × lat).

    Input shape: (channels, levels, lon, lat)
    Output shape: (output_channels, levels, lon, lat)

    Convolutions act on the spatial (lon, lat) dims. Periodic padding is
    applied along longitude and zero padding along latitude, matching the
    JAX ConvLonLat convention.

    Args:
        input_channels: number of input channels.
        output_channels: number of output channels.
        num_hidden_units: hidden layer channel count.
        num_hidden_layers: number of hidden conv layers.
        kernel_size: (height, width) = (lon, lat) kernel.
        activation: activation between layers.
        activate_final: whether to activate the last layer.
        use_bias: whether layers use bias.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_hidden_units: int = 32,
        num_hidden_layers: int = 2,
        kernel_size: tuple[int, int] = (3, 3),
        activation: Callable = torch.nn.functional.relu,
        activate_final: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()
        self.activation = activation
        self.activate_final = activate_final
        self.kernel_size = kernel_size

        # Padding: periodic in lon (dim=-2), zero in lat (dim=-1)
        self._pad_lon = (kernel_size[0] // 2, kernel_size[0] - kernel_size[0] // 2 - 1)
        self._pad_lat = (kernel_size[1] // 2, kernel_size[1] - kernel_size[1] // 2 - 1)

        output_sizes = [num_hidden_units] * num_hidden_layers + [output_channels]
        layers = []
        d_in = input_channels
        for d_out in output_sizes:
            layers.append(nn.Conv2d(
                d_in, d_out, kernel_size=kernel_size, padding=0, bias=use_bias,
            ))
            d_in = d_out
        self.layers = nn.ModuleList(layers)

    def _pad_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Apply periodic padding in lon and zero padding in lat."""
        # x: (batch, channels, lon, lat)
        # Periodic pad along lon (dim=-2)
        x = torch.nn.functional.pad(x, (0, 0, self._pad_lon[0], self._pad_lon[1]), mode='circular')
        # Zero pad along lat (dim=-1)
        x = torch.nn.functional.pad(x, (self._pad_lat[0], self._pad_lat[1], 0, 0), mode='constant')
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D conv tower.

        Args:
            x: (channels, levels, lon, lat)

        Returns:
            (output_channels, levels, lon, lat)
        """
        channels, levels, lon, lat = x.shape
        # Treat levels as batch: (levels, channels, lon, lat)
        x_r = x.permute(1, 0, 2, 3)  # (levels, channels, lon, lat)
        param = next(self.layers.parameters(), None)
        if param is not None and x_r.dtype != param.dtype:
            x_r = x_r.to(param.dtype)

        for i, layer in enumerate(self.layers):
            x_r = self._pad_inputs(x_r)
            x_r = layer(x_r)
            if i < len(self.layers) - 1 or self.activate_final:
                x_r = self.activation(x_r)

        out_c = x_r.shape[1]
        return x_r.permute(1, 0, 2, 3)  # (output_channels, levels, lon, lat)
