import einops
import torch.nn as nn
import torch


def block_images_einops(x, patch_size):
  """Image to patches."""
  batch, height, width, channels = x.shape
  grid_height = height // patch_size[0]
  grid_width = width // patch_size[1]
  x = einops.rearrange(
      x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
      gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
  return x


def unblock_images_einops(x, grid_size, patch_size):
  """patches to images."""
  x = einops.rearrange(
      x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
      gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
  return x


class GridGatingUnit(nn.Module):
  """A SpatialGatingUnit as defined in the gMLP paper.
  The 'spatial' dim is defined as the second last.
  If applied on other dims, you should swapaxes first.
  """
  
  def __init__(self, in_channel, spatial):
        # bn_layer not used
        super(GridGatingUnit, self).__init__()
        self.LN1 = nn.LayerNorm(in_channel // 2)
        self.linear = nn.Linear(spatial, spatial)

  def forward(self, x):
    u, v = torch.chunk(x, 2, dim=-1)
    v = self.LN1(v)
    v = v.transpose(-1, -3)
    v = self.linear(v)
    v = v.transpose(-1, -3)
    return u * (v + 1.)


class GridGmlpLayer(nn.Module):
  """Grid gMLP layer that performs global mixing of tokens."""
  def __init__(self, in_channel, out_channel, grid_size, factor = 2, dropout_rate = 0.0):
      # bn_layer not used
      super(GridGmlpLayer, self).__init__()
      self.grid_size = grid_size
      self.LN1 = nn.LayerNorm(in_channel)
      self.linear1 = nn.Linear(in_channel, in_channel * factor)
      self.gelu = nn.GELU()
      self.ggu = GridGatingUnit(in_channel * factor, grid_size[0] * grid_size[1])
      self.linear2 = nn.Linear(in_channel, in_channel)
      self.linear3 = nn.Linear(in_channel, out_channel)
      self.dropout_rate = dropout_rate

  def forward(self, x, deterministic=True):
    x = x.permute(0, 2, 3, 1)
    n, h, w, num_channels = x.shape
    gh, gw = self.grid_size
    fh, fw = h // gh, w // gw
    x = block_images_einops(x, patch_size=(fh, fw))
    # gMLP1: Global (grid) mixing part, provides global grid communication.
    y = self.LN1(x)
    y = self.linear1(y)
    y = self.gelu(y)
    y = self.ggu(y)
    y = self.linear2(y)
    x = x + y
    x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
    x = self.linear3(x)
    x = x.permute(0, 3, 1, 2)
    return x


class BlockGatingUnit(nn.Module):
  """A SpatialGatingUnit as defined in the gMLP paper.
  The 'spatial' dim is defined as the **second last**.
  If applied on other dims, you should swapaxes first.
  """
  def __init__(self, in_channel, spatial):
        # bn_layer not used
        super(BlockGatingUnit, self).__init__()
        self.LN1 = nn.LayerNorm(in_channel // 2)
        self.linear = nn.Linear(spatial, spatial)

  def forward(self, x):
    u, v = torch.chunk(x, 2, dim=-1)
    v = self.LN1(v)
    v = v.transpose(-1, -2)
    v = self.linear(v)
    v = v.transpose(-1, -2)
    return u * (v + 1.)


class BlockGmlpLayer(nn.Module):
  """Block gMLP layer that performs local mixing of tokens."""
  def __init__(self, in_channel, out_channel, block_size, factor = 2, dropout_rate = 0.0):
      # bn_layer not used
      super(BlockGmlpLayer, self).__init__()
      self.block_size = block_size
      self.LN1 = nn.LayerNorm(in_channel)
      self.linear1 = nn.Linear(in_channel, in_channel * factor)
      self.gelu = nn.GELU()
      self.bgu = BlockGatingUnit(in_channel * factor, block_size[0] * block_size[1])
      self.linear2 = nn.Linear(in_channel, in_channel)
      self.linear3 = nn.Linear(in_channel, out_channel)
      self.dropout_rate = dropout_rate

  def forward(self, x, deterministic=True):
    x = x.permute(0, 2, 3, 1)
    n, h, w, num_channels = x.shape
    fh, fw = self.block_size
    gh, gw = h // fh, w // fw
    x = block_images_einops(x, patch_size=(fh, fw))
    # gMLP1: Global (grid) mixing part, provides global grid communication.
    y = self.LN1(x)
    y = self.linear1(y)
    y = self.gelu(y)
    y = self.bgu(y)
    y = self.linear2(y)
    x = x + y
    x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
    x = self.linear3(x)
    x = x.permute(0, 3, 1, 2)
    return x