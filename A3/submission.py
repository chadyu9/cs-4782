import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    """
    h: Height of the patch.
    w: Width of the patch.
    dim: The dimension of the model embeddings.
    """

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"

    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def triplet_loss(queries, keys, margin=1.0):
    """
    Inputs:
    queries (b x D): A batch of training examples.
    keys (b x D): A batch of training examples. The ith example in keys is a positive
                  example for the ith example in queries.
    margin: The margin, m, in the equation above.

    Outputs:
    The triplet loss, calculated as described above.
    """
    b = queries.shape[0]  # batch size
    device = queries.device
    n = b * 2  # total number of examples

    # TODO1: Implement triplet loss
    # Hint: Whenever you create a new tensor, make sure to send it to the same
    #       location (device) your model and data are on.
    # Hint: How might you use matrices/matrix operations to keep track of distances between
    #       positive and negative pairs? (looking ahead to the instructions in part 1.2 maybe be useful)
    #################
    loss = None

    return loss


def nt_xent_loss(queries, keys, temperature=0.1):
    """
    Inputs:
    queries (b x D): A batch of training examples.
    keys (b x D): A batch of training examples. The ith example in keys is a
                  differently-augmented view of the ith example in queries.
    temperature: The temperature, tau, in the equation above.

    Outputs:
    The SimCLR loss, calculated as described above.
    """
    b, device = queries.shape[0], queries.device
    n = b * 2

    # TODO2: Implement the SimCLR loss
    # Hint: Whenever you create a new tensor, make sure to send it to the same
    #       location (device) your model and data are on.
    # Hint: Which loss function does the first equation in step 3 remind you of?
    #################
    loss = None

    return loss


class ViT(nn.Module):
    def __init__(self, d_model, num_layers, patch_size=4, img_side_length=32, p=0.05):
        """
        Inputs:
        d_model: The dimension of the encoder embeddings.
        num_layers: Number of encoder layers.
        patch_size: Side length of the square image patches.
        img_side_length: The height and width of the images.
        p: Dropout probability.
        """
        super(ViT, self).__init__()

        d_ff = 4 * d_model
        num_heads = d_model // 32

        # TODO3: define the ViT
        #################

        ################

    def forward(self, x, return_embedding=False):

        ## TODO4: Write the forward pass for the ViT
        #################

        #################

        return output
