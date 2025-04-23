
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
    # Normalize the queries and keys, compute the similarity matrix, and extract the positive similarities
    norm_q, norm_k = F.normalize(queries, p=2, dim=1).to(device), F.normalize(
        keys, p=2, dim=1
    ).to(device)
    sim_matrix = torch.mm(norm_q, norm_k.T).to(device)
    pos_sim = torch.diag(sim_matrix).to(device)

    # Compute the loss elements in place in the matrix
    loss_matrix = torch.maximum(
        torch.zeros(b, b).to(device), sim_matrix - pos_sim.reshape(b, 1) + margin
    ).to(device)
    torch.diagonal(loss_matrix, 0).zero_().to(device)

    # Average the loss elements
    loss = torch.sum(loss_matrix) / (b * (b - 1))

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
    total_training_examples_norm = F.normalize(
        torch.cat((queries, keys), dim=0), p=2, dim=1
    ).to(device)
    sim_matrix = (
        torch.mm(total_training_examples_norm, total_training_examples_norm.T)
        / temperature
    ).to(device)

    # Mask self similarities
    mask = torch.eye(n, dtype=torch.bool).to(device)
    sim_matrix.masked_fill_(mask, -1e9)

    # Derive ground truth labels (positive pairs are (k, k+b) and (k+b, k))
    target = torch.arange(n).to(device)
    target[0:b] += b
    target[b:] -= b
    ground_truth_labels = torch.scatter(
        torch.zeros(n, n).to(device),
        dim=1,
        index=target.reshape(n, 1),
        src=torch.ones(n, n).to(device),
    ).to(device)

    # Use cross entropy loss to get SimCLR loss
    loss = F.cross_entropy(sim_matrix, ground_truth_labels, reduction="mean").to(device)

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
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
            ),
            nn.LayerNorm(patch_size * patch_size * 3),
            nn.Linear(patch_size * patch_size * 3, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_embedding = posemb_sincos_2d(
            h=img_side_length // patch_size,
            w=img_side_length // patch_size,
            dim=d_model,
        )
        self.dropout = nn.Dropout(p)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=p,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.output_ln = nn.LayerNorm(d_model)
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        ################

    def forward(self, x, return_embedding=False):
        ## TODO4: Write the forward pass for the ViT
        #################
        x = self.to_patch_embedding(x)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.output_ln(x)
        if return_embedding:
            return x.mean(dim=1)
        return self.projection_head(x.mean(dim=1))
        #################

        # return output