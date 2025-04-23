import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import math
from einops import repeat
from functools import partial
from tqdm import tqdm
import numpy as np

torch.manual_seed(42)
torch.cuda.manual_seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def exists(val):
    return val is not None


ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


############## FILL IN THE TODO PARTS BELOW ##############

######################################
############### PART 1 ###############
######################################


############## VAE ##############
class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        ## TODO: Implement __init__
        pass

        #################

    def encoder(self, x):
        """
        Maps the input to mean and variance.
        """
        ## TODO: Implement encoder
        # Both outputs should be of shape batch_size x z_dim
        pass

        #################

    def decoder(self, z):
        """
        Maps the latent code to the reconstructed output.
        """
        ## TODO: Implement decoder
        # Output should be of shape batch_size x 784
        pass

        #################

    def sample(self, mu, log_var):
        """
        Sample vectors with mean mu and standard deviation sigma using the
        reparameterization trick.
        Sample noise form a standard Gaussian and then compute the latent
        vector by adding the mean and multiplying by the standard deviation
        """
        ## TODO: Implement sample
        pass

        #################

    def forward(self, x):
        ## TODO: Implement forward
        pass

        #################
        return output, mu, log_var


##### VAE Loss Function #####
# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    ## TODO: Implement the loss function
    pass


####### Sample Images ########
def sample_images(model, num_samples):
    with torch.no_grad():
        ## TODO: Sample num_samples images from the model
        pass


######################################
############### PART 2 ###############
######################################


def cosine_schedule(t, clip_min=1e-9):
    """
    Maps the time, t, to alpha_2
    """
    ## TODO: implement function

    #################
    pass


def predict_start_from_noise(z_t, alpha2, pred_noise, clamp_min=1e-8):
    """
    Given z_t, alpha_2, and the predicted noise, compute and return the estimate of the original data.
    """
    alpha2 = right_pad_dims_to(z_t, alpha2)
    ## TODO: implement function

    #################


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        sampling_timesteps=100,
    ):
        super().__init__()

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        self.diffusion_model = model

        self.alpha2_schedule = cosine_schedule
        self.sampling_timesteps = sampling_timesteps

    def diffusion_model_predictions(self, z_t, t):
        alpha2 = self.alpha2_schedule(t)
        model_output = self.diffusion_model(z_t, alpha2)

        pred_noise = model_output
        x_start = predict_start_from_noise(
            z_t,
            alpha2,
            pred_noise,
        )

        return ModelPrediction(pred_noise, x_start)

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1.0, 0.0, self.sampling_timesteps + 1, device=device)
        times = repeat(times, "t -> b t", b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    @torch.no_grad()
    def ddim_sample(self, n_samples):
        print("DDIM sampling")
        batch, device = n_samples, next(self.diffusion_model.parameters()).device

        time_pairs = self.get_sampling_timesteps(batch, device=device)

        z_t = torch.randn(batch, 2, device=device)

        x_start = None

        intermediate_samples = []

        sample_idx = 0

        for time, time_next in tqdm(
            time_pairs, desc="sampling loop time step", total=self.sampling_timesteps
        ):
            if sample_idx % 10 == 0 or sample_idx > 90:
                intermediate_samples.append(z_t.cpu().numpy())
            sample_idx += 1

            # Compute diffusion model prediction
            model_output = self.diffusion_model_predictions(z_t, time)
            # get alpha sigma of time and next time

            alpha2 = self.alpha2_schedule(time)
            alpha2_next = self.alpha2_schedule(time_next)
            alpha2, alpha2_next = map(
                partial(right_pad_dims_to, z_t), (alpha2, alpha2_next)
            )

            ## TODO: implement function

            #################

        return z_t, np.stack(intermediate_samples, axis=1)

    def sample(self, n_samples, batch_size):
        samples = []
        intermediate_samples = []
        for _ in range(n_samples // batch_size):
            sample, intermediate_sample = self.ddim_sample(batch_size)
            print(intermediate_sample.shape)
            samples.append(sample)
            intermediate_samples.append(intermediate_sample)

        samples = torch.cat(samples, dim=0)
        # Stack intermediate samples numpy arrays
        intermediate_samples = np.concatenate(intermediate_samples, axis=0)
        return samples, intermediate_samples

    def forward(self, point_coords):
        batch, d = point_coords.shape
        device = point_coords.device

        # sample random times

        times = torch.zeros((batch,), device=device).float().uniform_(0, 1.0)
        # noise sample

        noise = torch.randn_like(point_coords)

        alpha2 = self.alpha2_schedule(times)
        alpha2 = right_pad_dims_to(point_coords, alpha2)

        ## TODO: implement function

        #################
