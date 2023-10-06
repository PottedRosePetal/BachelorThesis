import torch
from torch.nn import Module

from .common import *
from .encoders import *
from .diffusion import *
from utils.dataset import cate_to_list


class GaussianVAE(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(args.latent_dim)

        emb_dict = {
            "cls_emb_dim":args.cls_dim,
            "ang_emb_dim":args.ang_dim,
            "num_disc_angles":args.num_disc_angles,
            "batch_norm_activated":args.batch_norm,
            "extended_residual":args.extended_residual,
            "batch_norm_threshold":args.batch_norm_threshold
        }


        self.diffusion = DiffusionPoint(
            net = PointwiseNet(
                point_dim=3, 
                context_dim=args.latent_dim, 
                residual=args.residual,
                num_classes=len(cate_to_list),
                **emb_dict
            ),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode,
                cosine_arg=args.cosine_factor
            )
        )
        
    def get_loss(self, x, class_index, angles, writer=None, it=None, kl_weight=1.0):
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        batch_size, _, _ = x.size()
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        log_pz = standard_normal_logprob(z).sum(dim=1)  # (B, ), Independence assumption <- what?
        entropy = gaussian_entropy(logvar=z_sigma)      # (B, )
        loss_prior = (- log_pz - entropy).mean()        # what does this loss do?

        loss_recons = self.diffusion.get_loss(x, z, class_index, angles)

        loss = kl_weight * loss_prior + loss_recons

        if writer is not None:
            writer.add_scalar('train/loss_entropy', -entropy.mean(), it)
            writer.add_scalar('train/loss_prior', -log_pz.mean(), it)
            writer.add_scalar('train/loss_recons', loss_recons, it)

        return loss

    def sample(self, z, num_points, flexibility, category_index, angles, truncate_std=None):
        """
        Args:
            z:  Input latent, normal random samples with mean=0 std=1, (B, F)
        """
        if truncate_std is not None:
            z = truncated_normal_(z, mean=0, std=1, trunc_std=truncate_std)
        samples = self.diffusion.sample(
            num_points, 
            context=z, 
            flexibility=flexibility, 
            category_index=category_index,
            angles=angles
            )
        return samples
