import torch
from torch.nn import Module, ModuleList, Embedding
import numpy as np

from models.unet_blocks import ConcatSquashLinear


class VarianceSchedule(Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear', cosine_arg=0.008):
        super().__init__()
        assert mode in ('linear', "cosine")
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == "linear":
            # Linear schedule from Ho et al, extended to work for any number of
            # diffusion steps.
            betas = np.linspace(beta_1, beta_T, num_steps)
        elif mode == "cosine":
            # Cosine schedule from Alex Nichol & Prafulla Dhariwal, https://arxiv.org/pdf/2102.09672.pdf, 
            # https://github.com/openai/improved-diffusion/blob/783b6740edb79fdb7d063250db2c51cc9545dcd1/improved_diffusion/gaussian_diffusion.py#L18
            # Also identical to https://github.com/openai/point-e/blob/main/point_e/diffusion/gaussian_diffusion.py
            betas = self.betas_for_alpha_bar(
                num_steps,
                lambda t: np.cos((t + cosine_arg) / (1+cosine_arg) * np.pi / 2) ** 2,
            )
        else:
            raise NotImplementedError(f"unknown beta schedule: {mode}")
        
        

        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        betas = torch.from_numpy(betas).float()

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, axis=0)
        
        # log_alphas = torch.log(alphas)
        # for i in range(1, log_alphas.size(0)):  # 1 to T
        #     log_alphas[i] += log_alphas[i - 1]
        # alpha_bars_0 = log_alphas.exp()

        # above implementation might be more numerically stable. However, way slower 
        # for large applications. 

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas
    
    def betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        """
        New cosine beta-schedule from OpenAI

        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].
        :param num_diffusion_timesteps: the number of betas to produce.
        :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                        produces the cumulative product of (1-beta) up to that
                        part of the diffusion process.
        :param max_beta: the maximum beta to use; use values lower than 1 to
                        prevent singularities.
        """
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)


class PointwiseNet(Module):     #unet

    def __init__(self, point_dim, context_dim, residual, num_classes = 1, **emb_dict):
        super().__init__()
        self.num_classes = num_classes
        self.batch_norm_activated = emb_dict["batch_norm_activated"]
        self.batch_norm_threshold = emb_dict["batch_norm_threshold"]
        self.extended_residual = emb_dict["extended_residual"]
        cls_emb_dim = emb_dict["cls_emb_dim"] if self.num_classes > 1 else 0
        self.cls_embedding = Embedding(num_classes, cls_emb_dim) if self.num_classes > 1 else lambda x: x

        self.num_disc_angles = emb_dict["num_disc_angles"]    #takes circle and splits it into this many discrete parts
        ang_emb_dim = emb_dict["ang_emb_dim"]
        self.angle_embedding_x = Embedding(self.num_disc_angles, ang_emb_dim)
        self.angle_embedding_y = Embedding(self.num_disc_angles, ang_emb_dim)
        self.angle_embedding_z = Embedding(self.num_disc_angles, ang_emb_dim)
        # dim could be roughly sqrt of num_classes, however, due to context being high-dimensional, 
        # I will take sqrt of whole ctx dim. sqrt(256+3) = 16
        self.act = torch.nn.functional.leaky_relu
        self.residual = residual

        combined_embedding_dim = context_dim + point_dim + cls_emb_dim + ang_emb_dim * 3

        self.layers = ModuleList([
            ConcatSquashLinear(3, 128, combined_embedding_dim),
            ConcatSquashLinear(128, 256, combined_embedding_dim),

            ConcatSquashLinear(256, 512, combined_embedding_dim),
            ConcatSquashLinear(512, 256, combined_embedding_dim),

            ConcatSquashLinear(256, 128, combined_embedding_dim),
            ConcatSquashLinear(128, 3, combined_embedding_dim)
        ])

        
    def forward(self, x:torch.Tensor, beta:torch.Tensor, context:torch.Tensor, class_index:torch.Tensor, angles:torch.Tensor):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F). F is defined as args.latent_dim
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1) #beta represents how much we vary the values depending on t.
        context = context.view(batch_size, 1, -1)   # (B, 1, F) #prep for .cat function, need to have same dimensions

        angles = angles.to(x.device)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        # sin/cos is helpful to give the "time" in my beta a more cyclic nature and more weight. 
        # connects start and end of beta schedule -> why? -> model cannot deal with just increasing numbers properly.

        # Apply the embedding layer to get the angle-specific embedding
        ang_emb_x = self.angle_embedding_x(angles[:,0])  # (B, 1, embedding_dim), 3 because of each rotation axis
        ang_emb_y = self.angle_embedding_y(angles[:,1])
        ang_emb_z = self.angle_embedding_z(angles[:,2])

        angle_embedding = torch.cat([ang_emb_x, ang_emb_y, ang_emb_z], dim=-1)
        angle_embedding = angle_embedding.view(batch_size, 1, -1)

        if self.num_classes > 1:
            # Apply the embedding layer to get the class-specific embedding
            class_embedding = self.cls_embedding(class_index)  # (B, 1, embedding_dim)
            class_embedding = class_embedding.view(batch_size, 1, -1)
            
            ctx_emb = torch.cat([time_emb, context, class_embedding, angle_embedding], dim=-1).to(x.device)    # (B, 1, F+cls_dim+ang_dim+3)   #combines beta vector and weights 
            
        else:
            ctx_emb = torch.cat([time_emb, context, angle_embedding], dim=-1).to(x.device)    #skips class embedding if training data is only one class

        out = x     # (B, N, d), batchsize, number of points, dimensions(x,y,z)

        # Initialize a list to store intermediate results for residual connections
        intermediate_outputs = []

        for i, layer in enumerate(self.layers):
            # Store the intermediate result before the current layer
            intermediate_outputs.append(out)
            out = layer(ctx=ctx_emb, x=out)

            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return out + x
        else:
            return out


class DiffusionPoint(Module):

    def __init__(self, net, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0, context, class_index, angles, t=None):
        """
        Args:
            x_0:  Input point cloud, (B, N, d).     #B = batch N = number of points, d = 3D
            context:  Shape latent, (B, F).         #B = batch F = z -> x but in latent space/encoded and reparametrized
        """
        batch_size, _, point_dim = x_0.size()       #B = batch size!, number of Points, dimensions of points, should be 3, coordinates
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, N, d)
        
        e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context, class_index=class_index, angles = angles) 
        #e_theta is forward diffusion process
        # this is q(x_T|x0) = N(x_T|sqrt(alpha_bar)*x_0, sqrt(1-alpha_bar)*I)

        loss = torch.nn.functional.mse_loss(
            e_theta.view(-1, point_dim), 
            e_rand.view(-1, point_dim), 
            reduction='mean'
            )
        return loss

    def sample(self, num_points, context, category_index, angles, point_dim=3, flexibility=0.0, ret_traj=False):
        batch_size = context.size(0)
        #creates gaussian random tensor of size [batch_size, num_points, point_dim], aka (B, N, d)
        x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)   
        traj = {self.var_sched.num_steps: x_T}
        #goes from T to zero, T is is random, zero complete object
        for t in range(self.var_sched.num_steps, 0, -1):    
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)   
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t]*batch_size]

            category_index = torch.ones((batch_size,), dtype=int, device=context.device) * category_index

            e_theta = self.net(x_t, beta=beta, context=context, class_index=category_index, angles=angles)
            # c0: scaling factor, c1: alpha probably learned, moves (more or less random) theta into right direction, 
            # sigma*z: learned movement in feature space
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]
        
        if ret_traj:
            return traj
        else:
            return traj[0]

