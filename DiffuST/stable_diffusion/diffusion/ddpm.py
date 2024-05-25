import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

# from contextlib import contextmanager
from functools import partial

# from diffusion.util import instantiate_from_config,count_params,exists
from diffusion.util import make_beta_schedule,extract_into_tensor


class Diffusion(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 load_only_unet=False,
                 image_size=256,
                 channels=1,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key="crossattn",
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        # self.clip_denoised = clip_denoised
        # self.log_every_t = log_every_t
        # self.first_stage_key = first_stage_key
        # self.image_size = image_size  # try conv?len(genes)
        # self.channels = channels
        # self.use_positional_encodings = use_positional_encodings
        ###########################################没用##########################################
        # self.use_ema = use_ema
        # if self.use_ema:
        #     self.model_ema = LitEma(self.model)
        #     print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        # self.use_scheduler = scheduler_config is not None
        # if self.use_scheduler:
        #     self.scheduler_config = scheduler_config
        self.v_posterior = v_posterior
        # self.original_elbo_weight = original_elbo_weight
        # self.l_simple_weight = l_simple_weight    
        # if monitor is not None:
        #     self.monitor = monitor
        ###########################################没用##########################################
        # if ckpt_path is not None:
        #     self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)
        # self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
        #                        linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        # self.loss_type = loss_type
        # self.learn_logvar = learn_logvar
        # self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        # if self.learn_logvar:
        #     self.logvar = nn.Parameter(self.logvar, requires_grad=True)           
        #############################有用######################################################3
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        
        
    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()
        
###########################################没用###############################################        
    # @contextmanager
    # def ema_scope(self, context=None):
    #     if self.use_ema:
    #         self.model_ema.store(self.model.parameters())
    #         self.model_ema.copy_to(self.model)
    #         if context is not None:
    #             print(f"{context}: Switched to EMA weights")
    #     try:
    #         yield None
    #     finally:
    #         if self.use_ema:
    #             self.model_ema.restore(self.model.parameters())
    #             if context is not None:
    #                 print(f"{context}: Restored training weights")    
###########################################没用###############################################      

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size),
                                  return_intermediates=return_intermediates)
    
    

    def q_sample(self, x_start, t, noise=None):
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

        

# class LatentDiffusion(DDPM):
#     """main class"""
#     def __init__(self,
#                  first_stage_config,
#                  cond_stage_config,
#                  num_timesteps_cond=None,
#                  cond_stage_key="image",
#                  cond_stage_trainable=False,
#                  concat_mode=True,
#                  cond_stage_forward=None,
#                  conditioning_key=None,
#                  scale_factor=1.0,
#                  scale_by_std=False,
#                  *args, **kwargs):
#         self.num_timesteps_cond = num_timesteps_cond
#         self.scale_by_std = scale_by_std
#         assert self.num_timesteps_cond <= kwargs['timesteps']
#         # for backwards compatibility after implementation of DiffusionWrapper
#         if conditioning_key is None:
#             conditioning_key = 'concat' if concat_mode else 'crossattn'
#         if cond_stage_config == '__is_unconditional__':
#             conditioning_key = None
#         ckpt_path = kwargs.pop("ckpt_path", None)
#         ignore_keys = kwargs.pop("ignore_keys", [])
#         super().__init__(conditioning_key=conditioning_key, *args, **kwargs)        
        
        
#         self.concat_mode = concat_mode
#         self.cond_stage_trainable = cond_stage_trainable
#         self.cond_stage_key = cond_stage_key
#         try:
#             self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
#         except:
#             self.num_downs = 0
#         if not scale_by_std:
#             self.scale_factor = scale_factor
#         else:
#             self.register_buffer('scale_factor', torch.tensor(scale_factor))

#         self.instantiate_first_stage(first_stage_config)
        
        
        
#     def instantiate_first_stage(self, config):
#         model = instantiate_from_config(config)
#         self.first_stage_model = model.eval()
#         self.first_stage_model.train = disabled_train
#         for param in self.first_stage_model.parameters():
#             param.requires_grad = False

#     def instantiate_cond_stage(self, config):
#         if not self.cond_stage_trainable:
#             if config == "__is_first_stage__":
#                 print("Using first stage also as cond stage.")
#                 self.cond_stage_model = self.first_stage_model
#             elif config == "__is_unconditional__":
#                 print(f"Training {self.__class__.__name__} as an unconditional model.")
#                 self.cond_stage_model = None
#                 # self.be_unconditional = True
#             else:
#                 model = instantiate_from_config(config)
#                 self.cond_stage_model = model.eval()
#                 self.cond_stage_model.train = disabled_train
#                 for param in self.cond_stage_model.parameters():
#                     param.requires_grad = False
#         else:
#             assert config != '__is_first_stage__'
#             assert config != '__is_unconditional__'
#             model = instantiate_from_config(config)
#             self.cond_stage_model = model   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


# class DiffusionWrapper(pl.LightningModule):
#     def __init__(self, diff_model_config, conditioning_key):
#         super().__init__()
#         self.diffusion_model = instantiate_from_config(diff_model_config)
#         self.conditioning_key = conditioning_key
#         assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

#     def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
#         if self.conditioning_key is None:
#             out = self.diffusion_model(x, t)
#         elif self.conditioning_key == 'concat':
#             xc = torch.cat([x] + c_concat, dim=1)
#             out = self.diffusion_model(xc, t)
#         elif self.conditioning_key == 'crossattn':
#             cc = torch.cat(c_crossattn, 1)
#             out = self.diffusion_model(x, t, context=cc)
#         elif self.conditioning_key == 'hybrid':
#             xc = torch.cat([x] + c_concat, dim=1)
#             cc = torch.cat(c_crossattn, 1)
#             out = self.diffusion_model(xc, t, context=cc)
#         elif self.conditioning_key == 'adm':
#             cc = c_crossattn[0]
#             out = self.diffusion_model(x, t, y=cc)
#         else:
#             raise NotImplementedError()

#         return out