import torch
from diffusers.schedulers import DEISMultistepScheduler

# 我猜是正向扩散过程
def get_velocity(
        self, sample: torch.FloatTensor, noise: torch.FloatTensor, timesteps: torch.IntTensor
) -> torch.FloatTensor:
    # 确保alphas_cumprod和timestep在相同的设备上和设定dtype
    self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
    timesteps = timesteps.to(sample.device)

    sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5     # 进行平方根处理
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()                 # 扁平化处理，输出一维数组
    while len(sqrt_alpha_prod.shape) < len(sample.shape):       # while 循环检查 sqrt_alpha_prod 张量的维度是否小于 sample 张量的维度。
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)         # 就使用 unsqueeze 操作在最后一维上添加一个维度。

    sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()        
    while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample     # 直接计算出最终加噪结果
    return velocity     # 返回结果

def compute_snr(timesteps: torch.IntTensor):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    noise_scheduler = DEISMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        trained_betas=None,
        solver_order=2,
        prediction_type="epsilon",
        thresholding=False,
        dynamic_thresholding_ratio=0.995,
        sample_max_value=1,
        algorithm_type="deis",
        solver_type="logrho",
        lower_order_final=True,
        use_karras_sigmas=False,
        timestep_spacing="linspace",
        steps_offset=0
    )
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # 展开张量。
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # 计算SNR
    snr = (alpha / sigma) ** 2
    return snr