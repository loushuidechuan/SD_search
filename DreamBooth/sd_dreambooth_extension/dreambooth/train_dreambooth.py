# Borrowed heavily from https://github.com/bmaltais/kohya_ss/blob/master/train_db.py and
# https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth
# With some custom bits sprinkled in and some stuff from OG diffusers as well.

import itertools
import json
import logging
import math
import os
import shutil
import time
import traceback
from contextlib import ExitStack
from decimal import Decimal
from pathlib import Path

import safetensors.torch
import tomesd
import torch
import torch.backends.cuda
import torch.backends.cudnn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils.random import set_seed as set_seed2
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    UNet2DConditionModel,
    DEISMultistepScheduler,
    UniPCMultistepScheduler, StableDiffusionXLPipeline, StableDiffusionPipeline
)
from diffusers.loaders import LoraLoaderMixin, text_encoder_lora_state_dict
from diffusers.models.attention_processor import LoRAAttnProcessor2_0, LoRAAttnProcessor
from diffusers.training_utils import unet_lora_state_dict
from diffusers.utils import logging as dl
from diffusers.utils.torch_utils import randn_tensor
from torch.cuda.profiler import profile
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from dreambooth import shared
from dreambooth.dataclasses.prompt_data import PromptData
from dreambooth.dataclasses.train_result import TrainResult
from dreambooth.dataset.bucket_sampler import BucketSampler
from dreambooth.dataset.sample_dataset import SampleDataset
from dreambooth.deis_velocity import get_velocity
from dreambooth.diff_lora_to_sd_lora import convert_diffusers_to_kohya_lora
from dreambooth.diff_to_sd import compile_checkpoint, copy_diffusion_model
from dreambooth.diff_to_sdxl import compile_checkpoint as compile_checkpoint_xl
from dreambooth.memory import find_executable_batch_size
from dreambooth.optimization import UniversalScheduler, get_optimizer, get_noise_scheduler
from dreambooth.shared import status
from dreambooth.utils.gen_utils import generate_classifiers, generate_dataset
from dreambooth.utils.image_utils import db_save_image, get_scheduler_class
from dreambooth.utils.model_utils import (
    unload_system_models,
    import_model_class_from_model_name_or_path,
    safe_unpickle_disabled,
    xformerify,
    torch2ify
)
from dreambooth.utils.text_utils import encode_hidden_state, save_token_counts
from dreambooth.utils.utils import (cleanup, printm, verify_locon_installed,
                                    patch_accelerator_for_fp16_training)
from dreambooth.webhook import send_training_update
from dreambooth.xattention import optim_to
from helpers.ema_model import EMAModel
from helpers.log_parser import LogParser
from helpers.mytqdm import mytqdm
from lora_diffusion.lora import (
    set_lora_requires_grad,
)

try:
    import wandb

    # 禁用烦人的魔杖弹出?
    wandb.config.auto_init = False
except:
    pass

logger = logging.getLogger(__name__)
# 定义一个处理程序，它将DEBUG消息或更高级别的消息写入sys.stderr
dl.set_verbosity_error()

last_samples = []
last_prompts = []


class ConditionalAccumulator:
    def __init__(self, accelerator, *encoders):
        self.accelerator = accelerator
        self.encoders = encoders
        self.stack = ExitStack()

    def __enter__(self):
        for encoder in self.encoders:
            if encoder is not None:
                self.stack.enter_context(self.accelerator.accumulate(encoder))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stack.__exit__(exc_type, exc_value, traceback)


"""
#######################################################################################################################
@@                                                                                                                   @@
@@                                                定义一些基础函数                                                    @@
@@                                                                                                                   @@
#######################################################################################################################
"""


# 检查和补丁调度程序
def check_and_patch_scheduler(scheduler_class):
    # 检查该类是否已经有了名为 'get_velocity' 的方法。
    if not hasattr(scheduler_class, 'get_velocity'):
        # 没有则添加 get_velocity 方法进去
        logger.debug(f"Adding 'get_velocity' method to {scheduler_class.__name__}...")
        scheduler_class.get_velocity = get_velocity


try:
    check_and_patch_scheduler(DEISMultistepScheduler)
    check_and_patch_scheduler(UniPCMultistepScheduler)
except:
    logger.warning("Exception while adding 'get_velocity' method to the schedulers.")

export_diffusers = False
user_model_dir = ""

# 设置随机种子是否固定
def set_seed(deterministic: bool):
    if deterministic:
        torch.backends.cudnn.deterministic = True
        seed = 0
        set_seed2(seed)
    else:
        torch.backends.cudnn.deterministic = False

# 定义一个全局列表
to_delete = []
# 清空 to_delete 列表
def clean_global_state():
    for check in to_delete:
        if check:
            try:
                obj_name = check.__name__
                del check
                # 记录被删除的东西的名称
                logger.debug(f"Deleted {obj_name}")
            except:
                pass

# 先验损失权重函数计算
def current_prior_loss(args, current_epoch):
    # 以下三个判断是进行初始化判断，确保后面计算有默认参数
    if not args.prior_loss_scale:           # 检测是否开启先验损失规模
        return args.prior_loss_weight
    if not args.prior_loss_target:          # 检测是否设置目标先验损失
        args.prior_loss_target = 150
    if not args.prior_loss_weight_min:      # 检测是否设置最小先验损失权重
        args.prior_loss_weight_min = 0.1

    if current_epoch >= args.prior_loss_target:
        return args.prior_loss_weight_min   # 如果大于则使用最小先验损失权重
    
    percentage_completed = current_epoch / args.prior_loss_target       # 计算当前轮数占总轮数的完成的百分比
    # 先验损失权重随着完成轮数增多，权重逐渐变小
    prior = (
            args.prior_loss_weight * (1 - percentage_completed)
            + args.prior_loss_weight_min * percentage_completed
    )
    printm(f"Prior: {prior}")
    return prior

# 停止上下文管理器
def stop_profiler(profiler):
    if profiler is not None:
        try:
            # 上传日志
            logger.debug("Stopping profiler.")
            profiler.stop()
        except:
            pass
"""
#######################################################################################################################
@@                                                                                                                   @@
@@                                          接收Web参数并实例化模型                                                    @@
@@                                                                                                                   @@
#######################################################################################################################
"""

# 训练主程序
def main(class_gen_method: str = "Native Diffusers", user: str = None) -> TrainResult:
    """
    @param class_gen_method:图像生成库。
    @param user:发送培训更新的用户(用于新UI)
    @return: 训练结果
    """
    args = shared.db_model_config
    status_handler = None
    logging_dir = Path(args.model_dir, "logging")
    global export_diffusers, user_model_dir

    # 尝试导入一些库，并实例化他们
    try:
        from core.handlers.status import StatusHandler
        from core.handlers.config import ConfigHandler
        from core.handlers.models import ModelHandler

        mh = ModelHandler(user_name=user)
        status_handler = StatusHandler(user_name=user, target="dreamProgress")
        export_diffusers = True
        user_model_dir = mh.user_path
        logger.debug(f"Export diffusers: {export_diffusers}, diffusers dir: {user_model_dir}")
        shared.status_handler = status_handler
        logger.debug(f"Loaded config: {args.__dict__}")
    except:
        pass
    
    # 实例化一个日志类
    log_parser = LogParser()
    
    # 更新SD状态栏状态的方法
    def update_status(data: dict):
        if status_handler is not None:
            if "iterations_per_second" in data:
                data = {"status": json.dumps(data)}
            status_handler.update(items=data)
            
    # 定义了一个TrainResult对象
    result = TrainResult
    # shared.db_model_config传入到TrainResult的config里
    result.config = args
    set_seed(args.deterministic)

    @find_executable_batch_size(
        starting_batch_size=args.train_batch_size,
        starting_grad_size=args.gradient_accumulation_steps,
        logging_dir=logging_dir,
        cleanup_function=clean_global_state()
    )

    def inner_loop(train_batch_size: int, gradient_accumulation_steps: int, profiler: profile):
        
        # 定义两个文本编码器，因为SDXL多加了一个OpenCLIP ViT-bigG
        text_encoder = None
        text_encoder_two = None
        global last_samples
        global last_prompts
        
        # 设置文本编码器的学习步数比例，为0则不训练
        stop_text_percentage = args.stop_text_encoder
        # 如果不训练Unet，则文本编码器的学习步数比例设为1，全程参与训练
        if not args.train_unet:
            stop_text_percentage = 1

        n_workers = 0
        # 设置最大词元长度
        args.max_token_length = int(args.max_token_length)
        # 如果不填充词元且词元长度大于75，发出警告
        if not args.pad_tokens and args.max_token_length > 75:
            logger.warning("Cannot raise token length limit above 75 when pad_tokens=False")

        verify_locon_installed(args)

        precision = args.mixed_precision if not shared.force_cpu else "no"
        
        # 选择混合精度
        weight_dtype = torch.float32
        if precision == "fp16":
            weight_dtype = torch.float16
        elif precision == "bf16":
            weight_dtype = torch.bfloat16
            
        # 选择使用的硬件
        try:
            # 尝试初始化 PyTorch Lighting 的 Accelerator
            # Accelerator 对象是 PyTorch Lightning 中用于管理分布式训练和硬件加速的实用工具
            accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,       # 梯度累积的步数
                mixed_precision=precision,      # 混合精度训练的开启与关闭
                log_with="all",                 # 指定日志的详细程度
                project_dir=logging_dir,        # 指定日志的详细程度
                cpu=shared.force_cpu,           # 选择是否使用CPU
            )

            run_name = "dreambooth.events"
            max_log_size = 250 * 1024  # 指定最大日志大小

        except Exception as e:
            if "AcceleratorState" in str(e):
                msg = "Change in precision detected, please restart the webUI entirely to use new precision."
            else:
                msg = f"Exception initializing accelerator: {e}"
            logger.warning(msg)
            result.msg = msg
            result.config = args
            stop_profiler(profiler)
            return result

        # 这是二级状态栏
        pbar2 = mytqdm(
            disable=not accelerator.is_local_main_process,
            position=1,
            user=user,
            target="dreamProgress",
            index=1
        )
        #目前，在训练两个模型时不可能进行梯度积累
        #加速。这将很快加速启用。现在，我们不允许梯度
        #训练两个模型时积累。
        # TODO (patil-suraj):当两个模型的梯度累积在加速中启用时，删除此检查。
        if (
                stop_text_percentage != 0
                and gradient_accumulation_steps > 1
                and accelerator.num_processes > 1
        ):
            msg = (
                '''
                在分布式训练中，文本编码器的训练不支持梯度积累。
                请将gradient_accumulation_steps设置为1。该功能将在未来得到支持。文本
                编码器培训将被禁用。
                '''
            )
            logger.warning(msg)
            status.textinfo = msg
            update_status({"status": msg})
            stop_text_percentage = 0
        pretrained_path = args.get_pretrained_model_name_or_path()
        logger.debug(f"Pretrained path: {pretrained_path}")
        count, instance_prompts, class_prompts = generate_classifiers(
            args, class_gen_method=class_gen_method, accelerator=accelerator, ui=False, pbar=pbar2
        )

        save_token_counts(args, instance_prompts, 10)

        if status.interrupted:
            result.msg = "Training interrupted."
            stop_profiler(profiler)
            return result

        num_components = 5
        if args.model_type == "SDXL":
            num_components = 7
        pbar2.reset(num_components)
        pbar2.set_description("Loading model components...")
        
        pbar2.set_postfix(refresh=True)
        if class_gen_method == "Native Diffusers" and count > 0:
            unload_system_models()

        def create_vae():
            vae_path = (
                args.pretrained_vae_name_or_path        # 查看vae名称或路径
                if args.pretrained_vae_name_or_path
                else args.get_pretrained_model_name_or_path()       # 没有则使用get方法获取
            )
            # 关闭安全反序列
            with safe_unpickle_disabled():
                # 预训练参数文件加载模型
                new_vae = AutoencoderKL.from_pretrained(    
                    vae_path,
                    subfolder=None if args.pretrained_vae_name_or_path else "vae",      # 确定是否使用子目录 
                    revision=args.revision,         # 可能是是模型的版本或 Git 版本控制的修订号
                )
            new_vae.requires_grad_(False)       # 不计算梯度，进行推断
            new_vae.to(accelerator.device, dtype=weight_dtype)
            return new_vae
        
        # 在with语句中安全反序列是关闭状态，执行完with语句将重新开启pytorch的安全反序列选项
        with safe_unpickle_disabled():
            # 加载分词器
            pbar2.set_description("Loading tokenizer...")
            pbar2.update()
            pbar2.set_postfix(refresh=True)
            tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(pretrained_path, "tokenizer"),
                revision=args.revision,
                use_fast=False,
            )

            tokenizer_two = None
            if args.model_type == "SDXL":
                pbar2.set_description("Loading tokenizer 2...")
                pbar2.update()
                pbar2.set_postfix(refresh=True)
                tokenizer_two = AutoTokenizer.from_pretrained(
                    os.path.join(pretrained_path, "tokenizer_2"),
                    revision=args.revision,
                    use_fast=False,
                )

            # 导入所使用的文本编码器
            text_encoder_cls = import_model_class_from_model_name_or_path(
                args.get_pretrained_model_name_or_path(), args.revision
            )
            
            # 加载词嵌入模型
            pbar2.set_description("Loading text encoder...")
            pbar2.update()
            pbar2.set_postfix(refresh=True)
            # 加载模型并为稳定扩散创建包装器
            text_encoder = text_encoder_cls.from_pretrained(
                args.get_pretrained_model_name_or_path(),
                subfolder="text_encoder",
                revision=args.revision,
                torch_dtype=torch.float32,
            )

            if args.model_type == "SDXL":
                # 导入正确的文本编码器类
                text_encoder_cls_two = import_model_class_from_model_name_or_path(
                    args.get_pretrained_model_name_or_path(), args.revision, subfolder="text_encoder_2"
                )

                pbar2.set_description("Loading text encoder 2...")
                pbar2.update()
                pbar2.set_postfix(refresh=True)
                # 加载模型并为稳定扩散创建包装器
                text_encoder_two = text_encoder_cls_two.from_pretrained(
                    args.get_pretrained_model_name_or_path(),
                    subfolder="text_encoder_2",
                    revision=args.revision,
                    torch_dtype=torch.float32,
                )

            printm("Created tenc")
            
            # 加载VAE模型
            pbar2.set_description("Loading VAE...")
            pbar2.update()
            vae = create_vae()
            printm("Created vae")
            
            # 加载UNet模型
            pbar2.set_description("Loading unet...")
            pbar2.update()
            unet = UNet2DConditionModel.from_pretrained(
                args.get_pretrained_model_name_or_path(),
                subfolder="unet",
                revision=args.revision,
                torch_dtype=torch.float32,
            )
            
            # 选择注意力模型
            if args.attention == "xformers" and not shared.force_cpu:
                xformerify(unet, use_lora=args.use_lora)
                xformerify(vae, use_lora=args.use_lora)
                
            # 查看Pytorch是否支持编译，如果可以则进行编译
            unet = torch2ify(unet)
            
            # 如果是全混合精度且精度为fp16，则直接加载UNet模型
            if args.full_mixed_precision:
                if args.mixed_precision == "fp16":
                    patch_accelerator_for_fp16_training(accelerator)
                unet.to(accelerator.device, dtype=weight_dtype)
            else:
                low_precision_error_string = (
                    "Please make sure to always have all model weights in full float32 precision when starting training - "
                    "even if doing mixed precision training. copy of the weights should still be float32."
                )
                
                # 检测UNet模型精度是不是混合精度
                if accelerator.unwrap_model(unet).dtype != torch.float32:
                    logger.warning(
                        f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
                    )
                    
                # 检测Tokenizer模型精度是不是混合精度
                if (
                        args.stop_text_encoder != 0
                        and accelerator.unwrap_model(text_encoder).dtype != torch.float32
                ):
                    logger.warning(
                        f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
                        f" {low_precision_error_string}"
                    )
                    
                # 检测word embedding模型精度是不是混合精度
                if (
                        args.stop_text_encoder != 0
                        and accelerator.unwrap_model(text_encoder_two).dtype != torch.float32
                ):
                    logger.warning(
                        f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder_two).dtype}."
                        f" {low_precision_error_string}"
                    )
                    
            # 查看是否使用梯度检查点
            if args.gradient_checkpointing:
                # 如果训练UNet就启用梯度
                if args.train_unet:
                    unet.enable_gradient_checkpointing()
                    
                # 查看训练文本编码器的步数比率，为0则不训练
                if stop_text_percentage != 0:
                    text_encoder.gradient_checkpointing_enable()
                    if args.model_type == "SDXL":
                        text_encoder_two.gradient_checkpointing_enable()
                    if args.use_lora:
                        # 我们需要在一个输入上启用梯度，以使梯度检查点工作
                        # 这将不会被优化，因为它不是优化器的参数
                        # 开启嵌入层的梯度计算
                        text_encoder.text_model.embeddings.position_embedding.requires_grad_(True)
                        if args.model_type == "SDXL":
                            text_encoder_two.text_model.embeddings.position_embedding.requires_grad_(True)
                # 不训练则直接丢入CUDA
                else:
                    text_encoder.to(accelerator.device, dtype=weight_dtype)
                    if args.model_type == "SDXL":
                        text_encoder_two.to(accelerator.device, dtype=weight_dtype)
                        
            # 选择是否要启动ema
            ema_model = None
            if args.use_ema:
                # 查看本地是否有ema模型
                if os.path.exists(
                        os.path.join(
                            args.get_pretrained_model_name_or_path(),
                            "ema_unet",
                            "diffusion_pytorch_model.safetensors",
                        )
                ):
                    # 实例化ema的Unet部分
                    ema_unet = UNet2DConditionModel.from_pretrained(
                        args.get_pretrained_model_name_or_path(),
                        subfolder="ema_unet",
                        revision=args.revision,
                        torch_dtype=weight_dtype,
                    )
                    # 启动xformers注意力机制且不在cp u上计算
                    if args.attention == "xformers" and not shared.force_cpu:
                        xformerify(ema_unet, use_lora=args.use_lora)

                    ema_model = EMAModel(
                        ema_unet, device=accelerator.device, dtype=weight_dtype
                    )
                    # 用完直接删掉，节省显存
                    del ema_unet
                else:
                    # 没有则直接使用Unet模型实例化ema
                    ema_model = EMAModel(
                        unet, device=accelerator.device, dtype=weight_dtype
                    )

            # 创建共享的unet/tenc学习率变量
            learning_rate = args.learning_rate
            txt_learning_rate = args.txt_learning_rate
            if args.use_lora:
                learning_rate = args.lora_learning_rate
                txt_learning_rate = args.lora_txt_learning_rate

            # 如果不使用Lora且不训练Unet的关闭Unet的梯度
            if args.use_lora or not args.train_unet:
                unet.requires_grad_(False)

            unet_lora_params = None
            
            # 选择是否使用lora
            if args.use_lora:
                pbar2.reset(1)
                pbar2.set_description("Loading LoRA...")
                # 现在我们将添加新的LoRA权重到注意层
                # 设置正确的lora层
                unet_lora_attn_procs = {}
                unet_lora_params = []
                rank = args.lora_unet_rank

                for name, attn_processor in unet.attn_processors.items():
                    cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
                    hidden_size = None
                    if name.startswith("mid_block"):
                        hidden_size = unet.config.block_out_channels[-1]
                    elif name.startswith("up_blocks"):
                        block_id = int(name[len("up_blocks.")])
                        hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                    elif name.startswith("down_blocks"):
                        block_id = int(name[len("down_blocks.")])
                        hidden_size = unet.config.block_out_channels[block_id]

                    lora_attn_processor_class = (
                        LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
                    )
                    if hidden_size is None:
                        logger.warning(f"Could not find hidden size for {name}. Skipping...")
                        continue
                    module = lora_attn_processor_class(
                        hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank
                    )
                    unet_lora_attn_procs[name] = module
                    unet_lora_params.extend(module.parameters())

                unet.set_attn_processor(unet_lora_attn_procs)

                # 文本编码器来自🤗变压器，所以我们不能直接修改它。
                # 所以，相反，我们monkey-patch它的注意力块的前向传播。
                if stop_text_percentage != 0:
                    # 确保dtype为float32，即使在fp16中加载了未训练的模型的其余部分
                    text_encoder_lora_params = LoraLoaderMixin._modify_text_encoder(
                        text_encoder, dtype=torch.float32, rank=args.lora_txt_rank
                    )

                    if args.model_type == "SDXL":
                        text_encoder_lora_params_two = LoraLoaderMixin._modify_text_encoder(
                            text_encoder_two, dtype=torch.float32, rank=args.lora_txt_rank
                        )
                        params_to_optimize = (
                            itertools.chain(unet_lora_params, text_encoder_lora_params, text_encoder_lora_params_two))
                    else:
                        params_to_optimize = (itertools.chain(unet_lora_params, text_encoder_lora_params))

                else:
                    params_to_optimize = unet_lora_params

                # 如果指定，加载LoRA权重
                if args.lora_model_name is not None and args.lora_model_name != "":
                    logger.debug(f"Load lora from {args.lora_model_name}")
                    lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(args.lora_model_name)
                    LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=unet)

                    LoraLoaderMixin.load_lora_into_text_encoder(
                        lora_state_dict, network_alphas=network_alphas, text_encoder=text_encoder)
                    if text_encoder_two is not None:
                        LoraLoaderMixin.load_lora_into_text_encoder(
                            lora_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_two)
                        
                        
            # CLIP/Unet二选一
            elif stop_text_percentage != 0:
                if args.train_unet:
                    if args.model_type == "SDXL":
                        # 创造一个拥有Unet和CLIP所有可训练参数的迭代器
                        params_to_optimize = itertools.chain(unet.parameters(), text_encoder.parameters(),
                                                             text_encoder_two.parameters())
                    else:
                        # 创造一个拥有Unet和CLIP所有可训练参数的迭代器
                        params_to_optimize = itertools.chain(unet.parameters(), text_encoder.parameters())
                else:
                    if args.model_type == "SDXL":
                        # 创造一个拥有Unet和CLIP所有可训练参数的迭代器
                        params_to_optimize = itertools.chain(text_encoder.parameters(), text_encoder_two.parameters())
                    else:
                        # 创造一个拥有Unet和CLIP所有可训练参数的迭代器
                        params_to_optimize = itertools.chain(text_encoder.parameters())
            else:
                params_to_optimize = unet.parameters()
                
            # 选择优化器
            optimizer = get_optimizer(args.optimizer, learning_rate, args.weight_decay, params_to_optimize)
            if len(optimizer.param_groups) > 1:
                try:
                    # 嵌入层权重衰减
                    optimizer.param_groups[1]["weight_decay"] = args.tenc_weight_decay
                    # 裁剪嵌入层梯度归一层
                    optimizer.param_groups[1]["grad_clip_norm"] = args.tenc_grad_clip_norm
                except:
                    logger.warning("Exception setting tenc weight decay")
                    traceback.print_exc()

            if len(optimizer.param_groups) > 2:
                try:
                    # XL嵌入层权重衰减
                    optimizer.param_groups[2]["weight_decay"] = args.tenc_weight_decay
                    # XL裁剪嵌入层梯度归一层
                    optimizer.param_groups[2]["grad_clip_norm"] = args.tenc_grad_clip_norm
                except:
                    logger.warning("Exception setting tenc weight decay")
                    traceback.print_exc()
                    
            # 设置图像生成调度器
            noise_scheduler = get_noise_scheduler(args)
            global to_delete
            to_delete = [unet, text_encoder, text_encoder_two, tokenizer, tokenizer_two, optimizer, vae]
            # 清理内存
            def cleanup_memory():
                try:
                    if unet:
                        del unet
                    if text_encoder:
                        del text_encoder
                    if text_encoder_two:
                        del text_encoder_two
                    if tokenizer:
                        del tokenizer
                    if tokenizer_two:
                        del tokenizer_two
                    if optimizer:
                        del optimizer
                    if train_dataloader:
                        del train_dataloader
                    if train_dataset:
                        del train_dataset
                    if lr_scheduler:
                        del lr_scheduler
                    if vae:
                        del vae
                    if unet_lora_params:
                        del unet_lora_params
                except:
                    pass
                cleanup(True)
                
            # 选择是否缓存潜在变量
            if args.cache_latents:
                vae.to(accelerator.device, dtype=weight_dtype)
                vae.requires_grad_(False)
                vae.eval()
                
            # 检测状态栏的中止按钮是否被点击过
            if status.interrupted:
                result.msg = "Training interrupted."
                stop_profiler(profiler)
                return result

            printm("Loading dataset...")
            pbar2.reset()
            pbar2.set_description("Loading dataset")
            
            # 将先前保存关闭
            with_prior_preservation = False
            # 赋值tokenizers并判断训练模型是否为XL模型
            tokenizers = [tokenizer] if tokenizer_two is None else [tokenizer, tokenizer_two]
            # 赋值text_encoders并判断训练模型是否为XL模型
            text_encoders = [text_encoder] if text_encoder_two is None else [text_encoder, text_encoder_two]
            # 返回一个DB数据集类，并进行赋值
            train_dataset = generate_dataset(
                model_name=args.model_name,
                instance_prompts=instance_prompts,
                class_prompts=class_prompts,
                batch_size=args.train_batch_size,
                tokenizer=tokenizers,
                text_encoder=text_encoders,
                accelerator=accelerator,
                vae=vae if args.cache_latents else None,
                debug=False,
                model_dir=args.model_dir,
                max_token_length=args.max_token_length,
                pbar=pbar2
            )
            # 如果训练数据集类里的类数量大于0
            if train_dataset.class_count > 0:
                # 将先前保存开启
                with_prior_preservation = True
            pbar2.reset()
            printm("Dataset loaded.")
            # 将最大词元进行赋值
            tokenizer_max_length = tokenizer.model_max_length
            # 如果要缓存潜在变量
            if args.cache_latents:
                printm("Unloading vae.")
                del vae
                # 保留对vae的引用以供以后检查
                vae = None
                # TODO:尝试在这里卸载标记器?
                del tokenizer
                if tokenizer_two is not None:
                    del tokenizer_two
                tokenizer = None
                tokenizer2 = None
                
            # 检测状态栏的中止按钮是否被点击过
            if status.interrupted:
                result.msg = "Training interrupted."
                stop_profiler(profiler)
                return result
            
            # 检查数据集是否为空
            if train_dataset.__len__ == 0:
                msg = "Please provide a directory with actual images in it."
                logger.warning(msg)
                status.textinfo = msg
                update_status({"status": status})
                cleanup_memory()
                result.msg = msg
                result.config = args
                stop_profiler(profiler)
                return result

            def collate_fn_db(examples):
                # 从examples提取 input_ids 、图像、父类型、权重
                input_ids = [example["input_ids"] for example in examples]
                pixel_values = [example["image"] for example in examples]
                types = [example["is_class"] for example in examples]
                weights = [
                    current_prior_loss_weight if example["is_class"] else 1.0
                    for example in examples
                ]
                # 设置损失为0
                loss_avg = 0
                # 计算损失平均值
                for weight in weights:
                    loss_avg += weight
                loss_avg /= len(weights)
                # 将图像堆叠为张量
                pixel_values = torch.stack(pixel_values)
                # 如果不缓存潜变量，就把图像转换为浮点张量
                if not args.cache_latents:
                    pixel_values = pixel_values.to(
                        memory_format=torch.contiguous_format
                    ).float()
                # 将图片进行列拼接，变成一个张量
                input_ids = torch.cat(input_ids, dim=0)
                
                # 一个批次的图像数据
                batch_data = {
                    "input_ids": input_ids,
                    "images": pixel_values,
                    "types": types,
                    "loss_avg": loss_avg,
                }
                # 如果参数里还有input_ids2，则为字典添加额外的字典到batch_data
                if "input_ids2" in examples[0]:
                    input_ids_2 = [example["input_ids2"] for example in examples]
                    input_ids_2 = torch.stack(input_ids_2)

                    batch_data["input_ids2"] = input_ids_2
                    batch_data["original_sizes_hw"] = torch.stack(
                        [torch.LongTensor(x["original_sizes_hw"]) for x in examples])
                    batch_data["crop_top_lefts"] = torch.stack([torch.LongTensor(x["crop_top_lefts"]) for x in examples])
                    batch_data["target_sizes_hw"] = torch.stack([torch.LongTensor(x["target_sizes_hw"]) for x in examples])
                return batch_data

            def collate_fn_sdxl(examples):
                # 从examples获取input_ids、图像、文本编码器、时间步
                input_ids = [example["input_ids"] for example in examples if not example["is_class"]]
                pixel_values = [example["image"] for example in examples if not example["is_class"]]
                add_text_embeds = [example["instance_added_cond_kwargs"]["text_embeds"] for example in examples if
                                   not example["is_class"]]
                add_time_ids = [example["instance_added_cond_kwargs"]["time_ids"] for example in examples if
                                not example["is_class"]]

                # Concat类和实例示例以保留之前的内容。
                # 这样做是为了避免两次向前传递。
                if with_prior_preservation:
                    input_ids += [example["input_ids"] for example in examples if example["is_class"]]
                    pixel_values += [example["image"] for example in examples if example["is_class"]]
                    add_text_embeds += [example["instance_added_cond_kwargs"]["text_embeds"] for example in examples if
                                        example["is_class"]]
                    add_time_ids += [example["instance_added_cond_kwargs"]["time_ids"] for example in examples if
                                     example["is_class"]]
                    
                # 将图片进行列拼接，变成一个张量，并转化为浮点数
                pixel_values = torch.stack(pixel_values)
                pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
                
                # 把input_ids、文本编码器、时间步变为一个张量
                input_ids = torch.cat(input_ids, dim=0)
                add_text_embeds = torch.cat(add_text_embeds, dim=0)
                add_time_ids = torch.cat(add_time_ids, dim=0)
                
                # 定义批数据
                batch = {
                    "input_ids": input_ids,
                    "images": pixel_values,
                    "unet_added_conditions": {"text_embeds": add_text_embeds, "time_ids": add_time_ids},
                }

                return batch
            
            # 根据train_batch_size进行数据集分批
            sampler = BucketSampler(train_dataset, train_batch_size)
            
            # 根据模型使用不同整理规范
            collate_fn = collate_fn_db
            if args.model_type == "SDXL":
                collate_fn = collate_fn_sdxl
            
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,                  # 训练集实例
                batch_size=1,                   # 批次大小
                batch_sampler=sampler,          # 设置批采样器
                collate_fn=collate_fn,          # 使用模型的整理规范，用于组装小批量数据
                num_workers=n_workers,          # 设置加载器工作线程数量，0为只用主程序加载
            )
            
            # 计算最大训练步数：训练轮数 * 训练数据类长度
            max_train_steps = args.num_train_epochs * len(train_dataset)

            # 这是独立的，因为优化器。Step在训练中每个“Step”只被调用一次，所以它不是
            # 受批量大小的影响
            # 计算预定训练步数：训练轮次 * 数据集数量
            sched_train_steps = args.num_train_epochs * train_dataset.num_train_images

            lr_scale_pos = args.lr_scale_pos
            if class_prompts:
                lr_scale_pos *= 2
                
            # 设置学习调度器
            lr_scheduler = UniversalScheduler(
                name=args.lr_scheduler,
                optimizer=optimizer,                        # 优化器
                num_warmup_steps=args.lr_warmup_steps,      # 学习率预热步数
                total_training_steps=sched_train_steps,     # 训练总步数
                min_lr=args.learning_rate_min,              # 最小学习率
                total_epochs=args.num_train_epochs,         # 总轮数
                num_cycles=args.lr_cycles,
                power=args.lr_power,
                factor=args.lr_factor,
                scale_pos=lr_scale_pos,
                unet_lr=learning_rate,                      # Unet学习率
                tenc_lr=txt_learning_rate,                  # 文本学习率
            )
            
            
            # 将模型和数据加载器移动到特定设备上，并进行赋值
            # 创造 ema, 防止 OOM（爆显存的意思
            if args.use_ema:
                # 如果训练CLIP
                if stop_text_percentage != 0:
                    (
                        ema_model.model,
                        unet,
                        text_encoder,
                        optimizer,
                        train_dataloader,
                        lr_scheduler,
                    ) = accelerator.prepare(
                        ema_model.model,
                        unet,
                        text_encoder,
                        optimizer,
                        train_dataloader,
                        lr_scheduler,
                    )
                else:
                    (
                        ema_model.model,
                        unet,
                        optimizer,
                        train_dataloader,
                        lr_scheduler,
                    ) = accelerator.prepare(
                        ema_model.model, unet, optimizer, train_dataloader, lr_scheduler
                    )
            # 不使用ema
            else:
                # 如果训练CLIP
                if stop_text_percentage != 0:
                    (
                        unet,
                        text_encoder,
                        optimizer,
                        train_dataloader,
                        lr_scheduler,
                    ) = accelerator.prepare(
                        unet, text_encoder, optimizer, train_dataloader, lr_scheduler
                    )
                else:
                    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                        unet, optimizer, train_dataloader, lr_scheduler
                    )
                    
            # 如果缓存潜在变量且有vae
            if not args.cache_latents and vae is not None:
                vae.to(accelerator.device, dtype=weight_dtype)

            if stop_text_percentage == 0:
                text_encoder.to(accelerator.device, dtype=weight_dtype)
            # 之后，我们重新计算训练步数
            # 我们需要初始化我们使用的追踪器，并存储我们的配置。
            # is_main_process方法是用来执行只执行一次的语句
            if accelerator.is_main_process:
                accelerator.init_trackers("dreambooth")

            """
            #######################################################################################################################
            @@                                                                                                                   @@
            @@                                         训练参数配置并定义保存模型函数                                               @@
            @@                                                                                                                   @@
            #######################################################################################################################
            """
            total_batch_size = (
                    train_batch_size * accelerator.num_processes * gradient_accumulation_steps
            )
            max_train_epochs = args.num_train_epochs
            # 我们计算文本编码器的训练步数（最大训练轮数 * 停止百分比）
            text_encoder_epochs = round(max_train_epochs * stop_text_percentage)
            global_step = 0             # 全局步数
            global_epoch = 0            # 全局轮数
            session_epoch = 0           # 当前轮数  
            first_epoch = 0             # 一轮
            resume_step = 0             # 恢复步数
            last_model_save = 0         # 最后一个保存模型的轮数
            last_image_save = 0         # 最后一个保存图片的轮数
            resume_from_checkpoint = False
            new_hotness = os.path.join(
                args.model_dir, "checkpoints", f"checkpoint-{args.snapshot}"
            )
            if os.path.exists(new_hotness):
                logger.debug(f"Resuming from checkpoint {new_hotness}")

                try:
                    # 导入modules.shared库
                    import modules.shared
                    # 将modules库里的安全反序列化部分赋值给no_safe
                    no_safe = modules.shared.cmd_opts.disable_safe_unpickle
                    # 将modules.shared安全反序列化开启
                    modules.shared.cmd_opts.disable_safe_unpickle = True
                except:
                    no_safe = False

                try:
                    import modules.shared
                    # 加载检查点
                    accelerator.load_state(new_hotness)
                    # 设置安全反序列化的状态
                    modules.shared.cmd_opts.disable_safe_unpickle = no_safe
                    # 设置全局步数和恢复步数。
                    global_step = resume_step = args.revision
                    # 设置是否从检查点回复
                    resume_from_checkpoint = True
                    # 设置总轮数
                    first_epoch = args.lifetime_epoch
                    global_epoch = args.lifetime_epoch
                except Exception as lex:
                    logger.warning(f"Exception loading checkpoint: {lex}")
            # 显示训练配置
            logger.debug("  ***** Running training *****")
            if shared.force_cpu:
                logger.debug(f"  TRAINING WITH CPU ONLY")
            logger.debug(f"  Num batches each epoch = {len(train_dataset) // train_batch_size}")
            logger.debug(f"  Num Epochs = {max_train_epochs}")
            logger.debug(f"  Batch Size Per Device = {train_batch_size}")
            logger.debug(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
            logger.debug(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
            logger.debug(f"  Text Encoder Epochs: {text_encoder_epochs}")
            logger.debug(f"  Total optimization steps = {sched_train_steps}")
            logger.debug(f"  Total training steps = {max_train_steps}")
            logger.debug(f"  Resuming from checkpoint: {resume_from_checkpoint}")
            logger.debug(f"  First resume epoch: {first_epoch}")
            logger.debug(f"  First resume step: {resume_step}")
            logger.debug(f"  Lora: {args.use_lora}, Optimizer: {args.optimizer}, Prec: {precision}")
            logger.debug(f"  Gradient Checkpointing: {args.gradient_checkpointing}")
            logger.debug(f"  EMA: {args.use_ema}")
            logger.debug(f"  UNET: {args.train_unet}")
            logger.debug(f"  Freeze CLIP Normalization Layers: {args.freeze_clip_normalization}")
            logger.debug(f"  LR{' (Lora)' if args.use_lora else ''}: {learning_rate}")
            if stop_text_percentage > 0:
                logger.debug(f"  Tenc LR{' (Lora)' if args.use_lora else ''}: {txt_learning_rate}")
            logger.debug(f"  V2: {args.v2}")
            
            # 将环境变量里的 CUDA_LAUNCH_BLOCKING 设置为1 
            # 启动 CUDA 的时候进行阻塞。在 GPU 执行完函数之前，CPU处于阻塞状态，多用于调试程序，检测潜在问题
            os.environ.__setattr__("CUDA_LAUNCH_BLOCKING", 1)
            
            # 检查保存
            def check_save(is_epoch_check=False):
                # 获取到上一层的变量
                nonlocal last_model_save
                nonlocal last_image_save
                # 进行传参，和参数初始化
                save_model_interval = args.save_embedding_every             # 保存模型间隔
                save_image_interval = args.save_preview_every               # 保存图像间隔
                save_completed = session_epoch >= max_train_epochs          # 判断当前轮数是否大于最大轮数
                save_canceled = status.interrupted                          # 查看是否已中止程序
                save_image = False
                save_model = False
                save_lora = False
                
                # 如果没有中止且当前轮数小于最大轮数
                if not save_canceled and not save_completed:
                    # 定点保存模型
                    if 0 < save_model_interval <= session_epoch - last_model_save:
                        save_model = True
                        # 如果使用lora
                        if args.use_lora:
                            save_lora = True
                        # 记录保存轮数
                        last_model_save = session_epoch

                    # 定点保存图片
                    if 0 < save_image_interval <= session_epoch - last_image_save:
                        save_image = True
                        # 记录保存轮数
                        last_image_save = session_epoch
                        
                # 否则进行模型和图像保存
                else:
                    logger.debug("\nSave completed/canceled.")
                    if global_step > 0:
                        save_image = True
                        save_model = True
                        if args.use_lora:
                            save_lora = True
                            
                # 初始化保存快照操作
                save_snapshot = False
                
                # 是否检查保存
                if is_epoch_check:
                    # 如果保存 样本 状态为True
                    if shared.status.do_save_samples:
                        save_image = True
                        shared.status.do_save_samples = False
                    # 如果保存 模型 状态为True
                    if shared.status.do_save_model:
                        if args.use_lora:
                            save_lora = True
                        save_model = True
                        shared.status.do_save_model = False

                save_checkpoint = False
                if save_model:
                    # 如果取消训练
                    if save_canceled:
                        # 如果训练步数大于0
                        if global_step > 0:
                            logger.debug("Canceled, enabling saves.")
                            save_snapshot = args.save_state_cancel
                            save_checkpoint = args.save_ckpt_cancel
                    # 如果训练已经完成
                    elif save_completed:
                        if global_step > 0:
                            logger.debug("Completed, enabling saves.")
                            save_snapshot = args.save_state_after
                            save_checkpoint = args.save_ckpt_after
                    # 训练中
                    else:
                        save_snapshot = args.save_state_during
                        save_checkpoint = args.save_ckpt_during
                    if save_checkpoint and args.use_lora:
                        save_checkpoint = False
                        save_lora = True
                # 如果使用lora
                if not args.use_lora:
                    save_lora = False
                    
                # 如果有要保存的模型
                if (
                        save_checkpoint
                        or save_snapshot
                        or save_lora
                        or save_image
                        or save_model
                ):
                    # 调用保存模型方法
                    save_weights(
                        save_image,
                        save_model,
                        save_snapshot,
                        save_checkpoint,
                        save_lora
                    )

                return save_model, save_image
            
            # 保存模型方法
            def save_weights(
                    save_image, save_diffusers, save_snapshot, save_checkpoint, save_lora
            ):
                global last_samples
                global last_prompts
                nonlocal vae
                nonlocal pbar2

                printm(" Saving weights.")
                pbar2.reset()
                pbar2.set_description("Saving weights/samples...")
                pbar2.set_postfix(refresh=True)

                # 使用经过训练的模块创建管道并保存它。
                if accelerator.is_main_process:
                    printm("Pre-cleanup.")
                    torch_rng_state = None
                    cuda_gpu_rng_state = None
                    cuda_cpu_rng_state = None
                    # 保存随机状态，这样样本生成不会影响训练。
                    if shared.device.type == 'cuda':
                        torch_rng_state = torch.get_rng_state()
                        cuda_gpu_rng_state = torch.cuda.get_rng_state(device="cuda")
                        cuda_cpu_rng_state = torch.cuda.get_rng_state(device="cpu")

                    optim_to(profiler, optimizer)

                    if profiler is None:
                        cleanup()

                    if vae is None:
                        printm("Loading vae.")
                        vae = create_vae()

                    printm("Creating pipeline.")
                    if args.model_type == "SDXL":
                        s_pipeline = StableDiffusionXLPipeline.from_pretrained(
                            args.get_pretrained_model_name_or_path(),
                            unet=accelerator.unwrap_model(unet, keep_fp32_wrapper=True),
                            text_encoder=accelerator.unwrap_model(
                                text_encoder, keep_fp32_wrapper=True
                            ),
                            text_encoder_2=accelerator.unwrap_model(
                                text_encoder_two, keep_fp32_wrapper=True
                            ),
                            vae=vae.to(accelerator.device),
                            torch_dtype=weight_dtype,
                            revision=args.revision,
                        )
                        xformerify(s_pipeline.unet,use_lora=args.use_lora)
                    else:
                        s_pipeline = DiffusionPipeline.from_pretrained(
                            args.get_pretrained_model_name_or_path(),
                            unet=accelerator.unwrap_model(unet, keep_fp32_wrapper=True),
                            text_encoder=accelerator.unwrap_model(
                                text_encoder, keep_fp32_wrapper=True
                            ),
                            vae=vae,
                            torch_dtype=weight_dtype,
                            revision=args.revision,
                        )
                        xformerify(s_pipeline.unet,use_lora=args.use_lora)
                        xformerify(s_pipeline.vae,use_lora=args.use_lora)

                    weights_dir = args.get_pretrained_model_name_or_path()

                    if user_model_dir != "":
                        loras_dir = os.path.join(user_model_dir, "Lora")
                    else:
                        model_dir = shared.models_path
                        loras_dir = os.path.join(model_dir, "Lora")
                    delete_tmp_lora = False
                    # 如果我们只需要保存图像，请更新临时路径
                    if save_image:
                        logger.debug("Save image is set.")
                        if args.use_lora:
                            if not save_lora:
                                logger.debug("Saving lora weights instead of checkpoint, using temp dir.")
                                save_lora = True
                                delete_tmp_lora = True
                                save_checkpoint = False
                                save_diffusers = False
                                os.makedirs(loras_dir, exist_ok=True)
                        elif not save_diffusers:
                            logger.debug("Saving checkpoint, using temp dir.")
                            save_diffusers = True
                            weights_dir = f"{weights_dir}_temp"
                            os.makedirs(weights_dir, exist_ok=True)
                        else:
                            save_lora = False
                            logger.debug(f"Save checkpoint: {save_checkpoint} save lora {save_lora}.")
                    # 这里需要inference_mode()来防止保存时出现问题吗?
                    logger.debug(f"Loras dir: {loras_dir}")

                    # 设置pt路径
                    if args.custom_model_name == "":
                        lora_model_name = args.model_name
                    else:
                        lora_model_name = args.custom_model_name

                    lora_save_file = os.path.join(loras_dir, f"{lora_model_name}_{args.revision}.safetensors")

                    with accelerator.autocast(), torch.inference_mode():

                        def lora_save_function(weights, filename):
                            metadata = args.export_ss_metadata()
                            logger.debug(f"Saving lora to {filename}")
                            safetensors.torch.save_file(weights, filename, metadata=metadata)

                        if save_lora:
                            # TODO: Add a version for the lora model?
                            pbar2.reset(1)
                            pbar2.set_description("Saving Lora Weights...")
                            # setup directory
                            logger.debug(f"Saving lora to {lora_save_file}")
                            unet_lora_layers_to_save = unet_lora_state_dict(unet)
                            text_encoder_one_lora_layers_to_save = None
                            text_encoder_two_lora_layers_to_save = None
                            if args.stop_text_encoder != 0:
                                text_encoder_one_lora_layers_to_save = text_encoder_lora_state_dict(text_encoder)
                            if args.model_type == "SDXL":
                                if args.stop_text_encoder != 0:
                                    text_encoder_two_lora_layers_to_save = text_encoder_lora_state_dict(text_encoder_two)
                                StableDiffusionXLPipeline.save_lora_weights(
                                    loras_dir,
                                    unet_lora_layers=unet_lora_layers_to_save,
                                    text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                                    text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
                                    weight_name=lora_save_file,
                                    safe_serialization=True,
                                    save_function=lora_save_function
                                )
                                scheduler_args = {}

                                if "variance_type" in s_pipeline.scheduler.config:
                                    variance_type = s_pipeline.scheduler.config.variance_type

                                    if variance_type in ["learned", "learned_range"]:
                                        variance_type = "fixed_small"

                                    scheduler_args["variance_type"] = variance_type

                                s_pipeline.scheduler = UniPCMultistepScheduler.from_config(s_pipeline.scheduler.config, **scheduler_args)
                                save_lora = False
                                save_model = False
                            else:
                                StableDiffusionPipeline.save_lora_weights(
                                    loras_dir,
                                    unet_lora_layers=unet_lora_layers_to_save,
                                    text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                                    weight_name=lora_save_file,
                                    safe_serialization=True
                                )
                                s_pipeline.scheduler = get_scheduler_class("UniPCMultistep").from_config(
                                    s_pipeline.scheduler.config)
                            s_pipeline.scheduler.config.solver_type = "bh2"
                            save_lora = False
                            save_model = False

                        elif save_diffusers:
                            # 我们正在节省权重，我们需要确保保存修订
                            if "_tmp" not in weights_dir:
                                args.save()
                            try:
                                out_file = None
                                status.textinfo = (
                                    f"Saving diffusion model at step {args.revision}..."
                                )
                                update_status({"status": status.textinfo})
                                pbar2.reset(1)

                                pbar2.set_description("Saving diffusion model")
                                s_pipeline.save_pretrained(
                                    weights_dir,
                                    safe_serialization=False,
                                )
                                if ema_model is not None:
                                    ema_model.save_pretrained(
                                        os.path.join(
                                            weights_dir,
                                            "ema_unet",
                                        ),
                                        safe_serialization=False,
                                    )
                                pbar2.update()

                                if save_snapshot:
                                    pbar2.reset(1)
                                    pbar2.set_description("Saving Snapshot")
                                    status.textinfo = (
                                        f"Saving snapshot at step {args.revision}..."
                                    )
                                    update_status({"status": status.textinfo})
                                    accelerator.save_state(
                                        os.path.join(
                                            args.model_dir,
                                            "checkpoints",
                                            f"checkpoint-{args.revision}",
                                        )
                                    )
                                    pbar2.update()

                                # 无论如何，我们都应该保存它，因为如果不存在快照，这是我们的备用方案。

                                # 将pt打包到检查点中
                                if save_checkpoint:
                                    pbar2.reset(1)
                                    pbar2.set_description("Compiling Checkpoint")
                                    snap_rev = str(args.revision) if save_snapshot else ""
                                    if export_diffusers:
                                        copy_diffusion_model(args.model_name, os.path.join(user_model_dir, "diffusers"))
                                    else:
                                        if args.model_type == "SDXL":
                                            compile_checkpoint_xl(args.model_name, reload_models=False,
                                                                  lora_file_name=out_file,
                                                                  log=False, snap_rev=snap_rev, pbar=pbar2)
                                        else:
                                            compile_checkpoint(args.model_name, reload_models=False,
                                                               lora_file_name=out_file,
                                                               log=False, snap_rev=snap_rev, pbar=pbar2)
                                    printm("Restored, moved to acc.device.")
                                    pbar2.update()

                            except Exception as ex:
                                logger.warning(f"Exception saving checkpoint/model: {ex}")
                                traceback.print_exc()
                                pass
                        save_dir = args.model_dir

                    if save_image:
                        logger.debug("Saving images...")
                        # Get the path to a temporary directory
                        del s_pipeline
                        logger.debug(f"Loading image pipeline from {weights_dir}...")
                        if args.model_type == "SDXL":
                            s_pipeline = StableDiffusionXLPipeline.from_pretrained(
                                weights_dir, vae=vae, revision=args.revision,
                                torch_dtype=weight_dtype
                            )
                        else:
                            s_pipeline = StableDiffusionPipeline.from_pretrained(
                                weights_dir, vae=vae, revision=args.revision,
                                torch_dtype=weight_dtype
                            )
                            if args.tomesd:
                                tomesd.apply_patch(s_pipeline, ratio=args.tomesd, use_rand=False)
                        if args.use_lora:
                            s_pipeline.load_lora_weights(lora_save_file)

                        try:
                            s_pipeline.enable_vae_tiling()
                            s_pipeline.enable_vae_slicing()
                            s_pipeline.enable_sequential_cpu_offload()
                            s_pipeline.enable_xformers_memory_efficient_attention()
                        except:
                            pass

                        samples = []
                        sample_prompts = []
                        last_samples = []
                        last_prompts = []
                        status.textinfo = (
                            f"Saving preview image(s) at step {args.revision}..."
                        )
                        update_status({"status": status.textinfo})
                        try:
                            s_pipeline.set_progress_bar_config(disable=True)
                            sample_dir = os.path.join(save_dir, "samples")
                            os.makedirs(sample_dir, exist_ok=True)

                            sd = SampleDataset(args)
                            prompts = sd.prompts
                            logger.debug(f"Generating {len(prompts)} samples...")

                            concepts = args.concepts()
                            if args.sanity_prompt:
                                epd = PromptData(
                                    prompt=args.sanity_prompt,
                                    seed=args.sanity_seed,
                                    negative_prompt=concepts[
                                        0
                                    ].save_sample_negative_prompt,
                                    resolution=(args.resolution, args.resolution),
                                )
                                prompts.append(epd)

                            prompt_lengths = len(prompts)
                            if args.disable_logging:
                                pbar2.reset(prompt_lengths)
                            else:
                                pbar2.reset(prompt_lengths + 2)
                            pbar2.set_description("Generating Samples")
                            ci = 0
                            for c in prompts:
                                c.out_dir = os.path.join(args.model_dir, "samples")
                                generator = torch.manual_seed(int(c.seed))
                                s_image = s_pipeline(
                                    c.prompt,
                                    num_inference_steps=c.steps,
                                    guidance_scale=c.scale,
                                    negative_prompt=c.negative_prompt,
                                    height=c.resolution[1],
                                    width=c.resolution[0],
                                    generator=generator,
                                ).images[0]
                                sample_prompts.append(c.prompt)
                                image_name = db_save_image(
                                    s_image,
                                    c,
                                    custom_name=f"sample_{args.revision}-{ci}",
                                )
                                shared.status.current_image = image_name
                                shared.status.sample_prompts = [c.prompt]
                                update_status({"images": [image_name], "prompts": [c.prompt]})
                                samples.append(image_name)
                                pbar2.update()
                                ci += 1
                            for sample in samples:
                                last_samples.append(sample)
                            for prompt in sample_prompts:
                                last_prompts.append(prompt)
                            del samples
                            del prompts
                        except:
                            logger.warning(f"Exception saving sample.")
                            traceback.print_exc()
                            pass

                        del s_pipeline
                        printm("Starting cleanup.")

                        if os.path.isdir(loras_dir) and "_tmp" in loras_dir:
                            shutil.rmtree(loras_dir)

                        if os.path.isdir(weights_dir) and "_tmp" in weights_dir:
                            shutil.rmtree(weights_dir)

                        if "generator" in locals():
                            del generator

                        if not args.disable_logging:
                            try:
                                printm("Parse logs.")
                                log_images, log_names = log_parser.parse_logs(model_name=args.model_name)
                                pbar2.update()
                                for log_image in log_images:
                                    last_samples.append(log_image)
                                for log_name in log_names:
                                    last_prompts.append(log_name)

                                del log_images
                                del log_names
                            except Exception as l:
                                traceback.print_exc()
                                logger.warning(f"Exception parsing logz: {l}")
                                pass

                        send_training_update(
                            last_samples,
                            args.model_name,
                            last_prompts,
                            global_step,
                            args.revision
                        )

                        status.sample_prompts = last_prompts
                        status.current_image = last_samples
                        update_status({"images": last_samples, "prompts": last_prompts})
                        pbar2.update()


                    if args.cache_latents:
                        printm("Unloading vae.")
                        del vae
                        # 再次保留引用
                        vae = None

                    status.current_image = last_samples
                    update_status({"images": last_samples})
                    cleanup()
                    printm("Cleanup.")

                    optim_to(profiler, optimizer, accelerator.device)

                    # 恢复所有随机状态，以避免进行采样影响训练。
                    if shared.device.type == 'cuda':
                        torch.set_rng_state(torch_rng_state)
                        torch.cuda.set_rng_state(cuda_cpu_rng_state, device="cpu")
                        torch.cuda.set_rng_state(cuda_gpu_rng_state, device="cuda")

                    cleanup()

                    # 如果我们要保存模型，则保存lora权重
                    if os.path.isfile(lora_save_file) and not delete_tmp_lora:
                        meta = args.export_ss_metadata()
                        convert_diffusers_to_kohya_lora(lora_save_file, meta, args.lora_weight)
                    else:
                        if os.path.isfile(lora_save_file):
                            os.remove(lora_save_file)

                    printm("Completed saving weights.")
                    pbar2.reset()

            # 在每台机器上只显示一次进度条，并且不将状态发送到新的UI。
            progress_bar = mytqdm(
                # 展示迭代范围 global_step 到 max_train_steps 
                range(global_step, max_train_steps),
                # 非本地运行不显示进度条
                disable=not accelerator.is_local_main_process,
                # 永远显示在命令行第一行
                position=0
            )
            # 设置进度条标签
            progress_bar.set_description("Steps")
            # 表示更新后刷新显示后缀
            progress_bar.set_postfix(refresh=True)
            # 将 args.revision 改为 int 类型
            args.revision = (
                args.revision if isinstance(args.revision, int) else
                int(args.revision) if str(args.revision).strip() else
                0
            )
            lifetime_step = args.revision       # 定义当前生命周期的步数
            lifetime_epoch = args.epoch         # 定义当前生命周期的轮数
            status.job_count = max_train_steps  # 
            status.job_no = global_step
            update_status({"progress_1_total": max_train_steps, "progress_1_job_current": global_step})
            training_complete = False
            msg = ""

            last_tenc = 0 < text_encoder_epochs
            if stop_text_percentage == 0:
                last_tenc = False

            cleanup()
            stats = {
                "loss": 0.0,
                "prior_loss": 0.0,
                "instance_loss": 0.0,
                "unet_lr": learning_rate,
                "tenc_lr": txt_learning_rate,
                "session_epoch": 0,                                         # 初始化轮数
                "lifetime_epoch": args.epoch,                               # 初始化当前生命周期的轮数
                "total_session_epoch": args.num_train_epochs,               # 
                "total_lifetime_epoch": args.epoch + args.num_train_epochs, # 计算整个训练生命周期的轮数
                "lifetime_step": args.revision,
                "session_step": 0,
                "total_session_step": max_train_steps,
                "total_lifetime_step": args.revision + max_train_steps,
                "steps_per_epoch": len(train_dataset),
                "iterations_per_second": 0.0,
                "vram": round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)
            }
            for epoch in range(first_epoch, max_train_epochs):
                if training_complete:
                    logger.debug("Training complete, breaking epoch.")
                    break
                
                # 训练Unet
                if args.train_unet:
                    unet.train()
                elif args.use_lora and not args.lora_use_buggy_requires_grad:
                    set_lora_requires_grad(unet, False)
                    
                # 判断是否继续进行CLIP训练
                train_tenc = epoch < text_encoder_epochs
                if stop_text_percentage == 0:
                    train_tenc = False
                    
                # 选择是否冻结clip的归一层
                if args.freeze_clip_normalization:
                    # 设置为评估模式，不训练
                    text_encoder.eval()
                    if args.model_type == "SDXL":
                        # 设置为评估模式，不训练
                        text_encoder_two.eval()
                else:
                    # 根据 train_tenc 判断是否进行训练模式
                    text_encoder.train(train_tenc)
                    if args.model_type == "SDXL":
                        text_encoder_two.train(train_tenc)

                if args.use_lora:
                    if not args.lora_use_buggy_requires_grad:
                        set_lora_requires_grad(text_encoder, train_tenc)
                        # 为了让渐变检查点工作，我们需要在输入上启用渐变
                        # 这不会被优化，因为它不是优化器的参数
                        text_encoder.text_model.embeddings.position_embedding.requires_grad_(train_tenc)
                        if args.model_type == "SDXL":
                            set_lora_requires_grad(text_encoder_two, train_tenc)
                            text_encoder_two.text_model.embeddings.position_embedding.requires_grad_(train_tenc)
                else:
                    # 根据 text_encoder 设置是否进行梯度计算
                    text_encoder.requires_grad_(train_tenc)
                    if args.model_type == "SDXL":
                        # 根据 text_encoder 设置是否进行梯度计算
                        text_encoder_two.requires_grad_(train_tenc)
                        
                # 根据当前轮数判断是否继续训练
                if last_tenc != train_tenc:
                    last_tenc = train_tenc
                    cleanup()
                    
                # 定义损失值
                loss_total = 0
                
                # 计算先验损失权重
                current_prior_loss_weight = current_prior_loss(
                    args, current_epoch=global_epoch
                )

                instance_loss = None    # 实例损失
                prior_loss = None       # 先验损失
                
                
                """
                #######################################################################################################
                @@@                                                                                                 @@@
                @@@                                           真正的训练部分                                          @@@
                @@@                                                                                                 @@@
                #######################################################################################################
                """
                
                
                for step, batch in enumerate(train_dataloader):
                    # 判断是否要从检查点开始训练
                    if (
                            resume_from_checkpoint
                            and epoch == first_epoch
                            and step < resume_step
                    ):
                        progress_bar.update(train_batch_size)
                        progress_bar.reset()
                        status.job_count = max_train_steps
                        status.job_no += train_batch_size
                        stats["session_step"] += train_batch_size
                        stats["lifetime_step"] += train_batch_size
                        update_status(stats)
                        continue

                    with ConditionalAccumulator(accelerator, unet, text_encoder, text_encoder_two):
                        # 将图像转换为潜空间
                        with torch.no_grad():       # 在上下文不进行梯度计算 
                            # 如果开启了缓存潜在变量
                            if args.cache_latents:
                                # 直接将图片扔进显卡
                                latents = batch["images"].to(accelerator.device)
                            # 没开就用vae进行编码后再扔进去
                            else:
                                latents = vae.encode(
                                    batch["images"].to(dtype=weight_dtype)
                                ).latent_dist.sample()  # 从潜在分布中随机采样一些值
                            latents = latents * 0.18215 # 对潜在空间进行缩放

                        # 我们将添加到模型输入的噪声样本
                        noise = torch.randn_like(latents, device=latents.device)    # 加噪
                        # 噪声偏移如果为0(简单来说是控制画面亮度的)
                        if args.offset_noise != 0:
                            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                            noise += args.offset_noise * torch.randn(
                                (latents.shape[0],
                                 latents.shape[1],
                                 1,
                                 1),
                                device=latents.device
                            )
                        # 依次传递参数
                        b_size, channels, height, width = latents.shape

                        # 对每个图像随机采样一个时间步长
                        timesteps = torch.randint(
                            0,
                            noise_scheduler.config.num_train_timesteps,
                            (b_size,),
                            device=latents.device
                        )
                        timesteps = timesteps.long()

                        # 根据每个时间步长的噪声大小向潜函数添加噪声
                        # (这是正向扩散过程)
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)    # 进行一次性加噪，获得 noisy_latents 最终加噪结果
                        # 如果训练 CLIP 查看是否进行文本填充
                        pad_tokens = args.pad_tokens if train_tenc else False
                        input_ids = batch["input_ids"]
                        encoder_hidden_states = None
                        if args.model_type != "SDXL" and text_encoder is not None:
                            # 获取隐藏状态
                            encoder_hidden_states = encode_hidden_state(
                                text_encoder,
                                batch["input_ids"],
                                pad_tokens,
                                b_size,
                                args.max_token_length,
                                tokenizer_max_length,
                                args.clip_skip,
                            )
                            
                            
                        # 如果Unet要的输入通道大于VAE的潜在空间通道
                        if unet.config.in_channels > channels:
                            # 计算差多少
                            needed_additional_channels = unet.config.in_channels - channels
                            # 随机一个张量
                            additional_latents = randn_tensor(
                                (b_size, needed_additional_channels, height, width),
                                device=noisy_latents.device,
                                dtype=noisy_latents.dtype,
                            )
                            # 将随机出来的张量和原来的噪声相加，得到一个新噪声
                            noisy_latents = torch.cat([additional_latents, noisy_latents], dim=1)
                        # 根据预测类型获得损失的目标
                        # epsilon 目标函数直接和噪声拟合
                        if noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        # v_prediction 目标函数根据潜在变量、噪声和时间步，拟合潜在变量在时间上的变化
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            target = noise_scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                        if args.model_type == "SDXL":
                            # 在上下文里使用混合精度进行训练
                            with accelerator.autocast():
                                model_pred = unet(
                                    noisy_latents, timesteps, batch["input_ids"],
                                    added_cond_kwargs=batch["unet_added_conditions"]
                                ).sample
                        else:
                            # 预测噪声残差并计算损失
                            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                        if args.model_type != "SDXL":
                            # 待办事项:设置一个优先保存标志，并使用它来确保这只发生在dreambooth中
                            if not args.split_loss and not with_prior_preservation:
                                # 使用均方误差计算损失
                                loss = instance_loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
                                # 
                                loss *= batch["loss_avg"]
                            else:
                                # 预测噪声残差
                                if model_pred.shape[1] == 6:
                                    model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                                if model_pred.shape[0] > 1 and with_prior_preservation:
                                        # 将噪声和model_pred分成两部分，分别计算每个部分的损失。
                                        print("model shape:")
                                        print(model_pred.shape)
                                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                                        target, target_prior = torch.chunk(target, 2, dim=0)

                                        # 计算实例损失
                                        loss = instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                                        # 计算先验损失
                                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(),
                                                                reduction="mean")
                                else:
                                    # 计算损失
                                    loss = instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                        else:
                            if with_prior_preservation:
                                # 将噪声和model_pred分成两部分，分别计算每个部分的损失。
                                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                                target, target_prior = torch.chunk(target, 2, dim=0)

                                # 计算实例丢失
                                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                                # 计算先验损失
                                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                                # 将之前的损失添加到实例损失中。
                                loss = loss + args.prior_loss_weight * prior_loss   # 损失 = 实例损失 + 损失权重 * 先验损失
                            else:
                                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                                
                        # 反向传播
                        accelerator.backward(loss)
                        
                        # 如果启用了梯度同步并且没有使用 Lora 则进行梯度裁剪
                        if accelerator.sync_gradients and not args.use_lora:
                            if train_tenc:
                                if args.model_type == "SDXL":
                                    params_to_clip = itertools.chain(unet.parameters(), text_encoder.parameters(),
                                                                     text_encoder_two.parameters())
                                else:
                                    params_to_clip = itertools.chain(unet.parameters(), text_encoder.parameters())
                            else:
                                params_to_clip = unet.parameters()
                            accelerator.clip_grad_norm_(params_to_clip, 1)
                            
                        # 更新参数
                        optimizer.step()
                        # 更新学习率
                        lr_scheduler.step(train_batch_size)
                        # 更新 ema模型 参数
                        if args.use_ema and ema_model is not None:
                            ema_model.step(unet)
                        # 进行性能分析
                        if profiler is not None:
                            profiler.step()
                            
                        # 判断是将梯度清零还是为空
                        optimizer.zero_grad(set_to_none=args.gradient_set_to_none)
                        
                    # 返回使用的显存
                    allocated = round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)
                    # 返回当前所有的显存
                    cached = round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)
                    # 返回最后一个学习率
                    lr_data = lr_scheduler.get_last_lr()
                    # 将最后一个学习率赋值给 last_lr 
                    last_lr = lr_data[0]
                    # 初始化最后一个文本学习率
                    last_tenc_lr = 0
                    # 为 stats 添加一个键值对
                    stats["lr_data"] = lr_data
                    # 获取文本编码器的学习率
                    try:
                        if len(optimizer.param_groups) > 1:
                            last_tenc_lr = optimizer.param_groups[1]["lr"] if train_tenc else 0
                    except:
                        logger.debug("Exception getting tenc lr")
                        pass
                    
                    
                    if 'adapt' in args.optimizer:
                        last_lr = optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]
                        if len(optimizer.param_groups) > 1:
                            try:
                                # 计算衰减后的文本编码器学习率
                                last_tenc_lr = optimizer.param_groups[1]["d"] * optimizer.param_groups[1]["lr"]
                            except:
                                logger.warning("Exception setting tenc weight decay")
                                traceback.print_exc()

                    update_status(stats)
                    del latents
                    del encoder_hidden_states
                    del noise
                    del timesteps
                    del noisy_latents
                    del target
                    
                    # 更新全局进度参数
                    global_step += train_batch_size
                    args.revision += train_batch_size
                    status.job_no += train_batch_size
                    # 将损失从计算图中分离出来，并转换为标量
                    loss_step = loss.detach().item()
                    # 跟踪训练过程中的损失值
                    loss_total += loss_step
                    
                    # 更新状态栏字段
                    stats["session_step"] += train_batch_size
                    stats["lifetime_step"] += train_batch_size
                    stats["loss"] = loss_step

                    logs = {
                        "lr": float(last_lr),
                        "loss": float(loss_step),
                        "vram": float(cached),
                    }
                    
                    # 为状态栏添加显示字段
                    stats["vram"] = logs["vram"]
                    stats["unet_lr"] = '{:.2E}'.format(Decimal(last_lr))
                    stats["tenc_lr"] = '{:.2E}'.format(Decimal(last_tenc_lr))

                    if args.split_loss and with_prior_preservation and args.model_type != "SDXL":
                        # 为 logs 新添实例损失字段
                        logs["inst_loss"] = float(instance_loss.detach().item())
                        
                        # 为 logs 新添先验损失字段
                        if prior_loss is not None:
                            logs["prior_loss"] = float(prior_loss.detach().item())
                        else:
                            logs["prior_loss"] = None  # 或者其他默认值
                            
                        # 为状态栏添加显示字段
                        stats["instance_loss"] = logs["inst_loss"]
                        stats["prior_loss"] = logs["prior_loss"]

                    if 'adapt' in args.optimizer:
                        status.textinfo2 = (
                            f"Loss: {'%.2f' % loss_step}, UNET DLR: {'{:.2E}'.format(Decimal(last_lr))}, TENC DLR: {'{:.2E}'.format(Decimal(last_tenc_lr))}, "
                            f"VRAM: {allocated}/{cached} GB"
                        )
                    else:
                        status.textinfo2 = (
                            f"Loss: {'%.2f' % loss_step}, LR: {'{:.2E}'.format(Decimal(last_lr))}, "
                            f"VRAM: {allocated}/{cached} GB"
                        )

                    progress_bar.update(train_batch_size)
                    
                    # 如果 mytqdm 里有 rate 就进行赋值，没有就设置为None
                    rate = progress_bar.format_dict["rate"] if "rate" in progress_bar.format_dict else None
                    if rate is None:
                        rate_string = ""
                    else:
                        # 选择显示速率的方式
                        if rate > 1:
                            rate_string = f"{rate:.2f} it/s"
                        else:
                            rate_string = f"{1 / rate:.2f} s/it" if rate != 0 else "N/A"
                    # 为状态栏添加速率字段
                    stats["iterations_per_second"] = rate_string
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=args.revision)

                    logs = {"epoch_loss": loss_total / len(train_dataloader)}
                    accelerator.log(logs, step=global_step)
                    stats["epoch_loss"] = '%.2f' % (loss_total / len(train_dataloader))

                    status.job_count = max_train_steps
                    status.job_no = global_step
                    stats["lifetime_step"] = args.revision
                    stats["session_step"] = global_step
                    # status0 = f"Steps: {global_step}/{max_train_steps} (Current), {rate_string}"
                    # status1 = f"{args.revision}/{lifetime_step + max_train_steps} (Lifetime), Epoch: {global_epoch}"
                    status.textinfo = (
                        f"Steps: {global_step}/{max_train_steps} (Current), {rate_string}"
                        f" {args.revision}/{lifetime_step + max_train_steps} (Lifetime), Epoch: {global_epoch}"
                    )
                    update_status(stats)
                    
                    # 检测 loss_step 是否为 NaN（Not a Number），如果是就中止训练
                    if math.isnan(loss_step):
                        logger.warning("Loss is NaN, your model is dead. Cancelling training.")
                        status.interrupted = True
                        if status_handler:
                            status_handler.end("Training interrrupted due to NaN loss.")

                    # 日志完成消息
                    if training_complete or status.interrupted:
                        # 不进行中止
                        shared.in_progress = False
                        # 将轮数和步数清零
                        shared.in_progress_step = 0
                        shared.in_progress_epoch = 0
                        logger.debug("  Training complete (step check).")
                        # 根据是否中止程序判断当前状态
                        if status.interrupted:
                            state = "canceled"
                        else:
                            state = "complete"
                            
                        # 训练状态
                        status.textinfo = (
                            f"Training {state} {global_step}/{max_train_steps}, {args.revision}"
                            f" total."
                        )
                        if status_handler:
                            status_handler.end(status.textinfo)
                        break
                    
                """
                #######################################################################################################
                @@@                                                                                                 @@@
                @@@                                     一批次训练部分结束                                            @@@
                @@@                                                                                                 @@@
                #######################################################################################################
                """
                
                
                
                # 等待每个进程结束
                accelerator.wait_for_everyone()
                
                # 进项参数更新
                args.epoch += 1
                global_epoch += 1
                lifetime_epoch += 1
                session_epoch += 1
                stats["session_epoch"] += 1
                stats["lifetime_epoch"] += 1
                lr_scheduler.step(is_epoch=True)
                status.job_count = max_train_steps
                status.job_no = global_step
                update_status(stats)
                # 运行保存函数
                check_save(True)

                if args.num_train_epochs > 1:
                    training_complete = session_epoch >= max_train_epochs

                if training_complete or status.interrupted:
                    logger.debug("  Training complete (step check).")
                    if status.interrupted:
                        state = "canceled"
                    else:
                        state = "complete"

                    status.textinfo = (
                        f"Training {state} {global_step}/{max_train_steps}, {args.revision}"
                        f" total."
                    )
                    if status_handler:
                        status_handler.end(status.textinfo)
                    break

                # 在时代的最后做这件事，在我们确定还没有完成之后
                if args.epoch_pause_frequency > 0 and args.epoch_pause_time > 0:
                    if not session_epoch % args.epoch_pause_frequency:
                        logger.debug(
                            f"Giving the GPU a break for {args.epoch_pause_time} seconds."
                        )
                        for i in range(args.epoch_pause_time):
                            if status.interrupted:
                                training_complete = True
                                logger.debug("Training complete, interrupted.")
                                if status_handler:
                                    status_handler.end("Training interrrupted.")
                                break
                            time.sleep(1)
            """
                #######################################################################################################
                @@@                                                                                                 @@@
                @@@                                     全部训练部分结束                                              @@@
                @@@                                                                                                 @@@
                #######################################################################################################
            """

            cleanup_memory()
            accelerator.end_training()
            result.msg = msg
            result.config = args
            result.samples = last_samples
            stop_profiler(profiler)
            return result

    return inner_loop()
