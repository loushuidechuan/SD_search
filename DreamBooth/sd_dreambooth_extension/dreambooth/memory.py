# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A collection of utilities for ensuring that training can always occur. Heavily influenced by the
[toma](https://github.com/BlackHC/toma) library.
"""

import functools
import gc
import inspect
import traceback

import torch
import torch.backends.cudnn

from dreambooth import shared
from dreambooth.utils.utils import cleanup


def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False


profiler = None


def find_executable_batch_size(function: callable = None, starting_batch_size: int = 128,
                               starting_grad_size: int = 128, logging_dir: str = "", cleanup_function: callable = None):
    """
        一个基本的装饰器，将尝试执行' function '。如果由于与内存不足或
        在CUDNN中,批大小减半并传递给“function”

        ' function '必须接受' batch_size '形参作为其第一个参数。

        参数:
        function(' callable ', *optional*):
        要包装的function
        Starting_batch_size (' int ', *可选*):
        尝试将批处理大小装入内存
        starting_grad_size:
        累加的起始步数要使用。每循环除以2。
        logging_dir:
        用于日志记录的目录。
        cleanup_function:
        每次循环后调用的function。用于清除内存。
    """

    global profiler
    try:
        profile_memory = shared.profile_db
    except Exception:
        profile_memory = False

    torch.backends.cudnn.benchmark = not profile_memory

    if profile_memory and profiler is None:
        from torch.profiler import profile

        cleanup(True)

        profiler = profile(
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=100, repeat=100),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'{logging_dir}'),
            with_stack=True,
            profile_memory=True)
        print("Starting profiler...")
        profiler.start()
    else:
        prof = None

    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size,
                                 starting_grad_size=starting_grad_size, logging_dir=logging_dir)

    batch_size = starting_batch_size
    grad_size = starting_grad_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        nonlocal grad_size
        nonlocal prof
        gc.collect()
        torch.cuda.empty_cache()
        params = list(inspect.signature(function).parameters.keys())
        # Guard against user error
        if len(params) < (len(args) + 1):
            arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
        while True:
            if batch_size == 0:
                gc.collect()
                torch.cuda.empty_cache()
                # Execute cleanup_function if it is not None
                if cleanup_function is not None:
                    cleanup_function()
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, grad_size, prof, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    grad_size //= 2
                    if grad_size == 0:
                        grad_size = 1
                    print(f"OOM Detected, reducing batch/grad size to {batch_size}/{grad_size}.")
                    traceback.print_exc()
                else:
                    raise

    return decorator
