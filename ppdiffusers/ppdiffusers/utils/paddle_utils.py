# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
Paddle utilities: Utilities related to Paddle
"""
import contextlib
import threading
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

from .import_utils import is_paddle_available
from .logging import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


# dummpy decorator, we do not use it
def maybe_allow_in_graph(cls):
    return cls


if is_paddle_available():
    import paddle

    class RNGStatesTracker:
        def __init__(self):
            self.states_ = {}
            self.mutex = threading.Lock()

        def reset(self):
            with self.mutex:
                self.states_ = {}

        def remove(self, generator_name=None):
            with self.mutex:
                if generator_name is not None:
                    del self.states_[generator_name]

        def manual_seed(self, seed, generator_name=None):
            with self.mutex:
                if generator_name is None:
                    generator_name = str(time.time())
                if generator_name in self.states_:
                    raise ValueError("state {} already exists".format(generator_name))
                orig_rng_state = paddle.get_cuda_rng_state()
                paddle.seed(seed)
                self.states_[generator_name] = paddle.get_cuda_rng_state()
                paddle.set_cuda_rng_state(orig_rng_state)
                return generator_name

        @contextlib.contextmanager
        def rng_state(self, generator_name=None):
            if generator_name is not None:
                if generator_name not in self.states_:
                    raise ValueError("state {} does not exist".format(generator_name))
                with self.mutex:
                    orig_cuda_rng_state = paddle.get_cuda_rng_state()
                    paddle.set_cuda_rng_state(self.states_[generator_name])
                    try:
                        yield
                    finally:
                        self.states_[generator_name] = paddle.get_cuda_rng_state()
                        paddle.set_cuda_rng_state(orig_cuda_rng_state)
            else:
                yield

    RNG_STATE_TRACKER = RNGStatesTracker()

    def get_rng_state_tracker(*args, **kwargs):
        return RNG_STATE_TRACKER

    paddle.Generator = get_rng_state_tracker

    randn = paddle.randn
    rand = paddle.rand
    randint = paddle.randint

    @paddle.jit.not_to_static
    def randn_pt(shape, dtype=None, name=None, **kwargs):
        generator = kwargs.get("generator", None)
        is_bfloat16 = "bfloat16" in str(dtype) or "bfloat16" in paddle.get_default_dtype()
        if is_bfloat16:
            if generator is None:
                return randn(shape, dtype="float16", name=name).cast(paddle.bfloat16)
            else:
                with get_rng_state_tracker().rng_state(generator):
                    return randn(shape, dtype="float16", name=name).cast(paddle.bfloat16)
        else:
            if generator is None:
                return randn(shape, dtype=dtype, name=name)
            else:
                with get_rng_state_tracker().rng_state(generator):
                    return randn(shape, dtype=dtype, name=name)

    @paddle.jit.not_to_static
    def rand_pt(shape, dtype=None, name=None, **kwargs):
        generator = kwargs.get("generator", None)
        if generator is None:
            return rand(shape, dtype=dtype, name=name)
        else:
            with get_rng_state_tracker().rng_state(generator):
                return rand(shape, dtype=dtype, name=name)

    @paddle.jit.not_to_static
    def randint_pt(low=0, high=None, shape=[1], dtype=None, name=None, **kwargs):
        generator = kwargs.get("generator", None)
        if generator is None:
            return randint(low=low, high=high, shape=shape, dtype=dtype, name=name)
        else:
            with get_rng_state_tracker().rng_state(generator):
                return randint(low=low, high=high, shape=shape, dtype=dtype, name=name)

    @paddle.jit.not_to_static
    def randn_like_pt(x, dtype=None, name=None, **kwargs):
        generator = kwargs.get("generator", None)
        if dtype is None:
            dtype = x.dtype
        return randn_pt(x.shape, dtype=dtype, generator=generator, name=name, **kwargs)

    paddle.randn = randn_pt
    paddle.rand = rand_pt
    paddle.randint = randint_pt
    paddle.randn_like = randn_like_pt

    def randn_tensor(
        shape: Union[Tuple, List],
        generator: Optional[Union[List["paddle.Generator"], "paddle.Generator"]] = None,
        dtype: Optional["paddle.dtype"] = None,
        *kwargs,
    ):
        """A helper function to create random tensors with the desired `dtype`. When
        passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
        is always created on the CPU.
        """
        # make sure generator list of length 1 is treated like a non-list
        if isinstance(generator, list) and len(generator) == 1:
            generator = generator[0]

        if isinstance(generator, (list, tuple)):
            batch_size = shape[0]
            shape = (1,) + tuple(shape[1:])
            latents = [randn_pt(shape, generator=generator[i], dtype=dtype) for i in range(batch_size)]
            latents = paddle.concat(latents, axis=0)
        else:
            latents = randn_pt(shape, generator=generator, dtype=dtype)

        return latents

    def rand_tensor(
        shape: Union[Tuple, List],
        generator: Optional[Union[List["paddle.Generator"], "paddle.Generator"]] = None,
        dtype: Optional["paddle.dtype"] = None,
        *kwargs,
    ):
        """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
        passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
        will always be created on CPU.
        """
        if isinstance(generator, (list, tuple)):
            batch_size = shape[0]
            shape = (1,) + tuple(shape[1:])
            latents = [rand_pt(shape, generator=generator[i], dtype=dtype) for i in range(batch_size)]
            latents = paddle.concat(latents, axis=0)
        else:
            latents = rand_pt(shape, generator=generator, dtype=dtype)

        return latents

    def randint_tensor(
        low=0,
        high=None,
        shape: Union[Tuple, List] = [1],
        generator: Optional["paddle.Generator"] = None,
        dtype: Optional["paddle.dtype"] = None,
        *kwargs,
    ):
        """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
        passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
        will always be created on CPU.
        """
        latents = randint_pt(low=low, high=high, shape=shape, dtype=dtype, generator=generator)

        return latents

    @contextmanager
    def dtype_guard(dtype="float32"):
        if isinstance(dtype, paddle.dtype):
            dtype = str(dtype).replace("paddle.", "")
        origin_dtype = paddle.get_default_dtype()
        paddle.set_default_dtype(dtype)
        try:
            yield
        finally:
            paddle.set_default_dtype(origin_dtype)

    paddle.dtype_guard = dtype_guard

    _init_weights = True

    @contextmanager
    def no_init_weights(_enable=True):
        """
        Context manager to globally disable weight initialization to speed up loading large models.

        TODO(Patrick): Delete safety argument `_enable=True` at next major version. .
        """
        global _init_weights
        old_init_weights = _init_weights
        if _enable:
            _init_weights = False
        try:
            yield
        finally:
            _init_weights = old_init_weights
