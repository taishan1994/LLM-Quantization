# coding=utf-8
# Copyright (c) Meta
# All rights reserved.

import functools
import inspect
from enum import Enum
from typing import Any, Dict, Union

from . import utils
import torch
from fairscale.nn.data_parallel import (
    FullyShardedDataParallel as FullyShardedDDP,
    ShardedDataParallel as ShardedDDP,
)
from fairscale.nn.wrap import auto_wrap
from torch import nn
from torch.nn import functional as F, MSELoss
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel, unwrap_model

# ---- Transformers imports with compatibility shims ----
try:
    # Old + some mid versions
    from transformers.trainer_utils import FSDPOption, has_length, ShardedDDPOption
except Exception:
    # Newer Transformers: ShardedDDPOption removed
    from transformers.trainer_utils import FSDPOption, has_length  # type: ignore

    class ShardedDDPOption(str, Enum):
        SIMPLE = "simple"
        ZERO_DP_2 = "zero_dp_2"
        ZERO_DP_3 = "zero_dp_3"
        OFFLOAD = "offload"
        AUTO_WRAP = "auto_wrap"

from transformers.utils import is_torch_neuroncore_available, logging

logger = logging.get_logger(__name__)
local_rank = utils.get_local_rank()

mse_loss = MSELoss()


def _to_opt_name(x) -> str:
    """
    Normalize an option (enum/str/other) to lower-case string name.
    - For Enum like ShardedDDPOption.ZERO_DP_3 -> 'zero_dp_3'
    - For 'ShardedDDPOption.ZERO_DP_3' or 'ZERO_DP_3' -> 'zero_dp_3'
    - For arbitrary str -> lower
    """
    if x is None:
        return ""
    s = str(x)
    # split by '.' to handle 'ShardedDDPOption.ZERO_DP_3'
    if "." in s:
        s = s.split(".")[-1]
    return s.lower()


def _as_opt_set(v) -> set:
    """
    Normalize self.args.sharded_ddp to a set of lower-case option names.
    Accepts None / str / enum / list/tuple/set.
    """
    if v is None:
        return set()
    if isinstance(v, (list, tuple, set)):
        return {_to_opt_name(x) for x in v}
    return {_to_opt_name(v)}


class KDTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ---------------- KD losses ----------------
    def ce_loss(self, size_average, student_logits, teacher_logits):
        model_output_log_prob = F.log_softmax(student_logits, dim=2)
        real_output_soft = F.softmax(teacher_logits, dim=2)
        loss = F.kl_div(model_output_log_prob, real_output_soft, reduction="batchmean")
        return loss

    def mse_loss(self, student_logits, teacher_logits):
        return mse_loss(student_logits, teacher_logits)

    def compute_loss_train(self, model, inputs, return_outputs=False):
        with torch.no_grad():
            teacher_outputs = model.teacher(
                **inputs
                # **inputs, output_hidden_states=True, output_attentions=True
            )
        teacher_logits = teacher_outputs.get("logits")
        del teacher_outputs

        student_outputs = model(**inputs)
        student_logits = student_outputs.get("logits")

        if not return_outputs:
            del student_outputs

        kd_loss = 0.0
        size_average = True
        if getattr(model, "kd_loss_scale", 0.0) > 0.0:
            kd_loss = self.ce_loss(size_average, student_logits, teacher_logits)

        del teacher_logits
        del student_logits

        tok_loss = model.kd_loss_scale * kd_loss
        return (tok_loss, student_outputs) if return_outputs else tok_loss

    # ---------------- training step ----------------
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss_train(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps

        # Native AMP / Deepspeed / FP32
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.deepspeed:
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    # ---------------- wrapping ----------------
    def _wrap_model(self, model, training=True, dataloader=None):
        if self.args.use_ipex:
            dtype = torch.bfloat16 if self.use_cpu_amp else torch.float32
            model = self.ipex_optimize_model(model, training, dtype=dtype)

        # Already wrapped?
        if unwrap_model(model) is not model:
            return model

        # DataParallel (8bit models do not support DDP)
        if self.args.n_gpu > 1 and not getattr(model, "is_loaded_in_8bit", False):
            model = nn.DataParallel(model)

        # --- FairScale ShardedDDP / FSDP (FairScale) path ---
        sharded_opts = _as_opt_set(getattr(self.args, "sharded_ddp", None))
        use_sharded = len(sharded_opts) > 0

        if use_sharded:
            # SIMPLE -> ShardedDDP
            if "simple" in sharded_opts:
                model = ShardedDDP(model, self.optimizer)
            else:
                mixed_precision = bool(self.args.fp16 or self.args.bf16)
                cpu_offload = "offload" in sharded_opts
                zero_3 = "zero_dp_3" in sharded_opts
                if "auto_wrap" in sharded_opts:
                    model = auto_wrap(model)
                self.model = model = FullyShardedDDP(
                    model,
                    mixed_precision=mixed_precision,
                    reshard_after_forward=zero_3,
                    cpu_offload=cpu_offload,
                ).to(self.args.device)

        # --- PyTorch FSDP path ---
        elif self.fsdp is not None:
            if not self.args.fsdp_config.get("xla", False):
                from torch.distributed.fsdp.fully_sharded_data_parallel import (
                    CPUOffload,
                    FullyShardedDataParallel as FSDP,
                    MixedPrecision,
                )
                from torch.distributed.fsdp.wrap import (
                    size_based_auto_wrap_policy,
                    transformer_auto_wrap_policy,
                )

                if FSDPOption.OFFLOAD in self.args.fsdp:
                    cpu_offload = CPUOffload(offload_params=True)
                else:
                    cpu_offload = CPUOffload(offload_params=False)

                auto_wrap_policy = None
                if FSDPOption.AUTO_WRAP in self.args.fsdp:
                    if self.args.fsdp_config.get("fsdp_min_num_params", 0) > 0:
                        auto_wrap_policy = functools.partial(
                            size_based_auto_wrap_policy,
                            min_num_params=self.args.fsdp_config["fsdp_min_num_params"],
                        )
                    elif self.args.fsdp_config.get(
                        "fsdp_transformer_layer_cls_to_wrap", None
                    ):
                        transformer_cls_to_wrap = set()
                        for layer_class in self.args.fsdp_config[
                            "fsdp_transformer_layer_cls_to_wrap"
                        ]:
                            transformer_cls = self._get_module_class_from_name_safe(
                                model, layer_class
                            )
                            if transformer_cls is None:
                                raise Exception(
                                    "Could not find the transformer layer class to wrap in the model."
                                )
                            transformer_cls_to_wrap.add(transformer_cls)
                        auto_wrap_policy = functools.partial(
                            transformer_auto_wrap_policy,
                            transformer_layer_cls=transformer_cls_to_wrap,
                        )

                mixed_precision_policy = None
                dtype = None
                if self.args.fp16:
                    dtype = torch.float16
                elif self.args.bf16:
                    dtype = torch.bfloat16
                if dtype is not None:
                    mixed_precision_policy = MixedPrecision(
                        param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype
                    )

                if type(model) != FSDP:
                    signature = inspect.signature(FSDP.__init__).parameters.keys()
                    kwargs = {}
                    for arg in [
                        "limit_all_gathers",
                        "forward_prefetch",
                        "backward_prefetch",
                    ]:
                        if arg in signature:
                            kwargs[arg] = getattr(self, arg, None)
                    kwargs["limit_all_gathers"] = True
                    self.model = model = FSDP(
                        model,
                        sharding_strategy=self.fsdp,
                        cpu_offload=cpu_offload,
                        auto_wrap_policy=auto_wrap_policy,
                        mixed_precision=mixed_precision_policy,
                        device_id=self.args.device,
                        ignored_modules=None
                        if getattr(model, "teacher", None) is None
                        else [model.teacher],
                        **kwargs,
                    )

        # --- Vanilla DDP (torch.distributed) ---
        elif self.args.local_rank != -1:
            kwargs = {}
            if self.args.ddp_find_unused_parameters is not None:
                kwargs["find_unused_parameters"] = self.args.ddp_find_unused_parameters
            elif isinstance(model, PreTrainedModel):
                kwargs["find_unused_parameters"] = not model.is_gradient_checkpointing
            else:
                kwargs["find_unused_parameters"] = True

            if self.args.ddp_bucket_cap_mb is not None:
                kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb
            if is_torch_neuroncore_available():
                return model
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank] if self.args._n_gpu != 0 else None,
                output_device=self.args.local_rank if self.args._n_gpu != 0 else None,
                **kwargs,
            )

        # torch.compile after wrapping
        if self.args.torch_compile:
            model = torch.compile(
                model,
                backend=self.args.torch_compile_backend,
                mode=self.args.torch_compile_mode,
            )

        return model

    # --- helper to keep original behavior but avoid import loops ---
    @staticmethod
    def _get_module_class_from_name_safe(model, name: str):
        from transformers.trainer_pt_utils import get_module_class_from_name

        return get_module_class_from_name(model, name)
