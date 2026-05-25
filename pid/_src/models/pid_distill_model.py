# PID distillation model — inference subset of the DMD2-distilled student.
#
# The training-time teacher / fake_score / discriminator / DMD-loss machinery has been
# stripped; what remains is the student net (`self.net`) plus the few-step sampler
# (`_get_t_list`, `_student_sample_loop`, `_velocity_to_x0`) consumed by
# `generate_samples_from_batch`.

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from contextlib import nullcontext
from typing import Optional

import attrs
import torch

from pid._src.models.pid_model import PidModel, PidModelConfig

logger = logging.getLogger(__name__)


@attrs.define(slots=False)
class PidDistillModelConfig(PidModelConfig):
    """Inference config for the distilled student."""

    # Few-step student schedule.
    student_timestep: float = 1.0
    student_sample_steps: int = 1
    student_sample_type: str = "sde"
    student_t_list: Optional[list] = None
    student_input_mode: str = "teacher_forcing"


class PidDistillModel(PidModel):
    """Inference-only PID distilled student."""

    def __init__(self, config: PidDistillModelConfig):
        # Stubs left in place so any parent code that probes for these attributes
        # gets None instead of AttributeError.
        self.teacher = None
        self.fake_score = None
        self.discriminator = None
        super().__init__(config)

    # ---------------------------------------------------------------------
    # Net output ↔ (x0, velocity) conversion
    # ---------------------------------------------------------------------

    def _net_output_to_x0(
        self,
        x_t: torch.Tensor,
        net_output: torch.Tensor,
        t: torch.Tensor,
        prediction_type: str,
    ) -> torch.Tensor:
        if prediction_type == "x0":
            return net_output.to(x_t.dtype)
        if prediction_type == "velocity":
            original_dtype = x_t.dtype
            s = [x_t.shape[0]] + [1] * (x_t.ndim - 1)
            t_shaped = t.double().view(*s)
            return (x_t.double() - t_shaped * net_output.double()).to(original_dtype)
        raise ValueError(f"Invalid prediction_type: {prediction_type}")

    def _net_output_to_velocity(
        self,
        x_t: torch.Tensor,
        net_output: torch.Tensor,
        t: torch.Tensor,
        prediction_type: str,
    ) -> torch.Tensor:
        if prediction_type == "velocity":
            return net_output
        if prediction_type == "x0":
            original_dtype = x_t.dtype
            s = [x_t.shape[0]] + [1] * (x_t.ndim - 1)
            t_shaped = t.double().view(*s).clamp(min=5e-2)
            return ((x_t.double() - net_output.double()) / t_shaped).to(original_dtype)
        raise ValueError(f"Invalid prediction_type: {prediction_type}")

    def _velocity_to_x0(self, x_t: torch.Tensor, net_output: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self._net_output_to_x0(x_t, net_output, t, self.config.prediction_type)

    # ---------------------------------------------------------------------
    # Multi-step student sampler
    # ---------------------------------------------------------------------

    def _get_t_list(self, device, num_steps: Optional[int] = None, shift: Optional[float] = None, scheduler: str = "default") -> torch.Tensor:
        target_steps = num_steps if num_steps is not None else self.config.student_sample_steps

        if scheduler == "default" and self.config.student_t_list is not None:
            full_t = torch.tensor(self.config.student_t_list, device=device, dtype=torch.float32)
            if target_steps != self.config.student_sample_steps:
                indices = torch.linspace(0, len(full_t) - 1, target_steps + 1).round().long()
                t_list = full_t[indices]
            else:
                t_list = full_t
        elif scheduler == "karras":
            # Karras schedule (rho=7)
            # t goes from 1.0 to 0.0
            steps = torch.linspace(0.0, 1.0, target_steps + 1, device=device)
            t_list = (1.0 - steps) ** 7
        elif scheduler == "exponential":
            # Exponential decay schedule
            epsilon = 0.002
            steps = torch.linspace(0.0, 1.0, target_steps + 1, device=device)
            t_list = torch.exp(steps * math.log(epsilon))
            t_list[-1] = 0.0
        elif scheduler == "cosine":
            # Cosine schedule
            steps = torch.linspace(0.0, 1.0, target_steps + 1, device=device)
            t_list = torch.cos(steps * (math.pi / 2))
        else: # "linear" or fallback
            t_list = torch.linspace(
                self.config.student_timestep,
                0.0,
                target_steps + 1,
                device=device,
                dtype=torch.float32,
            )

        if shift is not None and shift > 0.0:
            # Apply schedule shift formula: t_shifted = shift * t / (1 + (shift - 1) * t)
            t_list = shift * t_list / (1.0 + (shift - 1.0) * t_list)

        # Ensure t_list starts at student_timestep (e.g. 1.0 or whatever the first element was) and ends at 0.0
        t_list = torch.clamp(t_list, min=0.0, max=1.0)
        t_list[-1] = 0.0

        assert abs(t_list[-1].item()) < 1e-6, "t_list must end at 0"
        if num_steps is not None:
            logger.info(f"[distill inference] num_steps={num_steps}, scheduler={scheduler}, shift={shift}, t_list={t_list.tolist()}")
        return t_list

    def _student_sample_loop(
        self,
        noise: torch.Tensor,
        t_list: torch.Tensor,
        caption_embs: torch.Tensor,
        lq_video_or_image: Optional[torch.Tensor],
        lq_latent: Optional[torch.Tensor],
        degrade_sigma_tensor: Optional[torch.Tensor],
        generator: Optional[torch.Generator] = None,
        callback=None,
        sampler: str = "default",
    ) -> torch.Tensor:
        B = noise.shape[0]
        timescale = self.fm_trainer.timescale
        device = noise.device
        autocast_ctx = torch.autocast(device.type, dtype=self.autocast_dtype) if (self.autocast_dtype and device.type != "cpu") else nullcontext()
        x = noise
        net = self.net

        total_steps = len(t_list) - 1

        # Determine active sampler mode
        if sampler == "default":
            sampler_mode = self.config.student_sample_type
        elif sampler == "ode_euler":
            sampler_mode = "ode"
        elif sampler == "ode_heun":
            sampler_mode = "heun"
        elif sampler == "sde_ancestral":
            sampler_mode = "sde"
        else:
            sampler_mode = "sde"

        with autocast_ctx:
            for step, (t_cur, t_next) in enumerate(zip(t_list[:-1], t_list[1:])):
                t_cur_batch = t_cur.expand(B)
                t_cur_scaled = t_cur_batch * timescale

                # 1. Forward model at current state
                v_pred = net(
                    x,
                    t_cur_scaled,
                    caption_embs,
                    lq_video_or_image=lq_video_or_image,
                    lq_latent=lq_latent,
                    degrade_sigma=degrade_sigma_tensor,
                )

                if t_next.item() > 0:
                    if sampler_mode == "ode":
                        # Euler ODE step
                        v_for_step = self._net_output_to_velocity(x, v_pred, t_cur_batch, self.config.prediction_type)
                        dt = t_next - t_cur
                        x = x + dt * v_for_step
                    elif sampler_mode == "heun":
                        # Heun 2nd-order ODE predictor step
                        v_for_step = self._net_output_to_velocity(x, v_pred, t_cur_batch, self.config.prediction_type)
                        dt = t_next - t_cur
                        x_pred = x + dt * v_for_step
                        
                        # Corrector step
                        t_next_batch = t_next.expand(B)
                        t_next_scaled = t_next_batch * timescale
                        v_pred_next = net(
                            x_pred,
                            t_next_scaled,
                            caption_embs,
                            lq_video_or_image=lq_video_or_image,
                            lq_latent=lq_latent,
                            degrade_sigma=degrade_sigma_tensor,
                        )
                        v_for_step_next = self._net_output_to_velocity(x_pred, v_pred_next, t_next_batch, self.config.prediction_type)
                        x = x + dt * 0.5 * (v_for_step + v_for_step_next)
                    else: # "sde" ancestral
                        x0_pred = self._velocity_to_x0(x, v_pred, t_cur_batch)
                        eps_infer = torch.randn(
                            x0_pred.shape,
                            device=x0_pred.device,
                            dtype=x0_pred.dtype,
                            generator=generator,
                        )
                        s = [B] + [1] * (x.ndim - 1)
                        t_next_bcast = t_next.reshape(1).expand(s)
                        x = (1.0 - t_next_bcast) * x0_pred + t_next_bcast * eps_infer
                else:
                    x = self._velocity_to_x0(x, v_pred, t_cur_batch)

                if callback is not None:
                    callback(step, total_steps)

        return x

    # ---------------------------------------------------------------------
    # Inference entry point
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: dict,
        guidance: float = None,
        cfg_scale: float = None,
        num_steps: int = None,
        seed: int = 0,
        image_size=None,
        shift: float = None,
        is_negative_prompt: bool = False,
        callback=None,
        sampler: str = "default",
        scheduler: str = "default",
        **kwargs,
    ):
        device = next(self.net.parameters()).device

        # Encode any missing LQ_latent via the frozen VAE so callers can pass either
        # LQ_video_or_image or LQ_latent.
        if "LQ_latent" not in data_batch and "LQ_video_or_image" in data_batch and self.vae_encoder is not None:
            data_batch["LQ_latent"] = (
                self.encode_lq_latent(data_batch["LQ_video_or_image"]).contiguous().to(device=device, dtype=self.precision)
            )
        if "degrade_sigma" not in data_batch and "LQ_latent" in data_batch:
            B = data_batch["LQ_latent"].shape[0]
            data_batch["degrade_sigma"] = torch.zeros(B, device=data_batch["LQ_latent"].device, dtype=torch.float32)

        x0_key = self.config.input_data_key
        if image_size is None and x0_key in data_batch:
            x0_shape = data_batch[x0_key].shape
            img_h, img_w = x0_shape[-2], x0_shape[-1]
        else:
            image_size = image_size or self.config.image_size
            if isinstance(image_size, (list, tuple)):
                img_h, img_w = int(image_size[0]), int(image_size[1])
            else:
                img_h = img_w = int(image_size)

        # Determine shift: explicit arg > SD3-style dynamic_shift (if configured) > config default.
        if shift is None and self.config.dynamic_shift is not None:
            import math

            _ds = self.config.dynamic_shift
            shift = _ds["base_shift"] * math.sqrt(max(img_h, img_w) / _ds["base_image_size_for_shift_calc"])

        if "caption_embs" in data_batch:
            caption_embs = data_batch["caption_embs"].to(device=device, dtype=self.precision)
            captions = data_batch.get(self.config.input_caption_key, [])
            B = len(captions) if isinstance(captions, list) else 1
        else:
            captions = data_batch[self.config.input_caption_key]
            if isinstance(captions, str):
                captions = [captions]
            B = len(captions)
            if self.config.use_fixed_prompt:
                captions = [self.config.fixed_positive_prompt] * B
            caption_embs, _ = self._encode_text_raw(captions)
            caption_embs = caption_embs.to(device=device, dtype=self.precision)

        lq_video_or_image = None
        lq_latent = None
        if self.config.lq_condition_type in ("image", "image_latent"):
            lq_video_or_image = data_batch.get("LQ_video_or_image")
            if lq_video_or_image is not None:
                lq_video_or_image = lq_video_or_image.to(device=device, dtype=self.precision)
        if self.config.lq_condition_type in ("latent", "image_latent"):
            lq_latent = data_batch.get("LQ_latent")
            if lq_latent is not None:
                lq_latent = lq_latent.to(device=device, dtype=self.precision)

        sigma_val = data_batch.get("degrade_sigma", 0.0)
        if isinstance(sigma_val, torch.Tensor):
            degrade_sigma_tensor = sigma_val.to(device=device, dtype=torch.float32).reshape(-1)
            if degrade_sigma_tensor.numel() == 1:
                degrade_sigma_tensor = degrade_sigma_tensor.expand(B).contiguous()
            assert degrade_sigma_tensor.shape == (B,), (
                f"data_batch['degrade_sigma'] expected [B={B}], got {tuple(degrade_sigma_tensor.shape)}"
            )
        elif isinstance(sigma_val, (list, tuple)):
            degrade_sigma_tensor = torch.tensor(sigma_val, device=device, dtype=torch.float32)
            assert degrade_sigma_tensor.shape == (B,), (
                f"data_batch['degrade_sigma'] expected length {B}, got {len(sigma_val)}"
            )
        else:
            degrade_sigma_tensor = torch.full((B,), float(sigma_val), device=device, dtype=torch.float32)

        gen = torch.Generator(device=device).manual_seed(int(seed))
        noise = torch.randn(B, 3, img_h, img_w, device=device, generator=gen)

        autocast_ctx = torch.autocast(device.type, dtype=self.autocast_dtype) if (self.autocast_dtype and device.type != "cpu") else nullcontext()
        net = self.net
        net.eval()

        effective_steps = num_steps if num_steps is not None else self.config.student_sample_steps

        if effective_steps == 1:
            t_student = torch.full((B,), self.config.student_timestep, device=device, dtype=torch.float32)
            t_student_scaled = t_student * self.fm_trainer.timescale
            with autocast_ctx:
                v_student = net(
                    noise,
                    t_student_scaled,
                    caption_embs,
                    lq_video_or_image=lq_video_or_image,
                    lq_latent=lq_latent,
                    degrade_sigma=degrade_sigma_tensor,
                )
                x0_student = self._velocity_to_x0(noise, v_student, t_student)
            if callback is not None:
                callback(0, 1)
        else:
            t_list = self._get_t_list(device=device, num_steps=num_steps, shift=shift, scheduler=scheduler)
            x0_student = self._student_sample_loop(
                noise,
                t_list,
                caption_embs,
                lq_video_or_image,
                lq_latent,
                degrade_sigma_tensor,
                generator=gen,
                callback=callback,
                sampler=sampler,
            )

        return x0_student.clamp(-1, 1).unsqueeze(2)

    # ---------------------------------------------------------------------
    # Checkpoint helpers (only the student `net.` prefix matters at inference)
    # ---------------------------------------------------------------------

    def model_dict(self) -> dict:
        return {"net": self.net}

    def state_dict(self, *args, **kwargs):
        return self.net.state_dict(prefix="net.")

    def load_state_dict(self, state_dict, strict=True, assign=False, **kwargs):
        _net_sd = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("net.") and not k.startswith("net_ema."):
                _net_sd[k[len("net.") :]] = v
            elif k.startswith("net_ema.") or k.startswith("fake_score.") or k.startswith("discriminator."):
                continue
            else:
                _net_sd[k] = v

        missing, unexpected = self.net.load_state_dict(_net_sd, strict=False, assign=assign)
        if missing:
            lq_missing = [k for k in missing if "lq_proj" in k]
            other_missing = [k for k in missing if "lq_proj" not in k]
            if lq_missing:
                logger.info(f"Expected missing LQ keys ({len(lq_missing)} keys)")
            if other_missing and strict:
                logger.warning(f"Missing keys in net: {other_missing}")
        if unexpected:
            logger.warning(f"Unexpected keys in net: {unexpected}")

    def on_train_start(self, memory_format=torch.preserve_format) -> None:
        super().on_train_start(memory_format)
