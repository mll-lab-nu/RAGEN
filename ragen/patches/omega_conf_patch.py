"""Monkey patch `omega_conf_to_dataclass` to allow extra fields on VERL configs.

The upstream implementation instantiates dataclasses directly and therefore
rejects any unexpected keyword arguments. RAGEN users often want to attach
custom flags in YAML without editing VERL source code. This patch splits the
input config into two parts:

* fields belonging to the underlying dataclass → delegated to the original
  implementation, preserving all validation behaviour;
* any additional fields → attached to the resulting object via
  ``object.__setattr__`` so they remain accessible.

The patch is idempotent and can be safely applied multiple times.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Dict, Tuple

from hydra.utils import get_class
from omegaconf import DictConfig, ListConfig, OmegaConf

from verl.utils import config as verl_config


def apply_omega_conf_patch() -> None:
    """Apply the RAGEN override of ``omega_conf_to_dataclass`` if not present."""

    if getattr(verl_config, "_ragen_omega_conf_patch", False):
        return

    original_fn = verl_config.omega_conf_to_dataclass

    def _split_known_and_extra(
        cfg: DictConfig | dict,
        valid_fields: set[str],
        include_target: bool,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Split config into dataclass fields and extra attributes."""

        base: Dict[str, Any] = {}
        extras: Dict[str, Any] = {}

        # Resolve interpolations before splitting to avoid losing parent context
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)

        if isinstance(cfg, dict):
            iterator = cfg.items()
        else:
            return base, extras

        for key, value in iterator:
            if include_target and key == "_target_":
                base[key] = value
            elif key in valid_fields:
                base[key] = value
            else:
                extras[key] = value
        return base, extras

    def _attach_extras(obj: Any, extras: Dict[str, Any]) -> None:
        for key, value in extras.items():
            object.__setattr__(obj, key, value)

    def _resolve_dataclass(target: Any) -> Any:
        if not isinstance(target, str):
            return None
        try:
            cls = get_class(target)
        except Exception:
            return None
        return cls if is_dataclass(cls) else None

    def patched_omega_conf_to_dataclass(
        config: DictConfig | dict | None,
        dataclass_type: Any | None = None,
    ) -> Any:
        # Keep original behavior for non-mapping inputs (e.g., dataclass instances, lists).
        if not isinstance(config, (DictConfig, dict)):
            return original_fn(config, dataclass_type)

        # Resolve all interpolations early to avoid losing parent context
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)

        # Delegate directly if no dataclass is involved.
        if dataclass_type is not None:
            if not is_dataclass(dataclass_type):
                return original_fn(config, dataclass_type)

            if config is None:
                return original_fn(config, dataclass_type)

            base_dict, extras = _split_known_and_extra(config, {f.name for f in fields(dataclass_type)}, False)
            obj = original_fn(base_dict, dataclass_type)
            _attach_extras(obj, extras)
            return obj

        # No explicit dataclass type provided; rely on _target_.
        if config is None:
            return original_fn(config, dataclass_type)

        target = config.get("_target_") if isinstance(config, (DictConfig, dict)) else None

        dataclass_cls = _resolve_dataclass(target)
        if dataclass_cls is None:
            return original_fn(config, dataclass_type)

        valid_fields = {f.name for f in fields(dataclass_cls)}
        base_dict, extras = _split_known_and_extra(config, valid_fields, include_target=True)
        obj = original_fn(base_dict, dataclass_type)
        _attach_extras(obj, extras)
        return obj

    verl_config.omega_conf_to_dataclass = patched_omega_conf_to_dataclass
    verl_config._ragen_omega_conf_patch = True

    # Also patch any modules that have already imported omega_conf_to_dataclass directly
    import sys
    modules_to_patch = [
        "verl.workers.fsdp_workers",
        "verl.workers.megatron_workers",
        "verl.workers.rollout.base",
        "verl.workers.rollout.replica",
        "verl.workers.rollout.sglang_rollout.async_sglang_server",
        "verl.workers.rollout.vllm_rollout.vllm_async_server",
        "verl.trainer.ppo.ray_trainer",
        "verl.trainer.sft_trainer",
    ]
    for mod_name in modules_to_patch:
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            if hasattr(mod, "omega_conf_to_dataclass"):
                setattr(mod, "omega_conf_to_dataclass", patched_omega_conf_to_dataclass)


__all__ = ["apply_omega_conf_patch"]
