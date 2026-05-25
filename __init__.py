"""
ComfyUI custom node package: PiD (Pixel Diffusion Decoder).

PiD is a plug-and-play diffusion decoder from NVIDIA that replaces standard
VAE decoders, producing super-resolved images directly from latent space.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
