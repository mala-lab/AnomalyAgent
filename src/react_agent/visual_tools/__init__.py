"""The five general visual tools of the paper (Appendix A.1), exposed as LangChain tools in ``TOOLS``."""

from .brightness import image_brightness_enhancement
from .deblurring import image_deblurring
from .denoising import image_denoising
from .super_resolution import image_super_resolution
from .zooming import image_zooming

TOOLS = [
    image_denoising,
    image_deblurring,
    image_super_resolution,
    image_zooming,
    image_brightness_enhancement,
]

__all__ = ["TOOLS", "image_denoising", "image_deblurring", "image_super_resolution", "image_zooming", "image_brightness_enhancement"]
