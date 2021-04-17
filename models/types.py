"""
Module for unifying the typing system
"""
from typing import Any as _Any

__all__=["ColorImageSize","ImageSize"]

ColorImageSize = (_Any,_Any,3)
"""
Color image shaping
ColorImageSize, aka (Any, Any, 3)
"""

ImageSize = (_Any,_Any)
"""
One-chanel image shaping
ImageSize, aka (Any, Any)
"""
