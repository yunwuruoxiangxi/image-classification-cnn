"""
模型模塊導出
"""

from .mlp import create_mlp
from .simple_cnn import create_simple_cnn

__all__ = ['create_mlp', 'create_simple_cnn']