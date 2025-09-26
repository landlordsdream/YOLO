# -*- coding: utf-8 -*-
"""
从零实现 YOLOv8 的SPPF (Apatial Pyramid Pooling - Fast)
- 作用：扩大感受野、引入多尺度上下文特征
- 原理：输入 -> 1x1 conv 压缩 -> 多次 5x5 MaxPool 叠加 -> 拼接 -> 1x1 conv 融合
"""

import torch
import torch.nn as nn


class Conv(nn.Module):
	"""
	基础卷积块：Conv2d -> BN -> 激活
	"""
	
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, bias=False, act=True):
		super().__init__()
		self.conv = nn.Conv2d(
			in_channels, out_channels, kernel_size, stride,
			padding=(kernel_size - 1) // 2 if padding is None else padding,
			groups=groups, bias=bias
		)
		self.bn = nn.BatchNorm2d(out_channels)
		self.act = nn.SiLU() if act else nn.Identity()
	
	def forward(self, x):
		return self.act(self.bn(self.conv(x)))


class SPPF(nn.Module):
	"""
	SPPF 模块 （YOLOv8)
	输入： feature map
	输出： 拼接池化特征后的融合特征
	"""
	
	def __init__(self, in_channels, out_channels, pool_kernel=5):
		super().__init__()
		hidden = in_channels // 2  # 先降维一半，减少计算
		self.cv1 = Conv(in_channels, hidden, kernel_size=1, stride=1)  # 压缩通道
		self.cv2 = Conv(hidden * 4, out_channels, kernel_size=1, stride=1)  # 拼接后融合
		
		# 最大池化层，kernel=5, stride=1, padding=2 保持尺寸不变
		self.m = nn.MaxPool2d(kernel_size=pool_kernel, stride=1, padding=pool_kernel // 2)
	
	def forward(self, x):
		# 先降维，减少计算量
		x = self.cv1(x)  # [B, C/2, H, W]
		
		# 多次池化：每次在上一次结果上 做 5x5 MaxPool
		y1 = self.m(x)  # 第一次池化
		y2 = self.m(y1)  # 第二次池化
		y3 = self.m(y2)  # 第三次池化
		
		# 拼接： 把原始 + 三次池化结果拼接在一起
		out = torch.cat([x, y1, y2, y3], dim=1)
		
		# 再用1x1 conv融合
		return self.cv2(out)


# ------------------------ 测试代码 ------------------------
if __name__ == "__main__":
	# 构造一个假输入：batch=1, 通道=128, 分辨率=20x20
	x = torch.randn(1, 128, 20, 20)
	
	# 定义 SPPF 模块：输入128通道，输出256通道
	sppf = SPPF(in_channels=128, out_channels=256)
	
	# 前向传播
	y = sppf(x)
	
	# 打印形状
	print("输入形状: ", x.shape)  # torch.Size([1, 128, 20, 20])
	print("输出形状: ", y.shape)  # torch.Size([1, 256, 20, 20])
