# -*- coding: utf-8 -*-
"""
YOLOv8 卷积模块 Conv 实现
从头开始，包含卷积、批归一化和激活函数
"""

import torch      # 导入 PyTorch 库，提供张量操作和 GPU 支持
import torch.nn as nn         # 导入神经网络模块，包含常用层、损失函数、激活函数


class Conv(nn.Module):
	"""
	YOLOv8 中常用的卷积块
	conv + batchnorm + activation
	可复用在 Backbone、Neck、Head 中
	"""
	
	def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, bias=True, act=True):
		"""
		初始化卷积模块
		参数：
		in_channels    : 输入通道数
		out_channels   : 输出通道数
		kernel_size    : 卷积核大小，默认1
		stride         : 卷积步长，默认None，则自动计算same padding
		groups         : 分组卷积参数，默认为1
		bias           : 是否使用偏置，默认True
		act            : 是否使用激活函数，True使用SiLU，否则恒等函数
		"""
		
		super().__init__()    # 调用父类 nn.Module 的构造函数，保证模块能自动管理参数
		
		# 卷积层
		self.conv = nn.Conv2d(
			in_channels=in_channels,     # 输入通道
			out_channels=out_channels,   # 输出通道
			kernel_size=kernel_size,     # 卷积核大小
			stride=stride,               # 步长
			padding=(kernel_size - 1) // 2 if padding is None else padding,   # same padding
			groups=groups,               # 分组卷积
			bias=bias                    # 是否使用偏置
		)
		
		# 批归一化处理
		self.bn = nn.BatchNorm2d(out_channels)        # 对每个输出通道做标准化，提高训练稳定性
		
		# 激活函数
		self.act = nn.SiLU() if act else nn.Identity()
		# SiLU 是 YOLOv8 推荐激活函数，如果act=False 则使用恒等函数
	
	def forward(self, x):
		"""
		前向传播函数
		输入：x: 输入张量， 形状 [B, C, H, W]
		返回：输出张量，形状[B, out_channels, H_out, W_out]
		"""
		x = self.conv(x)     # 卷积操作
		x = self.bn(x)       # 批归一化
		x = self.act(x)      # 激活函数
		return x             # 返回结束
	
	
# 测试模块
if __name__ == "__main__":
	# 创建一个输入张量 [batch, channels, height, width]
	x = torch.randn(1, 3, 640, 640)   # 模拟 RGB 图像
	model = Conv(in_channels=3, out_channels=32, kernel_size=3, stride=1)
	
	y = model(x)    # 前向计算
	print(f"输入形状：{x.shape}")
	print(f"输出形状：{y.shape}")
	
	