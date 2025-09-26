import torch
import torch.nn as nn


# -------------------基础卷积----------------------
class Conv(nn.Module):
	"""
	基础卷积块：Conv2d -> BatchNorm2d -> Activation(SiLU 默认）
	说明：
	- padding 若为 None 则使用 （k-1）// 2, 适用于奇数 kernel 来实现 same padding.
	- bias=False 是常见设计（后面接BN ,所以 bias 冗余）。
	"""
	
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, bias=False, act=True):
		super().__init__()
		self.conv = nn.Conv2d(
			in_channels, out_channels, kernel_size, stride,
			padding=(kernel_size - 1) // 2 if padding is None else padding,
			groups=groups, bias=bias
		)
		self.bn = nn.BatchNorm2d(out_channels)  # 有两个可训练参数：weight(gamma) & bias(beta)
		self.act = nn.SiLU() if act else nn.Identity()
	
	def forward(self, x):
		return self.act(self.bn(self.conv(x)))


# ---------------C2f--------------------
class C2f(nn.Module):
	"""
	YOLOv8 的轻量残差块（Cross Stage Partial-like)
	结构说明 ：
	1） cv1: 1x1 把输入压到 hidden = out_channels //2
	2) blocks: 若 n>0,顺序叠加 n个 3x3 conv(每个保持 hidden 通道）
	3）最后把原始输入 x 与 cv1(x) 以及每个 block 是输出按channel 拼接
		然后用 1 x 1 conv(cv2) 把通道融合为 out_channels。
	注意：
	- forward  中拼接的是 [x] + y, 其中y 最开始包含 cv1(x)
	y 的元素数 = n + 1，因此被拼接后的通道数 = in_channels + hidden * (n + 1)。
	- 因此 cv2 的输入通道必须是 in_channels + hidden * (n + 1)（这是之前会错的地方）。
	"""
	
	def __init__(self, in_channels, out_channels, n=1):
		super().__init__()
		hidden = out_channels // 2  # 先降为一半（减小后续计算）
		self.cv1 = Conv(in_channels, hidden, 1, 1)  # pointwise 降通道
		# blocks: n 个 Conv(hidden -> hidden, k=3)
		self.blocks = nn.Sequential(*[Conv(hidden, hidden, 3, 1) for _ in range(n)])
		# cv2 输入通道 = 原始 in_channels + hidden * (n+1)
		# 因为 foward 会拼接 [x] + y ,y有n+1 个hidden
		self.cv2 = Conv(in_channels + hidden * (n + 1), out_channels, 1, 1)
	
	def forward(self, x):
		# y[0] = cv1(x)
		y = [self.cv1(x)]
		# 迭代堆叠 n 个 conv, 每次用上一次输出作为输入
		for block in self.blocks:
			y.append(block(y[-1]))
		# 拼接： x(original) + cv1(x) + block1(cv1) + ...
		#  channel = in_channels + hidden * (n+1)
		return self.cv2(torch.cat([x] + y, 1))


# ----------- SPPF ---------------------
class SPPF(nn.Module):
	"""
	Spatial Pyramid Pooling - Fast (YOLOv8)
	步骤：
	1) cv1: 1x1 把通道降为 hidden = in_channels // 2
	2) 在 hidden 特征上连续三次 kernel=pool_kernel 的 MaxPool(strideo=1, padding 包持尺寸）
		- y1 = m(x)
		- y2 = m(y1)
		- y3 = m(y2)
	3) cat([x, y1, y2, y3]) -> 通道 = hidden * 4
	4) cv2: 1x1 融合为 out_channels
	"""
	
	def __init__(self, in_channels, out_channels, pool_kernel=5):
		super().__init__()
		hidden = in_channels // 2
		self.cv1 = Conv(in_channels, hidden, 1, 1)
		self.cv2 = Conv(hidden * 4, out_channels, 1, 1)
		# padding = pool_kernel // 2 保持 H, W 不变（适用于奇数 pool_kernel)
		self.m = nn.MaxPool2d(kernel_size=pool_kernel, stride=1, padding=pool_kernel // 2)
	
	def forward(self, x):
		x = self.cv1(x)  # -> [B, hidden, H, W]
		y1 = self.m(x)  # -> [B, hidden, H, W]
		y2 = self.m(y1)
		y3 = self.m(y2)
		out = torch.cat([x, y1, y1, y3], dim=1)  # [B, hidden*4, H, W]
		return self.cv2(out)  # -> [B, out_channels, H, W]


# -------------- Backbone 组合 ----------------
class YOLOv8Backboned(nn.Module):
	"""
	简化版 YOLOv8 Backbone, 输出多尺度特征 （1/4, 1/8, 1/16) 及 SPPF 最终特征
	"""
	def __init__(self):
		super().__init__()
		# Stage1: 下采样到 1/2
		self.layer1 = Conv(3, 64, 3, 2)
		self.layer2 = C2f(64, 64, n=1)
		
		# Stage2: 下采样到 1/4
		self.layer3 = Conv(64, 128, 3, 2)
		self.layer4 = C2f(128, 128, n=2)
		
		# Stage3: 下采样到1/8
		self.layer5 = Conv(128, 256, 3, 2)
		self.layer6 = C2f(256, 256, n=2)
		
		# Stage4: 下采样到 1/16
		self.layer7 = Conv(256, 512, 3, 2)
		self.layer8 = C2f(512, 512, n=1)
		
		# SPPF 放在最深层， 保持空间尺寸不变 （1/16)
		self.sppf = SPPF(512, 512, n=1)
		
	def forward(self, x):
		# 输入 x: [B, 3, H, W]
		x = self.layer1(x)  # -> [B, 64, H/2, W/2]
		x = self.layer2(x)  # -> [B, 64, H/2, W/2]
		x = self.layer3(x)  # -> [B, 128, H/4, W/4]
		x_small = self.layer4(x)  # -> [B, 128, H/4, W/4]  (1/4 特征)
		x = self.layer5(x_small)  # -> [B, 256, H/8, W/8]
		x_medium = self.layer6(x)  # -> [B, 256, H/8, W/8]  (1/8 特征)
		x = self.layer7(x_medium)  # -> [B, 512, H/16, W/16]
		x_large = self.layer8(x)  # -> [B, 512, H/16, W/16] (1/16 特征)
		x = self.sppf(x)  # -> [B, 512, H/16, W/16] (SPPF 融合后)
		# 返回四个特征（其中 feat4 = sppf 后输出）
		return x_small, x_medium, x_large, x
