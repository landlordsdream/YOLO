import torch
import torch.nn as nn

from nn.modules.C2f import C2f
from nn.modules.conv import Conv


# ---------------------- Neck: FPN + PAN ------------------------
class YOLOv8Neck(nn.Module):
	"""
	简化版 YOLOv8 Neck: FPN + PAN, 融合多尺度特征
	"""
	
	def __init__(self, channels=[128, 256, 512], depth=3, width=1.0):
		"""
		channels: Backbone 输出特征的通道数
			顺序： [x_small, x_medium, x_large]
			- x_small -> 1/4 尺寸 （H/4, W/4)
			- x_medium -> 1/8 尺寸 （H/8, W/8)
			- x_large -> 1/16 尺寸 （H/16, W/16)
		depth:      控制 C2f 内部堆叠的 bottleneck 数量
		width:      控制通道扩展比例 (这里没展开用，留作接口)
		"""
		super().__init__()
		
		# 分别解包 Backbone 输出的通道数
		c3, c4, c5 = channels  # c1=128, c2=256, c3=512
		
		# ----------------- FPN: 自顶向下融合 -------------
		# 上采样操作，用最近邻插值，放大2 倍
		self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
		
		# 融合 p5 和 p4: 将 p5 上采样到 p4 的尺寸， 再 concat
		# 输入通道数 = c4 + c5
		# 输出通道数 = c4
		self.fpn_c2f1 = C2f(c4 + c5, c4, n=depth)
		
		# 融合 p4_fpn 和 p3: 同理， 将 p4_fpn 上采样到 p3 尺寸， 再concat
		# 输入通道数 = c3 + c4
		# 输出通道数 = c3
		self.fpn_c2f2 = C2f(c3 + c4, c3, n=depth)
		
		# ----------------- PAN: 自底向上路径 ---------------
		# 把 p3_fpn 下采样一倍 （stride = 2）， 尺寸对齐到 p4_fpn
		# 这里用 stride=2 的卷积代替 maxpool, 更灵活 （同时可学通道）
		self.down_c4 = Conv(c3, c3, 3, 2)
		
		# 融合 p3_down 和 p4_fpn, 得到新的 p4_pan
		self.pan_c2f1 = C2f(c3 + c4, c4, n=depth)
		
		# 再把 p4_pan 下采样一倍，尺寸对齐到 p5
		self.down_c5 = Conv(c4, c4, 3, 2)
		
		# 融合 p4_down 和 p5, 得到最终的 p5_pan
		self.pan_c2f2 = C2f(c4 + c5, c5, n=depth)
		
	def forward(self, x3, x4, x5):
		"""
		x3: p3 = Backbone 输出的 1/8尺度特征 （B, c3, H/8, W/8)
		x4: p4 = Backbone 输出的 1/16尺度特征 （B, c4, H/16, W/16)
		x5: p5 = Backbone 输出的 1/32尺度特征 （B, c5, H/32, W/32)
		"""
		
		# ----------------- FPN: 自顶向下 -----------------
		# Step1：p5 -> 上采样到 p4 的大小
		p5_upsampled = self.upsample(x5)     # (B, c5, H/16, W/16)
		# 与 p4 拼接， 再经过 C2f融合
		p4_fpn = self.fpn_c2f1(torch.cat([x4, p5_upsampled], dim=1))
		# 输出大小 （B, c4, H/16, W/16)
	
		# Step2: p4_fpn -> 上采样到 P3 的大小
		p4_upsampled = self.upsample(p4_fpn)  # (B, c4, H/8, W/8)
		# 与 p3 拼接， 再经过C2f 融合
		p3_fpn = self.fpn_c2f2(torch.cat([x3, p4_upsampled], dim=1))
		# 输出大小（B，c3, H/8, W/8)
	
		# ------------------ PAN: 自底向上 -------------------
		# Step1: 将 p3_fpn 下采样到 p4 的大小
		p3_down = self.down_c4(p3_fpn)  # (B, c3, H/16, W/16)
		# 与 p4_fpn 拼接， 再经过 C2f 融合
		p4_pan = self.pan_c2f1(torch.cat([p4_fpn, p3_down], dim=1))
		# 输出大小 （B， c4, H/16, W/16)
		
		# Step2: 将 p4_pan 下采样到 P5 的大小
		p4_down = self.down_c5(p4_pan)  # (B, c4, H/32, W/32)
		# 与p5 拼接，再经过C2f 融合
		p5_pan = self.pan_c2f2(torch.cat([x5, p4_down], dim=1))
		# 输出大小 （B, c5, H/32, W/32)
	
		# ------------------- 输出 --------------------
		# 返回 3 个尺度的特征图， 分别供检测头使用
		return p3_fpn, p4_pan, p5_pan
	
		
		
		
		