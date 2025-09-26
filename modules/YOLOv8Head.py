import torch
import torch.nn as nn
from nn.modules.conv import Conv   # 简单版只用到了Conv 模块


class YOLOv8Head(nn.Module):
	"""
	YOLOv8 Detection Head (Anchor-Free)
	
	说明：
	- 输入： 来自 Neck 的多尺度特征 [p3, p4, p5]
		p3: (B, c3, H3, W3)  e.g. (B, 128, 80, 80)
		p4: (B, c4, H4, W4)  e.g. (B, 256, 40, 40)
		p5: (B, c5, H5, W5)  e.g. (B, 512, 20, 20)
	
	- 输出：每个尺度的预测 tensor , 形状为（B, nc+4, H, W)
	通道维分布：前4通道维 bbox 回归 （cx, cy, w, h), 后nc 通道为类别 logits
	注意：这里的 (cx, cy, w, h) 的数值语义和解码方式取决于你训练/解码流程（下方有提示）
	"""
	
	def __init__(self, num_classes=80, channels=[128, 256, 512]):
		super().__init__()
		# num_classes: 类别数量
		# no (num outputs per location) = 4 (bbox) + nc （类别）
		self.nc = num_classes
		self.no = num_classes + 4  # 方便后续扩展/ 检查 （当前代码里没有直接用到 self.no, 但它是有用的元信息）
		
		# 为每个尺度准备独立的分类&回归小分支 （官方做法通常为每尺度单独 head)
		# 使用 ModuleList 便于迭代和按尺度索引访问
		self.cls_convs = nn.ModuleList()  # 分类分支集合 （输出 nc 通道）
		self.reg_convs = nn.ModuleList()  # 回归分支集合 （输出 4 通道）
		
		# channels 列表顺序要 和 Neck 输出顺序一致： [c3, c4, c5]
		for ch in channels:
			# 分类分支：先由 Conv(3x3) 增强特征 （保持通道数）， 再用 1x1 卷积把通道数映射到nc
			# Conv 是你项目中的轻量封装 （一般含 BN + 激活）。这样可以复用你已有的模块。
			self.cls_convs.append(
				nn.Sequential(
					Conv(ch, ch, kernel_size=3, stride=1),  # 保持分辨率 & 通道数， 做局部特征处理
					nn.Conv2d(ch, self.nc, 1)  # 1x1 卷积输出类别 logits (未经过 sigmoid）
				)
			)
			
			# 回归分支：同理，先 Conv 增强，再1x1 输出 4 通道（cx, cy, w, h)
			self.reg_convs.append(
				nn.Sequential(
					Conv(ch, ch, kernel_size=3, stride=1),  # 特征增强： 有助于回归稳定
					nn.Conv2d(ch, 4, 1)      # 输出 bbox 回归量 (通常是 raw offsets / logits)
				)
			)
			
	def forward(self, features):
		"""
		feature: list or tuple -> [p3, p4, p5]
				每个元素形状 (B, C_i, H_i, W_i)
				
		返回：
			output: list -> [p3_out, p4_out, p5_out]
					每个p?_out 的形状为 (B, 4+nc, H, W)
					通道顺序： torch.cat([reg_pred, cls_pred], dim=1)
							-> [0:4] = bbox, [4:] = class logits
		"""
		outputs = []
		# 这里假设 features 的顺序与 channels 一致 （即 channels = [c3, c4, c5])
		for i, x in enumerate(features):
			# x: 单个尺度的特征（B, ch, H, W)
	
			# 分类预测 (B, nc, H, W)
			# 注意：这里输入的是 logits (没用 Sigmoid/Softmax), 训练时通常用 BCEWithLogitsLoss
			cls_pred = self.cls_convs[i](x)
			
			# 回归预测 (B, 4, H, W)
			# 注意：回归输出的范围/激活策略取决与你的解码方式
			reg_pred = self.reg_convs[i](x)
			
			# 拼接： 先回归后分类 -- 这个顺序必须和你后面的loss/ 解码保持一致
			# 最后输出通道 为 4 + nc
			out = torch.cat([reg_pred, cls_pred], dim=1)  # dim=1 是 channel 维度
			outputs.append(out)
			
		return outputs
	
	
# ----------------- 调试用 -------------------
if __name__ == "__main__":
	# 少量示例数据， 用于检查前向维度是否正确
	head = YOLOv8Head(num_classes=10, channels=[128, 256, 512])
	x3 = torch.randn(1, 128, 80, 80)  # p3 (较大尺度特征）
	x4 = torch.randn(1,256, 40, 40)  # p4
	x5 = torch.randn(1, 512, 20, 20)  # p5
	
	preds = head([x3, x4, x5])
	for p in preds:
		print(p.shape)  # 期望：[1, 14, H, W]  (14 = 4 + 10)
