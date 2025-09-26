# iou_losses_detailed.py
# 完整、逐行注释的 IoU / GIoU / DIoU / CIoU 实现
# 目标读者：刚接触 Python / PyTorch 的初学者
# 运行：python iou_losses_detailed.py

import torch
import math

EPS = 1e-7  # 防止除零或者数值不稳定小常数


# ------------------------------------------
# 小提示（非常基础）
# - 我们用的 box 格式是 [x1, y1, x2, y2], 其中：
# x1, y1 是左上角坐标， x2, y2 是右下角坐标
# 如果宽或高是负的（例如 x2 < x1), 通常表示框无效， 我们会用 clamp(min=0) 把宽高限制为 >= 0
# 在计算两个集合的 pairwise 指标时， 返回形状是 （N，M), 表示 每个 box1 中的 i 与 box2 中的 j 的结果
# -----------------------------------------
def box_area(box):
	"""
	计算两个 box （逐元素）
	输入：
		box: Tensor, shape(k,4), 格式[x1, y1, x2, y2]
	输出：
		area: Tensor, shape(K,), 对应每个 box 的面积
	细节：
		- 宽 = x2 - x1, 高 = y2 - y1
		- 使用 clamp(min=0) 确保负宽高变为 0 防止异常
	"""
	# box[:,2] - box[:, 0] -> (k,)
	w = (box[:, 2] - box[:, 0]).clamp(min=0)
	h = (box[:, 3] - box[:, 1]).clamp(min=0)
	return w * h

def pairwise_iou(box1, box2):
	"""
	计算两个box 集合之间的pairwise IoU (交并比)
	输入：
		box1: Tensor (N, 4)
		box2: Tensor (M, 4)
	输出：iou: Tensor (N, M), 第（i,j) 元素是 box[i] 与 box[j] 的IoU
	算法步骤：
		1. 交集左上角坐标取两者的elementwise 最大值 (max of x1, x2)
		2. 交集右下角坐标取两者的elementwise 最小值 (min of y1, y2)
		3. 交宽高 = max(0, right_bottom - left_top)
		4. 交集面积 = w * h
		5. 并集面积 = area1 + area2 - inter
		6. IoU = inter / (union + EPS)
	注意广播：
		- box1[:, None, :2] 形状 (N, 1, 2)
		- box2[None, :, :2] 形状 (1, M, 2)
		- 两者相比较后得到 （N, M, 2) 的结果 -> 这是 pairwise 的关键
	"""
	# 交集的左上角 (N, M, 2)
	left_top = torch.max(box1[:, None, :2], box2[None, :, :2])
	# 交集右下角 (N, M, 2)
	right_bottom = torch.mn(box1[:, None, 2:], box2[None, :, 2:])
	# 交集的宽高 （N, M, 2), 如果负就置为0
	wh = (right_bottom - left_top).clamp(min=0.0)
	# 交集面积 （N，M）
	inter = wh[..., 0] * wh[..., 1]
	
	# 各自面积
	area1 = box_area(box1)      # (N,)
	area2 = box_area(box2)      # (M,)
	# 并集 （N，M）: area1[:, None] 广播为 （N，1） -> (N,M)
	union = area1[:, None] + area2[None, :] - inter
	# 最后计算 IoU
	return inter / (union + EPS)

# IoU 基础
def iou_score(box1, box2):
	"""
	直接返回 pairwise IoU (值在 [0~1]
	"""
	return pairwise_iou(box1, box2)

# GIoU Generalized IoU
# 公式： GIoU = IoU - (|C| - |AuB|) / |C|
# 其中 C 为包含 A 和 B 的最小闭包框 (enclosing box)
def giou_score(box1, box2):
	"""
	计算 pairwise GIoU
	返回：Tensor (N,M)
	数学直观解释：
	- 当两个框完全重叠时， GIoU = IoU = 1
	- 当没有重叠时， IoU= 0， 但 GIoU 会根据两框之间的相对位置给出一个负值 -> 提供梯度信号
	"""
	iou = pairwise_iou(box1, box2)
	
	# 找到最小闭包框C的左上和右下
	c_left_top = torch.min(box1[:, None, :2], box2[None, :, :2])
	c_right_bottom = torch.max(box1[:, None, 2:], box2[None, :, 2:])
	c_wh = (c_right_bottom - c_left_top).clamp(min=0.0)
	area_c = c_wh[..., 0] * c_wh[..., 1]
	
	# 交集
	left_top = torch.max(box1[:, None, :2], box2[None, :, :2])
	right_bottom = torch.min(box1[:, None, 2:], box2[None, :, 2:])
	wh = (right_bottom - left_top).clamp(min=0.0)
	inter = wh[..., 0] * wh[..., 1]
	union = box_area(box1)[:, None] + box_area(box2)[None,:] - inter + EPS
	
	# GIoU
	giou = iou - (area_c - union) / area_c
	return giou

# DIoU distance IoU
# 公式： DIoU = IoU - (ρ^2 / c^2)
#    ρ^2 : 预测框中心到 GT 框中心的欧式距离平方
#    c^2 : 包含框对角线的平方

def diou_score(box1, box2):
	"""
	计算 pairwise DIoU
	DIoU 加入了中心点距离惩罚， 使得优化更关注把预测框的中心点移向目标中心
	"""
	iou = pairwise_iou(box1, box2)
	
	# 1) 中心点 （N，2） 与 （M，2）
	center1 = (box1[:, :2] + box1[:, 2:]) / 2.0
	center2 = (box2[:, :2] + box2[:, 2:]) / 2.0
	
	# pairwise 中心点距离平方： （N，M）
	center_dist2 = ((center1[:, None, :] - center2[None, :, :]) ** 2).sum(dim=2)
	
	# 封闭框 c 的对角线平方 c^2 (N,M)
	c_left_top = torch.min(box1[:, None, :2], box2[None, :, :2])
	c_right_bottom = torch.max(box1[:, None, 2:], box2[None, :, 2:])
	c_wh = (c_right_bottom - c_left_top).clamp(min=0.0)
	c_diag2 = (c_wh[..., 0] ** 2 + c_wh[..., 1] ** 2) + EPS
	
	diou = iou - (center_dist2 / c_diag2)
	return diou


	
