import torch
import torch.nn.functional as F
from typing import cast

EPS = 1e-7  # 防止除零或 log(0) 出现 NaN 的小数   EPS 是一个很小的常数（如 1e-7）


# 防止 除以零，比如两个框面积都是 0 或完全重叠但面积计算异常时

#  保证数值稳定性

# ============================
# 坐标变换工具函数
# ============================
def cxcywh_to_xyxy(boxes):
	"""
	将中心点 + 宽高格式 [cx, cy, w, h] 转为 左上角 + 右下角格式 [x1, y1, x2, y2]
	输入：
		boxes： (N,4) 每行 [cx, cy, w, h]
	输出：
		（N，4） 每行 [x1, y1, x2, y2]
	作用：方便后续 IoU 计算， IoU 通常使用 xyxy 格式
	"""
	cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
	x1 = cx - w / 2  # 左上角 x
	y1 = cy - h / 2  # 左上角 y
	x2 = cx + w / 2  # 右下角 x
	y2 = cy + h / 2  # 右下角 y
	return torch.stack([x1, y1, x2, y2], dim=1)


def xyxy_to_cxcywh(boxes):
	"""
	将左上角 + 右下角格式 [x1, y1, x2, y2] 转为中心点 + 宽高格式 [cx, cy, w, h]
	输入：
		boxes: (N,4)
	输出：
		（N，4）
	作用：
		在 L1 回归损失中常用cxcywh 格式， 便于直接回归偏移量
	"""
	x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
	w = (x2 - x1).clamp(min=0)  # 防止宽为负
	h = (y2 - y1).clamp(min=0)
	cx = (x1 + x2) / 2
	cy = (y1 + y2) / 2
	
	return torch.stack([cx, cy, w, h], dim=1)


def box_area(boxes):
	"""
	计算每个 box 面积
	输入：
		boxes：(N, 4) xyxy
	输出：
		(N,) 面积
		clamp(min=0) 将负值截断为 0 保证宽高非负
		避免后续计算出现负数
	"""
	return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


# ========================
# 计算 N x M 的 IoU
# ========================
def pairwise_iou(box1, box2, eps=1e-7):
	"""
	计算两个 box 集合之间的pairwise IoU
	输入：
		box1: (N, 4)
		box2: (M, 4)
	输出：
		(N x M) 矩阵， 其中[i, j] 表示 box[i] 与 box[j] 的 IoU
	作用：
		IoU 是目标检测回归中常用衡量指标
	"""
	# 左上角取最大， 右下角取最小 -> 交集 box
	lt = torch.max(box1[:, None, :2], box2[None, :, :2])  # (N, M, 2)
	rb = torch.min(box1[:, None, 2:], box2[None, :, 2:])  # (N, M, 2)
	wh = (rb - lt).clamp(min=0)  # 交集宽高，负值置零
	inter = wh[..., 0] * wh[..., 1]  # 交集面积
	area1 = box_area(box1)  # box1 面积
	area2 = box_area(box2)  # box2 面积
	union = area1[:, None] + area2[None, :] - inter + eps  # 并集面积
	return inter / union


# ========================
# Step1: 分类 BCE 损失
# ========================
def classification_bce_loss(pred_logits, pos_mask, matched_labels, num_classes):
	"""
	pred_logits:  (N, C) 所有 anchor 的原始 logits
	pos_mask: (N,) bool, 标记那些是正样本
	matched_labels: (N,) int, 正样本对应gt 类别，负样本为 -1
	num_classes: 类别数
	返回：
		标量 BCE loss
	说明：
		使用 BCE 而不是 CrossEntropy, 更适合多标签 / 多类别二分类场景
	"""
	target = torch.zeros_like(pred_logits)  # 默认负样本全为零
	if pos_mask.any():
		# 找出正样本 anchor 的索引
		pos_idx = pos_mask.nonzero(as_tuple=False).squeeze(1)
		# 查表
		labels = matched_labels[pos_idx].long()
		target[pos_idx, labels] = 1.0  # 正样本对应类别为 1
	# 损失函数待理解
	loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='mean')
	return loss


# =======================
# Step2: bbox L1 回归
# =======================
def bbox_l1_loss(pred_cxcywh, target_cxcywh, pos_mask):
	"""
	L1 回归损失， 仅对正样本 anchor
	输入：
		pred_cxcywh: (N, 4) 预测框
		target_cxxywh: (N, 4) 对应gt 框
		pos_mask: (N,) bool
	返回：
		标量 L1 loss
	"""
	if pos_mask.sum() == 0:
		return torch.tensor(0., device=pred_cxcywh.device)
	pos_idx = pos_mask.nonzero(as_tuple=False).squeeze(1)
	return F.l1_loss(pred_cxcywh[pos_idx], target_cxcywh[pos_idx], reduction='mean')


# ========================
# Step3: IoU 损失
# ========================
def iou_loss(pred_cxcywh, target_xyxy, pos_mask):
	"""
	IoU 损失 = 1 - IoU， 仅对正样本
	输入：
		pred_cxcywh: (N, 4) 预测框
		target_xyxy: (N, 4) gt 框
		pos_mask: (N, 4)
	返回：
		标量 IoU loss
	"""
	if pos_mask.sum() == 0:
		return torch.tensor(0., device=pred_cxcywh.device)
	pos_idx = pos_mask.nonzero(as_tuple=False).squeeze(1)
	pred_xy = cxcywh_to_xyxy(pred_cxcywh[pos_idx])
	tgt = target_xyxy[pos_idx]
	ious = pairwise_iou(pred_xy, tgt).diag()  # 一一对应取对角线
	return (1.0 - ious).mean()


# ========================
# Step4: CIoU 损失
# ========================
def ciou_loss(pred_cxcywh, target_xyxy, pos_mask):
	"""
	CIoU 损失：结合IoU + 中心点距离 + 宽高比惩罚
	输入输出与 iou_loss 类似
	"""
	if pos_mask.sum() == 0:
		return torch.tensor(0., device=pred_cxcywh.device)
	pos_idx = pos_mask.nonzero(as_tuple=False).squeeze(1)
	pred = cxcywh_to_xyxy(pred_cxcywh[pos_idx])
	tgt = target_xyxy[pos_idx]
	
	ious = pairwise_iou(pred, tgt).diag().clamp(min=EPS)
	
	# 中心点距离平方
	pred_c = (pred[:, :2] + pred[:, 2:]) / 2
	tgt_c = (tgt[:, :2] + tgt[:, 2:]) / 2
	center_dist2 = ((pred_c - tgt_c) ** 2).sum(dim=1)
	
	# 包围框对角线平方
	enclose_x1 = torch.min(pred[:, 0], tgt[:, 0])
	enclose_y1 = torch.min(pred[:, 1], tgt[:, 1])
	enclose_x2 = torch.max(pred[:, 2], tgt[:, 2])
	enclose_y2 = torch.max(pred[:, 3], tgt[:, 3])
	c2 = ((enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2).clamp(min=EPS)
	
	# 宽高比惩罚项
	# 计算预测框的宽度和高度
	# pred[:, 2] = x2, pred[:, 0] = x1 → 宽度 = x2 - x1
	# pred[:, 3] = y2, pred[:, 1] = y1 → 高度 = y2 - y1
	# .clamp(min=EPS) 确保宽高不为 0，防止后续除零错误
	pw, ph = (pred[:, 2] - pred[:, 0]).clamp(min=EPS), (pred[:, 3] - pred[:, 1]).clamp(min=EPS)
	# pw: predicted width  (M,)
	# ph: predicted height (M,)
	
	# 计算真实框的宽度和高度
	# 同样防止宽高为 0
	gw, gh = (tgt[:, 2] - tgt[:, 0]).clamp(min=EPS), (tgt[:, 3] - tgt[:, 1]).clamp(min=EPS)
	# gw: ground truth width  (M,)
	# gh: ground truth height (M,)
	#  v 是宽高比一致性惩罚项
	# 公式：v = (4 / π²) * (arctan(gw/gh) - arctan(pw/ph))²
	
	# torch.atan(gw/gh): 真实框宽高比的“角度”表示
	# torch.atan(pw/ph): 预测框宽高比的“角度”表示
	# 差值的平方：衡量宽高比的差异程度
	# 乘以 （4.0 / (torch.pi ** 2)): 归一化到[0,1] 范围
	
	v = (4.0 / (torch.pi ** 2)) * (torch.atan(gw / gh) - torch.atan(pw / ph)) ** 2
	# alpan 是 v 的权重， 动态调整
	# 公式： alpha = v / (1 - IoU + v + EPS)
	
	# with torch.no_grad(): 表示这部分不参与梯度计算
	# 因为 alpha 是一个“ 控制信号”， 不是学习参数
	with torch.no_grad():
		alpha = v / ((1.0 - ious) + v + EPS)
	
	# CIoU 公式：
	# CIoU = IoU - (中心点距离惩罚） - （宽高比惩罚）
	
	# 第一项： ious 重叠度， 越大越好
	# 第二项： （center_dist2 / c2) 中心点距离归一化惩罚
	# 距离越远， 惩罚越大
	# 第三项： alpaa * v 动态宽高比惩罚
	# 形状越不匹配， 惩罚越大
	
	ciou = ious - (center_dist2 / c2) - alpha * v
	# CIoU 理论上应在[-1, 1] 但数值计算可能越界
	# clamp 保证 数值稳定，防止异常梯度
	# 损失 = 1 - CIoU
	# CIoU 越大 损失越小
	# CIoU = 1 损失等于 = 0 完美
	# CIoU = 0 损失等于 = 无重叠
	return (1.0 - ciou).mean()


# ==========================
# Step5 简单 center-in-box 分配器
# ==========================
def center_assign(anchor_centers_xy, gt_boxes_xyxy, gt_labels):
	"""
	这个函数的作用是：判断那些”候选点“ （anchor中心） 应该负责预测真实物体 （gt框）。
	
	想象你在地图上放了很多”探测点“ （anchor_centers),现在你要告诉每个探测点：
	- 它要不要去预测某个真实物体？ （是正样本吗？）
	- 如果要， 那它该预测哪一个真实物体？
	- 那个物体是什么类别？ （比如车、人、狗）
	
	判断标准很简单：只要一个“探测点” 落在某个真实物体的框内， 它就有资格成为正样本：
	如果它同时在多个物体框内， 就选离那个物体中心最近的那个。
	
	输入：
		anchor_centers_xy: (N, 2) 一堆探测点的坐标， 格式是[x, y]
		gt_boxes_xyxy:     (M, 4) 真实物体的边界框， 格式是 [左上x, 左上y, 右下x, 右下y]
		gt_labels:         (M, )  每个真实物体的类别编号， 比如0 = 猫， 1 = 狗...
		
	返回：
		pos_mask:          (N,) True/False -那些探测点是”正样本“ （需要负责预测）
		matched_gt_idx:    (N,) 数字 --每个探测点匹配到哪个真实物体 （-1 表示没匹配上）
		matched_labels:    (N,) 数字 --每个探测点应该预测的类别 （-1 表示不预测任何东西）
	"""
	
	N = anchor_centers_xy.shape[0]  # 探测点的数量
	M = gt_boxes_xyxy.shape[0]  # 真实物体的数量
	device = anchor_centers_xy.device  # 数据在哪块显卡 /cpu上
	dtype = anchor_centers_xy.dtype  # 数据类型 （如 float32)
	
	# 如果根本没有真实物体 （M=0), 那就没人可匹配，所有探测点都标记为”负样本“
	if M == 0:
		return (
			torch.zeros(N, dtype=torch.bool, device=device),  # 全 False: 都不是正样本
			torch.full((N,), -1, dtype=torch.long, device=device),  # 全 -1：都没有匹配到
			torch.full((N,), -1, dtype=torch.long, device=device)  # 全 -1：都没有类别
		)
	
	# 第一步：把数据重新整理一下， 方便后面”批量比较“
	# 我们要把每个探测点和每个真实框都比一遍， 看看谁在里面。
	# pytorch 能自动对不同形状的张量做“广播”计算，所以我们调整维度：
	ax = anchor_centers_xy[:, 0:1]  # （N，1）每个探测点的x坐标 （列向量）
	ay = anchor_centers_xy[:, 1:2]  # (N,1) 每个探测点的y坐标
	
	x1 = gt_boxes_xyxy[None, :, 0]  # (1, M) 所有真实框的左上角 x
	y1 = gt_boxes_xyxy[None, :, 1]  # (1, M) 所有真实框的左上角 y
	x2 = gt_boxes_xyxy[None, :, 2]  # (1, M) 所有真实框的右下角 x
	y2 = gt_boxes_xyxy[None, :, 3]  # (1, M) 所有真实框的右下角 y
	
	# 第二步：判断每个探测点是否落在每个真实框内部（含边线）
	# 条件： x 在左和右之间， 且 y 在上和下之间
	inside = cast(torch.Tensor, (ax >= x1) & (ay >= y1) & (ax <= x2) & (ay <= y2))
	# inside[i, j] 为 True 表示 第 i 个探测点 落在 第 j 个真实框 内部
	
	# any_inside[i] 表示第 i 个探测点 是否至少在一个真实框里面
	any_inside = inside.any(dim=1)
	
	# 初始化：假设所有探测点都没匹配到任何真实物体
	matched_gt_idx = -torch.ones(N, dtype=torch.long, device=device)
	
	# 第三步： 解决“一个探测点在多个框内的情况 比如两个物体挨得很近
	# 规则：挑一个离他最近的真实物体中心来匹配 （更合理）
	if any_inside.any():  # 只有确定有点落在框里才需要处理
		# 计算每个真实物体的中心点
		gt_centers = (gt_boxes_xyxy[:, :2] + gt_boxes_xyxy[:, 2:]) / 2.0  # (M, 2)
		
		# 计算每个探测点到每个真实物体中心的距离 平方 避免开根号
		# unsqueeze 是为了形成（N，1，2）和（1， M, 2), 能做广播减法
		dist2 = ((anchor_centers_xy.unsqueeze(1) - gt_centers.unsqueeze(0)) ** 2).sum(dim=2)  # (N, M)
		
		# 把”不在“ 框内的距离设为一个超大的数， 这样不会被误选为”最近“
		big = torch.tensor(1e9, device=device, dtype=dist2.dtype)
		dist2 = torch.where(inside, dist2, big)  # 只保留框内的距离有效
		
		# 对每一行找最小距离对应的 gt 索引（即最近的那个真实物体）
		matched_idx_all = dist2.argmin(dim=1)  # (N,) 每个探测点”理论上“ 该匹配谁
		
		# 但注意：只有那些确实在某个框里的探测点才允许匹配
		# 所以我们只更新这些点的匹配结果
		matched_gt_idx[any_inside] = matched_idx_all[any_inside]
	
	# 第四步：生成最终输出
	pos_mask = matched_gt_idx >= 0  # 匹配到了就是正样本 （True）， 否则就是负样本 （False）
	
	matched_labels = -torch.ones(N, dtype=torch.long, device=device)  # 默认无类别
	if pos_mask.any():
		# 找出正样本对应的真实物体索引，然后查它们的类别
		matched_labels[pos_mask] = gt_labels[matched_gt_idx[pos_mask]]
	
	return pos_mask, matched_gt_idx, matched_labels


# ===============================
# Step6: 组合 compute_loss
# ===============================
def compute_loss(pred_logits, pred_cxcywh, anchor_centers_xy, gt_boxes_xyxy, gt_labels,
				 weight_box=7.5, weight_cls=0.5, weight_l1=1.0, iou_type='ciou'):
	"""
	单张图片的完整损失计算函数 （用于目标检测模型训练）
	
	输入：
	pred_logits:(N,C)                 每个anchor 的类别预测 logits （未归一化分数）
	pred_cxcywh: (N, 4)               每个 anchor 预测的边界框， 格式为（cx, cy, w, h)
	anchor_centers_xy: (N,2)          每个 anchor 的中心坐标（x, y), 用于正样本匹配
	gt_boxes_xyxy: (M,4)              真实框坐标， 格式为 （x1,y1, x2, y2)
	gt_labels: （M,)                   真实类别标签 （整数， 从0开始）
	weight_box: float                 IoU/CIoU 损失的权重， 默认 7.5 （强调定位）
	weight_cls: float                 分类损失权重， 默认 0.5 防止负样本主导
	iou_type:str                      使用 'iou' 或 'ciou' 计算 bbox 损失
	
	输出：
		total_loss: 标量 Tensor         总损失（可反向传播）
		loss_dict: dict                 各项损失字典，便于调试和日志记录
	"""
	
	# 获取预测张量所在的设备 （如 CPU 或 CUDA) 用于后续创建张量时保持设备一致
	device = pred_logits.device
	
	# 获取类别总数 C ,即分类头的输出维度，用于构建分类目标
	num_classes = pred_logits.shape[1]
	
	# 1) 使用 center_assign 函数进行正负样本分配
	#   原理：若某个anchor 的中心点落在某个 GT 框内部， 则该 anchor 为正样本
	#   输出：
	#   	pos_mask:(N,) bool, 表示每个anchor 是否为正样本
	#       matched_gt_idx: （N, ) long, 记录每个anchor 应学习到的类别标签 （负样本标签不参与分类 loss)
	pos_mask, matched_gt_idx, matched_labels = center_assign(anchor_centers_xy, gt_boxes_xyxy, gt_labels)
	
	# 2) 计算分类损失：使用自定义的classification_bce_loss 函数
	#      - 所有 anchor 都参与计算
	#      - 正样本学习其对应类别的 one-hot 标签
	#      - 负样本学习“全为0” 的背景标签（多标签 BCE 设计）
	#      - 函数内部应处理 one-hot 构建和正负样本 mask
	loss_cls = classification_bce_loss(pred_logits, pos_mask, matched_labels, num_classes)
	
	# 3) 计算边界框回归相关的损失 （仅正样本参与）
	#    先判断是否存在正样本， 避免空 tensor 操作导致错误
	if pos_mask.sum() == 0:
		# 特殊情况：没有anchor 被分配为正样本 （罕见但可能发生）
		# 创建值为 0 的标量张量， 并确保与输入相同的设备上
		loss_l1 = torch.tensor(0., device=device)   # L1 回归损失为 0
		loss_box = torch.tensor(0., device=device)   # IoU/CIoU 损失为0
	else:
		# 提取正样本的索引 （非零位置）
		pos_idx = pos_mask.nonzero(as_tuple=False).squeeze(1)    # 得到一维索引列表
		
		# 根据正样本索引， 找到它们各自匹配的 GT 框
		# matched_gt_idx[pos_idx]  是这些正样本对应的 GT 索引
		# gt_boxes_xyxy[...] 取出对应的GT 框，形状为 （K，4）， K是正样本数量
		gt_for_pos = gt_boxes_xyxy[matched_gt_idx[pos_idx]]
		
		# 将GT框从 （x1, y1, x2, y2) 格式转换为 （cx,cy,w,h) 上的格式
		# 以便与 pred_cxcywh 的格式一致， 用于计算 L1 损失
		target_cxcywh = xyxy_to_cxcywh(gt_for_pos)
		
		# 计算 L1 损失：衡量预测框和目标框在 （cx, cy, w, h) 上的绝对坐标误差
		# pred_cxcywh[pos_idx] 是正样本的预测框
		# reduction='mean' 表示
		loss_l1 = F.l1_loss(pred_cxcywh[pos_idx], target_cxcywh, reduction='mean')
		
		# 计算 IoU或CIoU损失：衡量预测框与真实框的空间重叠质量
		# CIoU 比普通 IoU更好， 考虑了中心距离、长宽比等因素
		if iou_type.lower() == 'ciou':
			# 调用自定义 ciou_loss 函数， 传入所有预测框、所有 GT 和 正样本 mask
			# 函数内部应只对正样本 计算并返回平均 loss
			loss_box = ciou_loss(pred_cxcywh, gt_boxes_xyxy, pos_mask)
		else:
			# 使用普通 IoU loss
			loss_box = iou_loss(pred_cxcywh, gt_boxes_xyxy, pos_mask)
			
	# 4) 加权求和， 得到总损失
	#    各项损失乘以对应权重后相加
	#    权重设置体现了不同任务的重要性：
	#      - weight_box 较大（7.5）：强调定位精度
	#      - weight_cls 较小 （0.5）： 因负样本多，防止分类主导训练
	#      - weight_l1 适中 （1.0）：辅助坐标回归稳定新
	total_loss = weight_box * loss_box + weight_cls * loss_cls + weight_l1 * loss_l1
	
	# 返回总损失和各项损失字典 （用于训练日志、可视化、调试
	# 注意： loss_dict 中的各项使用 detach(), 避免保存计算图， 节省内存
	return total_loss, {
		'total': total_loss.detach(),    # 总损失（无梯度）
		'box': loss_box.detach(),        # IoU/CIou 损失
		'cls': loss_cls.detach(),        # 分类损失
		'l1': loss_l1.detach()           # l1 回归损失
	}
