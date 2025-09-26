import torch
from torchvision.ops import nms


def yolo_postprocess(preds, conf_thres=0.25, iou_thres=0.45, max_det=300):
	"""
	YOLOv8 后处理函数 （置信度计算 + NMS）
	
	输入：
		preds: torch.tensor, shape (B, N, 4+nc)
				decode_yolov8 输入的预料， B=batch size, N=预测框数量， nc=类别数
				每个框内容：[cx, cy, w, h, class_probs...]
		conf_thres: float
				置信度阈值, 小于该值的框会被丢掉
		iou_thres: float
				NMS IoU 阈值， 相同类别 IoU > iou_thres 的框只保留分数高的
		max_det: int
				每张图最多保留 max_det 个检测框
				
	输出：
		results： list[tensor] 长度 = B
				每个元素 shape: (num_det, 6)
				每个框内容： [x1, y1, x2, y2, score, class]
	"""
	
	B, N, C = preds.shape
	# C = 4 + nc, 前4是 bbox, 后 nc 是类别概率
	nc = C - 4
	
	results = []  # 用于存储 batch 每张图的最终检测结果
	
	# 遍历 batch 中的每一张图
	for b in range(B):
		# 取出 bbox 部分 (cx, cy, w, h), shape = (N, 4)
		boxes = preds[b, :, :4]
		# 取出类别概率部分， shape = (N, nc)
		scores = preds[b, :, 4:]
		
		# --------------------
		# 计算每个框的置信度和类别
		# 对每个预测框找到最高类别概率及对应类别索引
		# conf shape = (N,), cls shape = (N,)
		conf, cls = scores.max(dim=1)
		
		# 根据置信度阈值过滤掉低置信度框
		mask = conf > conf_thres  # bool tensor, shape = (N,)
		
		# 如果没有框通过阈值， 则返回空 tensor
		if mask.sum() == 0:
			results.append(torch.zeros((0, 6), device=preds.device))
			continue
		
		# 筛选出高置信度的 bbox, conf, class
		boxes = boxes[mask]  # shape = (num_kept, 4)
		conf = conf[mask]  # shape = (num_kept, )
		cls = cls[mask]  # shape = (num_kept, )
		
		# ---------------------
		# bbox 格式转换：cx, cy, w, h -> x1, y1, x2, y2
		# 原因：NMS 需要角点坐标
		cx, cy, w, h = boxes.unbind(dim=1)  # shape: (num_kept, )
		
		# 左上角 x1 = cx - w/2, y1 = cy - h/2
		x1 = cx - w / 2
		y1 = cy - h / 2
		# 右下角 x2 = cx + w/2, y2 = cy + w/2
		x2 = cx + w / 2
		y2 = cy + h / 2
		
		# stack 成 （num_kept, 4)
		boxes = torch.stack([x1, y1, x2, y2], dim=1)
		
		# ----------------------
		# NMS 非极大值抑制
		# 目的： 去掉重复预测
		# 原理： 对同类别的框， 按 score 排序， 从高到低迭代
		# 		如果IoU > iou_thres, 就舍弃分数低的
		# torchvision.ops.nms 内部做了以下步骤：
		# 1. 按分数排序
		# 2. 初始化 keep=[]
		# 3. 依次判断 IoU，如果 IoU > thres 则丢弃
		# 返回的是保留框的索引
		keep = nms(boxes, conf, iou_thres)  # shape: (num_kept_after_nms)
		
		# 限制最多 max_det 个框
		keep = keep[:max_det]
		
		# ---------------------
		# 拼接最终检测框
		# boxes[keep]: shape (num_det, 4)
		# conf[keep].unsqueeze(1): shape (num_det, 1)
		# cls[keep].unsqueeze(1).float(): shape (num_det,1)
		# 拼接后 shape = （num_det, 6)
		dets = torch.cat([
			boxes[keep],
			conf[keep].unsqueeze(1),
			cls[keep].unsqueeze(1),
		], dim=1)
		
		# 加入结果列表
		results.append(dets)
	
	# 返回整个 batch 的检测结果
	return results
