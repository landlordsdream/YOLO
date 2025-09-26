import torch
import torch.nn.functional as F

def decode_yolov8(outputs, strides=[8, 16, 32]):
	"""
	YOLOv8 Anchor-Free 解码
	
	参数：
		outputs: list of feature maps (来自 Head)
				[p3_out, p4_out, p5_out]
				每个 shape: (B, 4+nc, H, W)
		strides: 每个特征层对应的下采样步长
				p3: stride=8, p4: stride=16, p5: stride=32
				用于把格子坐标映射回原图坐标
				
	返回：
		tensor: 解码后的所有候选框和类别概率
		shape: (B, total_num_preds, 4+nc)
				每个预测框：[cx, cy, w, h, class_probs...]
	"""
	
	decoded = []  # 存储每个尺度解码结果
	
	# 遍历每个体征图
	for i, out in enumerate(outputs):
		B, C, H, W = out.shape
		nc = C - 4  # 类别数 = 通道数 - 4 （前4是bbox 回归）
		
		# ----- 分离回归和分类 ---------
		reg_pred = out[:, :4, :, :]  # 回归预测: (B, 4, H, W) -> [1, t, r, b] 或 [dx, dy, w,h]
		cls_pred = out[:, 4:, :, :]  # 分类预测：（B，nc, H, W)
		
		# ------- 分类解码 ----------
		# 直接用 sigmoid 转概率 （0~1)
		cls_prob = cls_pred.sigmoid()  # (B, nc, H, W)
		
		# -------- 构建网格坐标 --------
		# 每个特征图的每个格子都有一个固定中心坐标 （cx, cy)
		yv, xv = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
		# xv, yv shape: (H, W)
		grid = torch.stack((xv, yv), 2).to(out.device)  # (H, W, 2)
		# grid[..., 0] = x坐标， grid[...,1] = y坐标
	
		# -------- 回归解码 ----------
		reg_pred = F.relu(reg_pred)  # 保证预测的左/上/右/下或宽高 >= 0
		
		# 分离四个方向
		l = reg_pred[:, 0, :, :]  # 左
		t = reg_pred[:, 1, :, :]  # 上
		r = reg_pred[:, 2, :, :]  # 右
		b = reg_pred[:, 3, :, :]  # 下
		
		# 每个格子中心点坐标 （grid + 0.5 标识格子中心
		cx = (grid[..., 0] + 0.5).unsqueeze(0)  # (1, H, W) -> broadcast 到 B
		cy = (grid[..., 1] + 0.5).unsqueeze(0)  # (1, H, W)
		
		# ------ 转成真实 xywh -----
		# 将格子坐标乘 stride 放大到原图尺度
		x1 = (cx * strides[i]) - l
		y1 = (cy * strides[i]) - t
		x2 = (cx * strides[i]) - r
		y2 = (cy * strides[i]) + b
		
		# 计算中心点坐标和宽高
		bx = (x1 + x2) / 2
		by = (y1 + y2) / 2
		bw = x2 - x1
		bh = y2 - y1
		
		# 堆叠成 bbox tensor
		reg_box = torch.stack([bx, by, bw, bh], dim=-1)  # (B, H, W, 4)
		
		# ------- 展开到 （B，H*W， 4） -----
		reg_box = reg_box.view(B, -1, 4)
		cls_prob = cls_prob.permute(0, 2, 3, 1).contiguous().view(B, -1, nc)
		# permute: (B, nc, H, W) -> (B, H, W, nc) -> 展平成 （B, H*W, nc)
	
		# ------ 拼接回归和分类 -----
		out_decoded = torch.cat([reg_box, cls_prob], dim=-1)  # (B, H*W, 4+nc)
		decoded.append(out_decoded)
		
	# --- 合并所有尺度 ----
	# p3+p4+p5 拼接
	return torch.cat(decoded, dim=1)  # (B, sum(H*W), 4+nc)
