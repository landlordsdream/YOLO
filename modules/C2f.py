# -*- coding: utf-8 -*-
"""
从零实现 YOLOv8 风格的 C2f 模块 （含Bottleneck 与 Conv)
- Conv: 卷积 + BN + 激活
- Bottleneck: 残差瓶颈块（1x1 降/升维 + 3x3 卷积，带可选残差）
- C2f: ELAN/CSP 风格的轻量化特征融合块，YOLOv8 主干常用
"""

import torch
import torch.nn as nn


class Conv(nn.Module):
    """
    基础卷积块：Conv2d -> BatchNorm2d -> 激活
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, bias=False, act=True):
        # 继承 nn.Module,保证参数被框架管理
        super().__init__()
        # same padding: 若未显示传入，则根据kernel_size自动计算
        self.conv = nn.Conv2d(
            in_channels=in_channels,    # 输入通道
            out_channels=out_channels,   # 输出通道
            kernel_size=kernel_size,     # 卷积核大小
            stride=stride,               # 步长
            padding=(kernel_size - 1) // 2 if padding is None else padding,    # same padding
            groups=groups,                 # 分组卷积
            bias=bias                      # 是否使用偏置
        )
        
        # 批归一化层
        self.bn = nn.BatchNorm2d(out_channels)
        # 激活函数：默认 SiLU； 若 act = False, 则为恒等映射
        self.act = nn.SiLU() if act else nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)   # 卷积操作
        x = self.bn(x)     # 批归一化
        x = self.act(x)    # 激活函数
        return x
    
    
class Bottleneck(nn.Module):
    """
    残差瓶颈块：
    - 先用 1x1 卷积调整通道（降、升维）
    - 再用 3x3 卷积做局部特征抽取
    - 可选残差：输入与输出相加 （要求 in==out 且允许 shortcut)
    """
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super().__init__()
        # 隐藏通道数：用于 1x1 的瓶颈宽度， 默认0.5*out
        hidden_channels = int(out_channels * expansion)
        # 1x1 卷积：调整通道，计算量低
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1)
        # 3x3 卷积：提取空间邻域特征 （same padding 保持尺寸）
        self.cv2 = Conv(hidden_channels, out_channels, kernel_size=3, stride=1)
        # 是否使用残杀连接：只有 in 与 out 通道数相同时才安全相加
        self.use_shortcut = shortcut and (in_channels == out_channels)
        
    def forward(self, x):
        # 保存残差分支输入、
        identity = x
        # 主分支： 1x1 调整通道
        x = self.cv1(x)
        # 主分支： 3x3 抽特征
        x = self.cv2(x)
        # 残差连接 ：叠加输入，提高信息流通与梯度传播
        if self.use_shortcut:
            x = x + identity
        return x
    
    
class C2f(nn.Module):
    """
    YOLOv8 核心的 C2f （Cross-Stage Partial with 'f' used style) 块：
    直观理解（与官方思想一致，但自写实现）
    1. 先把输入通过 1x1 卷积，生成 2*hidden 通道
    2. 沿通道维切成两份：x1,x2 (各 hidden 通道）
    3. 用 n 个 Bottleneck 串行地“滚动”到 x2 上， 每次的输出都保留
    4. 把 [x1,x2,每次更新后的...]在通道维拼接
    5.再用 1x1 卷积融合到目标 out_channels
    这样就实现了“多分支保留 + 串行细化 + 最后融合”， 即高效又易训练
    """
    def __init__(self, in_channels, out_channels, n=3, shortcut=True, expansion=0.5):
        """
        参数：
        in_channels   : 输入通道数
        out_channels  : 输出通道
        n             : 内部 Bottleneck 的数量
        shortcut      : 是否在 Bottleneck 内启用残差
        expansion     : 隐藏通道宽度比例（相对 out_channels)
        """
        super().__init__()
        # 计算隐藏通道数：通常为out_channels 的一半左右
        self.hidden = int(out_channels * expansion)
        
        # 预卷积：把输入通道映射为 2* hidden，，便于后面一分为二
        self.cv1 = Conv(in_channels, 2 * self.hidden, kernel_size=1, stride=1)
        
        # 创建 n 个 Bottleneck, 逐个串行处理 第二支 x2
        self.m = nn.ModuleList([
            Bottleneck(self.hidden, self.hidden, shortcut=shortcut, expansion=1.0)
            for _ in range(n)
        ])
        # 注意：这里 Bottleneck 的 in/out 都是 hidden, expansion=1.0 代表瓶颈内部宽度=out
        
        # 融合卷积：把拼接后的通道数压回 out_channels
        # 拼接的通道总数 = x1(hidden) + x2(每次输出hidden) * (n + 1)
        # 其中（n + 1）包含“初始 x2" 与 n 次更新后的 n 个结果
        concat_channels = (2 + n)*self.hidden    # x1 + 初始 x2 + n次输出
        self.cv2 = Conv(concat_channels, out_channels, kernel_size=1, stride=1)
        
    def forward(self, x):
        # 先做 1x1 卷积得到 2* hidden 通道的特征
        y = self.cv1(x)
        # 在通道维（dim=1) 一分为二，得到 x1, x2(每个都是 hidden 通道）
        x1, x2 = y.split((self.hidden, self.hidden), dim=1)
        
        # 准备一个列表， 用于收集要拼接的特征
        outs = [x1, x2]     # 先放入 x1 与初始的 x2
        # 依次把 x2 送入每个 Bottleneck, 并把每次的输入也收集起来
        for b in self.m:
            x2 = b(x2)    # 串行细化
            outs.append(x2)   # 记录每次输出
            
        # 在通道维拼接：[x1, 初始x2, x2_第一次、、、]
        z = torch.cat(outs, dim=1)
        # 1x1 融合到目标通道数
        return self.cv2(z)
    
    
# ------------------------ 测试代码 ------------------------
if __name__ == "__main__":
    # 构造一个假输入：batch=1, 通道=64, 分辨率=80x80
    x = torch.randn(1, 64, 80, 80)

    # 示例1：C2f 把通道从 64 -> 128，内部 3 个 Bottleneck
    c2f = C2f(in_channels=64, out_channels=128, n=3, shortcut=True, expansion=0.5)
    y = c2f(x)
    print("输入形状: ", x.shape)  # torch.Size([1, 64, 80, 80])
    print("输出形状: ", y.shape)  # torch.Size([1, 128, 80, 80])

    # 示例2：保持通道不变，看看是否能正常前向
    c2f_same = C2f(in_channels=128, out_channels=128, n=2, shortcut=True, expansion=0.5)
    y2 = c2f_same(torch.randn(1, 128, 40, 40))
    print("输出形状(保持通道): ", y2.shape)  # torch.Size([1, 128, 40, 40])