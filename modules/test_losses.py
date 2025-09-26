import torch
from YOLOv8Loss import (
    cxcywh_to_xyxy, xyxy_to_cxcywh, box_area,
    pairwise_iou, iou_loss, ciou_loss,
    classification_bce_loss, bbox_l1_loss,
    center_assign, compute_loss
)

def test_coordinate_transform():
    boxes_cxcywh = torch.tensor([[5., 5., 4., 4.]])
    boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh)
    back_cxcywh = xyxy_to_cxcywh(boxes_xyxy)
    print("Original:", boxes_cxcywh)
    print("xyxy:", boxes_xyxy)
    print("Back:", back_cxcywh)
    assert torch.allclose(boxes_cxcywh, back_cxcywh, atol=1e-6)

def test_area_iou():
    box1 = torch.tensor([[0., 0., 2., 2.]])
    box2 = torch.tensor([[1., 1., 3., 3.]])
    iou = pairwise_iou(box1, box2)
    print("IoU:", iou.item())  # 应该是 1/7 ≈ 0.142857
    assert abs(iou.item() - 1/7) < 1e-6

def test_classification_bce():
    pred_logits = torch.tensor([[0.0, 1.0], [1.0, -1.0]])
    pos_mask = torch.tensor([True, False])
    matched_labels = torch.tensor([1, -1])
    loss = classification_bce_loss(pred_logits, pos_mask, matched_labels, num_classes=2)
    print("BCE Loss:", loss.item())

def test_bbox_l1():
    pred = torch.tensor([[5., 5., 2., 2.]])
    target = torch.tensor([[5., 5., 3., 3.]])
    pos_mask = torch.tensor([True])
    loss = bbox_l1_loss(pred, target, pos_mask)
    print("L1 Loss:", loss.item())  # 应该是 (1+1)/4 = 0.5

def test_center_assign():
    anchors = torch.tensor([[2., 2.], [6., 6.], [10., 10.]])
    gts = torch.tensor([[0., 0., 5., 5.], [5., 5., 9., 9.]])
    labels = torch.tensor([0, 1])
    pos_mask, matched_idx, matched_labels = center_assign(anchors, gts, labels)
    print("pos_mask:", pos_mask)
    print("matched_idx:", matched_idx)
    print("matched_labels:", matched_labels)

def test_compute_loss():
    pred_logits = torch.randn(5, 2)  # (N, C)
    pred_cxcywh = torch.rand(5, 4) * 10
    anchors = torch.rand(5, 2) * 10
    gt_boxes = torch.tensor([[2., 2., 6., 6.], [5., 5., 9., 9.]])
    gt_labels = torch.tensor([0, 1])
    total_loss, loss_dict = compute_loss(pred_logits, pred_cxcywh, anchors, gt_boxes, gt_labels)
    print("Total Loss:", total_loss.item())
    print("Loss Dict:", loss_dict)

if __name__ == "__main__":
    test_coordinate_transform()
    test_area_iou()
    test_classification_bce()
    test_bbox_l1()
    test_center_assign()
    test_compute_loss()
