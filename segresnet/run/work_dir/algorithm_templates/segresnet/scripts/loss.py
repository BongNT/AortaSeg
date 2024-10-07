from typing import Callable, Sequence
import warnings
from monai.losses.cldice import SoftclDiceLoss
from monai.losses.dice import DiceCELoss, GeneralizedDiceFocalLoss
from monai.losses import HausdorffDTLoss, LogHausdorffDTLoss

from torch.nn.modules.loss import _Loss
from monai.utils import DiceCEReduction, LossReduction, Weight, deprecated_arg, look_up_option, pytorch_after
from monai.networks import one_hot

import torch
import torch.nn as nn
import torch.nn.functional as F
class SoftclDice_GeneralizedDiceFocal_Loss(_Loss):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        w_type: Weight | str = Weight.SQUARE,
        reduction: LossReduction | str = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        gamma: float = 2.0,
        focal_weight: Sequence[float] | float | int | torch.Tensor | None = None,
        weight: Sequence[float] | float | int | torch.Tensor | None = None,
        lambda_gdl: float = 1.0,
        lambda_focal: float = 1.0,
    ) -> None:
        super().__init__()
        self.dice_ce = GeneralizedDiceFocalLoss(include_background,to_onehot_y, sigmoid, softmax, other_act, 
                                 reduction=reduction,smooth_dr=smooth_dr,smooth_nr=smooth_nr, batch=batch,
                                  weight=weight, lambda_gdl=lambda_gdl, lambda_focal=lambda_focal)
        self.softcl = SoftSkeletonRecallLoss()
        
    def forward(self, input: torch.Tensor, target: torch.Tensor, gt_skeleton: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].
        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )
        dice_loss = self.dice_ce(input, target)
        soft_cldice_loss = self.softcl(input, gt_skeleton)
        total_loss: torch.Tensor = 1 * dice_loss + 0.8 * soft_cldice_loss

        return total_loss
    
class SkeRecallDiceCELoss(_Loss):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        reduction: LossReduction | str = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        squared_pred:bool = False,
        dice_weight: Sequence[float] | float | int | torch.Tensor | None = None,
        ce_weight: Sequence[float] | float | int | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.dice_ce = DiceCELoss(include_background,to_onehot_y, sigmoid, softmax, other_act, 
                                 reduction=reduction,smooth_dr=smooth_dr,smooth_nr=smooth_nr, batch=batch,
                                 squared_pred=squared_pred)
        self.skeleton_loss = SoftSkeletonRecallLoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, gt_skeleton: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].
        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(pred.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {pred.shape} and {target.shape}."
            )
        dice_loss = self.dice_ce(pred, target)
        soft_cldice_loss = self.skeleton_loss(pred, gt_skeleton)
        total_loss: torch.Tensor = 1 * dice_loss + 1 * soft_cldice_loss

        return total_loss
    
    
class SkeRecallHausdorffDTLoss(_Loss):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        reduction: LossReduction | str = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        squared_pred:bool = False,
        weight: Sequence[float] | float | int | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.haus_loss = LogHausdorffDTLoss(include_background=include_background,to_onehot_y=to_onehot_y, sigmoid=sigmoid, softmax=softmax, 
                                 reduction=reduction, batch=batch)
        self.skeleton_loss = SoftSkeletonRecallLoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, gt_skeleton: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].
        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(pred.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {pred.shape} and {target.shape}."
            )
        dtloss = self.haus_loss(pred, target)
        soft_cldice_loss = self.skeleton_loss(pred, gt_skeleton)
        total_loss: torch.Tensor = 1 * dtloss + 1 * soft_cldice_loss

        return total_loss


class SoftSkeletonRecallLoss(_Loss):
    def __init__(self, batch: bool = False, include_background: bool = False, smooth: float = 1.):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(SoftSkeletonRecallLoss, self).__init__()

        self.batch = batch
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, pred, gt_skeleton):
        n_pred_ch = pred.shape[1]
        pred = torch.softmax(pred, dim=1)
        gt_skeleton = one_hot(gt_skeleton, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                gt_skeleton = gt_skeleton[:, 1:]
                pred = pred[:, 1:]

        assert pred.size() == gt_skeleton.size(), f'predict {pred.size()} & target {gt_skeleton.size()} shape do not match'
        
        reduce_axis: list[int] = torch.arange(2, len(pred.shape)).tolist() # BNH[WD]
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis
        
        # start cal loss
        sum_gt = gt_skeleton.sum(reduce_axis)
        inter_rec = (pred * gt_skeleton).sum(reduce_axis)
        
        rec = (inter_rec + self.smooth) / (torch.clip(sum_gt+self.smooth, 1e-8))

        rec = rec.mean()
        return -rec
        
        
    


class BoundaryDoULoss(nn.Module):
    def __init__(self, include_background):
        super(BoundaryDoULoss, self).__init__()
        self.include_background = include_background

    def _adaptive_size(self, pred_c, target, smooth=1e-5):
        # Create the kernel for boundary detection in 3D
        kernel = torch.Tensor([[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]).unsqueeze(0).unsqueeze(0).to(pred_c.device)
        kernel = kernel / kernel.sum()  # Normalize the kernel

        # Perform the convolution directly on the target to detect boundaries
        target = target.unsqueeze(1)  # Add channel dimension, shape becomes [B, 1, W, H, D]
        Y = F.conv3d(target, kernel, padding='same').to(pred_c.device)
        Y = torch.zeros((target.shape[0], target.shape[1], target.shape[2])).cuda()
        
        # Boundary intersection
        Y = (Y > 0).float() * target

        C = torch.count_nonzero(Y).to(pred_c.device)
        S = torch.count_nonzero(target).to(pred_c.device)

        # Compute alpha
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1
        alpha = min(alpha.item(), 0.8)  # Ensure alpha is a scalar, not a tensor.

        # Compute the loss
        intersect = torch.sum(pred_c * target).to(pred_c.device)
        y_sum = torch.sum(target).to(pred_c.device)
        z_sum = torch.sum(pred_c).to(pred_c.device)
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, pred, target):
        n_pred_ch = pred.shape[1]
        pred = torch.softmax(pred, dim=1)
        target = one_hot(target, num_classes=n_pred_ch).to(pred.device)

        c_range = range(0, n_pred_ch)
        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:].to(pred.device)
                pred = pred[:, 1:].to(pred.device)
                c_range = range(0, n_pred_ch - 1)

        assert pred.size() == target.size(), f'predict {pred.size()} & target {target.size()} shape do not match'

        loss = 0.0
        for i in c_range:
            loss += self._adaptive_size(pred[:, i], target[:, i])
        
        return loss / len(c_range)