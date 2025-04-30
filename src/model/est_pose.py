from typing import Tuple, Dict
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from ..config import Config

class EstPoseNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Directly estimate the translation vector and rotation matrix.
        """
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.fc_trans_1 = nn.Linear(1024, 256)
        self.bn_trans_1 = nn.BatchNorm1d(256)
        self.dropout_trans_1 = nn.Dropout(p=0.3)
        
        self.fc_trans_2 = nn.Linear(256, 128)
        self.bn_trans_2 = nn.BatchNorm1d(128)
        self.dropout_trans_2 = nn.Dropout(p=0.3)
        
        self.fc_trans_out = nn.Linear(128, 3)  # Output translation vector

        self.fc_rot_1 = nn.Linear(1024, 512)
        self.bn_rot_1 = nn.BatchNorm1d(512)
        self.dropout_rot_1 = nn.Dropout(p=0.3)
        
        self.fc_rot_2 = nn.Linear(512, 256)
        self.bn_rot_2 = nn.BatchNorm1d(256)
        self.dropout_rot_2 = nn.Dropout(p=0.3)
        
        self.fc_rot_out = nn.Linear(256, 6)  # Output flattened 3x3 rotation matrix

    def forward(
        self, pc: torch.Tensor, trans: torch.Tensor, rot: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstPoseNet

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        trans : torch.Tensor
            Ground truth translation vector in camera frame, shape \(B, 3\)
        rot : torch.Tensor
            Ground truth rotation matrix in camera frame, shape \(B, 3, 3\)

        Returns
        -------
        float
            The loss value according to ground truth translation and rotation
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        pred_trans,pred_rot = self.est(pc)
        
        def frobenius_loss(R_pred, R_gt):
            return ((R_pred - R_gt) ** 2).sum(dim=(1, 2)).mean()
        
        def rotation_error_rad(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
            """
            R_pred, R_gt: [B, 3, 3] rotation matrices
            Returns: [B] rotation error in radians
            """
            # Compute relative rotation
            R_rel = torch.matmul(R_pred.transpose(-1, -2), R_gt)
            trace = R_rel.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
            cos_theta = (trace - 1) / 2
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # numerical stability
            return torch.acos(cos_theta).mean()

        loss_trans = F.l1_loss(pred_trans,trans)
        loss_rot   = rotation_error_rad(pred_rot,rot)

        metric = dict(
            Transformation_Loss=loss_trans.item(),
            Rotation_Loss = (loss_rot.item()),
        )
        loss = loss_trans + loss_rot
        return loss, metric
    

    def rotation_6d_to_matrix(self, d6: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        Convert 6D rotation representation to 3x3 rotation matrix using Gram-Schmidt orthogonalization.
        Input:  (B, 6)
        Output: (B, 3, 3)
        """
        assert d6.shape[-1] == 6, "Input must have 6 dimensions"

        def safe_normalize(v: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
            norm = v.norm(dim=dim, keepdim=True).clamp(min=eps)
            return v / norm
        rmtype = d6.dtype
        # Use double precision to reduce numerical errors
        d6 = d6.to(dtype=torch.float64)

        # First basis vector
        a1 = safe_normalize(d6[:, 0:3], dim=1, eps=eps)

        # Orthogonalize second vector with respect to first
        a2 = d6[:, 3:6]
        proj = (a1 * a2).sum(dim=1, keepdim=True) * a1
        b2 = safe_normalize(a2 - proj, dim=1, eps=eps)

        # Third vector as cross product to ensure orthonormality
        b3 = torch.cross(a1, b2, dim=1)

        # Stack into rotation matrix
        R = torch.stack([a1, b2, b3], dim=-1)  # (B, 3, 3)

        # Return in original dtype if needed
        return R.to(dtype=rmtype)

    def est(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate translation and rotation in the camera frame

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)

        Returns
        -------
        trans: torch.Tensor
            Estimated translation vector in camera frame, shape \(B, 3\)
        rot: torch.Tensor
            Estimated rotation matrix in camera frame, shape \(B, 3, 3\)

        Note
        ----
        The rotation matrix should satisfy the requirement of orthogonality and determinant 1.
        """
        B, N, _ = x.size()
        x = x.transpose(2, 1)  # (B, 3, N)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = torch.max(x, 2)[0]  # global max pooling

        # ----------- Translation MLP -----------
        trans_feat = F.relu(self.bn_trans_1(self.fc_trans_1(x)))
        trans_feat = self.dropout_trans_1(trans_feat)
        trans_feat = F.relu(self.bn_trans_2(self.fc_trans_2(trans_feat)))
        trans_feat = self.dropout_trans_2(trans_feat)
        trans_out = self.fc_trans_out(trans_feat)  # Shape: (B, 3)

        # ----------- Rotation MLP -----------
        rot_feat = F.relu(self.bn_rot_1(self.fc_rot_1(x)))
        rot_feat = self.dropout_rot_1(rot_feat)
        rot_feat = F.relu(self.bn_rot_2(self.fc_rot_2(rot_feat)))
        rot_feat = self.dropout_rot_2(rot_feat)
        rot_out = self.fc_rot_out(rot_feat)  # Shape: (B, 9)
        # rot_out = rot_out.view(-1, 6)     # Shape: (B, 6)
        rot_out = self.rotation_6d_to_matrix(rot_out)

        return trans_out,rot_out

