from typing import Tuple, Dict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from ..config import Config
from ..vis import Vis


class EstCoordNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Estimate the coordinates in the object frame for each object point.
        """
        super().__init__()
        self.config = config
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, (1,3)), nn.BatchNorm2d(64), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, (1,1)), nn.BatchNorm2d(64), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, (1,1)), nn.BatchNorm2d(64), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128,(1,1)), nn.BatchNorm2d(128), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(128,1024,(1,1)), nn.BatchNorm2d(1024), nn.ReLU())

        # convs on concat of point & global features
        self.conv6 = nn.Sequential(nn.Conv2d(64+1024, 512,1), nn.BatchNorm2d(512), nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(512,256,1), nn.BatchNorm2d(256), nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(256,128,1), nn.BatchNorm2d(128), nn.ReLU())
        self.conv9 = nn.Sequential(nn.Conv2d(128,128,1), nn.BatchNorm2d(128), nn.ReLU())
        self.conv10= nn.Conv2d(128, 3, 1)
        self.previous = []

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
        d6 = d6.to(dtype=torch.float64)
        a1 = safe_normalize(d6[:, 0:3], dim=1, eps=eps)
        a2 = d6[:, 3:6]
        proj = (a1 * a2).sum(dim=1, keepdim=True) * a1
        b2 = safe_normalize(a2 - proj, dim=1, eps=eps)
        b3 = torch.cross(a1, b2, dim=1)
        R = torch.stack([a1, b2, b3], dim=-1)  # (B, 3, 3)
        return R.to(dtype=rmtype)
    
    def svd_transform(self, src, dst, calc_scale=False):
        B, N, D = src.shape
        centroid_src = src.mean(dim=1, keepdim=True)
        centroid_dst = dst.mean(dim=1, keepdim=True)
        src_centered = src - centroid_src
        dst_centered = dst - centroid_dst

        H = torch.einsum('bij,bik->bjk', src_centered, dst_centered)
        U, _, Vt = torch.linalg.svd(H)
        R = Vt.transpose(-2, -1) @ U.transpose(-2, -1)

        # Reflection correction
        det = torch.linalg.det(R)
        mask = det < 0
        Vt[mask] *= torch.tensor([1, 1, -1], device=Vt.device)

        R = Vt.transpose(-2, -1) @ U.transpose(-2, -1)

        if calc_scale:
            scale = (dst_centered.norm(dim=2) / src_centered.norm(dim=2)).mean(dim=1)
        else:
            scale = torch.ones(B, device=src.device)

        t = centroid_dst.squeeze(1) - scale.view(-1, 1) * torch.bmm(R, centroid_src.transpose(1, 2)).squeeze(2)

        return R, t, scale

    def ransac_rigid_transform(self, src_pts, dst_pts, max_iters=1000, threshold=0.01, min_inliers=5):
        B, N, D = src_pts.shape
        best_R, best_t = None, None
        best_inliers = torch.zeros((B,), dtype=torch.long, device=src_pts.device)
        best_mask = torch.zeros((B, N), dtype=torch.bool, device=src_pts.device)

        for _ in range(max_iters):
            # Sample minimal 3 points (for 3D rigid transform)
            indices = torch.randint(0, N, (B, 3), device=src_pts.device)
            src_sample = torch.gather(src_pts, 1, indices.unsqueeze(-1).expand(-1, -1, D))
            dst_sample = torch.gather(dst_pts, 1, indices.unsqueeze(-1).expand(-1, -1, D))

            R, t, _ = self.svd_transform(src_sample, dst_sample)
            src_transformed = torch.bmm(R, src_pts.transpose(1, 2)).transpose(1, 2) + t.unsqueeze(1)

            error = ((dst_pts - src_transformed) ** 2).sum(dim=2)
            mask = error < threshold ** 2
            inliers = mask.sum(dim=1)

            update = inliers > best_inliers
            if update.any():
                best_inliers = torch.where(update, inliers, best_inliers)
                best_mask = torch.where(update.unsqueeze(1), mask, best_mask)

        # Final recomputation using best inliers
        final_R, final_t, final_scale = [], [], []
        for b in range(B):
            inlier_mask = best_mask[b]

            if inlier_mask.sum() < min_inliers:
                R = torch.eye(3, device=src_pts.device).unsqueeze(0)
                t = torch.zeros(3, device=src_pts.device).unsqueeze(0)
            else:
                R, t, _ = self.svd_transform(src_pts[b:b+1, inlier_mask], dst_pts[b:b+1, inlier_mask])

            final_R.append(R)
            final_t.append(t)

        final_R = torch.cat(final_R, dim=0)
        final_t = torch.stack(final_t, dim=0)

        return final_R, final_t.squeeze(1), best_mask
    def forward(
        self, pc: torch.Tensor, coord: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstCoordNet

        Parameters
        ----------
        pc: torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        coord: torch.Tensor
            Ground truth coordinates in the object frame, shape \(B, N, 3\)

        Returns
        -------
        float
            The loss value according to ground truth coordinates
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        B, N, _ = pc.size()
        x = pc.unsqueeze(-1).permute(0,3,1,2) # B 1 N 3
        x = self.conv1(x)
        x = self.conv2(x)

        point_feat = x.contiguous() # B 64 N 1
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x_max = F.max_pool2d(x, (N,1))         # (B,1024,1,1)
        x_max_expanded = x_max.repeat(1, 1, N, 1)  # (B,1024,N,1)
        x_max = x_max.squeeze(-1).squeeze(-1)

        # # ----------- Translation MLP -----------
        # trans_feat = self.fc_trans_1(x_max)
        # trans_feat = self.fc_trans_2(trans_feat)
        # trans_out = self.fc_trans_out(trans_feat)  # Shape: (B, 3)

        # # ----------- Rotation MLP -----------
        # rot_feat = self.fc_rot_1(x_max)
        # rot_feat = self.fc_rot_2(rot_feat)
        # rot_out = self.fc_rot_out(rot_feat)  # Shape: (B, 6)
        # rot_out = self.rotation_6d_to_matrix(rot_out)

        # ---- conv6–conv10 ----
        concat_feat = torch.cat([point_feat, x_max_expanded], dim=1)
        x = self.conv6(concat_feat)  # (B,512,N,1)
        x = self.conv7(x)            # (B,256,N,1)
        x = self.conv8(x)            # (B,128,N,1)
        x = self.conv9(x)            # (B,128,N,1)
        x = self.conv10(x)           # (B,3,N,1)
        x_pred = x.squeeze(3).permute(0,2,1)  # → (B, N, 3)

        def rotation_error_rad(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
            R_rel = torch.matmul(R_pred.transpose(-1, -2), R_gt)
            trace = R_rel.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
            cos_theta = (trace - 1.0) / 2.0
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Ensure valid domain for arccos
            rot_error = torch.acos(cos_theta).mean()
            return rot_error
        
        rot,trans,_ = self.svd_transform(coord,pc)
        rot_out,trans_out,_= self.svd_transform(x_pred,pc)
        loss_trans = F.mse_loss(trans_out,trans)
        loss_rot   = rotation_error_rad(rot_out,rot)
        aux_loss = F.mse_loss(x_pred, coord)  # you can tune beta (default 1.0 in PyTorch <1.10)
        
        metric = dict(
            Point_l2_Loss = aux_loss.item(),
            Transformation_Loss=loss_trans.item(),
            Rotation_Loss = loss_rot.item(),
        )
        return aux_loss,metric

    def est(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

        We don't have a strict limit on the running time, so you can use for loops and numpy instead of batch processing and torch.

        The only requirement is that the input and output should be torch tensors on the same device and with the same dtype.
        """
        B, N, _ = pc.size()
        x = pc.unsqueeze(-1).permute(0,3,1,2) # B 1 N 3
        x = self.conv1(x)
        x = self.conv2(x)
        point_feat = x.contiguous() # B 64 N 1
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x_max = F.max_pool2d(x, (N,1))         # (B,1024,1,1)
        x_max_expanded = x_max.repeat(1, 1, N, 1)  # (B,1024,N,1)
        x_max = x_max.squeeze(-1).squeeze(-1)
        
        concat_feat = torch.cat([point_feat, x_max_expanded], dim=1)
        x = self.conv6(concat_feat)  # (B,512,N,1)
        x = self.conv7(x)            # (B,256,N,1)
        x = self.conv8(x)            # (B,128,N,1)
        x = self.conv9(x)            # (B,128,N,1)
        x = self.conv10(x)           # (B,3,N,1)
        x_pred = x.squeeze(3).permute(0,2,1)  # → (B, N, 3)

        rot,trans,_ = self.ransac_rigid_transform(x_pred,pc)

        return trans,rot
