"""SO101-specific image layout built on the repository's LeRobot dataset."""

from __future__ import annotations

import torch
import torchvision.transforms.functional as transforms_f

from ahawam.datasets.lerobot.robot_video_dataset import RobotVideoDataset


class SO101RobotVideoDataset(RobotVideoDataset):
    """Compose fixed and wrist cameras into one 384x320 model frame.

    The fixed camera occupies the top 256 pixels. The bottom row preserves the
    RoboCOIN two-wrist-camera slots: SO101 wrist on the left and black padding
    for the missing right wrist camera. This avoids stretching one wrist view
    across both slots and matches the layout expected by online deployment.
    """

    def _process_sampled_video_tensor(self, video: torch.Tensor) -> torch.Tensor:
        if self.concat_multi_camera != "so101":
            return super()._process_sampled_video_tensor(video)
        if video.ndim != 5 or int(video.shape[0]) != 2:
            raise ValueError(
                "`concat_multi_camera='so101'` requires [2,T,C,H,W] input in "
                "[fixed,wrist] order, got " + str(tuple(video.shape))
            )

        fixed = transforms_f.resize(
            video[0],
            size=[256, 320],
            interpolation=transforms_f.InterpolationMode.BILINEAR,
            antialias=True,
        )
        wrist = transforms_f.resize(
            video[1],
            size=[128, 160],
            interpolation=transforms_f.InterpolationMode.BILINEAR,
            antialias=True,
        )
        missing_right_wrist = torch.zeros_like(wrist)
        wrist_row = torch.cat([wrist, missing_right_wrist], dim=-1)
        composed = torch.cat([fixed, wrist_row], dim=-2)
        composed = self.resize_transform(composed)
        composed = self.crop_transform(composed)
        composed = self.normalize_transform(composed)
        return composed.permute(1, 0, 2, 3)
