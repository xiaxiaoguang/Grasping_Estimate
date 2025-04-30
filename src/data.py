import os
from typing import Dict, Optional
import random
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import pickle
import torch
from tqdm import tqdm
import math
from .config import Config
from .constants import DEPTH_IMG_SCALE, TABLE_HEIGHT, PC_MAX, PC_MIN, OBJ_INIT_TRANS
from .utils import get_pc, get_workspace_mask
from .vis import Vis
from .robot.cfg import get_robot_cfg

def preprocess_pose_data(config, mode='train', num_files=100):
    robot_cfg = get_robot_cfg(config.robot)
    data_root = os.path.join("data", config.obj_name, mode)
    files = sorted(os.listdir(data_root))
    
    output_dir = os.path.join("processed", config.obj_name, mode)
    os.makedirs(output_dir, exist_ok=True)

    processed_data = []
    total_saved = 0
    file_index = 0

    max_per_file = math.ceil(len(files) / num_files)

    for fname in tqdm(files):
        fdir = os.path.join(data_root, fname)
        try:
            obj_pose = np.load(os.path.join(fdir, "object_pose.npy"))
            if not np.linalg.norm(obj_pose[:2, 3] - OBJ_INIT_TRANS[:2]) < 0.1:
                continue

            camera_pose = np.load(os.path.join(fdir, "camera_pose.npy"))
            depth_array = (
                np.array(
                    cv2.imread(os.path.join(fdir, "depth.png"), cv2.IMREAD_UNCHANGED)
                ) / DEPTH_IMG_SCALE
            )

            full_pc_camera = get_pc(
                depth_array, robot_cfg.camera_cfg.intrinsics
            ) * np.array([-1, -1, 1])
            full_pc_world = (
                np.einsum("ab,nb->na", camera_pose[:3, :3], full_pc_camera)
                + camera_pose[:3, 3]
            )
            full_coord = np.einsum(
                "ba,nb->na", obj_pose[:3, :3], full_pc_world - obj_pose[:3, 3]
            )

            pc_mask = get_workspace_mask(full_pc_world)
            if np.sum(pc_mask) < config.point_num:
                continue

            processed_data.append({
                "full_pc_camera": full_pc_camera.astype(np.float32),
                "full_coord": full_coord.astype(np.float32),
                "pc_mask": pc_mask,
                "camera_pose": camera_pose.astype(np.float32),
                "obj_pose": obj_pose.astype(np.float32)
            })

            # Save batch if reached max size
            if len(processed_data) >= max_per_file:
                out_path = os.path.join(output_dir, f"{file_index:04d}.pkl")
                with open(out_path, "wb") as f:
                    pickle.dump(processed_data, f)
                print(f"Saved batch {file_index} with {len(processed_data)} samples to {out_path}")
                total_saved += len(processed_data)
                processed_data = []
                file_index += 1

        except Exception as e:
            print(f"Skipping {fdir}: {e}")

    # Save any remaining samples
    if processed_data:
        out_path = os.path.join(output_dir, f"{file_index:04d}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(processed_data, f)
        print(f"Saved final batch {file_index} with {len(processed_data)} samples to {out_path}")
        total_saved += len(processed_data)

    print(f"Total saved samples: {total_saved} in {file_index + 1} files.")

class PreprocessedPoseDataset(Dataset):
    def __init__(self, folder_path, config, scale=1):
        super().__init__()
        self.folder_path = folder_path
        self.config = config
        self.scale = scale

        # Index map: list of (file_idx, local_idx_in_file)
        self.index_map = []
        self.file_cache = {}  # Optional tiny cache
        self.file_list = []

        # First, index all files and count their entries without loading full data
        for fname in sorted(os.listdir(folder_path)):
            if fname.endswith(".pkl"):
                fpath = os.path.join(folder_path, fname)
                with open(fpath, "rb") as f:
                    file_data = pickle.load(f)
                    num_samples = len(file_data)
                file_idx = len(self.file_list)
                self.file_list.append(fpath)
                for i in range(num_samples):
                    self.index_map.append((file_idx, i))

        self.num_original = len(self.index_map)
        self.total_len = self.num_original * self.scale

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        base_idx = idx % self.num_original
        file_idx, local_idx = self.index_map[base_idx]
        file_path = self.file_list[file_idx]

        # Lazy load the file, no persistent memory
        with open(file_path, "rb") as f:
            records = pickle.load(f)
        record = records[local_idx]

        # Release reference ASAP
        del records

        mask = record["pc_mask"]
        valid_indices = np.flatnonzero(mask)

        if len(valid_indices) < self.config.point_num:
            raise ValueError("Not enough valid points for selection")

        sel_idx = np.random.choice(valid_indices, size=self.config.point_num, replace=False)

        pc_camera = record["full_pc_camera"][sel_idx]
        coord = record["full_coord"][sel_idx]
        rel_obj_pose = np.linalg.inv(record["camera_pose"]) @ record["obj_pose"]

        return dict(
            pc=pc_camera.astype(np.float32),
            coord=coord.astype(np.float32),
            trans=rel_obj_pose[:3, 3].astype(np.float32),
            rot=rel_obj_pose[:3, :3].astype(np.float32),
            camera_pose=record["camera_pose"],
            obj_pose_in_world=record["obj_pose"],
        )
# class PoseDataset(Dataset):
#     def __init__(self, config: Config, mode: str, scale: int = 1):
#         super().__init__()
#         self.config = config
#         self.robot_cfg = get_robot_cfg(config.robot)
#         self.data_root = os.path.join("data", config.obj_name, mode)

#         # Preload data
#         raw_files = sorted(os.listdir(self.data_root))
#         raw_files = raw_files * scale
#         random.shuffle(raw_files)

#         self.data = []
#         for i,fname in enumerate(raw_files):
#             print(i)
#             fdir = os.path.join(self.data_root, fname)
#             try:
#                 obj_pose = np.load(os.path.join(fdir, "object_pose.npy"))
#                 if not np.linalg.norm(obj_pose[:2, 3] - OBJ_INIT_TRANS[:2]) < 0.1:
#                     continue  # skip sample out of workspace

#                 camera_pose = np.load(os.path.join(fdir, "camera_pose.npy"))
#                 depth_array = (
#                     np.array(
#                         cv2.imread(os.path.join(fdir, "depth.png"), cv2.IMREAD_UNCHANGED)
#                     )
#                     / DEPTH_IMG_SCALE
#                 )

#                 full_pc_camera = get_pc(
#                     depth_array, self.robot_cfg.camera_cfg.intrinsics
#                 ) * np.array([-1, -1, 1])
#                 full_pc_world = (
#                     np.einsum("ab,nb->na", camera_pose[:3, :3], full_pc_camera)
#                     + camera_pose[:3, 3]
#                 )
#                 full_coord = np.einsum(
#                     "ba,nb->na", obj_pose[:3, :3], full_pc_world - obj_pose[:3, 3]
#                 )

#                 pc_mask = get_workspace_mask(full_pc_world)
#                 if np.sum(pc_mask) < self.config.point_num:
#                     continue  # skip insufficient points

#                 self.data.append(dict(
#                     full_pc_camera=full_pc_camera,
#                     full_coord=full_coord,
#                     pc_mask=pc_mask,
#                     camera_pose=camera_pose,
#                     obj_pose=obj_pose,
#                 ))

#             except Exception as e:
#                 print(f"Skipping {fdir} due to error: {e}")
#                 continue

#         print(f"Loaded {len(self.data)} valid samples into memory.")

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         record = self.data[idx]

#         # Sample `point_num` indices from valid mask
#         sel_pc_idx = np.random.randint(0, np.sum(record["pc_mask"]), self.config.point_num)
#         pc_camera = record["full_pc_camera"][record["pc_mask"]][sel_pc_idx]
#         coord = record["full_coord"][record["pc_mask"]][sel_pc_idx]

#         rel_obj_pose = np.linalg.inv(record["camera_pose"]) @ record["obj_pose"]

#         return dict(
#             pc=pc_camera.astype(np.float32),
#             coord=coord.astype(np.float32),
#             trans=rel_obj_pose[:3, 3].astype(np.float32),
#             rot=rel_obj_pose[:3, :3].astype(np.float32),
#             camera_pose=record["camera_pose"].astype(np.float32),
#             obj_pose_in_world=record["obj_pose"].astype(np.float32),
#         )

class PoseDataset(Dataset):
    def __init__(self, config: Config, mode: str, scale: int = 1):
        """
        Dataset for pose estimation

        Parameters
        ----------
        config: Config
            Configuration object
        mode: str
            Mode of the dataset (e.g. train or val)
        scale: int
            Scale of the dataset, used to make the dataset larger
            so that we don't need to wait for the restart of the dataloader
        """
        super().__init__()
        self.config = config
        self.robot_cfg = get_robot_cfg(config.robot)
        self.data_root = os.path.join("data", config.obj_name, mode)
        self.files = sorted(os.listdir(self.data_root))
        self.files = self.files * scale
        random.shuffle(self.files)

    def __len__(self) -> int:
        """
        For a torch dataset, a __len__ is required.

        Returns
        -------
        int
            Length of the dataset
        """
        return len(self.files)

    def __getitem__(self, idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        For a torch dataset, a __getitem__ is required.

        Parameters
        ----------
        idx: Optional[int]
            Index of the item to get. If None, a random index is used.

        Returns
        -------
        A dict of:

            pc: the point cloud in camera frame with shape (N, 3)

            trans: the ground truth translation vector with shape (3,)

            rot: the ground truth rotation matrix with shape (3, 3)

            coord: the ground truth coordinates in the object frame with shape (N, 3)

            camera_pose: the camera pose with shape (4, 4) (Used in simulation evaluation)

            obj_pose_in_world: the object pose in world frame with shape (4, 4) (Used in simulation evaluation)

        Note that they will be converted to torch tensors in the dataloader.

        The shape will be (B, ...) for batch size B when you get the data from the dataloader.
        """
        try:

            f = self.files[idx] if idx is not None else random.choice(self.files)
            fdir = os.path.join(self.data_root, f)

            obj_pose = np.load(os.path.join(fdir, "object_pose.npy"))
            if not np.linalg.norm(obj_pose[:2, 3] - OBJ_INIT_TRANS[:2]) < 0.1:
                # some times the object will be out of the workspace
                # so we need to skip this sample
                # this rarely happens so we don't need to worry about it
                return self.__getitem__()
            camera_pose = np.load(os.path.join(fdir, "camera_pose.npy"))
            depth_array = (
                np.array(
                    cv2.imread(os.path.join(fdir, "depth.png"), cv2.IMREAD_UNCHANGED)
                )
                / DEPTH_IMG_SCALE
            )

            full_pc_camera = get_pc(
                depth_array, self.robot_cfg.camera_cfg.intrinsics
            ) * np.array([-1, -1, 1])
            full_pc_world = (
                np.einsum("ab,nb->na", camera_pose[:3, :3], full_pc_camera)
                + camera_pose[:3, 3]
            )
            full_coord = np.einsum(
                "ba,nb->na", obj_pose[:3, :3], full_pc_world - obj_pose[:3, 3]
            )

            pc_mask = get_workspace_mask(full_pc_world)
            sel_pc_idx = np.random.randint(0, np.sum(pc_mask), self.config.point_num)

            pc_camera = full_pc_camera[pc_mask][sel_pc_idx]
            coord = full_coord[pc_mask][sel_pc_idx]
            rel_obj_pose = np.linalg.inv(camera_pose) @ obj_pose

            return dict(
                pc=pc_camera.astype(np.float32),
                coord=coord.astype(np.float32),
                trans=rel_obj_pose[:3, 3].astype(np.float32),
                rot=rel_obj_pose[:3, :3].astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                obj_pose_in_world=obj_pose.astype(np.float32),
            )

        except Exception as e:
            print(f"Error in {fdir}: {e}")
            return self.__getitem__()


class Loader:
    # a simple wrapper for DataLoader which can get data infinitely
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iter = iter(self.loader)

    def get(self) -> dict:
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            data = next(self.iter)
        return data
