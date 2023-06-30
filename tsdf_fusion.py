# Copyright (c) 2018 Andy Zeng
# This file is originally from (https://github.com/andyzeng/tsdf-fusion-python)
# and was modified by Noah Stier in 2023.
# The corresponding license is reproduced in ACKNOWLEDGEMENTS


import torch


def integrate(
    depth_im,
    cam_intr,
    cam_pose,
    obs_weight,
    world_c,
    vox_coords,
    weight_vol,
    tsdf_vol,
    sdf_trunc,
    im_h,
    im_w,
    deintegrate=False,
):
    # compute mask of voxels within the bounding box of the view frustum
    max_depth = torch.max(depth_im)
    min_depth = torch.min(depth_im)
    corner_uv = torch.tensor(
        [
            [0, 0, 1],
            [im_w, 0, 1],
            [0, im_h, 1],
            [im_w, im_h, 1],
        ],
        dtype=torch.float32,
        device=depth_im.device,
    )
    corner_vectors = corner_uv @ cam_intr.inverse().T
    corner_xyz_cam = torch.cat(
        (
            corner_vectors * max_depth,
            corner_vectors * min_depth,
        ),
        dim=0,
    )
    corner_xyz = (
        torch.cat(
            (
                corner_xyz_cam,
                torch.ones(
                    (len(corner_xyz_cam), 1),
                    dtype=corner_xyz_cam.dtype,
                    device=depth_im.device,
                ),
            ),
            dim=-1,
        )
        @ cam_pose.T
    )
    minbounds = torch.min(corner_xyz, dim=0)[0][:3]
    maxbounds = torch.max(corner_xyz, dim=0)[0][:3]
    inbounds = torch.all(world_c[:, :3] >= minbounds, dim=-1) & torch.all(
        world_c[:, :3] <= maxbounds, dim=-1
    )

    # Convert world coordinates to camera coordinates
    world2cam = torch.inverse(cam_pose)
    cam_c = (
        torch.matmul(world2cam, world_c[inbounds].transpose(1, 0))
        .transpose(1, 0)
        .float()
    )

    # Convert camera coordinates to pixel coordinates
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]
    pix_z = cam_c[:, 2]
    pix_x = torch.round((cam_c[:, 0] * fx / cam_c[:, 2]) + cx).long()
    pix_y = torch.round((cam_c[:, 1] * fy / cam_c[:, 2]) + cy).long()

    # Eliminate pixels outside view frustum
    valid_pix = (
        (pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (pix_z > 0)
    )

    vc_inbounds = vox_coords[inbounds]
    valid_vox = vc_inbounds[valid_pix]
    depth_val = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

    # Integrate tsdf
    depth_diff = depth_val - pix_z[valid_pix]
    dist = torch.clamp(depth_diff / sdf_trunc, max=1)
    valid_pts = (depth_val > 0) & (depth_diff >= -sdf_trunc)
    valid_vox_x, valid_vox_y, valid_vox_z = torch.unbind(valid_vox[valid_pts], dim=-1)
    valid_dist = dist[valid_pts]
    w_old = weight_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    tsdf_vals = tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z]

    if deintegrate:
        w_new = w_old - obs_weight
        tsdf_new = (w_old * tsdf_vals - obs_weight * valid_dist) / w_new
        tsdf_new[w_new == 0] = 1
        tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_new
    else:
        w_new = w_old + obs_weight
        tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] = (
            w_old * tsdf_vals + obs_weight * valid_dist
        ) / w_new
    weight_vol[valid_vox_x, valid_vox_y, valid_vox_z] = w_new

    return weight_vol, tsdf_vol


class TSDFVolumeTorch:
    """Volumetric TSDF Fusion of RGB-D Images."""

    def __init__(self, voxel_dim, origin, voxel_size, margin=3, device="cuda"):
        """Constructor.

        Args:
          vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
            xyz bounds (min/max) in meters.
          voxel_size (float): The volume discretization in meters.
        """
        self.device = device

        # Define voxel volume parameters
        self._voxel_size = float(voxel_size)
        self._sdf_trunc = margin * self._voxel_size
        self._const = 256 * 256
        self._integrate_func = integrate

        # Adjust volume bounds
        self._vol_dim = voxel_dim.long()
        self._vol_origin = origin
        self._num_voxels = torch.prod(self._vol_dim).item()

        # Get voxel grid coordinates
        xv, yv, zv = torch.meshgrid(
            torch.arange(0, self._vol_dim[0]),
            torch.arange(0, self._vol_dim[1]),
            torch.arange(0, self._vol_dim[2]),
        )
        self._vox_coords = (
            torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], dim=1)
            .long()
            .to(self.device)
        )

        # Convert voxel coordinates to world coordinates
        self._world_c = self._vol_origin + (self._voxel_size * self._vox_coords)
        self._world_c = torch.cat(
            [self._world_c, torch.ones(len(self._world_c), 1, device=self.device)],
            dim=1,
        )

        self.reset()

        # print("[*] voxel volume: {} x {} x {}".format(*self._vol_dim))
        # print("[*] num voxels: {:,}".format(self._num_voxels))

    def reset(self):
        self._tsdf_vol = torch.ones(*self._vol_dim).to(self.device)
        self._weight_vol = torch.zeros(*self._vol_dim).to(self.device)
        self._color_vol = torch.zeros(*self._vol_dim).to(self.device)

    def integrate(self, depth_im, cam_intr, cam_pose, obs_weight, deintegrate=False):
        """Integrate an RGB-D frame into the TSDF volume.
        Args:
          color_im (ndarray): An RGB image of shape (H, W, 3).
          depth_im (ndarray): A depth image of shape (H, W).
          cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
          cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
          obs_weight (float): The weight to assign to the current observation.
        """
        cam_pose = cam_pose.float().to(self.device)
        cam_intr = cam_intr.float().to(self.device)
        depth_im = depth_im.float().to(self.device)
        im_h, im_w = depth_im.shape
        weight_vol, tsdf_vol = self._integrate_func(
            depth_im,
            cam_intr,
            cam_pose,
            obs_weight,
            self._world_c,
            self._vox_coords,
            self._weight_vol,
            self._tsdf_vol,
            self._sdf_trunc,
            im_h,
            im_w,
            deintegrate=deintegrate,
        )
        self._weight_vol = weight_vol
        self._tsdf_vol = tsdf_vol

    def get_volume(self):
        return self._tsdf_vol, self._weight_vol

    @property
    def sdf_trunc(self):
        return self._sdf_trunc

    @property
    def voxel_size(self):
        return self._voxel_size
