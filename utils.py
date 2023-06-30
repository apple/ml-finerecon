import scipy.spatial
import skimage.measure
import trimesh
import torch
import numpy as np


def log_transform(tsdf):
    result = torch.log(tsdf.abs() + 1)
    result *= torch.sign(tsdf)
    return result


def tsdf2mesh(tsdf, voxel_size, origin, level=0):
    verts, faces, _, _ = skimage.measure.marching_cubes(tsdf, level=level)
    faces = faces[~np.any(np.isnan(verts[faces]), axis=(1, 2))]
    verts = verts * voxel_size + origin
    return trimesh.Trimesh(verts, faces)


def project(xyz, poses, K, imsize):
    """
    xyz: b x (*spatial_dims) x 3
    poses: b x nviews x 4 x 4
    K: (b x nviews x 3 x 3)
    imsize: (imheight, imwidth)
    """

    device = xyz.device
    batch_size = xyz.shape[0]
    spatial_dims = xyz.shape[1:-1]
    n_views = poses.shape[1]

    xyz = xyz.view(batch_size, 1, -1, 3).transpose(3, 2)
    xyz = torch.cat((xyz, torch.ones_like(xyz[:, :, :1])), dim=2)

    with torch.autocast(enabled=False, device_type=device.type):
        xyz_cam = (torch.inverse(poses) @ xyz)[:, :, :3]
        uv = K @ xyz_cam

    z = uv[:, :, 2]
    uv = uv[:, :, :2] / uv[:, :, 2:]
    imheight, imwidth = imsize
    """
    assuming that these uv coordinates have
        (0, 0) = center of top left pixel
        (w - 1, h - 1) = center of bottom right pixel
    then we allow values between (-.5, w-.5) because they are inside the border pixel
    """
    valid = (
        (uv[:, :, 0] >= -0.5)
        & (uv[:, :, 1] >= -0.5)
        & (uv[:, :, 0] <= imwidth - 0.5)
        & (uv[:, :, 1] <= imheight - 0.5)
        & (z > 0)
    )
    uv = uv.transpose(2, 3)

    uv = uv.view(batch_size, n_views, *spatial_dims, 2)
    z = z.view(batch_size, n_views, *spatial_dims)
    valid = valid.view(batch_size, n_views, *spatial_dims)
    return uv, z, valid


def sample_posed_images(
    imgs, poses, K, xyz, mode="bilinear", padding_mode="zeros", return_z=False
):
    """
    imgs: b x nviews x C x H x W
    poses: b x nviews x 4 x 4
    K: (b x nviews x 3 x 3)
    xyz: b x (*spatial_dims) x 3
    """

    device = imgs.device
    batch_size, n_views, _, imheight, imwidth = imgs.shape
    spatial_dims = xyz.shape[1:-1]

    """
    assuming that these uv coordinates have
        (0, 0) = center of top left pixel
        (w - 1, h - 1) = center of bottom right pixel

    adjust because grid_sample(align_corners=False) assumes
        (0, 0) = top left corner of top left pixel
        (w, h) = bottom right corner of bottom right pixel
    """
    uv, z, valid = project(xyz, poses, K, (imheight, imwidth))
    imsize = torch.tensor([imwidth, imheight], device=device)
    # grid = (uv + 0.5) / imsize * 2 - 1
    grid = uv / (0.5 * imsize) + (1 / imsize - 1)
    vals = torch.nn.functional.grid_sample(
        imgs.view(batch_size * n_views, *imgs.shape[2:]),
        grid.view(batch_size * n_views, 1, -1, 2),
        align_corners=False,
        mode=mode,
        padding_mode=padding_mode,
    )
    vals = vals.view(batch_size, n_views, -1, *spatial_dims)
    if return_z:
        return vals, valid, z
    else:
        return vals, valid


def sample_voxel_feats(img_feats, poses, K, xyz, imsize, invalid_fill_value=0):
    base_imheight, base_imwidth = imsize
    featheight = img_feats.shape[3]
    featwidth = img_feats.shape[4]
    _K = K.clone()
    _K[:, :, 0] *= featwidth / base_imwidth
    _K[:, :, 1] *= featheight / base_imheight

    voxel_feats, valid = sample_posed_images(
        img_feats,
        poses,
        _K,
        xyz,
        mode="bilinear",
        padding_mode="border",
    )
    voxel_feats.masked_fill_(~valid[:, :, None], invalid_fill_value)

    return voxel_feats, valid
