import itertools
import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
import tqdm

import data
import modules
import utils


class FineRecon(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        img_feature_dim = 47
        self.cnn2d = modules.Cnn2d(out_dim=img_feature_dim)
        self.fusion = modules.FeatureFusion(in_c=img_feature_dim)
        self.voxel_feat_dim = self.fusion.out_c

        # just a shorthand
        self.dg = self.config.depth_guidance

        if self.dg.enabled:
            if self.dg.density_fusion_channel:
                self.voxel_feat_dim += 1
            elif self.dg.tsdf_fusion_channel:
                self.voxel_feat_dim += 1

        self.cnn3d = modules.Cnn3d(in_c=self.voxel_feat_dim)

        if self.config.point_backprojection:
            self.cnn2d_pb_out_dim = img_feature_dim
            self.cnn2d_pb = modules.Cnn2d(out_dim=self.cnn2d_pb_out_dim)
            self.point_feat_mlp = torch.nn.Sequential(
                modules.ResBlock1d(self.cnn2d_pb_out_dim),
                modules.ResBlock1d(self.cnn2d_pb_out_dim),
            )
            self.point_fusion = modules.FeatureFusion(in_c=self.cnn2d_pb_out_dim)

        surface_pred_input_dim = occ_pred_input_dim = self.cnn3d.out_c
        if self.config.point_backprojection:
            surface_pred_input_dim += self.cnn2d_pb_out_dim

        if self.dg.enabled:
            if self.config.point_backprojection:
                if self.dg.density_fusion_channel:
                    surface_pred_input_dim += 1
                elif self.dg.tsdf_fusion_channel:
                    surface_pred_input_dim += 1

        self.surface_predictor = torch.nn.Sequential(
            torch.nn.Conv1d(surface_pred_input_dim, 32, 1),
            modules.ResBlock1d(32),
            modules.ResBlock1d(32),
            torch.nn.Conv1d(32, 1, 1),
        )
        self.occ_predictor = torch.nn.Sequential(
            torch.nn.Conv1d(occ_pred_input_dim, 32, 1),
            modules.ResBlock1d(32),
            modules.ResBlock1d(32),
            torch.nn.Conv1d(32, 1, 1),
        )

        if self.config.do_prediction_timing:
            self.init_time = 0
            self.per_view_time = 0
            self.final_step_time = 0
            self.n_final_steps = 0
            self.n_views = 0
            self.n_inits = 0

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=0.001)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda epoch: 1
            if self.global_step < (self.config.steps - self.config.finetune_steps)
            else 0.1,
            verbose=True,
        )
        return [opt], [sched]

    def density_fusion(self, pred_depth_imgs, poses, K_pred_depth, input_coords):
        depth, valid, z = utils.sample_posed_images(
            pred_depth_imgs[:, :, None],
            poses,
            K_pred_depth,
            input_coords,
            mode="nearest",
            return_z=True,
        )
        depth = depth.squeeze(2)
        valid.masked_fill_(depth == 0, False)

        dist = (z - depth).abs()
        in_voxel = valid & (dist < np.sqrt(3) * self.config.voxel_size / 2)

        weight = valid.sum(dim=1)
        density = in_voxel.sum(dim=1) / (weight + (weight == 0).to(weight.dtype))

        return density, weight

    def tsdf_fusion(self, pred_depth_imgs, poses, K_pred_depth, input_coords):
        depth, valid, z = utils.sample_posed_images(
            pred_depth_imgs[:, :, None],
            poses,
            K_pred_depth,
            input_coords,
            mode="nearest",
            return_z=True,
        )
        depth = depth.squeeze(2)
        valid.masked_fill_(depth == 0, False)
        margin = 3 * self.config.voxel_size
        tsdf = torch.clamp(z - depth, -margin, margin) / margin
        valid &= tsdf < 0.999
        tsdf.masked_fill_(~valid, 0)
        tsdf = torch.sum(tsdf, dim=1)
        weight = torch.sum(valid, dim=1)
        tsdf /= weight
        return tsdf, weight

    def get_img_voxel_feats_by_depth_guided_bp(
        self,
        rgb_imgs,
        pred_depth_imgs,
        poses,
        K_color,
        K_pred_depth,
        input_coords,
        use_highres_cnn=False,
        img_feats=None,
    ):
        img_voxel_feats, img_voxel_valid = self.get_img_voxel_feats_by_img_bp(
            rgb_imgs,
            poses,
            K_color,
            input_coords,
            use_highres_cnn=use_highres_cnn,
            img_feats=img_feats,
        )

        depth, depth_valid, z = utils.sample_posed_images(
            pred_depth_imgs[:, :, None],
            poses,
            K_pred_depth,
            input_coords,
            mode="nearest",
            return_z=True,
        )
        depth = depth.squeeze(2)

        depth_valid.masked_fill_(depth == 0, False)

        if "gaussian" in self.dg.bp_weighting:
            dist = (z - depth).abs()
            if self.dg.bp_weighting == "gaussian_12cm":
                weight = torch.exp(-((dist * 16) ** 2))
            elif self.dg.bp_weighting == "gaussian_24cm":
                weight = torch.exp(-((dist * 8) ** 2))
            else:
                raise NotImplementedError
            weight.masked_fill_(~depth_valid, 0)
            img_voxel_feats *= weight[:, :, None]

        elif "truncation" in self.dg.bp_weighting:
            dist = (z - depth).abs()
            if self.dg.bp_weighting == "truncation_3.5cm":
                weight = (dist < 0.035).float()
            elif self.dg.bp_weighting == "truncation_12cm":
                weight = (dist < 0.12).float()
            elif self.dg.bp_weighting == "truncation_24cm":
                weight = (dist < 0.24).float()
            elif self.dg.bp_weighting == "truncation_48cm":
                weight = (dist < 0.48).float()
            else:
                raise NotImplementedError

            weight.masked_fill_(~depth_valid, 0)
            img_voxel_feats *= weight[:, :, None]

        elif self.dg.bp_weighting == "none":
            ...
        else:
            raise NotImplementedError

        img_voxel_feats.masked_fill_(~img_voxel_valid[:, :, None], 0)

        return img_voxel_feats, img_voxel_valid

    def get_img_voxel_feats_by_img_bp(
        self,
        rgb_imgs,
        poses,
        K_color,
        input_coords,
        use_highres_cnn=False,
        img_feats=None,
    ):
        batch_size, n_imgs, _, imheight, imwidth = rgb_imgs.shape
        imsize = (imheight, imwidth)

        if img_feats is None:
            if use_highres_cnn:
                img_feats = self.cnn2d_pb(
                    rgb_imgs.view(batch_size * n_imgs, 3, imheight, imwidth)
                )
            else:
                img_feats = self.cnn2d(
                    rgb_imgs.view(batch_size * n_imgs, 3, imheight, imwidth)
                )

        img_feats = img_feats.view(batch_size, n_imgs, *img_feats.shape[1:])

        img_voxel_feats, img_voxel_valid = utils.sample_voxel_feats(
            img_feats, poses, K_color, input_coords, imsize
        )

        if (not self.training) and use_highres_cnn:
            # down-weight the high-res BP image features near the image border
            # to reduce boundary artifacts.
            # works at inference time, not tested with training

            xyz = input_coords
            batch_size = xyz.shape[0]
            xyz = xyz.view(batch_size, 1, -1, 3).transpose(3, 2)
            xyz = torch.cat((xyz, torch.ones_like(xyz[:, :, :1])), dim=2)

            featheight, featwidth = img_feats.shape[-2:]

            K = K_color.clone()
            K[:, :, 0] *= featwidth / imwidth
            K[:, :, 1] *= featheight / imheight
            with torch.autocast(enabled=False, device_type=self.device.type):
                xyz_cam = (torch.inverse(poses) @ xyz)[:, :, :3]
                uv = K @ xyz_cam
            uv = uv[:, :, :2] / uv[:, :, 2:]

            featsize = torch.tensor(
                [featwidth, featheight], device=self.device, dtype=uv.dtype
            )[None, None, :, None]
            uv[:, :, 0].clamp_(0, imwidth)
            uv[:, :, 1].clamp_(0, imheight)
            border_dist = ((uv / featsize).round() * featsize - uv).abs().min(dim=2)[0]
            pixel_margin = 20
            weight = (border_dist / pixel_margin).clamp(0, 1)
            weight = torch.sigmoid(weight * 12 - 6)
            img_voxel_feats *= weight[:, :, None]

        return img_voxel_feats, img_voxel_valid

    def sample_point_features_by_linear_interp(
        self, coords, voxel_feats, voxel_valid, grid_origin
    ):
        """
        coords: BN3
        voxel_feats: BFXYZ
        voxel_valid: BXYZ
        grid_origin: B3
        """
        crop_size_m = (
            torch.tensor(voxel_feats.shape[2:], device=self.device)
            * self.config.voxel_size
        )
        grid = (
            coords - grid_origin[:, None] + self.config.voxel_size / 2
        ) / crop_size_m * 2 - 1
        point_valid = (
            torch.nn.functional.grid_sample(
                voxel_valid[:, None].float(),
                grid[:, None, None, :, [2, 1, 0]],
                align_corners=False,
                mode="nearest",
                padding_mode="zeros",
            )[:, 0, 0, 0]
            > 0.5
        )

        point_feats = torch.nn.functional.grid_sample(
            voxel_feats,
            grid[:, None, None, :, [2, 1, 0]],
            align_corners=False,
            mode="bilinear",
            padding_mode="zeros",
        )[:, :, 0, 0]
        return point_feats, point_valid

    def augment_depth_inplace(self, batch):
        n_views = batch["pred_depth_imgs"].shape[1]
        n_augment = n_views // 2

        for i in range(len(batch["pred_depth_imgs"])):
            j = np.random.choice(
                batch["pred_depth_imgs"].shape[1], size=n_augment, replace=False
            )
            scale = torch.rand(len(j), device=self.device) * 0.2 + 0.9
            batch["pred_depth_imgs"][i, j] *= scale[:, None, None]

    def compute_loss(
        self,
        tsdf_logits,
        occ_logits,
        gt_tsdf,
        gt_occ,
        coarse_point_valid,
        fine_point_valid,
    ):
        occ_loss_mask = (~gt_occ.isnan()) & coarse_point_valid
        tsdf_loss_mask = (gt_occ > 0.5) & (~gt_tsdf.isnan()) & fine_point_valid

        occ_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            occ_logits[occ_loss_mask], gt_occ[occ_loss_mask]
        )

        loss = occ_loss
        if tsdf_loss_mask.sum() > 0:
            tsdf_loss = torch.nn.functional.l1_loss(
                utils.log_transform(torch.tanh(tsdf_logits[tsdf_loss_mask])),
                utils.log_transform(gt_tsdf[tsdf_loss_mask]),
            )
            loss += tsdf_loss
        else:
            tsdf_loss = torch.tensor(torch.nan)

        return loss, tsdf_loss, occ_loss

    def step(self, batch):
        if self.dg.enabled:
            if self.training and self.dg.depth_scale_augmentation:
                self.augment_depth_inplace(batch)

            voxel_feats, voxel_valid = self.get_img_voxel_feats_by_depth_guided_bp(
                batch["rgb_imgs"],
                batch["pred_depth_imgs"],
                batch["poses"],
                batch["K_color"][:, None],
                batch["K_pred_depth"][:, None],
                batch["input_coords"],
            )
            voxel_feats = self.fusion(voxel_feats, voxel_valid)
            voxel_valid = voxel_valid.sum(dim=1) > 1
            if self.config.no_image_features:
                voxel_feats = voxel_feats * 0

            if self.dg.density_fusion_channel:
                density, weight = self.density_fusion(
                    batch["pred_depth_imgs"],
                    batch["poses"],
                    batch["K_pred_depth"][:, None],
                    batch["input_coords"],
                )
                voxel_feats = torch.cat((voxel_feats, density[:, None]), dim=1)
            elif self.dg.tsdf_fusion_channel:
                tsdf, weight = self.tsdf_fusion(
                    batch["pred_depth_imgs"],
                    batch["poses"],
                    batch["K_pred_depth"][:, None],
                    batch["input_coords"],
                )
                tsdf.masked_fill_(weight == 0, 1)
                voxel_feats = torch.cat((voxel_feats, tsdf[:, None]), dim=1)
        else:
            voxel_feats, voxel_valid = self.get_img_voxel_feats_by_img_bp(
                batch["rgb_imgs"],
                batch["poses"],
                batch["K_color"][:, None],
                batch["input_coords"],
            )
            voxel_feats = self.fusion(voxel_feats, voxel_valid)
            voxel_valid = voxel_valid.sum(dim=1) > 1

        voxel_feats = self.cnn3d(voxel_feats, voxel_valid)

        if self.config.improved_tsdf_sampling:
            """
            interpolate the features to the points where we have GT tsdf
            """

            t = batch["crop_center"]
            R = batch["crop_rotation"]
            coords = batch["output_coords"]

            with torch.autocast(enabled=False, device_type=self.device.type):
                coords_local = (coords - t[:, None]) @ R
            coords_local += batch["crop_size_m"][:, None] / 2
            origin = torch.zeros_like(batch["gt_origin"])

            (
                coarse_point_feats,
                coarse_point_valid,
            ) = self.sample_point_features_by_linear_interp(
                coords_local, voxel_feats, voxel_valid, origin
            )
        else:
            """
            keep the voxel-center features that we have:
            GT tsdf has already been interpolated to these points
            """

            coarse_point_feats = voxel_feats.view(*voxel_feats.shape[:2], -1)
            coarse_point_valid = voxel_valid.view(voxel_valid.shape[0], -1)

        if self.config.point_backprojection:
            coords = batch["output_coords"]

            if self.dg.enabled:
                (
                    fine_point_feats,
                    fine_point_valid,
                ) = self.get_img_voxel_feats_by_depth_guided_bp(
                    batch["rgb_imgs"],
                    batch["pred_depth_imgs"],
                    batch["poses"],
                    batch["K_color"][:, None],
                    batch["K_pred_depth"][:, None],
                    coords,
                    use_highres_cnn=True,
                )
            else:
                (
                    fine_point_feats,
                    fine_point_valid,
                ) = self.get_img_voxel_feats_by_img_bp(
                    batch["rgb_imgs"],
                    batch["poses"],
                    batch["K_color"][:, None],
                    coords,
                    use_highres_cnn=True,
                )

            fine_point_feats = self.point_fusion(
                fine_point_feats[..., None, None], fine_point_valid[..., None, None]
            )[..., 0, 0]
            fine_point_valid = coarse_point_valid & (fine_point_valid.any(dim=1))
            fine_point_feats = self.point_feat_mlp(fine_point_feats)

            if self.config.no_image_features:
                fine_point_feats = fine_point_feats * 0

            if self.dg.enabled:
                if self.dg.density_fusion_channel:
                    density, weight = self.density_fusion(
                        batch["pred_depth_imgs"],
                        batch["poses"],
                        batch["K_pred_depth"][:, None],
                        coords,
                    )
                    fine_point_feats = torch.cat(
                        (fine_point_feats, coarse_point_feats, density[:, None]), dim=1
                    )
                elif self.dg.tsdf_fusion_channel:
                    tsdf, weight = self.tsdf_fusion(
                        batch["pred_depth_imgs"],
                        batch["poses"],
                        batch["K_pred_depth"][:, None],
                        coords,
                    )
                    tsdf.masked_fill_(weight == 0, 1)

                    fine_point_feats = torch.cat(
                        (fine_point_feats, coarse_point_feats, tsdf[:, None]), dim=1
                    )
                else:
                    fine_point_feats = torch.cat(
                        (fine_point_feats, coarse_point_feats), dim=1
                    )

            else:
                fine_point_feats = torch.cat(
                    (fine_point_feats, coarse_point_feats), dim=1
                )

        else:
            fine_point_feats = coarse_point_feats
            fine_point_valid = coarse_point_valid

        tsdf_logits = self.surface_predictor(fine_point_feats).squeeze(1)
        occ_logits = self.occ_predictor(coarse_point_feats).squeeze(1)

        loss, tsdf_loss, occ_loss = self.compute_loss(
            tsdf_logits,
            occ_logits,
            batch["gt_tsdf"],
            batch["gt_occ"],
            coarse_point_valid,
            fine_point_valid,
        )

        outputs = {
            "loss": loss,
            "tsdf_loss": tsdf_loss,
            "occ_loss": occ_loss,
            "tsdf_logits": tsdf_logits,
            "occ_logits": occ_logits,
        }
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.step(batch)

        logs = {}
        for k in ["loss", "tsdf_loss", "occ_loss"]:
            logs[f"loss_train/{k}"] = outputs[k].item()

        logs["lr"] = self.optimizers().param_groups[0]["lr"]

        self.log_dict(logs, rank_zero_only=True)
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self.step(batch)

        batch_size = batch["input_coords"].shape[0]
        assert batch_size == 1, "validation step assumes val batch size == 1"

        logs = {}
        for k in ["loss", "tsdf_loss", "occ_loss"]:
            logs[f"loss_val/{k}"] = outputs[k].item()

        self.log_dict(logs, batch_size=batch_size, sync_dist=True)

    def on_validation_epoch_end(self):
        if self.global_rank != 0:
            return

        if self.current_epoch % 10 != 0:
            return

        # every 10 epochs run inference on the first test scan

        loader = self.predict_dataloader(first_scan_only=True)
        for i, batch in enumerate(tqdm.tqdm(loader, desc="prediction", leave=False)):
            for k in batch:
                if k in self.transfer_keys:
                    batch[k] = batch[k].to(self.device)
            self.predict_step(batch, i)
        self.predict_cleanup()
        torch.cuda.empty_cache()

    def predict_cleanup(self):
        del self.global_coords
        del self.M
        del self.running_count

        del self.keyframe_rgb
        del self.keyframe_pose

        if self.dg.enabled:
            del self.keyframe_depth
            if self.dg.tsdf_fusion_channel:
                del self.running_tsdf
                del self.running_tsdf_weight

            if self.dg.density_fusion_channel:
                del self.running_density
                del self.running_density_weight

    def predict_init(self, batch):
        # setup before starting inference on a new scan

        if self.config.do_prediction_timing:
            torch.cuda.synchronize()
            t0 = time.time()

        vox4 = self.config.voxel_size * 4
        minbound = batch["gt_origin"][0]
        maxbound = batch["gt_maxbound"][0].float()
        maxbound = (torch.ceil((maxbound - minbound) / vox4) - 0.001) * vox4 + minbound

        x = torch.arange(
            minbound[0], maxbound[0], self.config.voxel_size, dtype=torch.float32
        )
        y = torch.arange(
            minbound[1], maxbound[1], self.config.voxel_size, dtype=torch.float32
        )
        z = torch.arange(
            minbound[2], maxbound[2], self.config.voxel_size, dtype=torch.float32
        )
        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        self.global_coords = torch.stack((xx, yy, zz), dim=-1).to(self.device)

        nvox = xx.shape
        self.running_count = torch.zeros(nvox, dtype=torch.float32, device=self.device)
        self.M = torch.zeros(
            (self.fusion.out_c, *nvox),
            dtype=torch.float32,
            device=self.device,
        )

        self.keyframe_rgb = []
        self.keyframe_pose = []

        if self.dg.enabled:
            self.keyframe_depth = []

            if self.dg.density_fusion_channel:
                self.running_density = torch.zeros(
                    nvox, dtype=torch.float32, device=self.device
                )
                self.running_density_weight = torch.zeros(
                    nvox, dtype=torch.int32, device=self.device
                )
            elif self.dg.tsdf_fusion_channel:
                self.running_tsdf = torch.zeros(
                    nvox, dtype=torch.float32, device=self.device
                )
                self.running_tsdf_weight = torch.zeros(
                    nvox, dtype=torch.int32, device=self.device
                )

        if self.config.do_prediction_timing:
            torch.cuda.synchronize()
            t1 = time.time()
            self.init_time += t1 - t0
            self.n_inits += 1

    def predict_per_view(self, batch):
        # fuse each view into the scene volume

        if self.config.do_prediction_timing:
            torch.cuda.synchronize()
            t0 = time.time()

        batch_size, n_imgs, _, imheight, imwidth = batch["rgb_imgs"].shape
        imsize = imheight, imwidth
        assert batch_size == 1 and n_imgs == 1

        uv, z, valid = utils.project(
            self.global_coords[None],
            batch["poses"][None],
            batch["K_color"][None],
            imsize,
        )
        valid = valid[0, 0]
        coords = self.global_coords[valid][None, None, None]

        if self.dg.enabled:
            (
                img_voxel_feats,
                img_voxel_valid,
            ) = self.get_img_voxel_feats_by_depth_guided_bp(
                batch["rgb_imgs"],
                batch["pred_depth_imgs"],
                batch["poses"][None],
                batch["K_color"][None],
                batch["K_pred_depth"][None],
                coords,
            )
            if self.dg.density_fusion_channel:
                density, density_weight = self.density_fusion(
                    batch["pred_depth_imgs"],
                    batch["poses"][None],
                    batch["K_pred_depth"][:, None],
                    coords,
                )
                density = density[0, 0, 0]
                density_weight = density_weight[0, 0, 0]
            elif self.dg.tsdf_fusion_channel:
                tsdf, tsdf_weight = self.tsdf_fusion(
                    batch["pred_depth_imgs"],
                    batch["poses"][None],
                    batch["K_pred_depth"][:, None],
                    coords,
                )
                tsdf = tsdf[0, 0, 0]
                tsdf_weight = tsdf_weight[0, 0, 0]
                tsdf.masked_fill_(tsdf_weight == 0, 0)
        else:
            (img_voxel_feats, img_voxel_valid,) = self.get_img_voxel_feats_by_img_bp(
                batch["rgb_imgs"],
                batch["poses"][None],
                batch["K_color"][None],
                coords,
            )

        """
        in get_img_voxel_feats_by_img_bp these values are already zeroed inside of utils.sample_voxel_feats
        zeroing again here just in case
        """
        img_voxel_feats.masked_fill_(~img_voxel_valid[:, :, None], 0)

        old_count = self.running_count[valid].clone()
        self.running_count[valid] += img_voxel_valid[0, 0, 0, 0]
        new_count = self.running_count[valid]

        x = img_voxel_feats[0, 0, :, 0, 0]
        old_m = self.M[:, valid]
        new_m = x / new_count[None] + (old_count / new_count)[None] * old_m
        self.M[:, valid] = new_m
        self.M.masked_fill_(self.running_count[None] == 0, 0)

        if self.dg.enabled:
            if self.dg.density_fusion_channel:
                old_count = self.running_density_weight[valid]
                self.running_density_weight[valid] += density_weight
                new_count = self.running_density_weight[valid]
                denom = new_count + (new_count == 0)
                self.running_density[valid] = (
                    density / denom + (old_count / denom) * self.running_density[valid]
                )
            elif self.dg.tsdf_fusion_channel:
                old_count = self.running_tsdf_weight[valid]
                self.running_tsdf_weight[valid] += tsdf_weight
                new_count = self.running_tsdf_weight[valid]
                denom = new_count + (new_count == 0)
                self.running_tsdf[valid] = (
                    tsdf / denom + (old_count / denom) * self.running_tsdf[valid]
                )

        if self.config.do_prediction_timing:
            torch.cuda.synchronize()
            t1 = time.time()
            self.per_view_time += t1 - t0
            self.n_views += 1

    def predict_final(self, batch):
        # final reconstruction: run point back-projection & 3d cnn

        if self.config.do_prediction_timing:
            torch.cuda.synchronize()
            t0 = time.time()

        global_feats = self.M
        global_feats = self.fusion.bn(global_feats[None]).squeeze(0)

        if self.config.no_image_features:
            global_feats = global_feats * 0

        if self.dg.enabled:
            if self.dg.density_fusion_channel:
                global_feats = torch.cat(
                    (global_feats, self.running_density[None]), dim=0
                )
            elif self.dg.tsdf_fusion_channel:
                self.running_tsdf.masked_fill_(self.running_tsdf_weight == 0, 1)

                extra = self.running_tsdf[None]
                global_feats = torch.cat((global_feats, extra), dim=0)

        global_feats = self.cnn3d(global_feats[None], self.running_count[None] > 0)
        global_valid = self.running_count > 0

        coarse_spatial_dims = np.array(global_feats.shape[2:])
        fine_spatial_dims = coarse_spatial_dims * self.config.output_sample_rate

        coarse_occ_logits = self.occ_predictor(
            global_feats.view(1, global_feats.shape[1], -1)
        ).view(global_feats.shape[2:])

        coarse_occ_mask = coarse_occ_logits > 0
        coarse_occ_idx = torch.argwhere(coarse_occ_mask)
        n_coarse_vox_occ = len(coarse_occ_idx)

        fine_surface = torch.full(
            tuple(fine_spatial_dims), torch.nan, device="cpu", dtype=torch.float32
        )

        coarse_voxel_size = self.config.voxel_size
        fine_voxel_size = self.config.voxel_size / self.config.output_sample_rate

        x = torch.arange(self.config.output_sample_rate)
        xx, yy, zz = torch.meshgrid(x, x, x, indexing="ij")
        fine_idx_offset = torch.stack((xx, yy, zz), dim=-1).view(-1, 3).to(self.device)
        fine_offset = (
            fine_idx_offset * fine_voxel_size
            - coarse_voxel_size / 2
            + fine_voxel_size / 2
        )

        coarse_voxel_chunk_size = (2**20) // (self.config.output_sample_rate**3)

        if self.config.point_backprojection:
            imheight, imwidth = self.keyframe_rgb[0].shape[1:]
            featheight = imheight // 4
            featwidth = imwidth // 4

            keyframe_chunk_size = 32
            highres_img_feats = torch.full(
                (
                    len(self.keyframe_rgb),
                    self.cnn2d_pb_out_dim,
                    featheight,
                    featwidth,
                ),
                torch.nan,
                dtype=torch.float32,
                device="cpu",
            )

            for keyframe_chunk_start in tqdm.trange(
                0,
                len(self.keyframe_rgb),
                keyframe_chunk_size,
                desc="highres img feats",
                leave=False,
            ):
                keyframe_chunk_end = min(
                    keyframe_chunk_start + keyframe_chunk_size,
                    len(self.keyframe_rgb),
                )

                rgb_imgs = torch.stack(
                    self.keyframe_rgb[keyframe_chunk_start:keyframe_chunk_end],
                    dim=0,
                )

                highres_img_feats[
                    keyframe_chunk_start:keyframe_chunk_end
                ] = self.cnn2d_pb(rgb_imgs)

        for coarse_voxel_chunk_start in tqdm.trange(
            0, n_coarse_vox_occ, coarse_voxel_chunk_size, leave=False, desc="chunks"
        ):
            coarse_voxel_chunk_end = min(
                coarse_voxel_chunk_start + coarse_voxel_chunk_size, n_coarse_vox_occ
            )

            chunk_coarse_idx = coarse_occ_idx[
                coarse_voxel_chunk_start:coarse_voxel_chunk_end
            ]
            chunk_coarse_coords = (
                chunk_coarse_idx * coarse_voxel_size + batch["gt_origin"]
            )

            chunk_fine_coords = chunk_coarse_coords[:, None].repeat(
                1, self.config.output_sample_rate**3, 1
            )
            chunk_fine_coords += fine_offset[None]
            chunk_fine_coords = chunk_fine_coords.view(-1, 3)

            (
                chunk_fine_feats,
                chunk_fine_valid,
            ) = self.sample_point_features_by_linear_interp(
                chunk_fine_coords,
                global_feats,
                global_valid[None],
                batch["gt_origin"],
            )

            if self.config.point_backprojection:
                img_feature_dim = self.M.shape[0]
                fine_bp_feats = torch.zeros(
                    (self.cnn2d_pb_out_dim, len(chunk_fine_coords)),
                    device=self.device,
                    dtype=self.M.dtype,
                )
                counts = torch.zeros(
                    len(chunk_fine_coords), device=self.device, dtype=torch.float32
                )

                if self.dg.enabled:
                    if self.dg.density_fusion_channel:
                        fine_density = torch.zeros(
                            len(chunk_fine_coords), device=self.device
                        )
                        fine_density_weights = torch.zeros(
                            len(chunk_fine_coords),
                            device=self.device,
                            dtype=torch.float32,
                        )
                    elif self.dg.tsdf_fusion_channel:
                        fine_tsdf = torch.zeros(
                            len(chunk_fine_coords), device=self.device
                        )
                        fine_tsdf_weights = torch.zeros(
                            len(chunk_fine_coords),
                            device=self.device,
                            dtype=torch.float32,
                        )

                for keyframe_chunk_start in range(
                    0, len(self.keyframe_rgb), keyframe_chunk_size
                ):
                    keyframe_chunk_end = min(
                        keyframe_chunk_start + keyframe_chunk_size,
                        len(self.keyframe_rgb),
                    )

                    chunk_highres_img_feats = highres_img_feats[
                        keyframe_chunk_start:keyframe_chunk_end
                    ].to(self.device)
                    rgb_img_placeholder = torch.empty(
                        1, len(chunk_highres_img_feats), 3, imheight, imwidth
                    )

                    poses = torch.stack(
                        self.keyframe_pose[keyframe_chunk_start:keyframe_chunk_end],
                        dim=0,
                    )

                    if self.dg.enabled:
                        pred_depth_imgs = torch.stack(
                            self.keyframe_depth[
                                keyframe_chunk_start:keyframe_chunk_end
                            ],
                            dim=0,
                        )
                        (
                            _fine_bp_feats,
                            valid,
                        ) = self.get_img_voxel_feats_by_depth_guided_bp(
                            rgb_img_placeholder,
                            pred_depth_imgs[None],
                            poses[None],
                            batch["K_color"][:, None],
                            batch["K_pred_depth"][:, None],
                            chunk_fine_coords[None],
                            use_highres_cnn=True,
                            img_feats=chunk_highres_img_feats,
                        )
                    else:
                        _fine_bp_feats, valid = self.get_img_voxel_feats_by_img_bp(
                            rgb_img_placeholder,
                            poses[None],
                            batch["K_color"][:, None],
                            chunk_fine_coords[None],
                            use_highres_cnn=True,
                            img_feats=chunk_highres_img_feats,
                        )

                    old_counts = counts.clone()
                    current_counts = valid.squeeze(0).sum(dim=0)
                    counts += current_counts

                    denom = torch.clamp_min(counts, 1)
                    _fine_bp_feats = _fine_bp_feats.squeeze(0)
                    _fine_bp_feats /= denom
                    _fine_bp_feats = _fine_bp_feats.sum(dim=0)
                    fine_bp_feats *= old_counts / denom
                    fine_bp_feats += _fine_bp_feats

                    if self.dg.enabled:
                        if self.dg.density_fusion_channel:
                            density, weight = self.density_fusion(
                                pred_depth_imgs[None],
                                poses[None],
                                batch["K_pred_depth"][:, None],
                                chunk_fine_coords[None],
                            )
                            old_count = fine_density_weights.clone()
                            fine_density_weights += weight.squeeze(0)
                            new_count = fine_density_weights
                            denom = torch.clamp_min(new_count, 1)
                            fine_density = (
                                density.squeeze(0) / denom
                                + (old_count / denom) * fine_density
                            )
                        elif self.dg.tsdf_fusion_channel:
                            tsdf, weight = self.tsdf_fusion(
                                pred_depth_imgs[None],
                                poses[None],
                                batch["K_pred_depth"][:, None],
                                chunk_fine_coords[None],
                            )
                            tsdf.masked_fill_(weight == 0, 0)

                            old_count = fine_tsdf_weights.clone()
                            fine_tsdf_weights += weight.squeeze(0)
                            new_count = fine_tsdf_weights
                            denom = torch.clamp_min(new_count, 1)
                            fine_tsdf = (
                                tsdf.squeeze(0) / denom
                                + (old_count / denom) * fine_tsdf
                            )

                fine_bp_feats = self.point_fusion.bn(
                    fine_bp_feats[None, ..., None, None]
                )[..., 0, 0]
                fine_bp_feats = self.point_feat_mlp(fine_bp_feats)

                if self.config.no_image_features:
                    fine_bp_feats = fine_bp_feats * 0

                if self.dg.enabled:
                    if self.dg.density_fusion_channel:
                        chunk_fine_feats = torch.cat(
                            (fine_bp_feats, chunk_fine_feats, fine_density[None, None]),
                            dim=1,
                        )
                    elif self.dg.tsdf_fusion_channel:
                        fine_tsdf.masked_fill_(fine_tsdf_weights == 0, 1)

                        extra = fine_tsdf[None]

                        chunk_fine_feats = torch.cat(
                            (fine_bp_feats, chunk_fine_feats, extra[None]), dim=1
                        )
                    else:
                        chunk_fine_feats = torch.cat(
                            (fine_bp_feats, chunk_fine_feats), dim=1
                        )
                else:
                    chunk_fine_feats = torch.cat(
                        (fine_bp_feats, chunk_fine_feats), dim=1
                    )

            chunk_fine_surface_logits = (
                self.surface_predictor(chunk_fine_feats)[0, 0].cpu().float()
            )

            chunk_fine_idx = chunk_coarse_idx[:, None].repeat(
                1, self.config.output_sample_rate**3, 1
            )
            chunk_fine_idx *= self.config.output_sample_rate
            chunk_fine_idx += fine_idx_offset[None]
            chunk_fine_idx = chunk_fine_idx.view(-1, 3).cpu()

            fine_surface[
                chunk_fine_idx[:, 0],
                chunk_fine_idx[:, 1],
                chunk_fine_idx[:, 2],
            ] = chunk_fine_surface_logits

        torch.tanh_(fine_surface)
        fine_surface *= 0.5
        fine_surface += 0.5

        if self.config.do_prediction_timing:
            torch.cuda.synchronize()
            t1 = time.time()
            self.final_step_time += t1 - t0
            self.n_final_steps += 1

        os.makedirs(self.logger.log_dir, exist_ok=True)
        name = batch["scan_name"][0]
        step = str(self.global_step).zfill(8)

        origin = (
            batch["gt_origin"].cpu().numpy()[0]
            - coarse_voxel_size / 2
            + fine_voxel_size / 2
        )

        try:
            pred_mesh = utils.tsdf2mesh(
                fine_surface.numpy(),
                voxel_size=fine_voxel_size,
                origin=origin,
                level=0.5,
            )
        except Exception as e:
            print(e)
        else:
            _ = pred_mesh.export(os.path.join(self.logger.log_dir, f"{name}.ply"))

    def predict_step(self, batch, batch_idx):
        if batch["initial_frame"][0]:
            self.predict_init(batch)

        self.predict_per_view(batch)

        if self.config.point_backprojection:
            # store any frames that are marked as keyframes for later point back-projection
            if batch["keyframe"][0]:
                self.keyframe_rgb.append(batch["rgb_imgs"][0, 0])
                self.keyframe_pose.append(batch["poses"][0])
                if self.dg.enabled:
                    self.keyframe_depth.append(batch["pred_depth_imgs"][0, 0])

        if batch["final_frame"][0]:
            self.predict_final(batch)

    def on_predict_epoch_end(self, _):
        if self.config.do_prediction_timing:
            per_init_time = self.init_time / self.n_inits
            per_view_time = self.per_view_time / self.n_views
            final_step_time = self.final_step_time / self.n_final_steps

            print("========")
            print("========")
            print(f"per_init_time: {per_init_time:.4f}")
            print(f"per_view_time: {per_view_time:.4f}")
            print(f"final_step_time: {final_step_time:.4f}")
            print("========")
            print("========")

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        self.transfer_keys = [
            "input_coords",
            "output_coords",
            "crop_center",
            "crop_rotation",
            "crop_size_m",
            "gt_tsdf",
            "gt_occ",
            "K_color",
            "K_pred_depth",
            "rgb_imgs",
            "pred_depth_imgs",
            "poses",
            "gt_origin",
            "gt_maxbound",
        ]
        self.no_transfer_keys = [
            "scan_name",
            "gt_tsdf_npzfile",
            "keyframe",
            "initial_frame",
            "final_frame",
        ]

        transfer_batch = {}
        no_transfer_batch = {}
        for k in batch:
            if k in self.transfer_keys:
                transfer_batch[k] = batch[k]
            elif k in self.no_transfer_keys:
                no_transfer_batch[k] = batch[k]
            else:
                raise NotImplementedError

        transfer_batch = super().transfer_batch_to_device(
            transfer_batch, device, dataloader_idx
        )
        transfer_batch.update(no_transfer_batch)
        return transfer_batch

    def get_scans(self):
        train_scans, val_scans, test_scans = data.get_scans(
            self.config.dataset_dir,
            self.config.tsdf_dir,
            self.dg.pred_depth_dir,
        )
        return train_scans, val_scans, test_scans

    def train_dataloader(self):
        train_scans, _, _ = self.get_scans()
        train_dataset = data.Dataset(
            train_scans,
            self.config.voxel_size,
            self.config.crop_size_nvox_train,
            self.config.n_views_train,
            improved_tsdf_sampling=self.config.improved_tsdf_sampling,
            random_translation=True,
            random_rotation=True,
            random_view_selection=True,
            image_augmentation=True,
            load_depth=self.dg.enabled,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size_per_device,
            num_workers=self.config.workers_train,
            persistent_workers=self.config.workers_train > 0,
            shuffle=True,
            drop_last=True,
        )
        return train_loader

    def val_dataloader(self):
        _, val_scans, test_scans = self.get_scans()
        val_dataset = data.Dataset(
            val_scans,
            self.config.voxel_size,
            self.config.crop_size_nvox_val,
            self.config.n_views_val,
            improved_tsdf_sampling=self.config.improved_tsdf_sampling,
            random_translation=True,
            random_rotation=True,
            load_depth=self.dg.enabled,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=self.config.workers_val,
        )
        return val_loader

    def predict_dataloader(self, first_scan_only=False):
        _, _, test_scans = self.get_scans()

        if first_scan_only:
            test_scans = test_scans[:1]

        predict_dataset = data.InferenceDataset(
            test_scans,
            load_depth=self.dg.enabled,
            keyframes_file=self.config.test_keyframes_file,
        )
        predict_loader = torch.utils.data.DataLoader(
            predict_dataset,
            batch_size=1,
            num_workers=self.config.workers_predict,
        )
        return predict_loader
