#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from gaussian_splatting.utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from gaussian_splatting.utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from gaussian_splatting.utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from gaussian_splatting.utils.graphics_utils import BasicPointCloud
from gaussian_splatting.utils.general_utils import strip_symmetric, build_scaling_rotation
from tqdm import tqdm
from torch.nn import functional as F

class ColorFieldModule(nn.Module):
    '''
    Color Field module that takes a bunch of coordinates and returns SH coefficients  (only supports degree 0)
    '''
    def __init__(self, cfg) -> None:
        super().__init__()
        # init frequencies
        freq = np.pi * 2.0**torch.linspace(0, cfg.max_freq, cfg.num_freq)[None]  # 2^i pi
        self.register_buffer('freq', freq)
        # mlp
        mlp = []
        layer_inds = [cfg.num_freq * 6] + cfg.hidden_layers + [3]
        for i in range(len(layer_inds) - 1):
            mlp.append(nn.Linear(layer_inds[i], layer_inds[i+1]))
            mlp.append(nn.LeakyReLU(0.01, inplace=True))
        mlp = mlp[:-1]
        self.mlp = nn.Sequential(*mlp)
    
    def forward(self, x):
        ''' x: [B, 3] coordinates '''
        freq_enc = []
        for i in range(3):
            omega = x[:, i:i+1] * self.freq
            freq_enc.append(torch.sin(omega))
            freq_enc.append(torch.cos(omega))
        freq_enc = torch.cat(freq_enc, dim=1)  # [B, freq]
        sh = self.mlp(freq_enc)  # [B, 3]
        return sh.unsqueeze(1)   # [B, 1, 3]

class SMPLGaussianModel:
    '''
    Gaussian model, where locations are replaced with the barycentric coordinates of the mesh 

    Here, the `xyz` parameter is the barycentric coordinates of the mesh, 
    and we have an extra *non-optimizable* parameter `face_idx` that keeps the face indices on the mesh

    `face_aligned` is a variable that tells us whether to constrain the splats 
    to the surface of the face. If not, then the rotation of the covariance matrix needs to be figured out

    '''
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, extra_rotation_matrix = None):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            # this rotation matrix can also be a rigid matrix (M)
            if extra_rotation_matrix is not None:  # [N, 3, 3]
                actual_covariance = extra_rotation_matrix @ actual_covariance @ extra_rotation_matrix.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        # set the activation functions
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, cfg, sh_degree : int, num_points_init: int = 50000):
        self.cfg = cfg
        # here cfg is the master cfg
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self.num_points_init = num_points_init
        self._faces = torch.empty(0)
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.setup_colorfield(cfg)
    
    def setup_colorfield(self, cfg):
        cf = cfg.network.colorfield
        self.cf_enabled = cf.enable
        if cf.enable == True:
            self.colorfield = ColorFieldModule(cf)
        else:
            self.colorfield = nn.Linear(1, 1) ## dummy
        self.colorfield = self.colorfield.cuda()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._faces,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.colorfield.state_dict(),
            self.optimizer.state_dict() if self.optimizer is not None else None,
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._faces,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        colorfield_dict,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        # set up training and gradient accumulation
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.colorfield.load_state_dict(colorfield_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features_dc(self):
        # get sh coefficients from colorfield
        if self.cf_enabled:
            return self.colorfield(self._xyz)
        else:
            return self._features_dc
    
    @get_features_dc.setter
    def get_features_dc(self, values):
        self.cf_enabled = False
        self._features_dc = values
    
    @property
    def get_features(self):
        # features_dc = [F, 1, 3]
        # features_rest = [F, N, 3]
        deg = self.active_sh_degree
        features_dc = self.get_features_dc
        if deg == 0:
            features_rest = torch.zeros_like(self._features_rest)
        else:
            # select only upto `degree` coefficients  (remove the dc component)
            tot_coeff = (deg + 1) ** 2  - 1
            features_active = self._features_rest[:, :tot_coeff, :]
            features_zero = torch.zeros_like(self._features_rest[:, tot_coeff:, :])
            features_rest = torch.cat([features_active, features_zero], dim=1)
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1, extra_rotation_matrix = None):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation, extra_rotation_matrix)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_SMPL(self, smpl_mesh, faces, per_face_color: torch.Tensor, spatial_lr_scale: float):
        ''' given smpl mesh and face color list, initialize the model 

        :smpl_mesh: Collection of vertices in canonical pose. Barycentric coordinates are initialized to 
            [0, 0, 0] and radius is equal to unsigned face area
        :per_face_color: RGB color of each face [Nf, 3]
        '''
        N = per_face_color.shape[0]
        self.spatial_lr_scale = spatial_lr_scale

        # set scales
        smpl_template = smpl_mesh.detach()  # [Nv, 3]
        f1, f2, f3 = faces.t()  # Nf each
        v1, v2, v3 = smpl_template[f1], smpl_template[f2], smpl_template[f3]  # [Nf, 3]
        device = v1.device
        facearea = torch.norm(torch.cross(v2 - v1, v3 - v1), dim=-1) / 2  # [Nf]   
        # select faces
        # in the free point version, we dont need to keep track of which face this came from
        self._faces = torch.multinomial(facearea*0 + 1, self.num_points_init, replacement=True).long()  # [N_init]
        ## starting the scales with a much smaller value
        scales = torch.log(facearea[self._faces, None].repeat(1, 3) + 1e-10)/2 - np.log(3)
        print("Scales", scales.min(), scales.max())
        # scales = torch.log(facearea[self._faces, None].repeat(1, 3) + 1e-10)*0 - 5
        rots = torch.zeros((self.num_points_init, 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.5 * torch.ones((self.num_points_init, 1), dtype=torch.float, device="cuda"))
        
        # Set color
        fused_color = RGB2SH(per_face_color[self._faces, :].float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color

        # sample points from the facearea
        if self.cfg.splat_options.face_aligned_init:
            print("Initializing points on the faces...")
            v1s, v2s, v3s = smpl_template[f1[self._faces]], smpl_template[f2[self._faces]], smpl_template[f3[self._faces]]  # [Np, 3]
            w = torch.rand((self.num_points_init, 1, 3), device=device)
            w /= w.sum(dim=-1, keepdim=True)  # [N, 1]
            xyz = (v1s * w[..., 0] + v2s * w[..., 1] + v3s * w[..., 2])  # [Np, 3]
        else:
            print("Initializing points randomly...")
            mincoord, maxcoord = smpl_template.min(0).values[None], smpl_template.max(0).values[None]  # [1, 3], [1, 3]
            u = torch.rand((self.num_points_init, 3), device=device)
            xyz = u * (maxcoord - mincoord) + mincoord  # [Np, 3]

        # set parameters
        self._xyz = nn.Parameter(xyz.clone().requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))  # [Np, 1, 3]
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # pretrain colorfield
        self.pretrain_colorfield()


    def training_setup(self, training_args):
        ''' here, training_args is the subconfig for the splats '''
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / training_args.sh_features_lr_scale, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self.colorfield.parameters(), 'lr': self.cfg.network.colorfield.lr, "name": "colorfield", "weight_decay": 1e-5}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-8)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_steps=training_args.position_lr_delay_steps,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    
    def pretrain_colorfield(self):
        ''' pretrain the colorfield using parameters obtained from SMPL init '''
        if not self.cf_enabled:
            print("No colorfield initialized, skipping pretraining.")
            return
        print("Pretraining color field using network parameters...")
        max_epochs = self.cfg.network.colorfield.pretrain_epochs
        pbar = tqdm(range(max_epochs))
        optim = torch.optim.Adam(self.colorfield.parameters(), lr=3e-4, weight_decay=1e-5)
        for i in pbar:
            optim.zero_grad()
            dc = self.get_features_dc
            gt_dc = self._features_dc.detach()
            loss = F.mse_loss(dc, gt_dc)
            pbar.set_description("Epoch: {}/{}, Loss: {}".format(i, max_epochs, loss.item()))
            loss.backward()
            optim.step()
        print("Finished pretraining...")

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        raise NotImplementedError
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        raise NotImplementedError
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # dont need to prune anything in the colorfield
            if group['name'] == 'colorfield':
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self._faces = self._faces[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # dont need to concat anything to colorfield
            if group['name'] == 'colorfield':
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation,
                              new_faces):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._faces = torch.cat((self._faces, new_faces), dim=0)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        ''' select points, repeat the selected points, and add them to the list
        for faces, we simply concat their index
        '''
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        # new_xyz = self.get_xyz[selected_pts_mask].repeat(N, 1) + samples
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        # new faces
        new_faces = self._faces[selected_pts_mask].repeat(N)
        # self._faces = torch.cat((self._faces, new_faces), dim=0)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_faces)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_faces = self._faces[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_faces)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

## Base Gaussian is deleted and moved to `gaussian_model_base.py`