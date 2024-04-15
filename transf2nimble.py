import os
import pickle

import numpy as np

from nimble.NIMBLELayer import NIMBLELayer
from nimble.utils import batch_to_tensor_device, smooth_mesh
import argparse

import open3d as o3d
import torch
from manotorch.manolayer import ManoLayer
from pytorch3d.structures.meshes import Meshes
from tqdm import tqdm
from transform import aa_to_rotmat, rotmat_to_aa


def fit_nimbel_params(nlayer, std_mano, mano_poses, mano_shapes, device):
    bn = mano_poses.shape[0]
    tex_param = torch.rand(bn, 10).to(device) - 0.5

    # * <<< trick!
    # * use the constant rotation to initialize the pose_param
    pose_param_np = pickle.load(open("const_rot.pkl", "rb"))
    pose_param_np = np.einsum("bij,bjk->bik", aa_to_rotmat(mano_poses[:, :3]), aa_to_rotmat(pose_param_np))
    pose_param_np = rotmat_to_aa(pose_param_np)
    pose_param_np = np.concatenate([pose_param_np, np.random.normal(0, 0.01, (bn, 60 - 3))], axis=1)
    shape_param_np = np.random.normal(0, 0.01, (bn, 20))
    # * >>>

    # * trick!
    # * calculate the translation of the root joint first
    # * then use it as the initial value of tsl_param
    _, _, _, bone_joints, _ = nlayer.forward(
        torch.tensor(pose_param_np, dtype=torch.float32).to(device),
        torch.tensor(shape_param_np, dtype=torch.float32).to(device),
        tex_param,
        handle_collision=False,
    )
    tsl_param_np = -bone_joints[:, 11, :].cpu().numpy() / 1000
    # * >>>

    pose_param = torch.tensor(pose_param_np, dtype=torch.float32, device=device, requires_grad=True)
    shape_param = torch.tensor(shape_param_np, dtype=torch.float32, device=device, requires_grad=True)
    tsl_param = torch.tensor(tsl_param_np, dtype=torch.float32, device=device, requires_grad=True)

    opt = torch.optim.Adam([pose_param, shape_param, tsl_param], lr=0.003)

    mano_out = std_mano(torch.tensor(mano_poses), torch.tensor(mano_shapes))
    mano_verts = mano_out.verts.to(device)

    bar = tqdm(range(800))
    for _ in bar:
        skin_v, muscle_v, bone_v, bone_joints, tex_img = nlayer.forward(
            pose_param,
            shape_param,
            tex_param,
            handle_collision=False,
        )

        skin_mano_v = nlayer.nimble_to_mano(skin_v, is_surface=True)
        nimble_verts = skin_mano_v / 1000 + tsl_param.unsqueeze(1)

        loss = torch.mean(torch.norm(nimble_verts - mano_verts, dim=-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        bar.set_description(f"loss: {loss.item():.4f}")
    return (
        pose_param.detach().cpu().numpy(),
        shape_param.detach().cpu().numpy(),
        tsl_param.detach().cpu().numpy(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nill")
    parser.add_argument("-v", "--vis", action="store_true")
    arg, custom_arg_string = parser.parse_known_args()

    device = f"cuda:0"

    pm_dict_name = r"nimble/assets/NIMBLE_DICT_9137.pkl"
    tex_dict_name = r"nimble/assets/NIMBLE_TEX_DICT.pkl"

    if os.path.exists(pm_dict_name):
        pm_dict = np.load(pm_dict_name, allow_pickle=True)
        pm_dict = batch_to_tensor_device(pm_dict, device)

    if os.path.exists(tex_dict_name):
        tex_dict = np.load(tex_dict_name, allow_pickle=True)
        tex_dict = batch_to_tensor_device(tex_dict, device)

    if os.path.exists(r"nimble/assets/NIMBLE_MANO_VREG.pkl"):
        nimble_mano_vreg = np.load("nimble/assets/NIMBLE_MANO_VREG.pkl", allow_pickle=True)
        nimble_mano_vreg = batch_to_tensor_device(nimble_mano_vreg, device)
    else:
        nimble_mano_vreg = None

    nlayer = NIMBLELayer(
        pm_dict, tex_dict, device, use_pose_pca=False, shape_ncomp=20, nimble_mano_vreg=nimble_mano_vreg
    ).to(device)

    standard_mano = ManoLayer(
        flat_hand_mean=True,
        side="right",
        center_idx=9,
        mano_assets_root="assets/mano_v1_2",
    )

    # * <<<< load the sample data
    sample_data = pickle.load(open("hand_sample.pkl", "rb"))

    # ** fit the nimble params
    res_mano_poses, res_mano_shapes, res_mano_tsls = fit_nimbel_params(
        nlayer, standard_mano, sample_data["mano_poses"], sample_data["mano_shapes"], device
    )
    # * >>>>

    # * do something with the results
    pass

    # * for example, visulize the results
    if arg.vis:
        vis_idx = 0
        mano_out = standard_mano(torch.tensor(sample_data["mano_poses"]), torch.tensor(sample_data["mano_shapes"]))
        mano_verts = mano_out.verts.to(device)

        mano_mesh = o3d.geometry.TriangleMesh()
        mano_mesh.vertices = o3d.utility.Vector3dVector(mano_verts[vis_idx].cpu().numpy())
        mano_mesh.triangles = o3d.utility.Vector3iVector(standard_mano.get_mano_closed_faces().cpu().numpy())
        mano_mesh.compute_vertex_normals()
        mano_mesh.paint_uniform_color([1, 0.5, 0.5])

        bn = res_mano_poses.shape[0]
        skin_v, muscle_v, bone_v, bone_joints, tex_img = nlayer.forward(
            torch.tensor(res_mano_poses, device=device),
            torch.tensor(res_mano_shapes, device=device),
            torch.rand(bn, 10).to(device) - 0.5,
            handle_collision=True,
        )

        skin_p3dmesh = Meshes(skin_v, nlayer.skin_f.repeat(bn, 1, 1))
        muscle_p3dmesh = Meshes(muscle_v, nlayer.muscle_f.repeat(bn, 1, 1))
        bone_p3dmesh = Meshes(bone_v, nlayer.bone_f.repeat(bn, 1, 1))

        skin_p3dmesh = smooth_mesh(skin_p3dmesh)
        muscle_p3dmesh = smooth_mesh(muscle_p3dmesh)
        bone_p3dmesh = smooth_mesh(bone_p3dmesh)

        tex_img = tex_img.detach().cpu().numpy()
        skin_v_smooth = skin_p3dmesh.verts_padded().detach().cpu().numpy()
        bone_joints = bone_joints.detach().cpu().numpy()

        skin_mesh = o3d.geometry.TriangleMesh()
        skin_mesh.vertices = o3d.utility.Vector3dVector(skin_v_smooth[vis_idx] / 1000 + res_mano_tsls[vis_idx])
        skin_mesh.triangles = o3d.utility.Vector3iVector(nlayer.skin_f.cpu().numpy())
        skin_mesh.compute_vertex_normals()
        skin_mesh.paint_uniform_color([1, 1, 0])
        o3d.visualization.draw_geometries([skin_mesh, mano_mesh])
