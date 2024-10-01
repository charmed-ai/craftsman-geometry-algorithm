import numpy as np
from PIL import Image
from craftsman.apps.utils import load_model
from craftsman.systems.base import BaseSystem
import trimesh
import torch
import argparse
from huggingface_hub import hf_hub_download
import os

def image2mesh(
    model: BaseSystem,
    view_front: np.ndarray,
    view_right: np.ndarray,
    view_back: np.ndarray,
    view_left: np.ndarray,
    guidance_scale: int = 7.5,
    step: int = 50,
    seed: int = 4,
    octree_depth: int = 7
):

    sample_inputs = {
        "mvimages": [[
            view_front,
            view_right,
            view_back,
            view_left
        ]]
    }

    latents = model.sample(
        sample_inputs,
        sample_times=1,
        guidance_scale=guidance_scale,
        return_intermediates=False,
        steps=step,
        seed=seed
    )[0]

    # decode the latents to mesh
    box_v = 1.1
    mesh_outputs, _ = model.shape_model.extract_geometry(
        latents,
        bounds=[-box_v, -box_v, -box_v, box_v, box_v, box_v],
        octree_depth=octree_depth
    )

    return trimesh.Trimesh(mesh_outputs[0][0], mesh_outputs[0][1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, default="./output/mesh.obj", help="Path to the obj file",)
    ############## model and mv model ##############
    parser.add_argument("--model", type=str, default="ckpts/image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6-aligned-vae", help="Path to the image-to-shape diffusion model",)
    ############## inference ##############
    parser.add_argument("--seed", type=int, default=4, help="Random seed for generating multi-view images",)
    parser.add_argument("--target-face-count", type=int, default=2000, help="Target face count for remeshing",)
    parser.add_argument("--guidance-scale", type=float, default=3, help="Guidance scale for 3D reconstruction",)
    parser.add_argument("--step", type=int, default=50, help="Number of steps for 3D reconstruction",)
    parser.add_argument("--octree-depth", type=int, default=7, help="Octree depth for 3D reconstruction",)
    ############## data preprocess ##############
    parser.add_argument("--no-rmbg", type=bool, default=False, help="Do NOT remove the background",)
    parser.add_argument("--rm-type", type=str, default="rembg", choices=["rembg", "sam"], help="Type of background removal",)
    parser.add_argument("--bkgd-type", type=str, default="Remove", choices=["Alpha as mask", "Remove", "Original"], help="Type of background",)
    parser.add_argument("--bkgd-color", type=str, default="[127,127,127,255]", help="Background color",)
    parser.add_argument("--fg-ratio", type=float, default=1.0, help="Foreground ratio",)
    parser.add_argument("--front-view", type=str, default="", help="Front view of the object",)
    parser.add_argument("--right-view", type=str, default="", help="Right view of the object",)
    parser.add_argument("--back-view", type=str, default="", help="Back view of the object",)
    parser.add_argument("--left-view", type=str, default="", help="Left view of the object",)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # load the shape diffusion model
    model = load_model(f"{args.model}/model.ckpt", f"{args.model}/config.yaml", device)
    mesh = image2mesh(model, Image.open(args.front_view), Image.open(args.right_view), Image.open(args.back_view), Image.open(args.left_view), args.guidance_scale, args.step,args.seed, args.octree_depth)
    mesh.export(args.output_path, include_normals=True)

if __name__=="__main__":
    main()
