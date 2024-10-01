# README
This repo contains a minimal application for running the CraftsMan 3D uplift model derived from https://github.com/wyysf-98/CraftsMan. This repository does not contain code for training, nor does it contain code for view consistent image generation.

## Pretrained models
Model weights are available from https://huggingface.co/wyysf/CraftsMan
They should be installed under ckpts/image-to-shape-diffusion/

## Running

```
prompt> craftsman_generate_geometry --output-path path/to/mesh.obj --front-view path/to/front.png --right-view path/to/left.png --back-view path/to/back.png --left-view path/to/left.png
```


# License
CraftsMan is under [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.en.html), so any downstream solution and products (including cloud services) that include CraftsMan code or a trained model (both pretrained or custom trained) inside it should be open-sourced to comply with the AGPL conditions. If you have any questions about the usage of CraftsMan, please contact [the authors](https://github.com/wyysf-98) first.


