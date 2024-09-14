# Runing the Example
## Install habitat env
```
conda create -n env_name python=3.9 cmake=3.14.0
conda install habitat-sim withbullet -c conda-forge -c aihabitat
pip install git+https://github.com/facebookresearch/habitat-lab.git
```


## Complete git submodules
See ``/gitmodules``.

``git submodule update <specific path to submodule>``.

Mannually ``mkdir`` and add pretrained detic model to ``perception/detection/detic/Detic/models``

cd ``perception/detection/detic/Detic``, ``git submodule  update --init ./third_party/CenterNet2/``.

## Install dependencies
- detectron2: 
``python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'``

- clip:
``pip install git+https://github.com/openai/CLIP.git``

## Run
``python NavVLM.py``

# Full dataset acquirement
HM3D: see [habitat-matterpord3D dataset (hm3d)](https://aihabitat.org/datasets/hm3d/).
Gibson: ``https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#gibson-and-3dscenegraph-datasets``
MP3D: ``https://github.com/niessner/Matterport``