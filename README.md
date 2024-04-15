# mano2nimble

This repository is a submodule of the CHORD project.

This repository contains the code for the conversion of the MANO model to the Nimble model. The conversion is done using the [Nimble library](https://github.com/reyuwei/NIMBLE_model) and the [MANO model](https://mano.is.tue.mpg.de/).

## Prerequisites
- Python 3
- pytorch
- numpy
- [manotorch](https://github.com/lixiny/manotorch)
- pytorch3d
- open3d
- tqdm
- trimesh

## Downloads
- Download the MANO model from the [MANO website](https://mano.is.tue.mpg.de/).
- Download the Nimble model from the [Nimble library](https://github.com/reyuwei/NIMBLE_model)

Your directory should look like this:
```
.
├── assets
│   └── mano_v1_2
│       ├── models
│           └── MANO_RIGHT.pkl
├── nimble
│   ├── assets
│   │   ├── NIMBLE_DICT_9137.pkl
│   │   ├── ...
```

## Usage
Simply run the following command:
```
python mano2nimble.py -v
```

## Aknowledgements
This code is based on the [manotorch](https://github.com/lixiny/manotorch) and [Nimble library](https://github.com/reyuwei/NIMBLE_model).

If you find this code useful, consider citing the following related works:
```
@inproceedings{li2023chord,
  title={Chord: Category-level hand-held object reconstruction via shape deformation},
  author={Li, Kailin and Yang, Lixin and Zhen, Haoyu and Lin, Zenan and Zhan, Xinyu and Zhong, Licheng and Xu, Jian and Wu, Kejian and Lu, Cewu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9444--9454},
  year={2023}
}
@article{li2024semgrasp,
  title={SemGrasp: Semantic Grasp Generation via Language Aligned Discretization},
  author={Li, Kailin and Wang, Jingbo and Yang, Lixin and Lu, Cewu and Dai, Bo},
  journal={arXiv preprint arXiv:2404.03590},
  year={2024}
}
```