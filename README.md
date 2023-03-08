## [Achieving Real Iterative Training via Iterative Click Loss for Interactive Image Segmentation](https://arxiv.org)

<p align="center">
  <img src="./assets/img/flowchart.png" alt="drawing", width="650"/>
</p>

<p align="center">
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-yellow.svg"/>
    </a>
</p>

## Environment
Training and evaluation environment: Python 3.9, PyTorch 1.13.1, CUDA 11.0. Run the following command to install required packages.
```
pip3 install -r requirements.txt
```

You need to configue the paths to the datasets in `config.yml` before training or testing. A script `download_datasets.sh` is prepared to download and organize required datasets.

## Demo
<p align="center">
  <img src="./assets/img/demo1.gif" alt="drawing", width="500"/>
</p>

An example script to run the demo. 
```
python demo.py --checkpoint=weights/cocolvis_icl_vit_huge.pth --gpu 0
```

## Evaluation

Before evaluation, please download the datasets and models, and then configure the path in `config.yml`.

Download our model: [click here]()

Use the following code to evaluate the huge model.

```
python scripts/evaluate_model.py NoBRS \
    --gpu=0 \
    --checkpoint=cocolvis_icl_vit_huge.pth \
    --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD \\
    --cf-n=4 \\
    --acf

# cf-n: CFR steps
# acf: adaptive CFR
```

## Training

Before training, please download the [MAE](https://github.com/facebookresearch/mae) pretrained weights (click to download: [ViT-Base](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth), [ViT-Large](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth), [ViT-Huge](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth)) and configure the dowloaded path in `config.yml`

Please also download the pretrained SimpleClick models from [here](https://github.com/uncbiag/SimpleClick).

Use the following code to train a huge model on C+L: 
```
python train.py models/plainvit_huge448_cocolvis.py \
    --batch-size=32 \
    --ngpus=4
```

## License
The code is released under the MIT License. It is a short, permissive software license. Basically, you can do whatever you want as long as you include the original copyright and license notice in any copy of the software/source. 

<!-- ## Citation
```bibtex
@article{liu2022simpleclick,
  title={SimpleClick: Interactive Image Segmentation with Simple Vision Transformers},
  author={Liu, Qin and Xu, Zhenlin and Bertasius, Gedas and Niethammer, Marc},
  journal={arXiv preprint arXiv:2210.11006},
  year={2022}
}
``` -->

## Acknowledgement
Our project is developed based on [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation) and [SimpleClick](https://github.com/uncbiag/SimpleClick)
