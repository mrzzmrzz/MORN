# MORN

CIKM 2022 accpeted full paper


This is the official implementation of *MORN: Molecular Property Prediction Based on Textual-Topological-Spatial Multi-View Learning.* In this work, we proposed a two-stage method for learning molecular representations to predict the molecular property from a multi-view perspective. In the first stage, textual-topological-spatial multi-views were proposed to learn the molecular representations, so as to capture both chemical system language and structure notation features simultaneously. In the second stage, an adaptive strategy was used to fuse molecular representations learned from multi-views to predict molecular properties.The proposed metalloenzyme inhibitors dataset is in `Dataset/iron`. If you find our work useful in your research, please cite:

```
@inproceedings{ma2022morn,
  title={MORN: Molecular Property Prediction Based on Textual-Topological-Spatial Multi-View Learning.},
  author={Ma, Runze and Zhang, Yidan and Wang, Xinye and Yu, Zhenyang and Duan, Lei},
  booktitle={Proceedings of the 31th {ACM} International Conference on Information and Knowledge Management,
            {CIKM} 2022},
  year={2022},
  doi={10.1145/3511808.3557401}
}
```




## Installation
```bash
=======
```python
# create a new environment
$ conda create --name morn python=3.10
$ conda activate morn

# install requirements
$ pip install -r requirements.txt

# clone the source code of MORN
$ git clone https://github.com/Mrz-zz/MORN
$ cd MORN
```



## Usage
1. Preprocess the raw data

```bash
python data_preprocessing.py
```

2. Train and evaluate the model:
```bash
python pipeline.py
```

