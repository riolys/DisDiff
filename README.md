# DisDiff  (ACM MM2024) 🌟

![arXiv](https://img.shields.io/badge/arxiv-2405.20584-brightgreen?style=flat&link=https%3A%2F%2Farxiv.org%2Fabs%2F2405.2058)

Official Code of [Disrupting Diffusion: Token-Level Attention Erasure Attack against Diffusion-based Customization(ACM MM2024)](https://arxiv.org/abs/2405.20584) 

<img src="https://github.com/user-attachments/assets/12a97d59-db3a-4ebc-8f96-855f0c371af4" alt="image" width="450" height="450"/>



## Environment
Please follow [Anti-DreamBooth](https://github.com/VinAIResearch/Anti-DreamBooth) to install the environment&base codes&datasets.

For visualizations, use this notebook of [Attend and excite](https://github.com/yuval-alaluf/Attend-and-Excite/blob/main/notebooks/explain.ipynb).

## Codes and Usage
Concat the attacks and scripts folders with the same name folders in [Anti-DreamBooth](https://github.com/VinAIResearch/Anti-DreamBooth).

Then conduct
```
bash scripts/disdiff.sh 
```

## Cite
If you find DisDiff is helpful to your work, please cite our work

```
@article{liu2024disrupting,
  title={Disrupting Diffusion: Token-Level Attention Erasure Attack against Diffusion-based Customization},
  author={Liu, Yisu and An, Jinyang and Zhang, Wanqian and Wu, Dayan and Gu, Jingzi and Lin, Zheng and Wang, Weiping},
  booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
  year={2024},
  series = {MM '24}
}
```
