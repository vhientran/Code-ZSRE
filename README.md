This is the official source code of our paper: [Enhancing Semantic Correlation between Instances and Relations for Zero-Shot Relation Extraction](https://www.jstage.jst.go.jp/article/jnlp/30/2/30_304/_article/-char/en) , Journal of Natural Language Processing, 2023.

To reproduce our reported experimental results, please follow steps below:

## 1. **Hardware Environment Information**

All our experiments were performed on 1 NVIDIA RTX A6000 GPU 48GB (CUDA version 11.6). 

## 2. **Prepare Conda Environment**

Create a new conda environment via commands:

conda create --name env_zsre python=3.6

Then, please install all required packages for this conda environment in the file: "Code-ZSRE/requirements.txt". Next, we active the conda environment env_zsre:

conda activate env_zsre

## 3. **How to Run Experiments**

### + Datasets:

Since the size of the original Wiki-ZSL dataset is large, please download this dataset at the [link](https://drive.google.com/file/d/1TMYvAbe9wsB5GiWcUL5bMAs9x6CpvnAj/view?usp=sharing) .
Then, put it in the folder: "Code-ZSRE/official_data/".

Now, both FewRel and Wiki-ZSL datasets are in the folder: "Code-ZSRE/official_data". Also, the relation labels and descriptions are available.
As introduced in the paper, we split the entire datasets into training and test sets, where the number of unseen relations in the test set is 15 (m=15).
We repeat this division 5 times for 5 different random selections of m.
Finally, the division result is saved in the file: "Code-ZSRE/official_data/split_train_test_sets/official_data_divisions.json"

### + Train and evaluate on Wiki-ZSL set:
We run 5 times with 5 different random selections according to id_round in [1,2,3,4,5].
Inside the folder: "Code-ZSRE/Wiki-ZSL"

cd Code-ZSRE/Wiki-ZSL

CUDA_VISIBLE_DEVICES=0 python train.py --alpha 1 --n_unseen 15 --K 5 --id_round 1

CUDA_VISIBLE_DEVICES=0 python train.py --alpha 1 --n_unseen 15 --K 5 --id_round 2

CUDA_VISIBLE_DEVICES=0 python train.py --alpha 1 --n_unseen 15 --K 5 --id_round 3

CUDA_VISIBLE_DEVICES=0 python train.py --alpha 1 --n_unseen 15 --K 5 --id_round 4

CUDA_VISIBLE_DEVICES=0 python train.py --alpha 1 --n_unseen 15 --K 5 --id_round 5

The final reported performance on Wiki-ZSL is avaraged of 5 the results by running 5 times above.


### + Train and evaluate on FewRel set:
Similarly, we also perform the experiment 5 times.
Inside the folder: "Code-ZSRE/FewRel"

cd Code-ZSRE/FewRel

CUDA_VISIBLE_DEVICES=0 python train.py --alpha 1 --n_unseen 15 --K 5 --id_round 1

CUDA_VISIBLE_DEVICES=0 python train.py --alpha 1 --n_unseen 15 --K 5 --id_round 2

CUDA_VISIBLE_DEVICES=0 python train.py --alpha 1 --n_unseen 15 --K 5 --id_round 3

CUDA_VISIBLE_DEVICES=0 python train.py --alpha 1 --n_unseen 15 --K 5 --id_round 4

CUDA_VISIBLE_DEVICES=0 python train.py --alpha 1 --n_unseen 15 --K 5 --id_round 5

The final reported performance on FewRel is avaraged of 5 the results by running 5 times above.


## Please kindly cite our paper if you find it useful. Thank you!
```bibtex
@article{tran2023enhancing,
  title={Enhancing Semantic Correlation between Instances and Relations for Zero-Shot Relation Extraction},
  author={Van-Hien Tran and Hiroki Ouchi and Hiroyuki Shindo and Yuji Matsumoto and Taro Watanabe},
  journal={Journal of Natural Language Processing},
  volume={30},
  number={2},
  pages={304-329},
  year={2023},
  doi={10.5715/jnlp.30.304}
}


