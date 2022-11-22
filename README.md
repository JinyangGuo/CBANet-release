# CBANet: Towards Complexity and Bitrate Adaptive Deep Image Compression using a Single Network

Created by [Jinyang Guo](https://jinyangguo.github.io/)

<!-- ## Introduction -->

<!-- This project is the official implementation of our accepted ICLR 2022 paper *BiBERT: Accurate Fully Binarized BERT* [[PDF](https://openreview.net/forum?id=5xEgrl_5FAJ)]. The large pre-trained BERT has achieved remarkable performance on Natural Language Processing (NLP) tasks but is also computation and memory expensive. As one of the powerful compression approaches, binarization extremely reduces the computation and memory consumption by utilizing 1-bit parameters and bitwise operations. Unfortunately, the full binarization of BERT (i.e., 1-bit weight, embedding, and activation) usually suffer a significant performance drop, and there is rare study addressing this problem. In this paper, with the theoretical justification and empirical analysis, we identify that the severe performance drop can be mainly attributed to the information degradation and optimization direction mismatch respectively in the forward and backward propagation, and propose BiBERT, an accurate fully binarized BERT, to eliminate the performance bottlenecks. Specifically, BiBERT introduces an efficient Bi-Attention structure for maximizing representation information statistically and a Direction-Matching Distillation (DMD) scheme to optimize the full binarized BERT accurately. Extensive experiments show that BiBERT outperforms both the straightforward baseline and existing state-of-the-art quantized BERTs with ultra-low bit activations by convincing margins on the NLP benchmark. As the first fully binarized BERT, our method yields impressive 59.2$\times$ and 31.2$\times$ saving on FLOPs and model size, demonstrating the vast advantages and potential of the fully binarized BERT model in real-world resource-constrained scenarios. -->

## Requirement

- Python 3.6+

- PyTorch 1.10+


## Usage
1. Download Kodak dataset
2. Change $data_dir in test.py to the path of Kodak dataset
3. Download pre-trained model from https://drive.google.com/file/d/1Uq6ctvM1cc9JA0J1Vb9Nm3aBOpgDqyEZ/view?usp=sharing and store it to current directory
4. run: 
```bash
python test.py --config config/OneEncoderPruner.json -p PATH_TO_PRETRAINED_MODEL
```

<!-- ## Datasets

We test CBANet on Kodak, which is available online:

- **Kodak**: https://github.com/nyu-mll/GLUE-baselines
- **SQuAD**: https://rajpurkar.github.io/SQuAD-explorer/

For data augmentation on GLUE, please follow the instruction in [TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT).

## Execution

Our experiments are based on the fine-tuned full-precision DynaBERT, which can be found [here](https://drive.google.com/file/d/1pYApaDcse5QIB6lZagWO0uElAavFazpA/view?usp=sharing). Complete running scripts and more detailed tips are provided in `./scripts`. Go through each script for more detail, and our corresponding well-trained BiBERT models are provided in [here](https://drive.google.com/drive/folders/1xEEIynvsYuqqG6wRlMhSySUusZWoR1FL?usp=sharing).

## Acknowledgement

The original code is borrowed from [BinaryBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/BinaryBERT) and [DynaBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/DynaBERT).

## Citation

If you find our work useful in your research, please consider citing:

```shell
@inproceedings{Qin:iclr22,
  author    = {Haotong Qin and Yifu Ding and Mingyuan Zhang and Qinghua Yan and 
  Aishan Liu and Qingqing Dang and Ziwei Liu and Xianglong Liu},
  title     = {BiBERT: Accurate Fully Binarized BERT},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022}
}
``` -->
