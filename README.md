# Lightweight Attention U-Net for Breast Cancer Semantic Segmentation

## Introduction

**Breast cancer** is one of the **most common** causes of death among **women worldwide**. Early detection helps in reducing the **number of early deaths**. The data reviews the **medical images of breast cancer** using ultrasound scan. **Breast Ultrasound Dataset** is categorized into **three classes** $:$ **normal, benign, and malignant images**. **Breast ultrasound images** can produce great results in **classification, detection, and segmentation** of breast cancer when combined with machine learning.

## Data

I'll be working with [Breast Ultrasound Images Dataset
](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset). * The data collected at baseline include **breast ultrasound images among women** in ages between 25 and 75 years old. This data was collected in **2018**. The number of patients is **600 female patients**. The dataset consists of **780 images** with an \*\*average image size of 500*500 pixels**. The images are in PNG format. The ground truth images are presented with original images. The images are categorized into **three classes, which are normal, benign, and malignant\*\*.

## Methodology

I'm continuing on the work done by [Oktay et al., 2018](https://arxiv.org/pdf/1804.03999.pdf) that's implemented by [this notebook on kaggle](https://www.kaggle.com/code/utkarshsaxenadn/breast-cancer-image-segmentation-attention-unet), [quantized](https://medium.com/codex/quantization-tutorial-in-tensorflow-to-optimize-a-ml-model-like-a-pro-cadf811482d9) the model parameters and applied [Depth-Wise Separable Convolutions](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728) instead of regular convolutions.

## Project Structure

You can find the code for the base model and its quantization in `attention-unet-base.ipynb` and the code for the depth-wise separable convolutions model and its quantization in `attention-unet-ds.ipynb`. The models are in the models directory along with their respective profiling using [MLTK Profiler](https://siliconlabs.github.io/mltk/docs/guides/model_profiler_utility.html).

## Results

| Names                            | Attention-U-Net-base | Attention-U-Net-quantized | Attention-U-Net-DS | Attention-U-Net-DS-quantized |
| -------------------------------- | -------------------- | ------------------------- | ------------------ | ---------------------------- |
| Flash, Model File Size (bytes)   | 42.9M                | 11.1M                     | 10.7M              | 3.0M                         |
| RAM, Runtime Memory Size (bytes) | -                    | 13.7M                     | 33.6M              | 13.7M                        |
| Operation Count                  | -                    | 29.2G                     | 5.8G               | 5.8G                         |
| Multiply-Accumulate Count        | -                    | 14.6G                     | 2.9G               | 2.9G                         |
| Accuracy                         | 94.78%               | 93.56%                    | 91.82%             | 91.82%                       |

The base model was too large for the profiler so most of
its metrics couldn’t be measured. However, you can see the strong advantages provided by these
methods to drop its size and make it easier to be used in an edge device context.

## Disclaimer

This work was made as a class project for CSE758 (TinyML - Edge Machine Learning) at Alexandria University for my Master's Degree.

## TODO

- [ ] Add **Open in colab** button for the two notebooks.
- [ ] Upload the research paper.

## References

[1]
Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg
Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew
Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath
Kudlur, and Josh Levenberg. 2016. TensorFlow: Large-Scale Machine Learning on Heterogeneous
Distributed Systems

[2]
Walid Al-Dhabyani, Mohammed Gomaa, Hussien Khaled, and Aly Fahmy. 2020. Dataset of breast
ultrasound images. Data in Brief, 28:104863.

[3] Gianni Brauwers and Flavius Frasincar. 2021. A general survey on attention mechanisms in deep
learning. IEEE Transactions on Knowledge and Data Engineering, page 1–1.

[4] Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, and Song Han. 2020. ONCE-FOR-ALL: TRAIN
ONE NETWORK AND SPE- CIALIZE IT FOR EFFICIENT DEPLOYMENT.

[5] François Chollet. 2017. Xception: Deep Learning with Depthwise Separable Convolutions.

[6] Andrew Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand,
and Marco Andreetto. 2017. MobileNets: Efficient Convolutional Neural Networks for Mobile
Vision Applications.

[7] Sergey Ioffe. 2015. Batch Normalization: Accelerating Deep Network Training by Reducing Internal
Covariate Shift.

[8] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig
Adam, and Dmitry Kalenichenko. 2017. Quantization and Training of Neural Networks for
Efficient Integer-Arithmetic-Only Inference.

[9] Jonathan Long, Evan Shelhamer, and Trevor Darrell. 2015. Fully Convolutional Networks for
Semantic Segmentation.

[10] Sachin Mehta and Mohammad Apple. 2022. MOBILEVIT: LIGHT-WEIGHT, GENERAL-PURPOSE,
AND MOBILE-FRIENDLY VISION TRANSFORMER.

[11] Ozan Oktay, Jo Schlemper, Loic Le Folgoc, Matthew Lee, Mattias Heinrich, Kazunari Misawa,
Kensaku Mori, Steven Mcdonagh, Nils Hammerla, Bernhard Kainz, Ben Glocker, and Daniel
Rueckert. 2018. Attention U-Net: Learning Where to Look for the Pancreas.
