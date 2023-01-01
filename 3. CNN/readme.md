
# 3. CNN
In this time, I re-produced the paper \[[Convolutional Neural Networks for Sentence Classification\]](https://arxiv.org/abs/1408.5882), from Yoon Kim. 2014.  
It tried to build text-classification model, with the **Single CNN** layer and **Smallest, Inevitable** gadgets for the model.  
My re-implementation is based on **[PyTorch](https://pytorch.org/)**.
### What is CNN?

### How can we use CNN, for Text Classification model?

### Re-Implementation Details
- In getting/preprocessing data, I adjusted one of it's [*PyTorch* implementation code](https://github.com/harvardnlp/sent-conv-torch.git).  
- Most of the setting followed original paper, and the only different part is **initialization** of **CNN/FC** layers. In my experiment, I found that using *"He"*  initialization(module *kaiming* in Torch) in **CNN** layer and *"Xavier"* initialization(*default*) in FC layer performs better, so I used that setting. But, it may not be Golden-Truth, since I only tested in **TREC dataset**, with only a few candidates.
- If you wanna imporove the model more, you might like its [deeper experiments](https://arxiv.org/abs/1510.03820), from Zhang and Wallace. 2016.
