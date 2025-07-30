# TiFiNet
Accurate signal classification is a key requirement for efficient spectrum sharing in integrated radar-communication systems, yet it is challenged by complex, non-stationary signal characteristics. In this work, we introduce TiFiNet, a lightweight deep neural network designed for robust waveform classification in such environments. Our approach leverages the smooth pseudo Wigner-Ville distribution to create high-resolution time-frequency inputs that capture meaningful signal features even under noisy conditions. The core of the TiFiNet architecture includes a dual-branch structure that refines informative time and frequency features, an attention mechanism that focuses on the most salient spectral regions, and a selective downsampling module that ensures computational efficiency. Evaluated on a diverse synthetic dataset with $12$ waveform types including radar and communications, TiFiNet demonstrates a superior balance of performance and efficiency, achieving $91.38\%$ accuracy with an inference time of $0.328$ ms and a model size of only $59$K parameters. These results show that     TiFiNet outperforms contemporary models, accordingly highlighting its suitability for advanced vehicular technology, including autonomous driving and Vehicle-to-Everything communication systems, where both high performance and efficiency on resource-constrained platforms are essential.

![](https://github.com/DatChanThanh/TiFiNet/blob/f30c4467c1d2650601e06a758f5908d36c3c4e3a/IRC_system.png)

![](https://github.com/DatChanThanh/TiFiNet/blob/c67d23808aa366ea217d679c75a1f493b4c40dfd/architecture.png)

The dataset can be download on [Google Drive](https://drive.google.com/drive/u/1/folders/14DWPcBVMQ7CrNo13gEoDppv6Rn0Ze0c3) (please report if not available)

