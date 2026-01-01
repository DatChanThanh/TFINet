# TFINet

With the growing demand for efficient spectrum utilization in integrated radar-communication (IRC) systems, driven by Internet-of-Things (IoT) and fifth-generation (5G) advancements, robust waveform classification techniques have become increasingly critical. This paper introduces TFINet, a cutting-edge deep learning (DL) architecture designed for waveform classification in spectrally congested environments. TFINet leverages time-frequency representations (TFRs) derived from the Smoothing Pseudo-Wigner-Ville Distribution (SPWVD) to improve feature quality and mitigate cross-term interference, enhancing classification accuracy. The network incorporates two key modules: the Dual-Temporal Frequency Extraction (DTFE) and Time-Frequency Selective Downsampling (TFSD). The DTFE module improves feature extraction by decoupling time and frequency features through dual-branch processing, while the TFSD module intelligently reduces dimensionality, preserving essential features without compromising performance. These innovations enable TFINet to balance computational efficiency and classification accuracy, enhancing its suitability for resource-constrained edge devices. On a diverse synthetic dataset of 12 waveform types, TFINet achieves $91.38\%$ overall classification accuracy with 59K parameters and 0.328 ms inference time. Compared to existing deep models, TFINet demonstrates superior performance in both accuracy and efficiency, validating its suitability for practical IRC systems.

![IRC systems](https://github.com/DatChanThanh/TFINet/blob/9d013a21558d37e2854a44b5b7fc743399377cef/IRC_system.png)

![Architecture](https://github.com/DatChanThanh/TFINet/blob/092a5741cf227248ceb1709a777d540a91fdb9b2/architecture.png)

The dataset can be download on [Google Drive](https://drive.google.com/drive/u/1/folders/14DWPcBVMQ7CrNo13gEoDppv6Rn0Ze0c3) (please report if not available).

 If there is any error or need to be discussed, please email to [Thanh-Dat Tran](https://github.com/DatChanThanh) via [trandatt21@gmail.com](mailto:trandatt21@gmail.com).
