# LaneSegmentationNetwork
Created by Dongkyu Yu and [Daijin Kim](http://imlab.postech.ac.kr/members_d.htm) at [POSTECH IM Lab](http://imlab.postech.ac.kr)

# Demo
[Youtube](https://www.youtube.com/watch?v=zwaLq1743J0)


![Lane segmentation demo](https://github.com/POSTECH-IMLAB/LaneSegmentationNetwork/blob/master/demo.gif)

# Overview
The semantic segmentaion network for lane segmentation. Which inspired by Google DeepLabV3. We use the [ResNet](https://github.com/KaimingHe/deep-residual-networks) as backbone network for high quailty feature extraction. And we design the module which employ atrous convolution with multi atrous rate which use same filters. It makes not only the network robust to multiple scales but also reduce the number of parameters for filters.

# Performance
We conduct the experiments on [Highway Driving Dataset for Semantic Video Segmentation](https://sites.google.com/site/highwaydrivingdataset/) from KAIST, achieving the test set performance 87.6% mIoU.

# Pre-trained model
You can download the pretrained model of network [here](https://drive.google.com/drive/folders/14TtrNFY94FS1fIDspzg4ZRPCFT5OXujc?usp=sharing).

# Acknowledgements
This work was supported by Institute for Information & communications Technology Promotion (IITP) grant funded by the Korea government (MSIP)(2014-0-00059, Development of Predictive Visual Intelligence Technology), MSIP (Ministry of Science, ICT and Future Planning), Korea, under the ICT Consilience Creative Program (IITP-R0346-16-1007) supervised by the IITP, and MSIP(Ministry of Science, ICT and Future Planning), Korea, under the ITRC (Information Technology Research Center) support program (IITP-2017-2016-0-00464) supervised by the IITP.
