10/9-10/10
prajna
c50871c7c1fd49eccd1bbb596408433c2215d85f
cifar100-train1
(pytorch) root@cdbbf1597042:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 | tee log

We trained cifar100 on the whole dataset. Model was not saved. Ran for 300 epochs. Took about 24hrs.

----------------------
10/12
prajna
c354337afbb39cdeba409abd080f36f8b040403b
cifar100-transfer1
(pytorch) root@cdbbf1597042:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans | tee log


We split cifar 100 into 50/50 classes. Trained on set A 175 epochs, then transfered to B by replacing the FC layer at the end and fine tuning the other layers. Ran for 100 epochs. LR for FC was .1, .01, .001 for 50, 25, 25 epochs. For the rest of the network it was .01, .01, .001.

---------------------------
10/16
prajna
bbb592e5c33afdf37ccf80e3dce2fcfea1f4f026
cifar100-transfer-finetune-blocks
(pytorch) root@cdbbf1597042:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --transBlocks | tee log

50/50 random split. Train on A, then fine tune on B. We fined tuned on B by either resetting dense blocks 3, 2-3, 1-3 or resetting just the final fc layer.

---------------------------
10/16
ayons PC
d9417cd5c3e07f50cc1e348c791b12321b89f7e6
cifar100-transfer-finetune-blocks-nat_vs._unnat
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --transBlocks --transNatSplit | tee log

65/35 natural vs human made split. Train on A, then fine tune on B. We fined tuned on B by either resetting dense blocks 3, 2-3, 1-3 or resetting just the final fc layer.

------------------------------
10/18
ayons PC
6242139ebcd546b9db803f00c6610cf06c2d8862
cifar100-transfer-finetune-blocks-65-35_split
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --transBlocks --transSplit 65 | tee log

65/35 random split. Train on A, then fine tune on B. We fined tuned on B by either resetting dense blocks 3, 2-3, 1-3 or resetting just the final fc layer.

-------------------------------
10/18
prajna
2871f954be2deec5694b754d0fccaa01d544ac8d
cifar100-transfer-finetune-blocks2
(pytorch) root@cdbbf1597042:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --transBlocks | tee log

Run again 50/50 random split. Train on A, then fine tune on B. We fined tuned on B by either resetting dense blocks 3, 2-3, 1-3 or resetting just the final fc layer.

-------------------------------
10/19
ayons PC
159a22399bbdd2c75f419fdcddde714dd5e105bc
cifar100-transfer-finetune-layers
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --nTransFTBlockLayersStep 2 --transFTBlock 2 --classes /root/Desktop/shared_docker/cs6784_2017/class_shuffled --preTrainedModel /root/Desktop/shared_docker/cs6784_2017/model_cifar100_base.t7 --transSplit 65 | tee log

65/35 random split. Train on A, then fine tune on B. We fined tuned on B by resetting the layers from dense block 2 starting from layer n for n in range(0, 16, 2). Where 16 is the number of layers in the block. The baseline model came from cifar100-transfer-finetune-blocks-65-35_split.

-------------------------------
10/19
prajna
3d34892b54775f61da2941ee49ddd7ac0d8b9c93
cifar100-transfer-finetune-layers2
(pytorch) root@cdbbf1597042:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --nTransFTBlockLayersStep 2 --transFTBlock 2 --classes /root/Desktop/shared-docker/python/cifar/class_shuffled --preTrainedModel /root/Desktop/shared-docker/python/cifar/model_cifar100_base.t7 | tee log

50/50 random split. Train on A, then fine tune on B. We fined tuned on B by resetting the layers from dense block 2 starting from layer n for n in range(0, 16, 2). Where 16 is the number of layers in the block. The baseline model came from cifar100-transfer-finetune-blocks2.

-------------------------------
10/23
prajna
87c3c55e925c7a8e97aaac35ee0dbbd5264e8462
cifar100-transfer-binary-classifier
(pytorch) root@cdbbf1597042:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --binClasses 1 --binWeight .9 | tee log

50/50 random split. Train on A by adding a single binary classifier with loss weight .9 and the cifar FC layer with weight .1.

-------------------------------
10/23
ayons PC
87c3c55e925c7a8e97aaac35ee0dbbd5264e8462
cifar100-transfer-binary-classifier2
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --binClasses 1 --binWeight 1 | tee log

50/50 random split. Train on A by adding a single binary classifier with no FC layer.

--------------------------------
10/25
Prajna
bf219f9a9ef204277fab08f54aeeac4b1aafc02c
cifar100-transfer-binary-classifier1.1
(pytorch) root@cdbbf1597042:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --classes /root/Desktop/shared-docker/python/cifar/class_shuffled --preTrainedModel /root/Desktop/shared-docker/python/cifar/model_cifar100_base.t7 --binWeight .9 --binClasses 1 | tee log

Fine-tuned on B the model create in cifar100-transfer-binary-classifier on A.

---------------------------------
10/25
Ayons PC
e7c5deebb0d1280df47cd98d12ba55d402cf1009
cifar100-transfer-binary-classifier2.2
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --classes /root/Desktop/shared_docker/cs6784_2017/class_shuffled --preTrainedModel /root/Desktop/shared_docker/cs6784_2017/model_cifar100_base.t7 --binClasses 1 --binWeight 1 | tee log

Fine-tuned on B the model create in cifar100-transfer-binary-classifier2 on A.

----------------------------------
10/26
Prajna
bfbe30ac1c6d815dff4ad859146ab784ec37f650
cifar100-transfer-binary-classifier3
(pytorch) root@cdbbf1597042:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --binWeight .67 --binClasses 1 | tee log

50/50 random split. Train on A by adding a single binary classifier with loss weight .67 and a FC layer then fine tune on B.

----------------------------------
10/26
Ayons PC
bfbe30ac1c6d815dff4ad859146ab784ec37f650
cifar100-transfer-binary-classifier4
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --binClasses 1 --binWeight .85 | tee log

50/50 random split. Train on A by adding a single binary classifier with loss weight .85 and a FC layer then fine tune on B.

----------------------------------
10/26
Prajna
bfbe30ac1c6d815dff4ad859146ab784ec37f650
cifar100-transfer-binary-classifier5
(pytorch) root@cdbbf1597042:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --binWeight .67 --binClasses 4 | tee log

50/50 random split. Train on A by adding 4 binary classifiers with loss weight .67 and a FC layer then fine tune on B.

----------------------------------
10/26
Ayons PC
bfbe30ac1c6d815dff4ad859146ab784ec37f650
cifar100-transfer-binary-classifier6
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --binClasses 1 --binWeight .4 | tee log

50/50 random split. Train on A by adding 1 binary classifier with loss weight .4 and a FC layer then fine tune on B.

----------------------------------
10/27
Prajna
0c30bc257f2b74e4753fd8d482e49ca44b09edd7
cifar100-transfer-binary-classifier7
(pytorch) root@cdbbf1597042:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --binWeight .1 --binClasses 1 | tee log

50/50 random split. Train on A by adding 1 binary classifier with loss weight .1 and a FC layer then fine tune on B.

----------------------------------
10/27
Ayons PC
0c30bc257f2b74e4753fd8d482e49ca44b09edd7
cifar100-transfer-binary-classifier8
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --binClasses 4 --binWeight .2 | tee log

50/50 random split. Train on A by adding 4 binary classifiers with loss weight .2 and a FC layer then fine tune on B.

----------------------------------
10/28
Prajna
0c30bc257f2b74e4753fd8d482e49ca44b09edd7
cifar100-transfer-binary-classifier9
(pytorch) root@cdbbf1597042:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --binWeight .2 --binClasses 1 | tee log

50/50 random split. Train on A by adding 1 binary classifier with loss weight .2 and a FC layer then fine tune on B.

----------------------------------
10/27
Ayons PC
0c30bc257f2b74e4753fd8d482e49ca44b09edd7
cifar100-transfer-binary-classifier10
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --binClasses 10 --binWeight .4 | tee log

50/50 random split. Train on A by adding 10 binary classifiers with loss weight .4 and a FC layer then fine tune on B.

----------------------------------
10/29
Prajna
cc68dc2f23820e6ff705c50c01da0929eaa5445d
imagenet100-transfer
158
(pytorch) root@fc95ae5d06e8:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data/imagenet --save results --imagenet --trans --batchSz 45 | tee log

imagenet 100 classes, 50/50 random split. Train on A and fine tune on B by resetting just the last classifier layer.

----------------------------------
10/29
Ayons PC
cc68dc2f23820e6ff705c50c01da0929eaa5445d
cifar100-transfer-binary-classifier-maml1
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --binClasses 1 --binWeight .4 --maml --batchSz 50 | tee log

50/50 random split. Train on A by adding 1 binary classifier with loss weight .4 and a FC layer using maml then fine tune on B.

-------------------------------
11/3
Ayons PC
acbc4eece8c2fb48f08b25c2f690b555ac2d7967
cifar100-transfer-finetune-layers-blocks
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --transBlocks --nTransFTBlockLayersStep 2 --transFTBlock 2 --classes /root/Desktop/shared_docker/cs6784_2017/class_shuffled --preTrainedModel /root/Desktop/shared_docker/cs6784_2017/model_cifar100_base.t7 --binClasses 1 --binWeight .4 --noRetrainAll | tee log

Fine tune on B. We fined tuned on B by resetting the layers from dense block 2 starting from layer n for n in range(0, 16, 2). Where 16 is the number of layers in the block. As well as on blocks 1-2. The baseline model came from cifar100-transfer-binary-classifier6.

-------------------------------
11/5
Ayons PC
709338192300767a42e88284abcf1f5168ef6e8d
cifar100-transfer-finetune-blocks2
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --classes /root/Desktop/shared_docker/cs6784_2017/class_shuffled --preTrainedModel /root/Desktop/shared_docker/cs6784_2017/model_cifar100_base.t7 --binClasses 1 --binWeight .4  --limitTransClsSize 60 | tee log

Fine tune on B by resetting only the last layer and reducing the training set size to 60 images per class. The baseline model came from cifar100-transfer-binary-classifier6.

-------------------------------
11/5
Ayons PC
709338192300767a42e88284abcf1f5168ef6e8d
cifar100-transfer-reduced-examples
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --classes /root/Desktop/shared_docker/cs6784_2017/class_shuffled --preTrainedModel /root/Desktop/shared_docker/cs6784_2017/model_cifar100_base.t7  --limitTransClsSize 60 | tee log

Fine tune on B by resetting only the last layer and reducing the training set size to 60 images per class. The baseline model came from cifar100-transfer-finetune-blocks2.

----------------------------------
11/3
EC2
709338192300767a42e88284abcf1f5168ef6e8d
cifar100-transfer-binary-classifier-maml2
python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --binClasses 4 --binWeight .4 --maml --batchSz 50 | tee log

50/50 random split. Train on A by adding 4 binary classifiers with loss weight .4 and a FC layer using maml then fine tune on B.

----------------------------------
11/3
Prajna
709338192300767a42e88284abcf1f5168ef6e8d
imagenet100-transfer-binary2
172
(pytorch) root@fc95ae5d06e8:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data/imagenet --save results --imagenet --trans --batchSz 45 --binWeight .4 --binClasses 1 | tee log

imagenet 100 classes, 50/50 random split. Train on A by adding 1 binary classifier with loss weight .4 and a FC layer and fine tune on B.

-------------------------------
11/9
Ayons PC
de45929d2ab71885ecb80e384a9802da7ebc2912
svhn-transfer-binary
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --classes /root/Desktop/shared_docker/cs6784_2017/class_shuffled --preTrainedModel /root/Desktop/shared_docker/cs6784_2017/model_cifar100_base.t7 --binClasses 1 --binWeight .4 --ftSVHN /root/Desktop/data/svhn | tee log

Fine tune on SVHN by resetting only the last layer and training/testing on svhn. The baseline model came from cifar100-transfer-binary-classifier6.

-------------------------------
11/9
Prajna
de45929d2ab71885ecb80e384a9802da7ebc2912
svhn-transfer
(pytorch) root@fc95ae5d06e8:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data/ --save results --cifar 100  --trans --classes class_shuffled --preTrainedModel model_cifar100_base.t7 --ftSVHN /root/Desktop/data/svhn | tee log

Fine tune on SVHN by resetting only the last layer and training/testing on svhn. The baseline model came from cifar100-transfer-finetune-blocks2.

-------------------------------
11/9
Ayons PC
b5d64f95c2fcbfc3ebad54e85a9f2255ca4659ce
cifar10-transfer-binary
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data/ --save results --cifar 100  --trans --binWeight .4 --binClasses 1 --classes class_shuffled --preTrainedModel model_cifar100_base.t7 --ftCIFAR10 | tee log

Fine tune on cifar10 by resetting only the last layer and training/testing on cifar10. The baseline model came from cifar100-transfer-binary-classifier6.

-------------------------------
11/9
Prajna
b5d64f95c2fcbfc3ebad54e85a9f2255ca4659ce
cifar10-transfer
(pytorch) root@fc95ae5d06e8:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data/ --save results --cifar 100  --trans --classes class_shuffled --preTrainedModel model_cifar100_base.t7 --ftCIFAR10 | tee log

Fine tune on cifar10 by resetting only the last layer and training/testing on cifar10. The baseline model came from cifar100-transfer-finetune-blocks2.

----------------------------------
11/14
Ayons PC
fbf58e60ef9da912113c069c5d496d1bda85e1f1
cifar100-transfer-binary-classifier11-lin
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --binClasses 4 --binWeight .4 --binWeightDecay | tee log

50/50 random split. Train on A by adding 4 binary classifiers with loss weight .4 and a FC layer then fine tune on B. The 4 units were linearly scaled.

----------------------------------
11/15
Ayons PC
fbf58e60ef9da912113c069c5d496d1bda85e1f1
cifar100-transfer-binary-classifier12-lin
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --binClasses 2 --binWeight .4 --binWeightDecay | tee log

50/50 random split. Train on A by adding 2 binary classifiers with loss weight .4 and a FC layer then fine tune on B. The 4 units were linearly scaled.

-------------------------------
11/14
Prajna
fbf58e60ef9da912113c069c5d496d1bda85e1f1
inat-rand-transfer
(pytorch) root@fc95ae5d06e8:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data/imagenet --save results --imagenet --trans --batchSz 45 --classes class_shuffled --preTrainedModel model_cifar10_base.t7 --ftINat /root/Desktop/data/inat --limitTransClsSize 600| tee log

Fine tune on inat by resetting only the last layer and training/testing on inat. The baseline model came from imagenet100-transfer. inat was randomly extracted to 50 classes, 600/class train and 50/class test.

-------------------------------
11/16
Prajna
fbf58e60ef9da912113c069c5d496d1bda85e1f1
inat-transfer-binary
(pytorch) root@fc95ae5d06e8:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data/imagenet --save results --imagenet  --trans --batchSz 45 --classes class_shuffled --preTrainedModel model_cifar10_base.t7 --ftINat /root/Desktop/shared-docker/python/cifar/inat-mini --binWeight .4 --binClasses 1 | tee log

Fine tune on inat by resetting only the last layer and training/testing on inat. The baseline model came from imagenet100-transfer-binary2. inat was inat-mini, 50 classes, 600/class train and 50/class test.

----------------------------------
11/16
Ayons PC
7b4fea52833a2ad0ab06aa57ad41cf3095b0d331
cifar100-binary-drop
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --binClasses 1 --binWeight .4 --dropBinaryAt 125 | tee log

50/50 random split. Train on A by adding 4 binary classifiers with loss weight .4 and a FC layer then fine tune on B. AFter 125 epochs on A, the binary loss was dropped for the remainder of training on A (and B of course).

-------------------------------
11/17
Prajna
fbf58e60ef9da912113c069c5d496d1bda85e1f1
inat-transfer
(pytorch) root@fc95ae5d06e8:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data/imagenet --save results --imagenet  --trans --batchSz 45 --classes class_shuffled --preTrainedModel model_cifar10_base.t7 --ftINat /root/Desktop/shared-docker/python/cifar/inat-mini | tee log

Fine tune on inat by resetting only the last layer and training/testing on inat. The baseline model came from imagenet100-transfer. inat was inat-mini, 50 classes, 600/class train and 50/class test.

----------------------------------
11/17
Ayons PC
7b4fea52833a2ad0ab06aa57ad41cf3095b0d331
cifar100-baseline
139
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans | tee log

50/50 random split. Train on A by adding a FC layer then fine tune on B.

-------------------------------
11/17
Prajna
7b4fea52833a2ad0ab06aa57ad41cf3095b0d331
inat10-transfer
(pytorch) root@fc95ae5d06e8:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data/imagenet --save results --imagenet  --trans --batchSz 45 --classes class_shuffled --preTrainedModel model_cifar10_base.t7 --ftINat /root/Desktop/data/inat/train --limitTransClsSize 1200 --ftCopySubset /root/Desktop/shared-docker/python/cifar/inat-mini10 --inatNClasses 10 | tee log

Fine tune on inat by resetting only the last layer and training/testing on inat. The baseline model came from imagenet100-transfer. inat was inat-mini, 10 classes, 1200/class train and 50/class test.

----------------------------------
11/17
Ayons PC
7b4fea52833a2ad0ab06aa57ad41cf3095b0d331
cifar100-transfer-binary-classifier13
165
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --binClasses 1 --binWeight .4 | tee log

50/50 random split. Train on A by adding 1 binary classifier with loss weight .4 and a FC layer then fine tune on B.

-------------------------------
11/17
Prajna
7b4fea52833a2ad0ab06aa57ad41cf3095b0d331
inat10-transfer-binary
(pytorch) root@fc95ae5d06e8:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data/imagenet --save results --imagenet  --trans --batchSz 45 --classes class_shuffled --preTrainedModel model_cifar10_base.t7 --ftINat /root/Desktop/shared-docker/python/cifar/inat-mini10 --binWeight .4 --binClasses 1 | tee log

Fine tune on inat by resetting only the last layer and training/testing on inat. The baseline model came from imagenet100-transfer-binary2. inat was inat-mini, 10 classes, 1200/class train and 50/class test.

----------------------------------
11/18
Prajna
61842fd9fa3f6aab4972d77cf982ba76c8483440
imagenet20-transfer
(pytorch) root@fc95ae5d06e8:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data/imagenet --save results --imagenet --imgnetNClasses 10 --trans --batchSz 45 --transSplit 10

imagenet 20 classes, 50/50 random split. Train on A and fine tune on B by resetting just the last classifier layer.

----------------------------------
11/19
Prajna
61842fd9fa3f6aab4972d77cf982ba76c8483440
imagenet20-transfer-binary
(pytorch) root@fc95ae5d06e8:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data/imagenet --save results --imagenet --imgnetNClasses 10 --trans --batchSz 45 --transSplit 10 --classes class_shuffled --binClasses 1 --binWeight .4 | tee log

imagenet 20 classes, 50/50 random split. Train on A by adding 1 binary classifier with loss weight .4 and a FC layer then fine tune on B.

----------------------------------
11/17
Ayons PC
61842fd9fa3f6aab4972d77cf982ba76c8483440
cifar100-baseline2
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans | tee log

50/50 random split. Train on A by adding a FC layer then fine tune on B.

----------------------------------
11/17
Ayons PC
61842fd9fa3f6aab4972d77cf982ba76c8483440
cifar100-baseline3
153
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans | tee log

50/50 random split. Train on A by adding a FC layer then fine tune on B.

----------------------------------
11/17
Ayons PC
61842fd9fa3f6aab4972d77cf982ba76c8483440
cifar100-transfer-binary-classifier14
135
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --binClasses 1 --binWeight .4 | tee log

50/50 random split. Train on A by adding 1 binary classifier with loss weight .4 and a FC layer then fine tune on B.

----------------------------------
11/22
Prajna
c3ea97504f4ccce60a0bcb7c12483b327ddbd1ae
tiny-imagenet-transfer
(pytorch) root@fc95ae5d06e8:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data/tiny-imagenet --save results --tinyImagenet --trans --batchSz 32 | tee log

100 out of the 200 tiny imagenet classes. 50/50 random split. Train on A then fine tune on B.

----------------------------------
11/23
Prajna
c3ea97504f4ccce60a0bcb7c12483b327ddbd1ae
tiny-imagenet-transfer-binary
(pytorch) root@fc95ae5d06e8:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data/tiny-imagenet --save results --tinyImagenet --trans --batchSz 32 --binClasses 1 --binWeight .4 | tee log

100 out of the 200 tiny imagenet classes. 50/50 random split. Train on A by adding 1 binary classifier with loss weight .4 and a FC layer then fine tune on B.

----------------------------------
11/24
Prajna
c3ea97504f4ccce60a0bcb7c12483b327ddbd1ae
tiny-imagenet-transfer-binary2
(pytorch) root@fc95ae5d06e8:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data/tiny-imagenet --save results --tinyImagenet --trans --batchSz 32 --binClasses 1 --binWeight .4 --classes class_shuffled | tee log

100 out of the 200 tiny imagenet classes. 50/50 random split. Train on A by adding 1 binary classifier with loss weight .4 and a FC layer then fine tune on B. Classes came from tiny-imagenet-transfer.

----------------------------------
11/27
Prajna
9b6e8d14ceb7b2c683effea2e3d7afcec7cb1c25
wrn-baseline
111
(pytorch) root@fc95ae5d06e8:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot/root/Desktop/data --save results --cifar100 --trans --wrn --nEpochs112 --nFTEpochs71 | tee log

Cifar 100 on wide resnets. 50/50 random split. Train on A then fine tune on B. We ran, stopped and resumed to reduce the number of epochs as it was converging earlier and took too long.

----------------------------------
11/27
Ayons PC
fcbc39d3daf8f6e642aa6919f199870534326515
results_v2
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train_v2.py | tee log

Baseline run, model for B was last of A. Binary layer was always present, just unused sometimes. Verification run using new code on cifar 100 50/50 split with and without binary classifier with weight .4.

----------------------------------
11/28
Ayons PC
fcbc39d3daf8f6e642aa6919f199870534326515
results_v2-best
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train_v2.py | tee log

Baseline run, model for B was best of A. Binary layer only present during binary training. Verification run using new code on cifar 100 50/50 split with and without binary classifier with weight .4. Classes are the same as results_v2.

----------------------------------
11/28
Ayons PC
3a2133d31d6b38efe3eb1a8d06d29786b7971750
results_v2-last2
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train_v2.py | tee log

Baseline run, model for B was last of A. Binary layer only present during binary training. Verification run using new code on cifar 100 50/50 split with and without binary classifier with weight .4. Classes were newly generated and not the same as results_v2.

----------------------------------
11/28
Prajna
9b6e8d14ceb7b2c683effea2e3d7afcec7cb1c25
wrn-binary
99
(pytorch) root@fc95ae5d06e8:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --wrn --nEpochs 112 --nFTEpochs 71 --binClasses 1 --binWeight .4 | tee log

Cifar 100 on wide resnets. 50/50 random split. Train on A by adding 1 binary classifier with loss weight .4 and a FC layer then fine tune on B.

----------------------------------
11/29
Prajna
9b6e8d14ceb7b2c683effea2e3d7afcec7cb1c25
cifar100-transfer-binary-classifier15
166
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --binClasses 1 --binWeight .4 --classes class_shuffled | tee log

50/50 random split. Train on A by adding 1 binary classifier with loss weight .4 and a FC layer then fine tune on B. classes is from cifar100-baseline.

----------------------------------
11/30
Prajna
9b6e8d14ceb7b2c683effea2e3d7afcec7cb1c25
cifar100-transfer-binary-classifier16
143
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --binClasses 1 --binWeight .4 --classes class_shuffled | tee log

50/50 random split. Train on A by adding 1 binary classifier with loss weight .4 and a FC layer then fine tune on B. classes is from cifar100-baseline3.

----------------------------------
11/30
Ayons PC
8838953601461d07953bcb4a36fbcc43be37aace
cifar100-transfer-binary-classifier17
148
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --binClasses 1 --binWeight .4 --classes class_shuffled | tee log

50/50 random split. Train on A by adding 1 binary classifier with loss weight .4 and a FC layer then fine tune on B. classes is from cifar100-baseline3, number of fc units was reduced from 200 to 100.

----------------------------------
11/30
Prajna
35ba154eb7fb6c6917a865c7777722c21422976f
cifar100-transfer-binary-classifier18
160
(pytorch) root@fc95ae5d06e8:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --binClasses 1 --binWeight .4 --classes class_shuffled | tee log

50/50 random split. Train on A by adding 1 binary classifier with loss weight .4 and a FC layer then fine tune on B. classes is from cifar100-transfer-binary-classifier14.

----------------------------------
11/28
Ayons PC
5341ae8b6023aa2b164a3899c8f79d87221d7037
results_v2-135
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train_v2.py | tee log

Baseline/binary run, model for B was of A at 135 epochs. Classes were newly generated and not the same as results_v2.

----------------------------------
11/30
Ayons PC
5341ae8b6023aa2b164a3899c8f79d87221d7037
cifar100-transfer-binary-classifier19
(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --binClasses 1 --binWeight .4 --classes class_shuffled --nEpochs 130 | tee log

50/50 random split. Train on A by adding 1 binary classifier with loss weight .4 and a FC layer then fine tune on B. classes is from cifar100-baseline3, only 130 epochs used.

----------------------------------
12/2
Prajna
5341ae8b6023aa2b164a3899c8f79d87221d7037
wrn-binary2
99
(pytorch) root@fc95ae5d06e8:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --wrn --nEpochs 112 --nFTEpochs 71 --binClasses 1 --binWeight .4 --classes class_shuffled | tee log

Cifar 100 on wide resnets. 50/50 split. Train on A by adding 1 binary classifier with loss weight .4 and a FC layer then fine tune on B. classes from cifar100-baseline3.

----------------------------------
12/2
Prajna
5341ae8b6023aa2b164a3899c8f79d87221d7037
wrn-baseline2
99
(pytorch) root@fc95ae5d06e8:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --wrn --nEpochs 112 --nFTEpochs 71 --classes class_shuffled | tee log

Cifar 100 on wide resnets. 50/50 split. Train on A then fine tune on B. classes from cifar100-baseline3.

----------------------------------
12/4
Prajna
5341ae8b6023aa2b164a3899c8f79d87221d7037
wrn-baseline3
99
(pytorch) root@fc95ae5d06e8:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --wrn --nEpochs 112 --nFTEpochs 71 --classes class_shuffled --seed 246 | tee log

Cifar 100 on wide resnets. 50/50 split. Train on A then fine tune on B. classes from wrn-baseline.

----------------------------------
12/4
Ayons PC
5341ae8b6023aa2b164a3899c8f79d87221d7037
cifar100-baseline4

(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --classes class_shuffled --seed 547 | tee log

50/50 random split. Train on A by adding a FC layer then fine tune on B. classes from cifar100-transfer-binary-classifier13

----------------------------------
12/4
Ayons PC
5341ae8b6023aa2b164a3899c8f79d87221d7037
cifar100-baseline5

(pytorch) root@63a8c9378964:~/Desktop/shared_docker/cs6784_2017# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --classes class_shuffled --seed 285 | tee log

50/50 random split. Train on A by adding a FC layer then fine tune on B. classes from cifar100-transfer-binary-classifier14

----------------------------------
12/6
Prajna
5341ae8b6023aa2b164a3899c8f79d87221d7037
wrn-baseline4

(pytorch) root@fc95ae5d06e8:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans --wrn --nEpochs 112 --nFTEpochs 71 --classes class_shuffled --seed 682 | tee log

Cifar 100 on wide resnets. 50/50 split. Train on A then fine tune on B. classes from wrn-binary.

