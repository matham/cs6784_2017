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

