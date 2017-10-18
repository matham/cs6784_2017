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

