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

