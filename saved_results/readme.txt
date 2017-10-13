10/9-10/10

---------------------
cifar100-train1
c50871c7c1fd49eccd1bbb596408433c2215d85f
(pytorch) root@cdbbf1597042:~/Desktop/shared-docker/python/cifar# python train.py --dataRoot /root/Desktop/data --save results --cifar 100 | tee log

We trained cifar100 on the whole dataset. Model was not saved. Ran for 300 epochs. Took about 24hrs.

----------------------
10/10
8aa1730df7659e8717c28afcdccf91849571e7ce
python train.py --dataRoot /root/Desktop/data --save results --cifar 100 --trans | tee log

