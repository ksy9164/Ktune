# KTUNE
  
### #
  
KTUNE is machine learning wrapper for ManyCore CPU which motivated by Keras
  
------------------------------------
# Compile
  
````
$ mv ./src
$ export LD_PRELOAD=libmkl_core.so:libmkl_sequential.so
$ make
$ mv Ktune.so ./../
````
-----------------------------------
# How to use?
  
-------------------
  
````
$ python3
>>>> import Ktune
>>>> t = Ktune.Sequential()
>>>> t.help()
  
You can use .help() to refer to it.
  
here is the function of Ktune
  
1. network
This is a function that determines what kind of network to use
 .networkd(kind of network)
example : .network(fully connected)
  
2. add
This is a function that add layer
 .add(number of layer , 1st layer number(input layer) , 2nd layer number ...., output layer number)
example : .add(4, 784 , 120 , 40 , 10)
  
3. action
This is a function that setting action function
 .action(action function name)
example : .action(sigmoid)
  
4. loss
This is a function that setting loss function
 .loss(loss function name)
example : .loss(mean_squared_error)
  
5. optimizer
This is a function that setting optimizer function
 .optimizer(optimizer function name)
example : .optimizerz(sgd)
  
6. data
This is a function that setting determines what kind of data to use
 .data(data name)
example : .data(mnist)
  
7. fit
This is a function that training data
 .fit(Batch_size,Epoch,Learning_rate)
example : .fit(5,100,0.8)
  
````
-------------------------------------------
# requirement
  
##### Intel C++ Compile ( icc )
  
##### MKL library
  
  
Our Execution environment is intel® Xeon Phi™ Processor 7210 (16GB, 1.30 GHz, 64 core)
  
-----------------------------------------------
# Paper
  
We published Paper whith Ktune you can find our Paper in this site (korean institute of information scientists and engineers)
http://www.dbpia.co.kr/Article/NODE07322653
  
