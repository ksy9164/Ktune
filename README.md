Ktune 
you can read paper in this site (korean institute of information scientists and engineers)
http://www.dbpia.co.kr/Article/NODE07322653


KTUNE is optimized in Manycore CPU for machinelearning!
It has a structure similar to keras
you can use it in python.

How can I start??
-> like this!!

# import Ktune
# t = Ktune.Sequential()
# t.help()


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
