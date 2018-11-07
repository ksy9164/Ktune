import Ktune
t = Ktune.Sequential()
t.network("fully connected")
t.layersize(3)
t.add(3072)
t.add(200)
t.add(10)
t.action("sigmoid")
t.loss("mean_squared_error")
t.optimizer("sgd")
t.data("cifar")
t.show();
t.fit(64,1,0.1);
