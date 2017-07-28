import Ktune

t = Ktune.Sequential()
t.network("fully connected")
t.layersize(3)
t.add(1)
t.add(2)
t.add(2)
t.action("reg")
t.loss("mean_squared_error")
t.optimizer("sgd")
t.data("mnist")
