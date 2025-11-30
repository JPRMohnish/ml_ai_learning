#forward pass
class Value:
  def __init__(self, value, parents = (), op = '', label = ''):
    self.value = value
    self.grad = 0
    self.parents = set(parents)
    self.operation = op
    self.label = label
    self.backward = lambda:None

  def __repr__(self):
    return f'Value: {self.value}'

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    val = Value(self.value + other.value, parents=(self, other), op='+')
    def backward():
      self.grad += 1.0 * val.grad
      other.grad += 1.0 * val.grad
    val.backward = backward
    return val

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    val = Value(self.value * other.value, parents=(self, other), op= '*')
    def backward():
      self.grad += other.value * val.grad
      other.grad += self.value * val.grad
    val.backward = backward
    return val

  def __rmul__(self, other):
    return self * other

  def __radd__(self, other):
    return self + other

  def __truediv__(self, other):
    return self * other ** -1

  def __neg__(self):
    return self * -1

  def __sub__(self, other):
    return self + (-other)



  def tanh(self):
    x = self.value
    # x must always be a number.
    tan_of_x = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    val =  Value(tan_of_x, parents = (self, ), op = 'tanh')
    def backward():
      self.grad += val.grad * (1 - tan_of_x * tan_of_x)
    val.backward = backward
    return val

  def exp(self):
    x = self.value
    exp_value = Value(math.exp(x), parents = (self, ), op = 'exp')
    def backward():
      self.grad += exp_value.grad * exp_value.value
    exp_value.backward = backward
    return exp_value


  def __pow__(self, other):
    assert (isinstance(other, (int, float)), "only ints ot floats allowed...")
    x = self.value
    power_value = Value(x ** other, parents = (self, ), op = f'**{other}')
    def backward():
      self.grad += power_value.grad * other * (x ** (other - 1))
    power_value.backward = backward
    return power_value


  # dfs backwards to automaticlally apply backward and calucate gradients till inputs
  def backwards(self):
    visited = set() # its a directed graph ideally not needed
    topo_order = []
    def dfs(u):
      # print(u.label, u in visited)
      if u not in visited:
        visited.add(u)
        for p in u.parents:
          dfs(p)
        topo_order.append(u)

    dfs(self)

    self.grad = 1.0
    for node in reversed(topo_order):
      node.backward()
      
class Neuron:
  def __init__(self, num_inputs):
    self.weights = [Value(random.uniform(-0.9999, 0.9999))  for _ in range(num_inputs)]
    self.bias = Value(random.uniform(-0.9999, 0.9999))

  def __call__(self, x):
    # call this neuron on input x;
    # neural network function is w*x +b

    activation = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
    nonLinearity = activation.tanh()

    # then introduce non linearity
    return nonLinearity

  def paramters(self):
    return self.weights + [self.bias]

class Layer:
  def __init__(self, num_inputs, num_neurons):
    self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]

  def __call__(self, x):
    # remember the output o hiddlen layer 1 is an input to the hidden layer 2
    outputs = [neuron(x) for neuron in self.neurons]
    return outputs[0] if len(outputs) == 1 else outputs

  def parameters(self):
    #collate all parameters together.
    return [p for neuron in self.neurons for p in neuron.paramters()]


class MultiLayerPerceptron:
  def __init__(self, num_inputs, neuron_layer_sizes):
    sizes = [num_inputs] + neuron_layer_sizes
    self.layers = [Layer(sizes[i], neuron_layer_sizes[i])for i in range(len(neuron_layer_sizes))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):

    return [p for layer in self.layers for p in layer.parameters()]


def loss_function_mean_square_error(y_expected, y_predicted):
  return sum((y_pred - y_exp) **2 for y_exp, y_pred in zip(y_expected, y_predicted))

train_set_x = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
train_set_y = [1.0, -1.0, -1.0, 1.0]

mlp = MultiLayerPerceptron(len(train_set_x[0]), [4, 4, 1])

num_epochs = 2000;
learning_rate = 0.05
for iteration in range(num_epochs):
  
  y_pred = [mlp(x) for x in train_set_x]
  loss = loss_function_mean_square_error(train_set_y, y_pred)

  #backward pass
  for p in mlp.parameters():
    p.grad = 0.0 # zero grad before each backward.
  loss.backwards()

  #update
  for p in mlp.parameters():
    p.value -= learning_rate * p.grad

  #print
  print(loss, y_pred)

