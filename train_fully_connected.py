from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable
from torch import optim
from torch.utils.tensorboard import SummaryWriter

summary_writer = SummaryWriter('logs', flush_secs=5)


device = torch.device('cuda')


iris = datasets.load_iris()  # datasets are stored in a dictionary containing an array of features and targets
iris.keys()

labels = iris['target']
preprocessed_features = (iris['data'] - iris['data'].mean(axis=0)) / iris['data'].std(axis=0)
# train_test_split takes care of the shuffling and splitting process
train_features, test_features, train_labels, test_labels = train_test_split(preprocessed_features, labels, test_size=1/3)

features = {
    'train': torch.tensor(train_features, dtype=torch.float32),
    'test': torch.tensor(test_features, dtype=torch.float32),
}

features["train"] = features["train"].to(device)
features["test"] = features["test"].to(device)

labels = {
    'train': torch.tensor(train_labels, dtype=torch.long),
    'test': torch.tensor(test_labels, dtype=torch.long),
}

labels["train"] = labels["train"].to(device)
labels["test"] = labels["test"].to(device)

class MLP(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_layer_size: int,
                 output_size: int,
                 activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_layer_size)
        self.l2 = nn.Linear(hidden_layer_size, output_size)
        self.activation_fn = activation_fn
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.l1(inputs)
        x = self.activation_fn(x)
        x = self.l2(x)
        return x
    

feature_count = 4
hidden_layer_size = 100
class_count = 3
# model = MLP(feature_count, hidden_layer_size, class_count)

# logits = model.forward(features['train'])

# loss_function = nn.CrossEntropyLoss()

# loss = loss_function(logits, labels['train'])

# loss.backward()

def accuracy(probs: torch.FloatTensor, targets: torch.LongTensor) -> float:
  ## First work out which class has been predicted for each data sample. Hint: use argmax
  predicted_class = torch.argmax(probs, dim = 1)
  ## Second count how many of these are correctly predicted
  wrongly_predicted = torch.count_nonzero(predicted_class - targets, dim = 0).item()
  correctly_predicted = targets.size()[0] - wrongly_predicted
  ## Finally return the accuracy, i.e. the percentage of samples correctly predicted
  accuracy = correctly_predicted / targets.size()[0]
  return accuracy
  #print(accuracy)
  
  
def check_accuracy(probs: torch.FloatTensor,
                   labels: torch.LongTensor,
                   expected_accuracy: float):
    actual_accuracy = float(accuracy(probs, labels))
    assert actual_accuracy == expected_accuracy, f"Expected accuracy to be {expected_accuracy} but was {actual_accuracy}"

check_accuracy(torch.tensor([[0, 1],
                             [0, 1],
                             [0, 1],
                             [0, 1],
                             [0, 1]]),
               torch.ones(5, dtype=torch.long),
               1.0)
check_accuracy(torch.tensor([[1, 0],
                             [0, 1],
                             [0, 1],
                             [0, 1],
                             [0, 1]]),
               torch.ones(5, dtype=torch.long),
               0.8)
check_accuracy(torch.tensor([[1, 0],
                             [1, 0],
                             [0, 1],
                             [0, 1],
                             [0, 1]]),
               torch.ones(5, dtype=torch.long),
               0.6)
check_accuracy(torch.tensor([[1, 0],
                             [1, 0],
                             [1, 0],
                             [1, 0],
                             [1, 0]]),
               torch.ones(5, dtype=torch.long),
               0.0)
print("All test cases passed")


# Define the model to optimze
model = MLP(feature_count, hidden_layer_size, class_count)
model = model.to(device)

# The optimizer we'll use to update the model parameters
# SGD = stochastic gradient descent; efficient approach to fitting
# linear classifiers; Stochastic Gradient Descent (SGD) is a simple yet efficient optimization algorithm
# used to find the values of parameters/coefficients of functions that minimize a cost function. 
# In other words, it is used for discriminative learning of linear classifiers under convex loss functions
# such as SVM and Logistic regression. 
# It is a simple hill descending algorithm, taking a step in the steepest downhill direction (the negative of the gradient)
# in order to reduce the loss.
optimizer = optim.SGD(model.parameters(), lr=0.08)  # lr = learning rate

# Now we define the loss function.
criterion = nn.CrossEntropyLoss() 

# Now we iterate over the dataset a number of times. Each iteration of the entire dataset 
# is called an epoch.
for epoch in range(0, 100):
    # We compute the forward pass of the network
    logits = model.forward(features['train'])
    # Then the value of loss function 
    loss = criterion(logits,  labels['train'])
    
    train_accuracy = accuracy(logits, labels['train']) * 100
    summary_writer.add_scalar('accuracy/train', train_accuracy, epoch)
    summary_writer.add_scalar('loss/train', loss.item(), epoch)
    
    # How well the network does on the batch is an indication of how well training is 
    # progressing
    print("epoch: {} train accuracy: {:2.2f}, loss: {:5.5f}".format(
        epoch,
        accuracy(logits, labels['train']) * 100,
        loss.item()
    ))
    
    # Now we compute the backward pass, which populates the `.grad` attributes of the parameters
    loss.backward()
    # Now we update the model parameters using those gradients
    optimizer.step()
    # Now we need to zero out the `.grad` buffers as otherwise on the next backward pass we'll add the 
    # new gradients to the old ones.
    optimizer.zero_grad()
    
# Finally we can test our model on the test set and get an unbiased estimate of its performance.    
logits = model.forward(features['test'])    
test_accuracy = accuracy(logits, labels['test']) * 100
print("test accuracy: {:2.2f}".format(test_accuracy))

summary_writer.close()