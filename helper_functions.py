import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm # type: ignore
import numpy as np
from timeit import default_timer as timer
import torchvision 
from torchvision import datasets
from torchvision.transforms import ToTensor

device = "cuda" if torch.cuda.is_available() else "cpu"

def visualize_image_batch(image_data: torch.tensor,class_names:list, row=1 , cols=1, cmap = "gray"):
  """
  Visualize a batch of data from a data loader with specified number of rows and columns
  by default row = 1  col = 1
  """
  fig = plt.figure(figsize = (9,9))
  for i in range(1 , row*cols+1):
    random_idx = torch.randint(0, len(image_data) , size = [1]).item()
    img , label = image_data[random_idx]
    fig.add_subplot(row, cols, i)
    plt.imshow(img.squeeze(), cmap = cmap)
    plt.title(class_names[label])
    plt.axis(False)

def get_train_time(start: float,
                   end: float,
                   device: torch.device = device):
  """
  gives the time taken between start and end time of a model
  """
  total_time = end - start
  print(f"Train time on {device}: {total_time:.3f} seconds")
  return total_time

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device=device):
  '''
  Returns a dictionary containing the results of the model predicting on data_loader
  '''
  loss, acc = 0 , 0
  model.eval()
  model.to(device)
  with torch.inference_mode():
    #device agnostic
    for X,y in tqdm(data_loader):
      X,y = X.to(device) , y.to(device)
      y_pred = model(X)
      loss += loss_fn(y_pred , y)
      acc += accuracy_fn(y_true = y , y_pred = y_pred.argmax(dim=1))

    #scale the loss and acc to find the avg loss/acc per batch
    loss /= len(data_loader)
    acc /= len(data_loader)

  return {"model_name":model.__class__.__name__ ,
          "model_loss":loss.item() ,
          "model_acc":acc}


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc

#Training loop


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
  """
    Performs training with model trying to learn on data loader
  """
  train_loss , train_acc = 0 , 0
  test_loss_stack = []
  train_loss_stack = []
  model.train()

  for batch, (X,y) in enumerate(data_loader):

    X,y = X.to(device) , y.to(device)
    y_pred = model(X)
    loss = loss_fn(y_pred , y)
    train_loss_stack.append(loss)
    train_loss += loss
    train_acc += accuracy_fn(y_true = y , y_pred = y_pred.argmax(dim=1))#Logits -> prediction labels

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%\n")
  return train_loss_stack

#Creating a testing loop

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
  """
  Perform a testing loop step on model going over data loader
  """
  test_loss , test_acc = 0 , 0
  test_loss_stack = []

  model.eval()

  with torch.inference_mode():
    for X,y in data_loader:
      X,y = X.to(device) , y.to(device)
      test_pred = model(X)
      loss = loss_fn(test_pred , y)
      test_loss_stack.append(loss)
      test_loss += loss
      test_acc += accuracy_fn(y_true = y , y_pred = test_pred.argmax(dim=1))

    #Adjust metrics and print out
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n")
    return test_loss_stack
  

def make_single_prediction(model: torch.nn.Module,
                           img: torch.tensor,
                           class_names: list,
                           display: bool= True):
  """
  Makes a single prediction on a single image
  """
  pred_logits = model(img.to(device).unsqueeze(dim = 0)).to(device)
  pred_probs = pred_logits.argmax(dim = 1)
  if display:
    print(f"Predicted Output: {class_names[pred_probs]}")
  return pred_probs

import random
def make_random_predictions(model: nn.Module,
                            count: int,
                            test_data: torchvision.datasets,
                            rows:int , 
                            cols: int,
                            class_names: list,
                            seed:int = 42):
  '''
    This function will give random predctions of amount of count given by the user

    Args:
      model: the model to make prediction
      count: the number of predictions
      test_data: The data on which predictions are made use torchvision.datasets
      class_names: A lsit containing all the labels
      seed: random seeding , 42 by default
  '''
  torch.manual_seed(seed)
  random_numbers = [random.randint(0, len(test_data) - 1) for _ in range(count)]
  plt.figure(figsize=(20,20))

  for idx, i in enumerate(random_numbers):
      sample, label = test_data[i]

      # Plot the sample
      plt.subplot(rows, cols, idx + 1)  # idx + 1 for correct subplot positioning
      plt.imshow(sample.squeeze(), cmap="gray")
      plt.axis('off')  # Hide axes

      # Make prediction
      pred = make_single_prediction(model, sample , class_names, False)

      # Title with color coding (green for correct, red for incorrect)
      if pred == label:
          plt.title(f"Predicted:{class_names[pred]}|Actual:{class_names[label]}",  color="g")
      else:
          plt.title(f"Predicted:{class_names[pred]}|Actual:{class_names[label]}", color="r")

  # Adjust the layout to prevent overlap
  plt.show()


def get_label(data: torchvision.datasets):
  """
    This function returns the class names or labels in a torchvision dataset
  """
  mnist_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=ToTensor())

  return mnist_dataset.classes