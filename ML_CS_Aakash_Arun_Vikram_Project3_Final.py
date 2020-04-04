#!/usr/bin/env python
# coding: utf-8

# # ML in Cybersecurity: Project III
# 
# ## Team
#   * **Team name**:  *HARP*
# * **Members**:   
#    * Aakash Rajpal - s8aarajp@stud.uni-saarland.de [2581266]
#    * Arun Sivadasan - s8arsiva@stud.uni-saarland.de [2581838]
#    * Vikram Vashisth - s8vivash@stud.uni-saarland.de [2581724]
# 
# ## Logistics
#   * **Due date**: 12th December 2019, 13:59:59 
#   * Email the completed notebook to mlcysec_ws1920_staff@lists.cispa.saarland 
#   * Complete this in the previously established **teams of 3**
#   * Feel free to use the course [mailing list](https://lists.cispa.saarland/listinfo/mlcysec_ws1920_stud) to discuss.
#   
# ## Timeline
#   * 28-Nov-2019: Project 3 hand-out
#   * **12-Dec-2019** (13:59:59): Email completed notebook to mlcysec_ws1920_staff@lists.cispa.saarland
# 
#   * 19-Dec-2019: Project 3 discussion and summary
#   
#   
# ## About this Project
# In this project, we dive into the vulnerabilities of machine learning models and the difficulties of defending against them. To this end, we require you to implement an evasion attack (craft adversarial examples) yourselves, and defend your own model.   
# 
# 
# ## A Note on Grading
# The total number of points in this project is 100. We further provide the number of points achievable with each excercise. You should take particular care to document and visualize your results, though.
# 
# 
#  
# ## Filling-in the Notebook
# You'll be submitting this very notebook that is filled-in with (all!) your code and analysis. Make sure you submit one that has been previously executed in-order. (So that results/graphs are already visible upon opening it). 
# 
# The notebook you submit **should compile** (or should be self-contained and sufficiently commented). Check tutorial 1 on how to set up the Python3 environment.
# 
# It is extremely important that you **do not** re-order the existing sections. Apart from that, the code blocks that you need to fill-in are given by:
# ```
# #
# #
# # ------- Your Code -------
# #
# #
# ```
# Feel free to break this into multiple-cells. It's even better if you interleave explanations and code-blocks so that the entire notebook forms a readable "story".
# 
# 
# ## Code of Honor
# We encourage discussing ideas and concepts with other students to help you learn and better understand the course content. However, the work you submit and present **must be original** and demonstrate your effort in solving the presented problems. **We will not tolerate** blatantly using existing solutions (such as from the internet), improper collaboration (e.g., sharing code or experimental data between groups) and plagiarism. If the honor code is not met, no points will be awarded.
# 
#  
#  ## Versions
#   * v1.0: Initial notebook
#   * v1.1: Clarifications at 1.1.2, 1.2.2, 2.1
#  
#   ---

# In[1]:


import torch


# In[2]:


torch.cuda.is_available()


# In[4]:


import time 
 
import numpy as np 
import matplotlib.pyplot as plt 

import json 
import time 
import pickle 
import sys 
import csv 
import os 
import os.path as osp 
import shutil 

import pandas as pd

from IPython.display import display, HTML
 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots 
plt.rcParams['image.interpolation'] = 'nearest' 
plt.rcParams['image.cmap'] = 'gray' 
 
# for auto-reloading external modules 
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython 
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[5]:


# Some suggestions of our libraries that might be helpful for this project
from collections import Counter          # an even easier way to count
from multiprocessing import Pool         # for multiprocessing
from tqdm import tqdm                    # fancy progress bars
import time as timer

# Load other libraries here.
# Keep it minimal! We should be easily able to reproduce your code.
# We only support sklearn and pytorch.
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
#import foolbox

# We preload pytorch as an example
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset


# In[3]:


compute_mode = 'cpu'

if compute_mode == 'cpu':
    device = torch.device('cpu')
elif compute_mode == 'gpu':
    # If you are using pytorch on the GPU cluster, you have to manually specify which GPU device to use
    # It is extremely important that you *do not* spawn multi-GPU jobs.
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'    # Set device ID here
    device = torch.device('cuda')
else:
    raise ValueError('Unrecognized compute mode')


# In[4]:


import foolbox


# #### Helpers
# 
# In case you choose to have some methods you plan to reuse during the notebook, define them here. This will avoid clutter and keep rest of the notebook succinct.

# # 1. Attacking an ML-model
# 
# In this section, we implement an attack ourselves. We then leverage the Foolbox library to craft adversarial examples. First, however, you need a model you can attack. Feel free to choose the DNN/ConvNN from project 1.
# 
# Hint: you might want to save the trained model to save time later.

# ### 1.1.1: Setting up the model (5 Points)
# 
# Re-use the model from project 1 here and train it until it achieves reasonable accuracy (>92%).

# For this project we will use the lenet model pretrained on the MNIST dataseta and published by Pytorch. 
# 
# Source for model: PyTorch's official documentation site (https://pytorch.org/tutorials/beginner/fgsm_tutorial.html) provides the following link for the Lenet model which is pre-trained on the MNIST dataset: 
# https://drive.google.com/drive/folders/1fn83DF14tWmit0RTKWRhPq5uVXt73e0h?usp=sharing 
# 
# This file needs to be saved as "Models/lenet_mnist_model.pth"

# #### a) Load data

# In[5]:


# MNIST Test dataset and dataloader declaration
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data/MNIST/', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data/MNIST/', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)


# #### b) Define model

# In[6]:


# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    
    
pretrained_model = "Models/lenet_mnist_model.pth"
use_cuda=False
    
    
# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
#device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()


# #### c) define loss, optimizer

# The lenet model uses ‘categorical_crossentropy’ as its loss function and uses SGD as its optimizer. 

# #### d) train

# Since this is already pre-trained on the MNIST dataset, we will just evaluate how it performs on the training set. As seen below, it classifies more than 92% images correctly. 

# In[7]:


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data/MNIST/', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=4, shuffle=False, num_workers=2)

correct = 0
total = 0
with torch.no_grad():
    for data in train_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the {} training images: {} %'.format(total, round((100 * correct / total),2)))


# #### d) evaluate

# Since this is already pre-trained on the MNIST dataset, we will just evaluate how it performs on the test set. As seen below, it classifies more than 92% images correctly as required by the project mandate. We can proceed with the various attacks 

# In[8]:


import torchvision.transforms as transforms


# In[9]:


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data/MNIST/', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=4, shuffle=False, num_workers=2)

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the {} test images: {} %'.format(total, (100 * correct / total)))


# ### 1.1.2: Implementing an attack (15 Points)
# 
# We now want you to attack the model trained in the previous step. Please implement the FGSM attack mentioned in the lecture. 

# ### 1.1.2 - A) FGSM attack function 

# In[10]:


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


# ### 1.1.2 - B) Attack, Generate Samples function 

# In[11]:


def attack_generate_samples( model, device, data_loader, epsilon, target_examples):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in data_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < target_examples):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < target_examples:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(data_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(data_loader), final_acc))
    print ("Number of adversarial examples: {} ".format(len(adv_examples)))

    # Return the accuracy and an adversarial example
    
    return final_acc, adv_examples


# In[12]:


def test_epsilon( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


# ### 1.1.2 - A) Selection of epsilon

# In[13]:


# MNIST Test dataset and dataloader for attacks 
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data/MNIST/', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data/MNIST/', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)


# In[14]:


accuracies = []
epsilons = [0, .05, .1, .15, .2, .25, .3]

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test_epsilon(model, device, test_loader, eps)
    accuracies.append(acc)


plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()
    


# ### 1.1.2 - A) Run attack with selected epsilon

# The epsilon value of 0.05 was chosen as we found that it introduces minimum noise without too much distortion to the image. Hence we felt this was a better value to run the attack and select the samples.

# In[15]:


accuracies = []
examples = []
selected_epsilon = 0.05

target_samples = 1000

acc, examples = attack_generate_samples(model, device, train_loader, selected_epsilon, target_samples)

len(examples)


# ### 1.1.3: adversarial sample set (5 Points)
# 
# Please additionally generate a dataset containing at least 1,000 adversarial examples using FGSM.

# ### 1.1.3 - A) Storing 1024 adversarials FGSM attack for use for Defence Model 

# In[17]:


from time import time

timestamp=time()
file_name = "data/Adversarial/adv_examples" + ".pickle"
import pickle 
pickle_file = open(file_name,"wb")
pickle.dump(examples, pickle_file)
pickle_file.close()



# ### 1.1.3: Visualizing the results (5 Points)
# 
# Please chose one sample for each class (for example the first when iterating the test data) and plot the (ten) adversarial examples as well as the predicted label (before and after the attack)

# In[18]:


# Sort the examples by original label
def getKey(item):
     return item[0]
examples = sorted(examples, key=getKey)



example_counter={
    0:0,
    1:0,
    2:0,
    3:0,
    4:0,
    5:0,
    6:0,
    7:0,
    8:0,
    9:0    
    
}
current_label = 0 
fig, axes = plt.subplots(10,10,figsize=(20,20))
plt.setp(axes, xticks=[], yticks=[])
plt.title("Adversarial examples orignal --> converted data")

for example in examples: 
    orig,adv,ex = example

    if current_label!=orig:
        current_label=orig
    if example_counter[orig]<10: 
        row=current_label
        column=example_counter[orig]
        axes[row,column].set_title("{} -> {}".format(orig, adv))
        axes[row,column].imshow(ex, cmap="gray")
        example_counter[orig]+=1


plt.show()


# ### 1.2.1: Using libraries for attacks (10 Points)
# As the field of evasion attacks (in particular for DNN) is very active research field, several libraries have been published that contain attacks. We will work here with the Foolbox (https://github.com/bethgelab/foolbox) library. Please implement two other (recent, advanced) attacks of your choice using this library. 

# ### 1.2.1 A) Overview of approach 

# The below diagram gives an overview of how we demonstrate various attacks. 

# In[19]:


from IPython.display import Image
Image("Images/attack.jpg")


# ### 1.2.1 B) Generate 10 sample images for demo 

# In[20]:


# MNIST Test dataset and dataloader declaration
#To optimize: change the logic for loading unique images 
sample_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data/MNIST/', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=False)
i=0
images_ori_v2 = []

for batch in sample_loader:
    image, label = batch
   
    if i==label.item():
        images_ori_v2.append([image, label])
        i+=1 
    if i==10:
        break 


# ### 1.2.1 C) FGSM attack - against 10 sample images   

# #### A function to run FGSM against a single image 

# In[21]:


from torch.autograd import Variable
def attack_single_image( model, device, data_loader, epsilon):
    '''
    This function crafts preturbed images for all images passed through the data_loader 
    '''

    adv_examples = []


    # Loop over all examples in the given data set
    for data, target in data_loader:
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        candidate=data

        # Set requires_grad attribute of tensor. Important for Attack
        candidate.requires_grad = True
        found_preturbation = False 


        for x in range(10):

            # Forward pass the data through the model
            output = model(candidate)
            init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            # Calculate the loss
            loss = F.nll_loss(output, target)

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = candidate.grad.data

            # Call FGSM Attack
            perturbed_data = fgsm_attack(candidate, epsilon, data_grad)

            # Re-classify the perturbed image
            output = model(perturbed_data)

            # Check for success
            final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            if final_pred.item() != target.item(): # We got a successful perturbation, save the data and break the loop to go to next image in batch
                adv_image = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_image,(x+1)) )
                break
            else:  
                candidate = Variable(perturbed_data.data, requires_grad=True)

    #=======================================================================
    # Return adv_examples which contains in each row: 
    # 1. Actual label 
    # 2. Incorrect Prediction made by model on the adversarial image 
    # 3. Adversarial image 
    # 4. Number of iterations it took to generate this image 

    return adv_examples


# #### Run the attack against 10 sample images 

# In[22]:


start = timer.time()

adv_FGSM = []
# Run attack against the sample image set 
adv_FGSM = attack_single_image(model, device, images_ori_v2, selected_epsilon)

# Sort the examples by original label
def getKey(item):
     return item[0]
adv_FGSM = sorted(adv_FGSM, key=getKey)

print("FGSM Time taken: ", round((timer.time() - start),2)) 


# In[23]:


adv_FGSM_actual_labels = [] 
adv_FGSM_iterations = []
adv_FGSM_predicted_labels = []
for row in adv_FGSM:
    
    actual_label, predicted_label, adv_FGSM_image, iterations = row
    adv_FGSM_actual_labels.append(actual_label)
    adv_FGSM_predicted_labels.append(predicted_label)
    adv_FGSM_iterations.append(iterations)
    


# In[24]:


import matplotlib.pyplot as plt
import numpy as np 
get_ipython().run_line_magic('matplotlib', 'inline')


def barchart_singlebar (data_labels,data_values,x_axis_title,y_axis_title,chart_title):
    """
    This function creates a single bar chart from data_labels and data_values  
    
    """
    # Get the y limits
    ymin, ymax = min(data_values), max(data_values)

    xpos = np.arange(len(data_labels))
    fig, ax = plt.subplots()
    bar=ax.bar(xpos,data_values)
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for p in ax.patches:
        h = p.get_height()
        x = p.get_x()+p.get_width()/2.
        if h != 0:
            ax.annotate("%g" % p.get_height(), xy=(x,h), xytext=(0,2),  
                       textcoords="offset points", ha="center", va="bottom")

    plt.xticks(xpos,data_labels)
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)

    # Set the y limits making the maximum 5% greater
    plt.ylim(ymin, 1.10 * ymax)
    plt.title(chart_title)


# In[25]:


barchart_singlebar(adv_FGSM_actual_labels,adv_FGSM_iterations,"Actual labels","Count of iterations","FGSM: Iterations taken to develop an adversarial image")


# ### 1.2.1 D) Foolbox attack 1 (CarliniWagnerL2Attack) - against 10 sample images   

# In[26]:


import foolbox

start = timer.time()

fmodel1 = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10)
from numpy import array 
adv_Method1=[]
adv_Method1_labels=[]
image_list = np.array(images_ori_v2)
 
for index, item in enumerate(image_list):


    input_images = image_list[index][0].data
    input_images = input_images.numpy()
    
    input_labels=image_list[index][1].data
    input_labels = input_labels.numpy() 

    
    attack1 = foolbox.attacks.CarliniWagnerL2Attack(fmodel1)
    adversarial1 = attack1(input_images,input_labels)
    adv_Method1.append(adversarial1)
    adv_Method1_labels.append(fmodel1.forward(adversarial1).argmax(axis=-1))    

print("Attack 1 Time taken: ", round(timer.time() - start))  


# ### 1.2.1 E) Foolbox attack 2 (DeepFoolAttack) - against 10 sample images   

# In[27]:


start = timer.time()

fmodel2 = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10)
from numpy import array 
adv_Method2=[]
adv_Method2_labels=[]
image_list = np.array(images_ori_v2)

for index, item in enumerate(image_list):
    input_images = image_list[index][0].data
    input_images = input_images.numpy()
    
    input_labels=image_list[index][1].data
    input_labels = input_labels.numpy() 

    attack2 = foolbox.attacks.DeepFoolAttack(fmodel2)
    adversarial2 = attack2(input_images,input_labels)
    adv_Method2.append(adversarial2)
    adv_Method2_labels.append(fmodel2.forward(adversarial2).argmax(axis=-1))    
    
    

print("Attack 2 Time taken: ", round(timer.time() - start))  


# ### 1.2.2: Visualizing the results (20 Points)
# As before, please plot the new adversarial examples. Compare all crafting techniques (FGSM, 2 methods from Foolbox).
# 

# In[28]:



col_titles = ['Ori','FGSM','Method 1', 'Method 2'] 
nsamples = 10
nrows = nsamples+1
ncols = len(col_titles)

fig, axes = plt.subplots(nrows,ncols,figsize=(15,15))  # create the figure with subplots
plt.subplots_adjust(hspace=0.5)

[ax.set_axis_off() for ax in axes.ravel()]  # remove the axis

axes[0,0].text(0.0,0.2, "Original image", size=20, horizontalalignment='left',verticalalignment='center')
axes[0,1].text(0.1,0.2, "FGSM image", size=20, horizontalalignment='left',verticalalignment='center')
axes[0,2].text(0.2,0.2, "Foolbox 1", size=20, horizontalalignment='left',verticalalignment='center')
axes[0,3].text(0.2,0.2, "Foolbox 2", size=20, horizontalalignment='left',verticalalignment='center')

for i in range(nsamples):
    #Original images
    row=i+1
    
    #Show original image 
    image, label = images_ori_v2[i]
    image_np = image.detach().numpy() 
    axes[row,0].imshow(image_np.reshape(28,28))
    axes[row,0].set_title("Label:{}".format(label.item()))

    #Show FGSM image 
    actual_label, predicted_label, adv_FGSM_image, iterations = adv_FGSM[i]
    image_np = adv_FGSM_image
    axes[row,1].imshow(image_np.reshape(28,28))
    axes[row,1].set_title("Label:{}".format(predicted_label))

    #Show Foolbox1 image 
    image_np = adv_Method1[i] 
    axes[row,2].imshow(image_np.reshape(28,28))
    axes[row,2].set_title("Label:{}".format(np.asscalar(adv_Method1_labels[i])))
    
    #Show Foolbox2 image 
    image_np = adv_Method2[i] 
    axes[row,3].imshow(image_np.reshape(28,28))
    axes[row,3].set_title("Label:{}".format(np.asscalar(adv_Method2_labels[i])))


plt.show()


# The above diagram shows the original image and the adversarial variants along with the model's wrong prediction 

# * Differences between the different attack methods: 
#     * The foolbox approach is easier to implement as the library has a standardized set of attacks. However, foolbox sample documentation showcases FGSM as the example. Sample implementations of non-FGSM attacks are not available. However, it is still relatively a good resource for anyone needing to implement an attack methods. 
# 
# * Does the attack always succeed (the model make wrong prediction on the adversarial sample)?
#     * In Foolbox - yes, the model always fails to classify. But it could also be because foolbox may be iterating internally until it finds a successful adversarial attack. 
#     * In FGSM - it depends on the number of iterations. In earlier iterations, it was able to classify the image correctly. Refer: 1.2.1 C) FGSM attack - against 10 sample images
# 

# | Attack:                              | FSGM     | Foolbox attack 1   (CarliniWagnerL2Attack)  | Foolbox attack 2   (DeepFoolAttack) |
# |--------------------------------------|----------|---------------------------------------------|-------------------------------------|
# | Time taken to execute on sample data | 0.11 sec | 88 sec                                      | 2 sec                               |
# | Most common   missclassification     | 8        | 8                                           | 8                                   |
# | No. of times                         | 5 times  | 5 times                                     | 4 times                             |

# As seen from the above table: 
# 1. The model seems to misclassify most adversarial digits as 8 in all 3 attacks 
# 2. The FGSM code in section 1.2.1 ran the fastest while CarliniWagerL2 attack took the longest 
# 

# These type of attacks is a white box attack as all three requires access to the model. These attacks would be more useful as part of the effort to identify a model's weakness during its development phase

# # 2. Defending an ML model
# 
# So far, we have focused on attacking an ML model. In this section, we want you to defend your model. As before concerning the attack, you can chose an example from the lecture, or experiment with any idea you have.
# 
# We do not require the defense to work perfectly - but what we want you to understand is why it works or why it does not work.

# ### 2.1: Implementing a defense of your choice (25 Points)
# As stated before, feel free to implement a defense or mitigation of your choice. Evaluate the defense on adversarial examples. This entails at least the 1,000 examples crafted from FGSM.   
# Also, you are encouraged (optional) to defend against the two other attack methods, i.e. you are free to increase this special test set (for example by >30 examples (>10 from your FGSM attack, >10 from both the two other attacks of the library)).

# Here we are implementing a Conditional Variational Autoencoder Model (VAE) to Defend against an adversarial attack. The idea is that the adverserial images created by FGSM or any other attack is first passed to this Autoencoder Defence Model. The VAE Model then generates an image correspoding to the adverserial image that fits the data distrubtion of the MNIST training data on which the VAE Model is pretrained.
# 
# The Model is Referenced from the below paper, the code here is in keras. We have implemented a similiar one in pytorch 
# 
# Model Refernce : https://colab.research.google.com/drive/1ky8foTDlb2OeQ1ckgxuydBEAgkex2Ir2#scrollTo=yyLvjD4CVPCm
# 
# Code Reference : https://graviraja.github.io/conditionalvae/

# Defining the VAE Model with the hyperparameters and it's encoder and decoder classes

# In[29]:


##Model Hyperparameters

BATCH_SIZE = 64         # number of data points in each batch
N_EPOCHS = 12          # times to run the model on complete data
INPUT_DIM = 28 * 28     # size of each input
HIDDEN_DIM = 256        # hidden dimension
LATENT_DIM = 75         # latent vector dimension
N_CLASSES = 10          # number of classes in the data
lr = 1e-3               # learning rate



##Given a class label, we will convert it into one-hot encoding
def idx2onehot(idx, n=N_CLASSES):

    assert idx.shape[1] == 1
    assert torch.max(idx).item() < n

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx.data, 1)

    return onehot


 
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.linear = nn.Linear(input_dim + n_classes, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim + n_classes]

        hidden = F.relu(self.linear(x))
        # hidden is of shape [batch_size, hidden_dim]

        # latent parameters
        mean = self.mu(hidden)
        # mean is of shape [batch_size, latent_dim]
        log_var = self.var(hidden)
        # log_var is of shape [batch_size, latent_dim]

        return mean, log_var
##The decoder takes a sample from the latent dimension and uses that as an input to output X. We will see how to sample from latent parameters later in the code.
class Decoder(nn.Module):
    ''' This the decoder part of VAE

    '''
    def __init__(self, latent_dim, hidden_dim, output_dim, n_classes):
        '''
        Args:
            latent_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the size of output (in case of MNIST 28 * 28).
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.latent_to_hidden = nn.Linear(latent_dim + n_classes, hidden_dim)
        self.hidden_to_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim + num_classes]
        x = F.relu(self.latent_to_hidden(x))
        # x is of shape [batch_size, hidden_dim]
        generated_x = F.sigmoid(self.hidden_to_out(x))
        # x is of shape [batch_size, output_dim]

        return generated_x

class CVAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.

    '''
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, n_classes)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, n_classes)

    def forward(self, x, y):

        x = torch.cat((x, y), dim=1)

        # encode
        z_mu, z_var = self.encoder(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        z = torch.cat((x_sample, y), dim=1)

        # decode
        generated_x = self.decoder(z)

        return generated_x, z_mu, z_var


# Instantising the Model with the hyperparameters and using Adam as our optimizer

# In[30]:


# model
model_cvae = CVAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, N_CLASSES)

#optimizer
optimizer = optim.Adam(model_cvae.parameters(), lr=lr)




# VAE consists of two loss functions
# 
# Reconstruction loss
# KL divergence
# So the final objective is
# 
# loss = reconstruction_loss + kl_divergence

# In[31]:


def calculate_loss(x, reconstructed_x, mean, log_var):
    # reconstruction loss
    RCL = F.binary_cross_entropy(reconstructed_x, x, size_average=False)
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return RCL + KLD


# Optimizing the loss function and training our VAE Model on MNIST Dataset so that it can learn the Data distriution of the MNIST Training Data

# In[32]:


transforms = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms)

test_dataset = datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=transforms
    )

## Define the iterator for the training, testing data.

train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# In[33]:


def train():
    # set the train mode
    model_cvae.train()

    # loss of the epoch
    train_loss = 0

    for i, (x, y) in enumerate(train_iterator):
        # reshape the data into [batch_size, 784]
        x = x.view(-1, 28 * 28)
        x = x.to(device)

        # convert y into one-hot encoding
        y = idx2onehot(y.view(-1, 1))
        y = y.to(device)

        # update the gradients to zero
        optimizer.zero_grad()

        # forward pass
        reconstructed_x, z_mu, z_var = model_cvae(x, y)

        # loss
        loss = calculate_loss(x, reconstructed_x, z_mu, z_var)

        # backward pass
        loss.backward()
        train_loss += loss.item()

        # update the weights
        optimizer.step()

    return train_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test():
    # set the evaluation mode
    model_cvae.eval()

    # test loss for the data
    test_loss = 0

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        for i, (x, y) in enumerate(test_iterator):
            # reshape the data
            x = x.view(-1, 28 * 28)
            x = x.to(device)

            # convert y into one-hot encoding
            y = idx2onehot(y.view(-1, 1))
            y = y.to(device)

            # forward pass
            reconstructed_x, z_mu, z_var = model_cvae(x, y)

            # loss
            loss = calculate_loss(x, reconstructed_x, z_mu, z_var)
            test_loss += loss.item()

    return test_loss

best_test_loss = 10000000000.0000
for e in range(N_EPOCHS):

    train_loss = train()
    test_loss = test()

    train_loss /= len(train_dataset)
    test_loss /= len(test_dataset)

    print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')

    if best_test_loss > test_loss:
        best_test_loss = test_loss
        patience_counter = 1
    else:
        patience_counter += 1

    if patience_counter > 3:
        break


# Now our VAE Model is trained on the MNIST Dataset and it has learned the data distribution.
# Importing the pickle files for Adverarial Images generated by FGSM Attack. I have used pickle file as We were working seprately hence it was easier. This pickle file corresponds to 1024 adversarial images generated by FGSM.
# 
# 

# In[34]:


import pickle
pickle_file_adverserial_fgsm_defence = open("data/Adversarial/1024_FGSM_adv_examples.pickle","rb")
test_features_fgsm = pickle.load(pickle_file_adverserial_fgsm_defence)
pickle_file_adverserial_fgsm_defence.close()
#print(test_features_fgsm[0])
print(len(test_features_fgsm))


# Converting the Test data from the pickle file for 1024 FGSM Advarsarial Examples into tensors and reshaping as required for the CNN Model and the VAE_generation Model

# In[35]:



batch_fgsm=[]
actual_fgsm_label = []
pertuared_fgsm_label = []

for i in range(len(test_features_fgsm)):
    actual_fgsm_label.append(torch.tensor(test_features_fgsm[i][0]))
    pertuared_fgsm_label.append(torch.tensor(test_features_fgsm[i][1]))
    batch_fgsm.append(torch.from_numpy(test_features_fgsm[i][2]))

print(batch_fgsm[0].shape)


# In[36]:


batch_fgsm_tensor = torch.stack(batch_fgsm)
actual_fgsm_lab_tensor = torch.stack(actual_fgsm_label)
pertuared_fgsm_lab_tesnor = torch.stack(pertuared_fgsm_label)
print(batch_fgsm_tensor.shape)
#batch_rel = batch_fgsm_tensor.reshape(960,28*28)
#print(batch_rel.shape)
print(actual_fgsm_lab_tensor.shape)


# In[37]:


fgsm_images_dataloader = DataLoader(batch_fgsm_tensor, batch_size=64)
fgsm_actual_labels = DataLoader(actual_fgsm_lab_tensor, batch_size=64)


# Testing the FGSM Images on the CNN Model used above to calculate the classification error

# In[38]:


correct = 0
total = 0
with torch.no_grad():
    for (data,labels) in zip(fgsm_images_dataloader,fgsm_actual_labels):
        images = data.reshape(64,1,28,28)
        outputs = model(images)
        #print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the {} FGSM adverserial test images: {} %'.format(total, (100 * correct / total)))


# Generating the Variational Autoencoder (VAE) images from the FGSM Advarsarial Images

# In[39]:


# convert y into one-hot encoding
vae_generataed_images_list = []
for (data,labels) in zip(fgsm_images_dataloader,fgsm_actual_labels):
    y_adv = idx2onehot(labels.view(-1, 1))
    y_adv = y_adv.to(device)
    data = data.reshape(64,784)
    reconstructed_x, z_mu, z_var = model_cvae(data, y_adv)
    vae_generataed_images_list.append(reconstructed_x)
    #print(reconstructed_x.shape)
    #print(data.shape)
    #print(y_adv.shape)
print(len(vae_generataed_images_list))


# Reshaping the VAE_Encoder Generated Images as needed by the CNN model for Classification

# In[40]:


vae_generataed_images=torch.stack(vae_generataed_images_list)
print(vae_generataed_images.shape)
vae_generataed_images=vae_generataed_images.reshape(1024,28,28)
print(vae_generataed_images.shape)


# In[41]:


vae_images_dataloader = DataLoader(vae_generataed_images, batch_size=64)


# Testing the VAE_Encoder Generated Images on the CNN Model for Classification

# In[42]:


correct = 0
total = 0
with torch.no_grad():
    for (data_vae,labels) in zip(vae_images_dataloader,fgsm_actual_labels):
        images = data_vae.reshape(64,1,28,28)
        outputs = model(images)
        #print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the {} VAE Generated test images from the FGSM Advarsial ones: {} %'.format(total, (100 * correct / total)))


# ### 2.2: Conclusions (15 Points)
# Please interpret the results of your defense here. 
# 
# * What did you try to make the classifier more robust against FGSM? 
# * Why did it work? 
# * Is the classifier now robust against FGSM?  
# * ...
# 
# Feel free to state any interesting finding you encountered during this project.

# 

# Q1) What did you try to make the classifier more robust against FGSM?
# 
# We implemented a Defence Mechanism that involves training a Conditional Variational Autoencoder (VAE) Model on the MNIST Training Dataset. This model converts any images ( advarsarial or clean) representation to the data distribution that it has been trained on using the encoder. This representation is then fed to the decoder, which generates sample from the distrubtion so as it gets correctly classified. We can see from our results that when the VAE Generated Images from the FGSM Advarsarial images where given as test data to our MNIST CNN Model it performs really well with an accuracy of almost 72%. This is great.
# 
# Below we can see the  VAE Generated Images from a few Adverserail FGSM Images
# 
# 
# 
# 

# In[6]:


Image("Images/adv.png")


# Q2) Why did it work?
# 
# The model works on a simple fact that is autoencoders are very good at capturing data distribution. They can reconstruct inputs with very low error if their input belongs to the training data distribution. This fact can be exploited to check if the input (adversarial or clean) belongs to the distribution that the classification model understands. When an adversarial input is fed to the encoder, it produces incorrect representation. This representation is then fed to the decoder, which generates sample from incorrect class . If there is statistical mismatch between the adversarial input and its generated output, it can be implied that input does not belong to training data distribution and was intended to fool the classification model. Here, we use Variational Autoencoder owing to their smooth latent representation space which helps avoid degenrate reconstruction. Below is a an image reperesentation stating the same

# In[3]:


from IPython.display import Image
Image("Images/vae.jpg")


# Q3)Is the classifier now robust against FGSM? 
# 
# This Defence Model can be used against all Advarasarial Images generated by any attack method(White box or Black Box). This is because this Model is a Detection mechanism Model which takes a different route by predicting if an input contains adversarial pretubations, instead of modifying the classification model(MNIST). The advantage of this is that they can be brought in as the first line of defense in practical settings without disturbing the main model.
# 
# 
# 

# We learned that there are various defense techniques out there. But the problem is the most common defence consists of introducing adversarial images to train a more robust network, which are generated using the target model.This is know as  Adversarial Training which  modifies the loss function of the actual classification model MNIST to augment training data with peturbed samples using FGSM technique during training of the MNIST CNN Model. I realised that this method won't be that robust as we are training only on FGSM images and if an advarsarial image from a different Method attacks then this would fail misrebaly. Hence it won't be robust enough. It has been shown that this approach has some limitations — in particular, this kind of defence is less effective against black-box attacks than white-box attacks in which the adversarial images are generated using a different model
# 
# Also we could have used a GAN Model to generate images but 

# In[ ]:




