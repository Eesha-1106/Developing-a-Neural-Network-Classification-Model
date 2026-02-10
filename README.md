# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="800" height="1000" alt="image" src="https://github.com/user-attachments/assets/7832d7ae-54d9-4bcb-9ac6-19b1b349c3ce" />


## DESIGN STEPS
### Step 1: Load and Preprocess Data
Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).

### Step 2: Feature Scaling and Data Split
Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.

### Step 3: Convert Data to PyTorch Tensors
Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.

### Step 4: Define the Neural Network Model
Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.

### Step 5: Train the Model
Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.

### Step 6: Evaluate and Predict
Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.


## PROGRAM

### Name:Eesha Ranka

### Register Number:212224240040

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1=nn.Linear(input_size,36)
        self.fc2=nn.Linear(36,16)
        self.fc3=nn.Linear(16,8)
        self.fc4=nn.Linear(8,4)


    def forward(self, x):
      x=F.relu(self.fc1(x))
      x=F.relu(self.fc2(x))
      x=F.relu(self.fc3(x))
      x=self.fc4(x)
      return x
        
# Initialize the Model, Loss Function, and Optimizer

def train_model(model, train_loader, criterion, optimizer, epochs):
  model.train()
  for epoch in range(epochs):
    for inputs, targets in train_loader: 
        optimizer.zero_grad()
        outputs = model(inputs) 
        loss = criterion(outputs, targets) 
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

```

### Dataset Information
<img width="1566" height="381" alt="image" src="https://github.com/user-attachments/assets/072b6eed-b1c9-4238-a715-4a693f7b1d91" />


### OUTPUT

## Confusion Matrix

<img width="539" height="455" alt="download" src="https://github.com/user-attachments/assets/9a696835-faf7-4e37-88ce-2da2cea4f721" />


## Classification Report
<img width="801" height="525" alt="image" src="https://github.com/user-attachments/assets/ca0cdaa3-d4b7-46b7-9c4a-3ad9a8cc8cf9" />


### New Sample Data Prediction
<img width="1536" height="466" alt="image" src="https://github.com/user-attachments/assets/26d8e0cb-e2d1-4b46-ba17-81483dd6e654" />


## RESULT
This program has been executed successfully.
