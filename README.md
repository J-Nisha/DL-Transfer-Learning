# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Develop an image classification model using transfer learning with the pre-trained VGG19 model.

## DESIGN STEPS
### STEP 1:
Import required libraries and define image transforms.

### STEP 2:
Load training and testing datasets using ImageFolder.

### STEP 3:
Visualize sample images from the dataset.

### STEP 4:
Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5:
Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

### STEP 6:
Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions





## PROGRAM

### Name: Nisha J

### Register Number: 212223040133

```python
# Load Pretrained Model and Modify for Transfer Learning
model=models.vgg19(weights=VGG19_Weights.DEFAULT)


# Modify the final fully connected layer to match the dataset classes
model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)



# Include the Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)



# Train the model
def train_model(model, train_loader,test_loader,num_epochs=100):
    # Write your code here
    train_losses=[]
    val_losses=[]
    model.train()
    for epoch in range(num_epochs):
      running_loss=0.0
      for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
      train_losses.append(running_loss/len(train_loader))

      model.eval()
      val_loss=0.0
      with torch.no_grad():
        for images,labels in test_loader:
          images,labels=images.to(device),labels.to(device)
          outputs=model(images)
          loss=criterion(outputs,labels.unsqueeze(1).float())
          val_loss+=loss.item();
      val_losses.append(val_loss/len(test_loader))
      model.train()
      print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: Nisha J")
    print("Register Number: 212223040133")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot<img width="547" height="352" alt="image" src="https://github.com/user-attachments/assets/d98b2807-6bbe-4c8d-af39-9ff3fa8e22bc" />
<img width="547" height="352" alt="image" src="https://github.com/user-attachments/assets/24fedd1a-61df-47a0-a70e-47c910cfef0f" />
<img width="547" height="352" alt="image" src="https://github.com/user-attachments/assets/a9882294-46db-4797-a97c-c1153b30f03c" />





## Confusion Matrix
<img width="706" height="565" alt="image" src="https://github.com/user-attachments/assets/a9cf427f-2234-480f-85c8-ed2214ea5880" />



## Classification Report
<img width="484" height="209" alt="image" src="https://github.com/user-attachments/assets/4073fd4e-f5cd-4713-a926-36bd863f11aa" />



### New Sample Data Prediction
<img width="423" height="381" alt="image" src="https://github.com/user-attachments/assets/2334c34c-fb6d-41ff-a3f6-b0023b2663a8" />

<img width="424" height="369" alt="image" src="https://github.com/user-attachments/assets/a3a77d27-39cf-4778-9d24-a73bdcb04483" />




## RESULT
VGG19 model was fine-tuned and tested successfully. The model achieved good accuracy with correct predictions on sample test images.
