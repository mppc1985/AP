import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torch.nn import functional as F
import torchvision.models as models
import timm
import os
from MLP import MLP
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc

# Define image size
IMG_SIZE = 224

# Define class labels
class_labels = {
    "Hatchback": 0,
    "Other": 1,
    "Pickup": 2,
    "Seden": 3,
    "SUV": 4
}

# Preprocess images and labels
def preprocess_data(image_dir, class_labels, batch_size=32, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageFolder(image_dir, transform=transform)
    
    # Encode class labels
    labels = [class_labels[class_name] for class_name in dataset.classes]
    
    # Split dataset into train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create DataLoader for train and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, torch.tensor(labels)

def generate_embeddings(dataloader, model):
    model.eval()
    embeddings = []
    targets = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)  # move images to GPU
            labels = labels.to(device)  # move labels to GPU
            features = model(images)
            embeddings.append(features)
            targets.append(labels)
    embeddings = torch.cat(embeddings)
    targets = torch.cat(targets)
    return embeddings, targets


# Example usage
image_dir = "vehicles"
train_loader, test_loader, labels = preprocess_data(image_dir, class_labels)


# Criar o modelo(pre-treinado)
vit_model = timm.create_model('vit_base_patch16_224', pretrained=True)

print("1")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
vit_model = vit_model.to(device)

train_embeddings, train_targets = generate_embeddings(train_loader, vit_model)
test_embeddings, test_targets = generate_embeddings(test_loader, vit_model)


print(train_embeddings.shape, train_targets.shape)

valid_embeddings = train_embeddings.to(device)
valid_targets = train_targets.to(device)

# Determine the size of the embeddings and the number of classes
input_dim = train_embeddings.shape[1]
output_dim = len(labels)

# Criar MLP
mlp = MLP(input_dim, hidden_dim=512, output_dim=output_dim).to(device)
# Funcao de Loss, Otimizador e epochs
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters())
num_epochs = 100


# Initialize lists to monitor test loss and accuracy
import matplotlib.pyplot as plt
train_loss = []
valid_loss = []
# Train the MLP
for epoch in range(num_epochs):
    # Move the embeddings and the targets to the GPU
    train_embeddings = train_embeddings.to(device)
    train_targets = train_targets.to(device)

    # Forward pass
    outputs = mlp(train_embeddings)
    loss = criterion(outputs, train_targets)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save the current training loss
    train_loss.append(loss.item())

    # Validation
    mlp.eval()
    with torch.no_grad():
        outputs = mlp(valid_embeddings.to(device))
        loss = criterion(outputs, valid_targets.to(device))
        valid_loss.append(loss.item())

    print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Create count of the number of epochs
epoch_count = range(1, num_epochs + 1)

# Visualize loss history
plt.plot(epoch_count, train_loss, 'r--')
plt.plot(epoch_count, valid_loss, 'b-')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()



 
    
test_embeddings = test_embeddings.to(device)
test_targets = test_targets.to(device)

# Make predictions on the test set

mlp.eval()
with torch.no_grad():
    outputs = mlp(test_embeddings)
    _, predicted = torch.max(outputs.data, 1)
    total = test_targets.size(0)
    correct = (predicted == test_targets).sum().item()

    # Cálculo de métricas adicionais
    precision = precision_score(test_targets.cpu(), predicted.cpu(), average='macro')
    recall = recall_score(test_targets.cpu(), predicted.cpu(), average='macro')
    f1 = f1_score(test_targets.cpu(), predicted.cpu(), average='macro')
    conf_matrix = confusion_matrix(test_targets.cpu(), predicted.cpu())

print(f'Test Accuracy of the model on the test images: {100 * correct / total:.2f} %')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')
print('Confusion Matrix:')
print(conf_matrix)

