import torch
from torchvision import transforms
from PIL import Image
from model.alexnet import AlexNet

import __main__
setattr(__main__, "AlexNet", AlexNet)

def predict_image(image_path):
    '''Predicts the class of an image given its path.'''
    # Define your transformations (adjust according to your model's needs)
    transform = transforms.Compose([transforms.Resize((227, 227)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    model =  torch.load('model/melanoma_CNN.pt', map_location=torch.device('cpu'))
    with torch.no_grad():
        outputs = model(image)
        _ , predicted = torch.max(outputs, 1)
    
    # Return the predicted class (you may need to adjust this)
    return "Malignant Skin Cancer" if predicted.item() == 1 else "Benign Skin Cancer"
