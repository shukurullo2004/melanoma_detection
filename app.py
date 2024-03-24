import gradio as gr
import os
import torch
from PIL import Image
from timeit import default_timer as timer

from model import create_model
from typing import Tuple, Dict

class_names = ['Benign', 'Malignant']

model, transform = create_model()

# Load saved weights
model.load_state_dict(
    torch.load(
        f="melanoma_model1.pth",
        map_location=torch.device("cpu"),  # load to CPU
    )
)


### 3. Predict function ###

# Create predict function

def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()

    # Apply transformations to the image
    img_tensor = transform(img).unsqueeze(0).to(next(model.parameters()).device)

    # Put model into evaluation mode
    model.eval()

    # Pass the image through the model
    with torch.no_grad():
        y_logits = model(img_tensor).squeeze()
        y_pred_probs = torch.sigmoid(y_logits)

    # Round the prediction probabilities to get binary predictions
    y_pred_binary = torch.round(y_pred_probs).item()

    # Create a dictionary with the class label and the corresponding prediction probability
    pred_label = class_names[int(y_pred_binary)]

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time
    return {pred_label: float(y_pred_probs)}, pred_time

# Create title, description and article strings
title = "Melanoma Cancer Detection"
description = "An Vision Tranformer feature extractor computer vision model to classify images of MELANOMA CANCER.."
article = " model is built by Shukurullo Meliboev using Kaggle's Melanoma disease datasets."
example_list = [["examples/" + example] for example in os.listdir("examples")]
# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=1, label="Predictions"), # what are the outputs?
                             gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch(False) # generate a publically shareable URL?
