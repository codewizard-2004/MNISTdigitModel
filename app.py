import gradio as gr
from PIL import Image
import torch
from torchvision import transforms , datasets
from torchvision.transforms import ToTensor
from cnn import HandwrittenDigitClassifier
import helper_functions as hf

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

try:
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=ToTensor())
    print("Dataset Already Downloaded")
except:
    print("Dataset Not Downloaded")
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())


class_names = mnist_dataset.classes


model = HandwrittenDigitClassifier(
    input_shape = 1,
    hidden_units = 10,
    output_shape = len(class_names)
).to(device)

import torch

# Load the model state dict and map it to the CPU
if device == "cpu":
    model.load_state_dict(torch.load("model/HandwrittenModel.pth", map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load("model/HandwrittenModel.pth"))


# Function to resize, convert the image to grayscale, and then to a PyTorch tensor
def process_image(image):
    if image is None:
        return "No image uploaded. Please upload an image."

    try:
        # Resize the image to 28x28
        resized_image = image.resize((28, 28))
        
        # Convert the image to grayscale (1 color channel)
        grayscale_image = resized_image.convert('L')
        
        # Convert the grayscale image to a PyTorch tensor
        to_tensor = transforms.ToTensor()  # Converts a PIL Image to tensor
        image_tensor = to_tensor(grayscale_image)

        preds = hf.make_single_prediction(model , image_tensor , class_names).item()
        

        
        # Return a success message including tensor shape
        return f"The number is {preds}."
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create the Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            # File upload component
            image_input = gr.Image(type="pil", label="Upload Image")
            
            # Submit button
            submit_button = gr.Button("Submit")
        
        with gr.Column():
            # Output text
            output_text = gr.Textbox(label="Message", placeholder="Your message will appear here.")
    
    # Link the button to trigger the function and display the output message
    submit_button.click(fn=process_image, inputs=image_input, outputs=output_text)

# Launch the Gradio app
demo.launch()
