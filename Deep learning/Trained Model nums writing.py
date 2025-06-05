import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import keras

# Load the trained model
model = keras.models.load_model('your_trained_model.h5')

# Define a function to preprocess the image
def preprocess_image(img):
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img = img.convert('L')  # Convert to grayscale
    img = np.array(img)  # Convert to numpy array
    img = img / 255.0  # Normalize the pixel values
    img = img.reshape(1, 28, 28, 1)  # Reshape for the model
    return img

# Define a function to make predictions
def predict_digit(img):
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction)  # Get the predicted class
    return predicted_class

# Create a tkinter GUI for drawing
class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")

        # Create a canvas to draw on (white background, black for drawing)
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg="white")
        self.canvas.pack()

        # Create a button to clear the canvas
        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        # Label to show the predicted digit
        self.label = tk.Label(self.root, text="Draw a digit...", font=("Helvetica", 24))
        self.label.pack()

        # Initialize PIL image for drawing (start with white background)
        self.image = Image.new("RGB", (280, 280), (0, 0, 0))  # White background
        self.draw = ImageDraw.Draw(self.image)

        # Bind mouse events to canvas
        self.canvas.bind("<B1-Motion>", self.draw_digit)
        self.canvas.bind("<ButtonRelease-1>", self.predict_digit)

    def draw_digit(self, event):
        x, y = event.x, event.y
        # Black brush strokes on white background
        self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="white", width=6)
        self.draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill="white")

    def predict_digit(self, event):
        # Make a prediction when the mouse button is released
        digit = predict_digit(self.image)
        self.label.config(text=f"Predicted: {digit}")

    def clear_canvas(self):
        # Clear the canvas and reset the PIL image
        self.canvas.delete("all")
        self.image = Image.new("RGB", (280, 280), (0, 0, 0))  # Reset to white background
        self.draw = ImageDraw.Draw(self.image)
        # Clear the predicted number label
        self.label.config(text="Draw a digit...")

# Create the tkinter application
root = tk.Tk()
app = DigitApp(root)
root.mainloop()
