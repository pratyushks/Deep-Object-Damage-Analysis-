from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import tensorflow as tf
from keras.models import load_model

model = load_model('model_car_11.h5')

class_names = ['undamaged', 'damaged']

def classify_image(file_path, threshold=0.5):
    image = Image.open(file_path)
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)[0]

    class_index = np.argmax(predictions)
    confidence = predictions[class_index]

    if confidence >= threshold:
        class_name = class_names[class_index]
    else:
        class_name = 'unknown'
    return class_name, confidence

def select_image():
    file_path = filedialog.askopenfilename()
    label_name, confidence = classify_image(file_path, threshold=0.5)
    img = Image.open(file_path)
    img = img.resize((300, 300))
    img = ImageTk.PhotoImage(img)
    panel.configure(image=img)
    panel.image = img
    if label_name == 'unknown':
        label.configure(text=f'Damaged \nConfidence: {100-confidence*100:.2f}%')
    else:
        label.configure(text=f'{label_name.capitalize()}\nConfidence: {confidence*100:.2f}%')


root = Tk()
root.title('Object Damage Detection')

label = Label(root, text='Please select an image', font=('Arial', 16))
panel = Label(root)
btn = Button(root, text='Select Image', command=select_image)

label.pack(padx=10, pady=10)
panel.pack(padx=10, pady=10)
btn.pack(padx=10, pady=10)

root.mainloop()
