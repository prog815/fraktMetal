import tensorflow
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt

model = tensorflow.keras.models.load_model("/kaggle/input/liner-metal-keras-model-250/model_keras.h5")
classes = [  "blue" ,  "green" ,  "red" ,  ]

img = Image.open( "/kaggle/input/liner-metal-keras-model-250/Images/1.png" ).convert('RGB')
img = img.resize((256 , 256), Image.LANCZOS)
print('Исходное изображение')
# plt.imshow(img)
# plt.show()


inp_numpy = np.array( img )[None]

print('Размеченное изображение')
img_annot = Image.open( "/kaggle/input/liner-metal-keras-model-250/Annotations/1.png" ).convert('RGB')
img_annot = img_annot.resize((256 , 256), Image.LANCZOS)
# plt.imshow(img_annot)
# plt.show()

print('Распознанное изображение')
segmentation_output = model.predict( inp_numpy )[0].argmax(-1)

# Исходный массив
arr = segmentation_output

rgb_arr = np.zeros((128, 128, 3), dtype=int)

dicRGB = {2: (255,0,0), 1: (0,255,0), 0:(0,0,255)}

for i in range(128):
    for j in range(128):
        rgb_arr[i, j] = dicRGB[arr[i,j]]

# plt.imshow(rgb_arr)
print("завершено")