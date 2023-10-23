import tensorflow
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt

model = tensorflow.keras.models.load_model("model_keras.h5")
classes = [  "blue" ,  "green" ,  "red" ,  ]

img = Image.open( "1.png" ).convert('RGB')
img = img.resize((256 , 256), Image.LANCZOS)
print('Исходное изображение')
# plt.imshow(img)
# plt.show()

inp_numpy = np.array( img )[None]

print('Распознанное изображение')
segmentation_output = model.predict( inp_numpy )[0].argmax(-1)

# Исходный массив
arr = segmentation_output

rgb_arr = np.zeros((128, 128, 3), dtype=np.uint8)

dicRGB = {2: (255,0,0), 1: (0,255,0), 0:(0,0,255)}

for i in range(128):
    for j in range(128):
        rgb_arr[i, j] = dicRGB[arr[i,j]]

# plt.imshow(rgb_arr)

res_image = Image.fromarray(rgb_arr,'RGB')
res_image.save('res_image.png')

print("завершено")