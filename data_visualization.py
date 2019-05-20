from matplotlib import image as img
from matplotlib import pyplot as plt
from constants import TRAIN_DATA_DIR
import os

expressions = os.listdir(TRAIN_DATA_DIR)

rows = len(expressions)
columns = 6
index = 0

fig = plt.figure(figsize=(10,10))

for expression in expressions:
    expression_dir = TRAIN_DATA_DIR + expression + '/'
    image_files = os.listdir(expression_dir)
    for i in range(1, 7):
        image = img.imread(expression_dir + image_files[i])
        index += 1
        fig.add_subplot(rows, columns, index)
        image_plot = plt.imshow(image, cmap='gray')

    print(expression + ' files:' + str(len(image_files)))

plt.show()
