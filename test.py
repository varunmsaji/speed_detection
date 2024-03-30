import ultralytics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pickle
image = Image.open('traffic.jpg')

# plt.imshow(image)
# plt.axis('off')
# plt.show()

with open('cached_model.pkl','rb') as f:
    model = pickle.load(f)

# print(model)

results = model.predict(image)

results = results[0].boxes.data
results = results.detach().cpu().numpy()
df = pd.DataFrame(results).astype("float")



