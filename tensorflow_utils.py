import base64
from io import BytesIO

import numpy as np
from PIL import Image
from tensorflow import expand_dims
from tensorflow.keras.utils import get_file, img_to_array, load_img
from tensorflow.nn import softmax

class_predictions = np.array([
    'apple pie', 'baby back ribs', 'baklava', 'beef carpaccio', 'beef tartare',
    'beet salad', 'beignets', 'bibimbap', 'bread pudding', 'breakfast burrito',
    'bruschetta', 'caesar salad', 'cannoli', 'caprese salad', 'carrot cake',
    'ceviche', 'cheesecake', 'cheese plate', 'chicken curry',
    'chicken quesadilla', 'chicken wings', 'chocolate cake',
    'chocolate mousse', 'churros', 'clam chowder', 'club sandwich',
    'crab cakes', 'creme brulee', 'croque madame', 'cup cakes', 'deviled eggs',
    'donuts', 'dumplings', 'edamame', 'eggs benedict', 'escargots', 'falafel',
    'filet mignon', 'fish and chips', 'foie gras', 'french fries',
    'french onion soup', 'french toast', 'fried calamari', 'fried rice',
    'frozen yogurt', 'garlic bread', 'gnocchi', 'greek salad',
    'grilled cheese sandwich', 'grilled salmon', 'guacamole', 'gyoza',
    'hamburger', 'hot and sour soup', 'hot dog', 'huevos rancheros', 'hummus',
    'ice cream', 'lasagna', 'lobster bisque', 'lobster roll sandwich',
    'macaroni and cheese', 'macarons', 'miso soup', 'mussels', 'nachos',
    'omelette', 'onion rings', 'oysters', 'pad thai', 'paella', 'pancakes',
    'panna cotta', 'peking duck', 'pho', 'pizza', 'pork chop', 'poutine',
    'prime rib', 'pulled pork sandwich', 'ramen', 'ravioli', 'red velvet cake',
    'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed salad',
    'shrimp and grits', 'spaghetti bolognese', 'spaghetti carbonara',
    'spring rolls', 'steak', 'strawberry shortcake', 'sushi', 'tacos',
    'takoyaki', 'tiramisu', 'tuna tartare', 'waffles'
])


def model_predict(img_array, model):
    pred = model.predict(img_array)
    score = softmax(pred[0])
    return score


def load_image(image_link, image_data):
    if image_link != "":
        img_path = get_file(origin=image_link)
        img = load_img(img_path, target_size=(224, 224))

    elif image_data:
        img = Image.open(BytesIO(base64.b64decode(str(image_data))))
        img = img.resize((224, 224))

    img_array = img_to_array(img)
    img_array = expand_dims(img_array, 0)
    return img_array
