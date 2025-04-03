import numpy as np
from PIL import ImageOps, Image

def classify(image, model, class_names):
    try:
        # Preprocess the image
        image = ImageOps.fit(image, (150, 150), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = image_array.astype(np.float32) / 255.0

        # Prepare data for prediction
        data = np.expand_dims(normalized_image_array, axis=0)

        # Perform prediction
        prediction = model.predict(data)
        class_index = int(prediction[0] > 0.5)  # 0 or 1 based on threshold

        class_name = class_names[class_index]
        confidence_score = prediction[0][0] if class_index == 1 else 1 - prediction[0][0]

        return class_name, confidence_score
    except Exception as e:
        raise ValueError(f"Error in classify function: {e}")