import base64
from openai import OpenAI
from PIL import Image
import os
import io
import base64



class GPT:
    def __init__(self, key, model_name):
        self.client = OpenAI(api_key=key)
        self.model_name = model_name
    
    def resize_image(self, image):
        # Load and resize the image
        img_resized = image.resize((480, 480))

        # Save the resized image to a bytes buffer instead of a file
        buffer = io.BytesIO()
        img_resized.save(buffer, format="JPEG")
        buffer.seek(0)

        # Option 1: Get raw bytes
        image_bytes = buffer.getvalue()

        # Option 2: Get base64-encoded string (useful for JSON or web APIs)
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        return image_base64

    def get_response(self, text, image=None):
        # print(f' path: {os.path.join(self.images_dir, image_fname)}')
        if image is None:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            { "type": "text", "text": text }
                        ],
                    }
                ],
            )
            return response.choices[0].message.content
        if isinstance(image, str): # If boxed image path is sent instead of raw image, it has to be read from the path
            if os.path.exists(os.path.join(image)) is False:
                return None
            image = Image.open(image)
        image = self.resize_image(image=image)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": text },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    

