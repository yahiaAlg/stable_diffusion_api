import requests
import base64
from PIL import Image
from io import BytesIO
import time


def test_api(
    prompt,
    api_url="http://localhost:8000",
    api_key="your-secret-key-here",
    negative_prompt="",
    guidance_scale=7.5,
    num_inference_steps=30,
):
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
    }

    try:
        print("Sending request...")
        start_time = time.time()

        response = requests.post(
            f"{api_url}/generate/",
            headers=headers,
            json=data,
            timeout=300,  # 5 minutes timeout
        )
        response.raise_for_status()

        result = response.json()
        elapsed_time = time.time() - start_time
        print(f"Generation took {elapsed_time:.2f} seconds")

        if result["status"] == "success":
            image_data = base64.b64decode(result["image"])
            image = Image.open(BytesIO(image_data))
            return image
        else:
            raise Exception(result.get("message", "Unknown error"))

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {str(e)}")
        return None


if __name__ == "__main__":
    # Test the API
    prompt = "a beautiful sunset over the ocean, hyperrealistic, 8k"
    image = test_api(prompt)

    if image:
        image.save("generated_image.jpg", quality=90, optimize=True)
        print("Image saved as generated_image.jpg")
