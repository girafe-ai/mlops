import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import triton_python_backend_utils as pb_utils



OUTPUT_DTYPE = np.float32

class TritonPythonModel:
    def initialize(self, args):
        self.output_dtype = OUTPUT_DTYPE
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

    def _preprocess_batch(self, image_paths):
        processed_images = []

        for path in image_paths:
            path_str = path.decode("utf-8")
            img = Image.open(path_str).convert("RGB")
            img_tensor = self.transform(img)
            processed_images.append(img_tensor.numpy())

        return np.stack(processed_images).astype(self.output_dtype)

    def execute(self, requests):
        responses = []

        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE_PATH")
            image_paths = input_tensor.as_numpy()
            img_array = self._preprocess_batch(image_paths)

            output_tensor = pb_utils.Tensor("PREPROCESSED_IMAGE", img_array)
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)

        return responses

