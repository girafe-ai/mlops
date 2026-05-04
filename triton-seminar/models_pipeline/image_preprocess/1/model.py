import io

import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image

HEIGHT = 1024
WIDTH = 2048
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            image_bytes = pb_utils.get_input_tensor_by_name(
                request, "IMAGE_BYTES"
            ).as_numpy()
            encoded = image_bytes.reshape(-1).astype(np.uint8).tobytes()

            image = Image.open(io.BytesIO(encoded)).convert("RGB")
            image = image.resize((WIDTH, HEIGHT), Image.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
            chw = np.transpose(array, (2, 0, 1))
            normalized = ((chw - MEAN) / STD)[None, ...].astype(np.float32)

            output = pb_utils.Tensor("IMAGE_TENSOR", normalized)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output]))
        return responses
