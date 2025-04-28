import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import triton_python_backend_utils as pb_utils



OUTPUT_DTYPE = np.int32


class TritonPythonModel:
    def initialize(self, args):
        self.output_dtype = OUTPUT_DTYPE

    def _get_classes(self, logits):
        preds = np.argmax(logits, axis=1)
        return preds.astype(self.output_dtype)

    def execute(self, requests):
        responses = []
        for request in requests:
            logits_tensor = pb_utils.get_input_tensor_by_name(request, "LOGITS")
            logits = logits_tensor.as_numpy()

            preds = self._get_classes(logits)
            preds = np.expand_dims(preds, axis=1)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("PREDICTED_CLASS", preds),
                ]
            )
            responses.append(inference_response)

        return responses