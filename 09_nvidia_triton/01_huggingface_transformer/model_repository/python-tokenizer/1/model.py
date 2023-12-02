import numpy as np
from transformers import AutoTokenizer
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/assets/rubert-tokenizer", local_files_only=True
        )

    def tokenize(self, texts):
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            max_length=16,
            truncation=True,
        )
        input_ids = np.array(encoded["input_ids"], dtype=np.int64)
        attention_mask = np.array(encoded["attention_mask"], dtype=np.int64)
        return input_ids, attention_mask

    def execute(self, requests):
        responses = []
        for request in requests:
            texts = pb_utils.get_input_tensor_by_name(request, "TEXTS").as_numpy()
            texts = [el.decode() for el in texts]

            tokens, mask = self.tokenize(texts)

            output_tensor_tokens = pb_utils.Tensor("INPUT_IDS", tokens)
            output_tensor_mask = pb_utils.Tensor("ATTENTION_MASK", mask)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor_tokens, output_tensor_mask]
            )
            responses.append(inference_response)
        return responses
