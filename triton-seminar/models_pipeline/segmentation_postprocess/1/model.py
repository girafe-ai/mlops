import numpy as np
import triton_python_backend_utils as pb_utils

CITYSCAPES_TRAIN_ID_TO_COLOR = np.asarray(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ],
    dtype=np.uint8,
)


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            logits = pb_utils.get_input_tensor_by_name(request, "LOGITS").as_numpy()
            pred = logits.argmax(axis=1).astype(np.uint8)
            color = CITYSCAPES_TRAIN_ID_TO_COLOR[pred]
            hist = np.bincount(pred.reshape(-1), minlength=19).astype(np.uint64)

            shifted = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(shifted).astype(np.float32)
            probs = exp / exp.sum(axis=1, keepdims=True)
            mean_confidence = probs.max(axis=1).mean(dtype=np.float32)
            mean_confidence = np.asarray([mean_confidence], dtype=np.float32)

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("PRED_MASK", pred),
                        pb_utils.Tensor("COLOR_MASK", color),
                        pb_utils.Tensor("CLASS_HISTOGRAM", hist),
                        pb_utils.Tensor("MEAN_CONFIDENCE", mean_confidence),
                    ]
                )
            )
        return responses
