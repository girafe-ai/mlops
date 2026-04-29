import numpy as np
import tritonclient.http as httpclient


TRITON_URL = "localhost:8000"
MODEL_NAME = "cityscapes"

BATCH_SIZE = 2
HEIGHT = 1024
WIDTH = 2048

client = httpclient.InferenceServerClient(url=TRITON_URL)

input_data = np.random.rand(
    BATCH_SIZE, 3, HEIGHT, WIDTH
).astype(np.float32)

inputs = [
    httpclient.InferInput(
        "input",
        input_data.shape,
        "FP32",
    )
]
inputs[0].set_data_from_numpy(input_data)

outputs = [
    httpclient.InferRequestedOutput("logits")
]

response = client.infer(
    model_name=MODEL_NAME,
    inputs=inputs,
    outputs=outputs,
)

logits = response.as_numpy("logits")

print("Input shape:", input_data.shape)
print("Output shape:", logits.shape)
print("Output dtype:", logits.dtype)
print("Output min:", logits.min())
print("Output max:", logits.max())
print("Output mean:", logits.mean())