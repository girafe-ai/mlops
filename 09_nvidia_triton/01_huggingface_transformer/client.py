from functools import lru_cache

import numpy as np
import torch
from transformers import AutoTokenizer
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def call_triton_embedder_ensembele(text: str):
    triton_client = get_client()
    text = np.array([text.encode("utf-8")], dtype=object)

    input_text = InferInput(
        name="TEXTS", shape=text.shape, datatype=np_to_triton_dtype(text.dtype)
    )
    input_text.set_data_from_numpy(text, binary_data=True)

    infer_output = InferRequestedOutput("EMBEDDINGS", binary_data=True)
    query_response = triton_client.infer(
        "ensemble-onnx", [input_text], outputs=[infer_output]
    )
    embeddings = query_response.as_numpy("EMBEDDINGS")[0]
    return embeddings


def call_triton_tokenizer(text: str):
    triton_client = get_client()
    text = np.array([text.encode("utf-8")], dtype=object)

    input_text = InferInput(
        name="TEXTS", shape=text.shape, datatype=np_to_triton_dtype(text.dtype)
    )
    input_text.set_data_from_numpy(text, binary_data=True)

    query_response = triton_client.infer(
        "python-tokenizer",
        [input_text],
        outputs=[
            InferRequestedOutput("INPUT_IDS", binary_data=True),
            InferRequestedOutput("ATTENTION_MASK", binary_data=True),
        ],
    )
    input_ids = query_response.as_numpy("INPUT_IDS")[0]
    attention_massk = query_response.as_numpy("ATTENTION_MASK")[0]
    return input_ids, attention_massk


def main():
    texts = [
        "Шла Маша по шоссе и сосала сушку",
        "Переходила Маша по путям откусывая печенье",
        "Переходила Маша по путям откусывая крендель",
        "Я сажаю алюминевые огурцы",
    ]
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruBert-base")
    encoded = tokenizer(
        texts[0],
        padding="max_length",
        max_length=16,
        truncation=True,
    )
    input_ids, attention_mask = encoded["input_ids"], encoded["attention_mask"]
    _input_ids, _attention_mask = call_triton_tokenizer(texts[0])
    assert (input_ids == _input_ids).all() and (attention_mask == _attention_mask).all()

    embeddings = torch.tensor(
        [call_triton_embedder_ensembele(row).tolist() for row in texts]
    )
    distances = torch.cdist(
        x1=embeddings,
        x2=embeddings,
        p=2,
    )
    print(distances)


if __name__ == "__main__":
    main()
