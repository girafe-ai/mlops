FROM nvcr.io/nvidia/tritonserver:23.12-py3

COPY requirements.txt .
RUN pip3 install -r requirements.txt --index-url https://pypi.org/simple

ENTRYPOINT [ "tritonserver" ]
