FROM python:3.13.3

COPY requirements.txt /tmp/requirements.txt

WORKDIR /workspace

RUN pip install --upgrade pip && \
    python -m pip install -r /tmp/requirements.txt

COPY . /workspace

CMD ["bash"]