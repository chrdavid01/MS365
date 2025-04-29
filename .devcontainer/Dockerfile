FROM python:3.13.3

WORKDIR /workspace

RUN pip install --upgrade pip && \
    pip install black pandas kagglehub scikit-learn matplotlib ipykernel

COPY . /workspace

CMD ["bash"]