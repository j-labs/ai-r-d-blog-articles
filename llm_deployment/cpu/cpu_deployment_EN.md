# Deployment of LLM Inference Engines on CPU – Experiences with Text Generation Inference (TGI)

## Introduction

Can running models on CPUs be an alternative to GPUs or TPUs? A lower hourly unit cost, higher availability, and simpler
configuration are reasons we might consider it. In this article, I focus on attempting to run selected models on
different machines in GCP (Google Cloud Platform) and checking how quickly they can generate tokens.

## Models Used in the Experiments

Parameters in language models are numerical values (weights) in a neural network that are created during training. They
store linguistic patterns and knowledge, allowing the model to recognize context, generate responses, or solve tasks.
The larger the number of parameters, the greater the model’s potential representational capacity (*model capacity*,
i.e., the ability to memorize, represent, and use complex patterns in data), but at the same time the computational cost
and hardware requirements increase.

OpenAI-class models such as o4-mini and o3-mini (more precisely gpt-oss-120b and gpt-oss-20b) have roughly 117B and 21B
parameters respectively. The models we tested are smaller, but still offer functionality sufficient for many practical
applications. For comparison, I will use several models of different scales:

| Model       | Number of parameters | Typical use cases                          |
|-------------|---------------------:|--------------------------------------------|
| Qwen2.5-3B  |                   3B | Lightweight assistant, text analysis       |
| Qwen2.5-7B  |                   7B | Code generation, mid-tier chatbots         |
| Qwen2.5-14B |                  14B | Quality comparable to commercial solutions |

In an article comparing inference engines, we can find information that Text Generation Inference (TGI) has good CPU
support and is intended for production use, so I focus here on using it with language models.

## Model Testing

[Hugging Face](https://huggingface.co/) provides several standard builds. One of them is
`ghcr.io/huggingface/text-generation-inference:latest-intel-cpu`, and that is the one I use first.

The full Kubernetes manifest used in the experiment is as follows:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: text-generation-inference
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: text-generation-inference
  template:
    metadata:
      labels:
        app.kubernetes.io/name: text-generation-inference
    spec:
      containers:
        - name: text-generation-inference
          image: ghcr.io/huggingface/text-generation-inference:sha-9f38d93-intel-cpu
          livenessProbe:
            httpGet:
              path: /health
              port: 80
            initialDelaySeconds: 10
            periodSeconds: 10
            timeoutSeconds: 2
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health
              port: 80
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 2
            failureThreshold: 3
          startupProbe:
            httpGet:
              path: /health
              port: 80
            failureThreshold: 30
            periodSeconds: 30
          resources:
            requests:
              cpu: "30"
              memory: "55Gi"
            limits:
              cpu: "31"
              memory: "55Gi"
          args:
            - --model-id
            - "{{ .Values.modelId }}"
          ports:
            - name: http
              containerPort: 80
          securityContext:
            privileged: false
          volumeMounts:
            - name: model-volume
              mountPath: /model-repository
              readOnly: true
      volumes:
        - name: model-volume
          persistentVolumeClaim:
            claimName: model-sync-pvc
```

Model quality was tested using the following request, and performance using a similar configuration with different
inputs:

```shell
curl --location 'http://127.0.0.1:3000/generate' \
--header 'Content-Type: application/json' \
--data '{
  "inputs": "Explain what quantum coherence is.",
  "parameters": {
    "adapter_id": "null",
    "best_of": 1,
    "decoder_input_details": false,
    "details": false,
    "do_sample": true,
    "frequency_penalty": 0.1,
    "grammar": null,
    "max_new_tokens": 100,
    "repetition_penalty": 1.03,
    "return_full_text": false,
    "seed": null,
    "temperature": 0.5,
    "top_k": 1,
    "top_p": 0.95,
    "truncate": null,
    "typical_p": 0.95,
    "watermark": true
  }
}'
```

The outputs for individual models are as follows:

* Qwen-2.5-14B:

```json
{
  "generated_text": " Quantum coherence is a concept from quantum physics and string theory that refers to a theoretical consequence arising from attempts to unify different physical theories. Here are a few key points that may help in understanding this concept:\n\n1. **Unification of fundamental forces**"
}
```

* Qwen-2.5-7B:

```json
{
  "generated_text": " Quantum coherence is a concept in quantum physics that refers to the relationships between quantum states. It is especially important in quantum field theory and quantum electrodynamics (QED). Here are a few key points that explain what quantum coherence is:\n\n1. Q"
}
```

* Qwen-2.5-3B:

```json
{
  "generated_text": " Give an example in which quantum coherence is crucial to solving a problem.\nQuantum coherence is a physical concept related to quantum mechanics theory that describes the connection between two signals or objects such that changes in one signal are at least partially transmitted to the other. This sp"
}
```

| Model        | Machine type  | vCPU | RAM  | Time for 100 tokens | Max time in queue | CPU usage | Memory usage | Monthly cost |
|--------------|---------------|------|------|---------------------|-------------------|-----------|--------------|--------------|
| Qwen2.5-14B  | c4-highcpu-32 | 32   | 64GB | 23.97 s             | ~0.1s             | 50.89%    | 87.60%       | 1094 USD     |
| Qwen2.5-7B   | c4-highcpu-32 | 32   | 64GB | 15.72 s             | ~0.1s             | 52.7%     | 21%          | 1094 USD     |
| Qwen2.5-3B   | c4-highcpu-32 | 32   | 64GB | 7.1 s               | ~0.1s             | 48.37%    | 20.97%       | 1094 USD     |
| Qwen2.5-1.5B | c4-highcpu-32 | 32   | 64GB | 4.73 s              | ~0.1s             | 33.98%    | 12.576%      | 1094 USD     |
| GPT-4        | -             | -    | -    | 8 s                 | -                 | -         | -            | -            |
| GPT-4o       | -             | -    | -    | 0.97 s              | -                 | -         | -            | -            |

The above tests were conducted for 10 users. They show that the server handles such load without issues, and generating
100 tokens takes about as long as reading a similar text aloud at a moderate pace (around 7–8 seconds). It is therefore
clear that only a model of about 3 billion parameters reaches a speed close to the subjectively perceived “read-aloud
speed,” i.e., similar to GPT-4.

## Increasing the Load

In the next test, I checked how increasing the load to 100 users would affect these parameters. This has a significant
impact on the scalability of such an application.

| Model        | Machine type  | vCPU | RAM  | Time for 100 tokens | Max time in queue | CPU usage | Memory usage | Monthly cost |
|--------------|---------------|------|------|---------------------|-------------------|-----------|--------------|--------------|
| Qwen2.5-14B  | c4-highcpu-32 | 32   | 64GB | 53.2 s              | ~0.4s             | 51.5%     | 90.48%       | 1094 USD     |
| Qwen2.5-7B   | c4-highcpu-32 | 32   | 64GB | 23.7 s              | ~0.3s             | 50.97     | 47.51%       | 1094 USD     |
| Qwen2.5-1.5B | c4-highcpu-32 | 32   | 64GB | 7.1 s               | ~0.3s             | 51.98%    | 12.98%       | 1094 USD     |

Increasing the load caused roughly a twofold increase in response generation time and slightly affected the time
requests spent in the queue. It did not significantly affect CPU and memory usage.

### Increasing the Load to 1,000 Users

Increasing the load to 1,000 users increased the request queue to 2 and led to dropped requests. It did not affect CPU
and memory, so unfortunately it is difficult to find a metric for autoscaling here (even though Hugging Face recommends
using a metric related to the request queue).

## Quantization

A commonly used technique for reducing resource requirements and accelerating inference is quantization. It involves
reducing the precision of the numbers in which model weights are stored, which results in:

* a smaller model size,
* lower memory requirements,
* faster inference.

The trade-off is a loss in output quality.

I ran a few quantization tests using TGI and CPU to check whether this technique could help us achieve better results or
lower costs.

### Preparing TGI

At the time of writing, all “on-the-fly” quantization types in TGI are designed for the GPU backend. The only
theoretically working option is to use the [Llama.cpp](https://github.com/ggml-org/llama.cpp) backend, which supports
the [GGUF](https://huggingface.co/docs/hub/gguf) format that enables running pre-quantized models.

However, this requires building a dedicated version of TGI, since Hugging Face does not provide ready-made images. The [
`Dockerfile` provided in the TGI repository](https://github.com/huggingface/text-generation-inference/blob/main/Dockerfile_llamacpp)
unfortunately did not work and takes up a lot of space, so I created a custom one:

```dockerfile
FROM ubuntu:24.04 AS deps

ARG llamacpp_version=b4827
ARG llamacpp_native=ON
ARG llamacpp_cpu_arm_arch=native

WORKDIR /opt/src

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt upgrade -y && apt install -y \
    clang \
    cmake \
    curl \
    git \
    python3-dev \
    libssl-dev \
    pkg-config \
    tar \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev

ADD https://github.com/ggml-org/llama.cpp/archive/refs/tags/${llamacpp_version}.tar.gz /opt/src/
RUN mkdir -p llama.cpp \
 && tar -xzf ${llamacpp_version}.tar.gz -C llama.cpp --strip-components=1 \
 && cd llama.cpp \
 && cmake -B build \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DCMAKE_INSTALL_LIBDIR=/usr/lib \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DGGML_NATIVE=${llamacpp_native} \
    -DGGML_CPU_ARM_ARCH=${llamacpp_cpu_arm_arch} \
    -DLLAMA_BUILD_COMMON=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=OpenBLAS \
    -DGGML_BACKEND_BLAS=ON \
    -DBUILD_SHARED_LIBS=ON \
 && cmake --build build --parallel --config Release \
 && cmake --install build

WORKDIR /app
COPY rust-toolchain.toml rust-toolchain.toml
RUN curl -sSf https://sh.rustup.rs | sh -s -- --no-modify-path --default-toolchain 1.85.1 --profile minimal -y
ENV PATH="/root/.cargo/bin:$PATH"
RUN cargo install cargo-chef --locked

FROM deps AS planner
COPY ../projects/model_deployment .
RUN cargo chef prepare --recipe-path recipe.json

FROM deps AS builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook \
    --recipe-path recipe.json \
    --profile release \
    --package text-generation-router-llamacpp
COPY ../projects/model_deployment .
RUN cargo build \
    --profile release \
    --package text-generation-router-llamacpp --frozen

FROM ubuntu:24.04
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt upgrade -y && apt install -y \
    python3-venv \
    python3-pip \
    libopenblas0 \
    libblas3 \
    liblapack3

RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY backends/llamacpp/requirements.txt requirements.txt
COPY --from=builder /opt/src/llama.cpp/gguf-py gguf-py
COPY --from=builder /opt/src/llama.cpp/convert_hf_to_gguf.py /bin/

RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0 \
    && pip3 install --no-cache-dir -r requirements.txt -e gguf-py

COPY --from=builder /usr/lib/libllama.so /usr/lib/
COPY --from=builder /usr/lib/libggml*.so /usr/lib/
COPY --from=builder /app/target/release/text-generation-router-llamacpp /usr/bin/

ENV HF_HUB_ENABLE_HF_TRANSFER=1

ENTRYPOINT ["text-generation-router-llamacpp"]
```

Next, I built it using the following parameters:

```shell
docker build \
    -t tgi-llamacpp \
    --build-arg llamacpp_native=OFF \
    --build-arg llamacpp_cuda=OFF \
    --build-arg llamacpp_cpu_arm_arch=x86-64 \
    https://github.com/huggingface/text-generation-inference.git \
    -f Dockerfile_llamacpp_custom
```

### Running the Image

Below is the full Kubernetes manifest used to run the built image. Note the GGUF-related model parameters that appear (
`model-gguf`, `tokenizer-config-path`):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: text-generation-inference
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: text-generation-inference
  template:
    metadata:
      labels:
        app.kubernetes.io/name: text-generation-inference
    spec:
      containers:
        - name: text-generation-inference
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          livenessProbe:
            httpGet:
              path: /health
              port: metrics
            initialDelaySeconds: 10
            periodSeconds: 10
            timeoutSeconds: 2
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health
              port: metrics
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 2
            failureThreshold: 3
          startupProbe:
            httpGet:
              path: /health
              port: metrics
            failureThreshold: 30
            periodSeconds: 30
          resources:
            requests:
              cpu: "30"
              memory: "55Gi"
            limits:
              cpu: "31"
              memory: "55Gi"
          args:
            - --model-id
            - /model-repository/Qwen/Qwen2.5-14B-Instruct
            - --model-gguf
            - /model-repository/bartowski/Qwen2.5-14B-Instruct-GGUF/Qwen2.5-14B-Instruct-Q4_K_M.gguf
            - --tokenizer-config-path
            - /model-repository/Qwen/Qwen2.5-14B-Instruct/tokenizer_config.json
          ports:
            - name: http
              containerPort: 3000
            - name: metrics
              containerPort: 3000
          securityContext:
            privileged: false
          volumeMounts:
            - name: model-volume
              mountPath: /model-repository
              readOnly: true
      volumes:
        - name: model-volume
          persistentVolumeClaim:
            claimName: model-sync-pvc
```

### Measurements

Below are the measurements I obtained for models quantized in Q4_K_M format. The tests were conducted for 10 concurrent
users. I also included measurements on weaker machines due to the good 100-token generation times.

| Model                | Machine type  | vCPU | RAM  | Time for 100 tokens | Max time in queue | CPU usage | Memory usage | Monthly cost |
|----------------------|---------------|------|------|---------------------|-------------------|-----------|--------------|--------------|
| Qwen2.5-14B (q4_k_m) | c4-highcpu-32 | 32   | 64GB | 10.65 s             | 81 s              | 100%      | 34.29%       | 1094 USD     |
| Qwen2.5-7B (q4_k_m)  | c4-highcpu-32 | 32   | 64GB | 4.73 s              | 35 s              | 80%       | 14.44%       | 1094 USD     |
| Qwen2.5-3B (q4_k_m)  | c4-highcpu-32 | 32   | 64GB | 3.16 s              | 22.37 s           | 66.59%    | 7.9%         | 1094 USD     |
| Qwen2.5-14B (q4_k_m) | c4-highcpu-16 | 16   | 32GB | 18 s                | 145 s             | 96.74%    | 41.06%       | 547 USD      |
| Qwen2.5-7B (q4_k_m)  | c4-highcpu-16 | 16   | 32GB | 6.9 s               | 53 s              | 68.99%    | 24.07%       | 547 USD      |
| Qwen2.5-14B (q4_k_m) | n2-standard-8 | 8    | 32GB | 35.95 s             | 145 s             | 56.36%    | 34.29%       | 312 USD      |

You can see a significant improvement in 100-token generation time and RAM usage. However, *Max Queue Duration*—the time
spent waiting in the queue to be processed—increased dramatically compared to non-quantized models. Unfortunately, the
GGUF format is optimized for a single client, which means throughput for this configuration is unacceptable for
production solutions.

## Summary

The tests showed that running LLMs on CPU has, in practice, only one real advantage—significantly higher availability of
machines in the cloud compared to GPU and TPU. Inference time quickly becomes a bottleneck, especially for models above
7 billion parameters. Quantization in TGI is currently not useful in production scenarios due to scaling issues and poor
support for many concurrent users. CPU may therefore work only for prototyping, as a fallback solution, and for
lightweight models up to around 7 billion parameters—everywhere else, GPU remains the practical choice.

## Useful Links

* [Explanation of the quantization concept](https://www.digitalocean.com/community/tutorials/model-quantization-large-language-models)
* [Details on individual quantization model types](https://huggingface.co/docs/transformers/main/en//quantization)
* [Llamacpp backend](https://huggingface.co/docs/text-generation-inference/en/backends/llamacpp)
