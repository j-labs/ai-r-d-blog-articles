# Deployment of LLM Inference Engines on TPU – Experiences with vLLM

## Introduction

Tensor Processing Units (TPUs) are processors dedicated to computations commonly found in artificial intelligence
workloads. They offer high performance, a low cost per computation, and are increasingly well supported by popular
frameworks. In this article, I focus on running large language models on TPUs in Google Cloud using vLLM, and I evaluate
the benefits and limitations of this approach.

## A Brief Overview of TPUs

TPUs were designed by Google engineers and optimized primarily for matrix/tensor multiplication and addition—operations
that account for over 90% of machine learning computations. Their core component is the MXU (Matrix Multiply Unit), a
massive matrix engine capable of performing billions of such operations in a single clock cycle. TPUs have integrated
HBM (High Bandwidth Memory), which feeds data to the MXU at extremely high speeds. In consumer GPUs, communication
between the GPU and its memory is slower due to the high cost of HBM and the need for specialized cooling.

TPUs require specially compiled code, which led to the creation of the XLA (Accelerated Linear Algebra) compiler for
numerical and tensor operations. Its goal is to translate high-level ML operations into highly optimized machine code.
Currently, XLA natively supports TensorFlow and JAX, while support for PyTorch is provided through the PyTorch/XLA
project.

## Inference Engine Support for TPUs

Our main focus is on vLLM and HF TGI due to their popularity and platform independence. It is also worth mentioning that
Google has developed Jetstream—an inference engine optimized for high throughput and cost efficiency, running on TPUs.

| Engine            | Base framework | TPU support     | Notes                                                                          |
|-------------------|----------------|-----------------|--------------------------------------------------------------------------------|
| vLLM              | PyTorch        | Via PyTorch/XLA | Stable, official tutorials in Google documentation for TPU deployment          |
| TGI (Optimum-TPU) | PyTorch        | Via PyTorch/XLA | The optimum-tpu project exists but has not been updated since January 21, 2025 |
| JetStream         | JAX/XLA        | Native          | Developed specifically for TPU                                                 |

## Deployment

Google published
an [article on deploying vLLM](https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-vllm-tpu), but it
contains a few pitfalls to be aware of:

1. Availability of the latest Trillium TPU (v6) is very limited, and you may not receive approval to use it
   (by default, the quota for v6e is 0 and you must request an increase from Google).
2. The article suggests using GCS Fuse, which is a relatively expensive and inefficient solution—this is discussed in
   the next section.

### Infrastructure Setup

TPUs are theoretically offered in [many regions](https://cloud.google.com/tpu/docs/regions-zones), but in practice they
are not always available “off the shelf.” I most often succeeded in the us-central1 region, which is why I created the
Kubernetes cluster there.

#### Model Storage: NFS Server vs GCS Fuse

Models take up tens of gigabytes, and to avoid downloading them on every deployment, it is worth storing them on a disk
that can later be attached to a given instance.

In its article, Google recommends connecting the deployment to GCS (Google Cloud Storage) via GCS Fuse. This technique
allows Google Cloud Storage to be treated like a disk. In practice, however, this solution turned out to be very
inefficient and expensive: downloading the gemma-3-27b-it model (about 55 GB) took 4 hours, loading it into the TPU took
40 minutes, and the cost amounted to USD 104.

In response, I decided to create an NFS (Network File System) server, which offers lower latency, higher throughput, and
whose cost is limited to maintaining a VM with such a server. In my case, downloading the model and loading it into the
TPU together took about 10 minutes.

Let’s now create a virtual machine (VM) with an NFS server installed. In the Compute Engine service, create a new VM
instance and attach an SSD disk. After connecting to the machine via SSH, install and configure the NFS server:

```shell
# --- Install NFS server ---
sudo apt install -y nfs-kernel-server

# --- Set variables: disk device and mount path ---
DISC_ID="/dev/sdb1"
MOUNT_PATH="/home/nfs_data"

# --- Create mount directory and mount the device ---
sudo mkdir -p "$MOUNT_PATH"
sudo mount "$DISC_ID" "$MOUNT_PATH"

# --- Add entry to /etc/fstab to make the mount persistent ---
UUID=$(sudo blkid -s UUID -o value "$DISC_ID")
FSTAB_LINE="UUID=$UUID $MOUNT_PATH ext4 defaults,nofail 0 2"
grep -qF "$FSTAB_LINE" /etc/fstab || echo "$FSTAB_LINE" | sudo tee -a /etc/fstab >/dev/null

# --- Set ownership for NFS data (anonymous NFS user) ---
sudo chown nobody:nogroup "$MOUNT_PATH"

# --- Configure export of the directory for NFS clients ---
echo "/home/nfs_data *(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports

# --- Restart NFS server to apply the new configuration ---
sudo systemctl restart nfs-kernel-server

# --- Verification ---
echo "=== Mount ==="; findmnt "$MOUNT_PATH" || true
echo "=== NFS exports ==="; sudo exportfs -v | grep "$MOUNT_PATH" || true
echo "=== NFS port ==="; sudo ss -tulnp | grep :2049 || true
```

#### Creating a Kubernetes Cluster with TPU

1. First, set a few environment variables in the console (adjust them to your project and cluster):

```shell
gcloud config set project ai-r-d-467405 && \
gcloud config set billing/quota_project ai-r-d-467405 && \
export PROJECT_ID=$(gcloud config get project) && \
export PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)") && \
export CLUSTER_NAME=cluster-9 && \
export CONTROL_PLANE_LOCATION=us-central1 && \
export ZONE=us-central1-a && \
export CLUSTER_VERSION=1.33.2-gke.1240000
```

2. Next, create a node pool with TPUv5e:

```shell
gcloud container node-pools create tpunodepool \
    --location=${CONTROL_PLANE_LOCATION} \
    --node-locations=${ZONE} \
    --num-nodes=1 \
    --machine-type=ct5lp-hightpu-4t \
    --cluster=${CLUSTER_NAME} \
    --enable-autoscaling --total-min-nodes=0 --total-max-nodes=1
```

3. We also need to create a standard node pool to ensure that DNS is enabled in the cluster:

```shell
gcloud beta container --project ${PROJECT_ID} node-pools create "dns-pool" \
    --cluster=${CLUSTER_NAME} \
    --region ${CONTROL_PLANE_LOCATION} \
    --node-version ${CLUSTER_VERSION} \
    --machine-type "e2-medium" \
    --image-type "COS_CONTAINERD" \
    --disk-type "pd-balanced" \
    --disk-size "20" \
    --metadata disable-legacy-endpoints=true \
    --num-nodes "1" \
    --enable-autoscaling \
    --min-nodes "0" \
    --max-nodes "1" \
    --location-policy "BALANCED" \
    --enable-autoupgrade \
    --enable-autorepair \
    --max-surge-upgrade 1 \
    --max-unavailable-upgrade 0 \
    --shielded-integrity-monitoring \
    --no-shielded-secure-boot
```

#### Application Deployment

Before the deployment, create a Secret to store the Hugging Face access token:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: hf-token-secret
type: Opaque
data:
  HUGGINGFACE_TOKEN: "insert-your-base64-encoded-token-here"
```

Provide a Persistent Volume:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-sync-pv
spec:
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: nfs-storageclass
  mountOptions:
    - hard
    - nfsvers=4.1
  nfs:
    server: 10.128.0.43 # NFS server IP here
    path: /home/nfs_data
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-sync-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  volumeName: model-sync-pv
  storageClassName: nfs-storageclass
```

Next, create the Deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: release-name-vllm-tpu
  labels:
    app: release-name-vllm-tpu
spec:
  replicas: 1
  selector:
    matchLabels:
      app: release-name-vllm-tpu
  template:
    metadata:
      labels:
        app: release-name-vllm-tpu
    spec:
      nodeSelector:
        cloud.google.com/gke-tpu-topology: 2x2
        cloud.google.com/gke-tpu-accelerator: tpu-v5-lite-podslice
      containers:
        - name: vllm-tpu
          image: "docker.io/vllm/vllm-tpu:73aa7041bfee43581314e6f34e9a657137ecc092"
          imagePullPolicy: IfNotPresent
          command: [ "python3", "-m", "vllm.entrypoints.openai.api_server" ]
          args:
            - --host=0.0.0.0
            - --port=8000
            - --tensor-parallel-size=4
            - --max-model-len=4096
            - --model=Bedovyy/Qwen3-32B.w8a8
            - --download-dir=/data
            - --max-num-batched-tokens=512
            - --max-num-seqs=64
          env:
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token-secret
                  key: HUGGINGFACE_TOKEN
            - name: VLLM_XLA_CACHE_PATH
              value: "/data"
            - name: VLLM_USE_V1
              value: "1"
          ports:
            - containerPort: 8000
          resources:
            limits:
              google.com/tpu: 4
          readinessProbe:
            tcpSocket:
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 10
          volumeMounts:
            - name: model-volume
              mountPath: /data
            - name: dshm
              mountPath: /dev/shm
      volumes:
        - name: model-volume
          persistentVolumeClaim:
            claimName: model-sync-pvc
        - name: dshm
          emptyDir:
            medium: Memory
```

Add a Service:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: release-name-vllm-tpu-service
  labels:
    app: release-name-vllm-tpu
spec:
  selector:
    app: release-name-vllm-tpu
  type: LoadBalancer
  ports:
    - name: http
      protocol: TCP
      port: 8000
      targetPort: 8000
```

Done! To communicate with the deployed LLM, simply use port forwarding:

```shell
kubectl port-forward svc/release-name-vllm-tpu-service 3000:8000
```

## Benchmarks

### Impact of Concurrency and the max-num-seqs Parameter

To test performance, I used a benchmarking tool provided by vLLM. The parameters were chosen to reflect typical RAG
system usage: long input and short output.

```shell
vllm bench serve \
  --backend vllm \
  --base-url http://34.171.196.175:8000 \
  --endpoint /v1/completions \
  --model Bedovyy/Qwen3-32B \
  --tokenizer Qwen/Qwen3-32B \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 500 \
  --max-concurrency 32 \
  --ramp-up-strategy linear \
  --ramp-up-start-rps 1 \
  --ramp-up-end-rps 8 \
  --num-prompts 150
```

The tests were conducted on the `Bedovyy/Qwen3-32B.w8a8` model, with `max-num-batched-tokens=512` (the maximum number of
tokens that can be processed simultaneously), for different values of `max-num-seqs` (the maximum number of concurrent
inference requests processed by vLLM) and max-concurrency (the number of requests sent in parallel by the benchmark).

#### max-concurrency = 32

| max-num-seqs | max-num-batched-tokens | Request throughput | Output throughput | Total throughput | P50 TTFT (ms) | Time per output token (ms) | Inter-token Latency Median (ms) |
|--------------|------------------------|--------------------|-------------------|------------------|---------------|----------------------------|---------------------------------|
| 64           | 512                    | 0.45               | 188.19            | 3800.10          | 47276.68      | 56.49                      | 83.85                           |
| 128          | 512                    | 0.46               | 188.09            | 3862.02          | 46343.69      | 56.38                      | 82.23                           |
| 256          | 512                    | 0.46               | 187.69            | 3835.79          | 46493.28      | 56.66                      | 84.20                           |
| 512          | 512                    | 0.45               | 189.58            | 3805.25          | 46776.08      | 56.06                      | 83.92                           |

#### max-concurrency = 8

| max-num-seqs | max-num-batched-tokens | Request throughput | Output throughput | Total throughput | P50 TTFT (ms) | Time per output token (ms) | Inter-token Latency Median (ms) |
|--------------|------------------------|--------------------|-------------------|------------------|---------------|----------------------------|---------------------------------|
| **64**       | 512                    | 0.41               | 161.40            | 3420.13          | 1740.06       | 43.00                      | 63.81                           |
| **128**      | 512                    | 0.41               | 160.74            | 3402.02          | 1738.32       | 43.02                      | 62.83                           |
| **256**      | 512                    | 0.40               | 165.94            | 3359.19          | 1749.00       | 42.58                      | 63.35                           |
| **512**      | 512                    | 0.40               | 158.44            | 3367.61          | 1750.68       | 43.21                      | 64.73                           |

#### Observations

When running the above tests, I expected to observe dependencies between `max-num-seqs` and performance, as suggested by
the documentation. However, the data revealed some discrepancies.

1. **Request and token throughput**
    * Throughput depends on max-concurrency (as expected).
    * Throughput is practically independent of `max-num-seqs` (as expected).
    * Superficial tests with changing `max-num-batched-tokens` did not show a dependency of throughput on this parameter
      (contrary to the documentation).
2. **TTFT (Time To First Token) – lack of expected improvement**
    * In **this** configuration, `TTFT` barely changed with `max-num-seqs`, even though the documentation suggests that
      with increasing `max-num-seqs`, `TTFT` may decrease if requests were previously queued.
3. **Latency and single-token generation time**
    * The `max-num-seqs` parameter has virtually no impact on `TTFT` and `ITL` (Inter-token latency).
    * Both metrics increase only with max-concurrency.

### Impact of max-model-len and Prompt Length

If `max-model-len` is too large and forces pre-allocation of matrices or affects the shape of the input/output matrices,
which are then divided into computation blocks, while the actual sequence length is small, this could lead to
inefficient
use of compute units (so-called tail overhead or padding).

I therefore ran tests for different max-model-len values with the following base parameters:

* `max-num-seqs`: 512
* `max-num-batched-tokens`: 512
* `number of parallel requests`: 300

| max-model-len | Prompt length | Prompt tokens / s | Generation tokens/s | P50 TTFT (ms) | Time per output token (ms) | Inter-token Latency Median (ms) |
|---------------|---------------|-------------------|---------------------|---------------|----------------------------|---------------------------------|
| **16384**     | **3000**      | 8991              | 843                 | 577           | 0.35                       | 89                              |
| **16384**     | **8000**      | 9682              | 466                 | 1342          | 0.62                       | 179                             |
| **16384**     | **16000**     | 16163             | 280                 | 1750          | 1.75                       | 356                             |
| **32768**     | **3000**      | 7773              | 759                 | 624           | 0.356                      | 89                              |
| **32768**     | **8000**      | 10047             | 442                 | 1608          | 0.625                      | 167                             |
| **32768**     | **16000**     | 16003             | 270                 | 1750          | 1.75                       | 360                             |

For the same prompt length with different `max-model-len` values (16384 and 32768), I obtained very similar
measurements.
For this specific configuration, changing `max-model-len` from 16384 to 32768 had no noticeable impact on performance.
In general, `max-model-len` affects memory usage and the maximum values of `max-num-seqs` / `max-num-batched-tokens`,
so under different settings it may be more significant.

This suggests that `max-model-len` is not a parameter that affects performance in this system. Potentially, this may be
due to other optimizations such as PagedAttention (in vLLM)
and [Dynamic Shapes](https://docs.pytorch.org/xla/master/learn/dynamic_shape.html).

## Summary

Running LLMs on TPUs using vLLM in Google Cloud turns out to be a viable and cost-effective approach, provided the
infrastructure is properly prepared. TPUs offer high throughput and a favorable cost-to-performance ratio, but their
use comes with certain challenges, including limited availability of the latest versions, the need to configure the
environment using XLA, and suboptimal default approaches to model storage (GCS Fuse).

Replacing GCS Fuse with a custom NFS server significantly reduced model loading times and lowered costs. Tests showed
that a properly configured vLLM on TPUv5e can handle a large volume of requests with high performance—even over
1,000 generated tokens per second—at a monthly cost of around USD 1,576 (with a 3-year commitment) or USD 3,504
without a 3-year commitment.

Conclusions:

* TPUs are a good option for high-performance LLM inference, especially in production environments.
* It is worth investing time in infrastructure optimization (NFS, appropriate vLLM parameters).
* In the conducted tests on `TPUv5e`, for the `Qwen3-32B.w8a8` model and `max-num-batched-tokens=512`, changes to
  `max-num-seqs` had a negligible impact on `TTFT` and `throughput`—in practice, `max-concurrency` proved to be the key
  factor. In other configurations (different models, longer prompts, larger `max-num-batched-tokens`), vLLM scheduler
  parameters may affect performance as described in the documentation.

## Useful Links

* [How TPU works](https://medium.com/@ruslanbredun007/what-is-tpu-and-how-why-it-works-9a0a4a59399e)
* [Monitoring vLLM in GCP](https://cloud.google.com/stackdriver/docs/managed-prometheus/exporters/vllm)
* [vLLM parameter optimization diagram](https://github.com/vllm-project/vllm/issues/2492)
* [vLLM optimization documentation](https://docs.vllm.ai/en/latest/configuration/optimization.html#optimization-and-tuning)
* [Similar benchmarks by Miko.ai on other models and compute units](https://engineering.miko.ai/navigating-the-ai-compute-maze-a-deep-dive-into-google-tpus-nvidia-gpus-and-llm-benchmarking-5332339e4c9b)
