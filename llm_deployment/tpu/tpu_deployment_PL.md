# Deployment silników inferencji LLM na TPU – doświadczenia z vLLM

## Wprowadzenie

Tensor Processing Unit (TPU) to procesory dedykowane pod obliczenia często występujące w zagadnieniach związanych ze
sztuczną inteligencją. Oferują wysoką wydajność, niski koszt przeliczeniowy, a popularne frameworki coraz lepiej je
wspierają. W tym artykule skupię się na uruchamianiu modeli językowych na TPU w Google Cloud przy użyciu vLLM, ocenię
korzyści i ograniczenia, jakie ze sobą niesie to podejście.

## Krótko o TPU

TPU zostały zaprojektowane przez inżynierów Google i zoptymalizowane przede wszystkim do mnożenia i dodawania
macierzy/tensorów – to ponad 90% obliczeń w uczeniu maszynowym. Ich sercem jest MXU (Matrix Multiply Unit), czyli
ogromna macierz wykonująca biliony takich operacji w jednym takcie zegara. Mają one zintegrowaną pamięć
HBM (High Bandwidth Memory), która dostarcza dane do MXU z ogromną szybkością. W konsumenckich GPU komunikacja między
GPU a jej pamięcią jest wolniejsza z uwagi na koszty HBM i konieczność stosowania specjalnego chłodzenia.

TPU wymaga specjalnie skompilowanego kodu, dlatego powstał kompilator XLA (Accelerated Linear Algebra) do operacji
numerycznych i tensorowych. Jego celem jest przetłumaczenie wysokopoziomowych operacji ML na wysoce zoptymalizowany
kod maszynowy. Obecnie XLA obsługuje TensorFlow i JAX natywnie, a wsparcie dla PyTorch jest zapewniane przez projekt
PyTorch/XLA.

## Wsparcie silników inferencji dla TPU

W okręgu naszych zainteresowań jest głównie vLLM i HF TGI z uwagi na ich popularność i niezależność od platformy.
Warto jednak wspomnieć, że poza nimi Google opracowało Jetstream — silnik inferencyjny zoptymalizowany pod
kątem wysokiej przepustowości i efektywności kosztowej działający na TPU.

| Silnik            | Framework bazowy | Wsparcie TPU      | Uwagi                                                                              |
|-------------------|------------------|-------------------|------------------------------------------------------------------------------------|
| vLLM              | Pytorch          | Przez Pytorch/XLA | Stabilne, oficjalne tutoriale w dokumentacji Google związane z deploymentem na TPU |
| TGI (Optimum-TPU) | Pytorch          | Przez Pytorch/XLA | Istnieje projekt optimum-tpu, ale jest nieaktualizowany od 21 stycznia 2025        |
| JetStream         | JAX/XLA          | Natywne           | Opracowane specjalnie dla TPU                                                      |

## Deployment

Google opublikowało [artykuł o deploymencie vLLM](https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-vllm-tpu)
jednak zawiera on kilka pułapek, na które należy uważać:

1. Dostępność najnowszego TPU Trillium (v6) jest mocno ograniczona i można nie dostać zgody na jego użycie
   (domyślnie Quota pod v6e wynosi 0 i trzeba prosić Google o jej zwiększenie)
2. Artykuł sugeruje użycie GCS Fuse, jednak jest to rozwiązanie relatywnie kosztowne i niezbyt wydajne — o tym w
   kolejnej części artykułu

### Tworzenie infrastruktury

TPU jest teoretycznie oferowane w [wielu regionach](https://cloud.google.com/tpu/docs/regions-zones), ale praktyka
pokazuje, że nie wszędzie można dostać TPU "od ręki". Najczęściej udawało mi się to w regionie us-central1, dlatego
właśnie tam utworzyłem klaster K8s.

#### Przechowywanie modeli: NFS Server vs GCS Fuse

Modele zajmują kilkanaście-kilkadziesiąt gigabajtów i żeby nie pobierać ich przy każdym deploymencie, warto je
przechowywać na dysku, który można następnie przyłączyć do danej instancji.

Google w swoim artykule wskazuje, żeby deployment był połączony z GCS (Google Cloud Storage) przez GCS Fuse. Jest to
technika pozwalająca traktować Google Clouds Storage (GCS) jak dysk. Rozwiązanie to w praktyce okazało się bardzo
nieefektywne i drogie: pobranie modelu gemma-3-27b-it (zajmującego ok. 55GB) zajęło 4 godziny, załadowanie go do
TPU 40 minut, a koszt wyniósł 379 zł.

W odpowiedzi postanowiłem stworzyć serwer NFS (Network File System), który oferuje mniejszą latencję, większą
przepustowość a koszt to jedynie utrzymanie VM z takim serwerem. W moim przypadku pobranie modelu i załadowanie go do
TPU łącznie zajęło ok. 10 minut.

Stwórzmy teraz maszynę wirtualną (VM) z zainstalowanym serwerem NFS. W usłudze Compute Engine utwórzmy nową instancję
VM i dołączmy do niej dysk SSD. Po nawiązaniu połączenia z maszyną za pomocą SSH zainstalujmy i skonfigurujmy serwer NFS:

```shell
# --- Instalacja serwera NFS ---
sudo apt install -y nfs-kernel-server

# --- Ustawienie zmiennych: urządzenie dyskowe i ścieżka montowania ---
DISC_ID="/dev/sdb1"
MOUNT_PATH="/home/nfs_data"

# --- Utworzenie katalogu do montowania i montowanie urządzenia ---
sudo mkdir -p "$MOUNT_PATH"
sudo mount "$DISC_ID" "$MOUNT_PATH"

# --- Dodanie wpisu do /etc/fstab, aby montowanie było trwałe ---
UUID=$(sudo blkid -s UUID -o value "$DISC_ID")
FSTAB_LINE="UUID=$UUID $MOUNT_PATH ext4 defaults,nofail 0 2"
grep -qF "$FSTAB_LINE" /etc/fstab || echo "$FSTAB_LINE" | sudo tee -a /etc/fstab >/dev/null

# --- Nadanie właściciela dla danych NFS (użytkownik anonimowy NFS) ---
sudo chown nobody:nogroup "$MOUNT_PATH"

# --- Konfiguracja eksportu katalogu dla klientów NFS ---
echo "/home/nfs_data *(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports

# --- Restart serwera NFS, aby wczytać nową konfigurację ---
sudo systemctl restart nfs-kernel-server

# --- Weryfikacja ---
echo "=== Montowanie ==="; findmnt "$MOUNT_PATH" || true
echo "=== Eksporty NFS ==="; sudo exportfs -v | grep "$MOUNT_PATH" || true
echo "=== Port NFS ==="; sudo ss -tulnp | grep :2049 || true
```

#### Stworzenie klastra w k8s z TPU

1. Najpierw w konsoli ustawmy klika zmiennych środowiskowych (należy je dostosować do swojego projektu i klastra):

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

2. Następnie stwórzmy node-pool z TPUv5e:

```shell
gcloud container node-pools create tpunodepool \
    --location=${CONTROL_PLANE_LOCATION} \
    --node-locations=${ZONE} \
    --num-nodes=1 \
    --machine-type=ct5lp-hightpu-4t \
    --cluster=${CLUSTER_NAME} \
    --enable-autoscaling --total-min-nodes=0 --total-max-nodes=1
```

3. Musimy jeszcze stworzyć standardowy node-pool, aby zapewnić, że DNS jest włączony w klastrze:

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

#### Deployment aplikacji

Przed Deploymentem stwórzmy Secret, w którym przechowamy access token do Hugging Face:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: hf-token-secret
type: Opaque
data:
  HUGGINGFACE_TOKEN: "insert-your-base64-encoded-token-here"
```

Zapewnijmy Persistent Volume:

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
    server: 10.128.0.43 # tutaj IP serwera NFS
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

Następnie utwórzmy Deployment:

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
        command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
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

Dodajmy serwis:

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

Gotowe! Aby móc się komunikować się z tak postawionym LLMem wystarczy zrobić port-forwarding:

```shell
kubectl port-forward svc/release-name-vllm-tpu-service 3000:8000
```

## Testy

### Wpływ concurrency i parametry max-num-seqs

Do wytestowania performance użyłem narzędzia przygotowanego przez vLLM służącego do benchmarkowania.
Parametry zostały dobrane tak aby wytestować typowe użycie modelu w systemie RAG: długie wejście i krótkie wyjście.

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

Testy zostały przeprowadzone na modelu `Bedovyy/Qwen3-32B.w8a8`, z parametrem `max-num-batched-tokens=512` (maksymalna
liczbę tokenów, która może być przetwarzana jednocześnie) dla różnych `max-num-seqs` (maksymalna liczba jednoczesnych
żądań inferencji przetwarzana przez vLLM) i max-concurrency (liczba requestów równolegle wysyłanych przez test).

#### max-concurrency = 32

| max-num-seqs | max-num-batched-tokens | Request throughput | Output throughput | Total throughput | P50 TTFT (ms) | Time per output token (ms) | Inter-token Latency Mediam (ms) |
|--------------|------------------------|--------------------|-------------------|------------------|---------------|----------------------------|---------------------------------|
| 64           | 512                    | 0.45               | 188.19            | 3800.10          | 47276.68      | 56.49                      | 83.85                           |
| 128          | 512                    | 0.46               | 188.09            | 3862.02          | 46343.69      | 56.38                      | 82.23                           |
| 256          | 512                    | 0.46               | 187.69            | 3835.79          | 46493.28      | 56.66                      | 84.20                           |
| 512          | 512                    | 0.45               | 189.58            | 3805.25          | 46776.08      | 56.06                      | 83.92                           |

#### max-concurrency = 8

| max-num-seqs | max-num-batched-tokens | Request throughput | Output throughput | Total throughput | P50 TTFT (ms) | Time per output token (ms) | Inter-token Latency Mediam (ms) |
|--------------|------------------------|--------------------|-------------------|------------------|---------------|----------------------------|---------------------------------|
| **64**       | 512                    | 0.41               | 161.40            | 3420.13          | 1740.06       | 43.00                      | 63.81                           |
| **128**      | 512                    | 0.41               | 160.74            | 3402.02          | 1738.32       | 43.02                      | 62.83                           |
| **256**      | 512                    | 0.40               | 165.94            | 3359.19          | 1749.00       | 42.58                      | 63.35                           |
| **512**      | 512                    | 0.40               | 158.44            | 3367.61          | 1750.68       | 43.21                      | 64.73                           |

#### Obserwacje

Przeprowadzając powyższe testy spodziewałem się, że będzie można zaobserwować zależności między `max-num-seqs` a
wydajnością co wynikało z dokumentacji. Jednak powyższe dane pokazały pewne rozbieżności.
1. **Przepustowość requestów i tokenów (request/output throughput)**
    * Throughput jest zależny od max-concurrency (zgodnie z oczekiwaniami)
    * Throughput jest praktycznie niezależny od `max-num-seqs` (zgodnie z oczekiwaniami)
    * Powierzchowne testy ze zmienianiem `max-num-batched-tokens` nie wykazały zależności throughput od tego parametru
      (niezgodnie z dokumentacją)
2. **TTFT (Time To First Token) – brak spodziewanej poprawy**
    * W **tej** konfiguracji `TTFT` praktycznie nie zmieniało się wraz z `max-num-seqs` – mimo że dokumentacja sugeruje,
      że przy rosnącym `max-num-seqs` `TTFT` może spadać, jeśli wcześniej mieliśmy kolejki requestów.
3. **Latencja i czas generacji pojedynczego tokena**
    * Parametr `max-num-seqs` nie ma praktycznie żadnego wpływu na `TTFT` i `ITL` (Inter-token latency)
    * Oba parametry rosną jedynie wraz z max-concurrency

### Wpływ parametru max-model-len i długości prompta

Jeśli `max-model-len` jest zbyt duże i wymuszałoby pre-alokację macierzy lub wpływałoby na kształt macierzy
wejściowej/wyjściowej, która jest następnie dzielona na bloki obliczeniowe, a rzeczywista długość sekwencji jest mała,
mogłoby to prowadzić do nieoptymalnego wykorzystania jednostek obliczeniowych (tzw. tail overhead lub padding)

Przeprowadziłem więc testy dla różnych max-model-len przy następujących parametrach bazowych:
- `max-num-seqs`: 512
- `max-num-batched-tokens`: 512
- `ilość równoległych requestów`: 300

| max-model-len   | Prompt length | Prompt tokens / s | Generation tokens/s | P50 TTFT (ms) | Time per output token (ms) | Inter-token Latency Mediam (ms) |
|-----------------|---------------|-------------------|---------------------|---------------|----------------------------|---------------------------------|
| **16384**       | **3000**      | 8991              | 843                 | 577           | 0.35                       | 89                              |
| **16384**       | **8000**      | 9682              | 466                 | 1342          | 0.62                       | 179                             |
| **16384**       | **16000**     | 16163             | 280                 | 1750          | 1.75                       | 356                             |
| **32768**       | **3000**      | 7773              | 759                 | 624           | 0.356                      | 89                              |
| **32768**       | **8000**      | 10047             | 442                 | 1608          | 0.625                      | 167                             |
| **32768**       | **16000**     | 16003             | 270                 | 1750          | 1.75                       | 360                             |

Dla analogicznej długości prompta przy różnych `max-model-len` (16384 i 32768) otrzymałem bardzo zbliżone pomiary.
Dla tej konkretnej konfiguracji zmiana `max-model-len` z 16384 do 32768 nie miała zauważalnego wpływu na wydajność.
W ogólności `max-model-len` wpływa na zużycie pamięci i maksymalne wartości `max-num-seqs` / `max-num-batched-tokens`,
więc przy innych ustawieniach może mieć większe znaczenie.
Wynika z tego, że `max-model-len` nie jest parametrem, który wpływa na wydajność w tym systemie. Potencjalnie może to
być spowodowane innymi optymalizacjami, takimi jak PagedAttention (w vLLM) i [Dynamic Shapes](https://docs.pytorch.org/xla/master/learn/dynamic_shape.html).

## Podsumowanie

Uruchamianie modeli językowych LLM na TPU z użyciem vLLM w Google Cloud okazuje się realnym i opłacalnym podejściem,
pod warunkiem odpowiedniego przygotowania infrastruktury. TPU oferują dużą przepustowość i korzystny stosunek kosztów
do wydajności, ale ich wykorzystanie wiąże się z pewnymi wyzwaniami, m.in. ograniczoną dostępnością najnowszych wersji,
koniecznością konfiguracji środowiska z użyciem XLA oraz nieoptymalnym domyślnym podejściem do przechowywania modeli
(GCS Fuse).

Zastąpienie GCS Fuse własnym serwerem NFS znacząco skróciło czas ładowania modeli i obniżyło koszty. W testach wykazano,
że odpowiednio skonfigurowany vLLM na TPUv5e potrafi obsługiwać duży wolumen żądań z wysoką wydajnością – nawet ponad
1000 tokenów generowanych na sekundę – przy miesięcznym koszcie około 1576 USD (w przypadku 3-letniego zobowiązania)
lub 3504 USD bez zobowiązania 3-letniego.

Wnioski:
- TPU to dobra opcja dla wysokowydajnej inferencji LLM, zwłaszcza w środowiskach produkcyjnych.
- Warto poświęcić czas na optymalizację infrastruktury (NFS, odpowiednie parametry vLLM).
- W przeprowadzonych testach na `TPUv5e`, dla modelu `Qwen3-32B.w8a8` i` max-num-batched-tokens=512`, zmiany
  `max-num-seqs` miały znikomy wpływ na `TTFT` i `throughput` – w praktyce kluczowy okazał się `max-concurrency`.
  W innych konfiguracjach (inne modele, dłuższe prompty, większe m`ax-num-batched-tokens`) parametry scheduler’a vLLM
  mogą jednak wpływać na wydajność, zgodnie z dokumentacją.

## Linki warte przejrzenia

- [Jak działa TPU](https://medium.com/@ruslanbredun007/what-is-tpu-and-how-why-it-works-9a0a4a59399e)
- [Monitoring vLLM w GCP](https://cloud.google.com/stackdriver/docs/managed-prometheus/exporters/vllm)
- [Optymalizacja parametrów w vLLM na diagramie](https://github.com/vllm-project/vllm/issues/2492)
- [Dokumentacja o optymalizacji parametrów w vLLM](https://docs.vllm.ai/en/latest/configuration/optimization.html#optimization-and-tuning)
- [Podobne testy przeprowadzone przez firmę Miko.ai na innych modelach i jednostkach obliczeniowych](https://engineering.miko.ai/navigating-the-ai-compute-maze-a-deep-dive-into-google-tpus-nvidia-gpus-and-llm-benchmarking-5332339e4c9b)