# Deployment silników inferencji LLM na CPU – doświadczenia z Text Generation Inference (TGI)

## Wprowadzenie

Czy uruchamianie modeli na CPU może być alternatywą dla GPU lub TPU? Niższy koszt jednostkowy godzinny, większa
dostępność i prostsza konfiguracja są powodami dla których moglibyśmy się o to pokusić. W tym artykule
skupię się na próbie uruchomienia wybranych modeli na różnych maszynach w GCP (Google Cloud Platform) i sprawdzeniu jak
szybko są w stanie generować tokeny.

## Modele użyte w eksperymentach

Parametry w modelach językowych to wartości liczbowe (wagi) w sieci neuronowej, które powstają w procesie trenowania.
To właśnie one przechowują wzorce językowe i wiedzę, pozwalając modelowi rozpoznawać kontekst, generować odpowiedzi
czy rozwiązywać zadania. Im większa liczba parametrów, tym model ma potencjalnie większą pojemność reprezentacyjną
(_ang. model capacity_, czyli zdolność do zapamiętywania, odwzorowywania i wykorzystywania złożonych wzorców w danych),
ale jednocześnie rosną koszty obliczeniowe i wymagania sprzętowe.

Modele klasy OpenAI o4-mini i o3-mini (a dokładniej gpt-oss-120b i gpt-oss-20b) mają odpowiednio ok. 117 mld i 21 mld
parametrów.  Testowane przez nas modele są mniejsze, ale nadal oferują funkcjonalności wystarczające w wielu praktycznych
zastosowaniach.
Do porównań wykorzystam kilka modeli o różnej skali:

| Model       | Liczba parametrów | Charakterystyka zastosowań                       |
| ----------- | ----------------: | ------------------------------------------------ |
| Qwen2.5-3B  |             3 mld | Lekki asystent, analiza tekstu                   |
| Qwen2.5-7B  |             7 mld | Generacja kodu, chatboty średniej klasy          |
| Qwen2.5-14B |            14 mld | Zbliżona jakość do komercyjnych rozwiązań        |


W artykule porównującym silniki inferencji znajdziemy informację, że Text Generation Inference (TGI) ma dobre wsparcie
dla CPU i jest przeznaczony do produkcyjnego zastosowania, dlatego też tutaj skupię się na jego użyciu z modelami
językowymi.


## Testowanie modeli

[Hugging Face](https://huggingface.co/) udostępnia kilka standardowych kompilacji. Jedną z nich jest
`ghcr.io/huggingface/text-generation-inference:latest-intel-cpu` i to z niej skorzystam w pierwszej kolejności.
Pełny deskryptor Kubernetes użyty do eksperymentu prezentuje się następująco:

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

Jakość modeli została wytestowana używając następującego requestu, a wydajność używając podobnej konfiguracji z
różnymi inputami:
```shell
curl --location 'http://127.0.0.1:3000/generate' \
--header 'Content-Type: application/json' \
--data '{
  "inputs": "Wytłumacz czym jest kwantowa spójność.",
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

Odpowiedzi dla poszczególnych modeli prezentują się następująco:

- Qwen-2.5-14B:

```json
{
    "generated_text": " Kwantowa spójność to pojęcie z zakresu fizyki kwantowej i teorii strun, które odnosi się do konsekwencji teoretycznej, wynikającej z prób zjednoczenia różnych teorii fizycznych. Oto kilka kluczowych punktów, które mogą pomóc w zrozumieniu tego pojęcia:\n\n1. **Zjednoczenie sił fundamentalnych"
}
```

- Qwen-2.5-7B:

```json
{
    "generated_text": " Kwantowa spójność to pojęcie z dziedziny fizyki kwantowej, które odnosi się do związków między stanami kwantowymi. Jest ona szczególnie ważna w teorii kwantowej pola i teorii kwantowej elektrodynamiki (QED). Oto kilka kluczowych punktów, które wyjaśniają, czym jest kwantowa spójność:\n\n1. Z"
}
```

- Qwen-2.5-3B:

```json
{
    "generated_text": " Podaj przykład, w którym kwantowa spójność jest kluczowa do rozwiązywania problemu.\nKwantowa spójność to koncept fizyczny związany z teorią kwantowej mechaniki, który opisuje połączenie dwóch sygnałów lub obiektów w taki sposób, że zmiany w jednym sygnale są przynajmniej częściowo przesłane do drugiego. Ta sp"
}
```

| Model        | Typ maszyny   | vCPU | RAM  | Czas 100 tokenów | Maksymalny czas w kolejce | Użycie CPU | Użycie pamięci | Koszt miesięczny |
|--------------|---------------|------|------|------------------|---------------------------|------------|----------------|------------------|
| Qwen2.5-14B  | c4-highcpu-32 | 32   | 64GB | 23.97 s          | ~0.1s                     | 50.89%     | 87.60%         | 1094 USD         |
| Qwen2.5-7B   | c4-highcpu-32 | 32   | 64GB | 15.72 s          | ~0.1s                     | 52.7%      | 21%            | 1094 USD         |
| Qwen2.5-3B   | c4-highcpu-32 | 32   | 64GB | 7.1 s            | ~0.1s                     | 48.37%     | 20.97%         | 1094 USD         |
| Qwen2.5-1.5B | c4-highcpu-32 | 32   | 64GB | 4.73 s           | ~0.1s                     | 33.98%     | 12.576%        | 1094 USD         |
| GPT-4        | -             | -    | -    | 8 s              | -                         | -          | -              | -                |
| GPT-4o       | -             | -    | -    | 0.97 s           | -                         | -          | -              | -                |

Powyższe testy zostały przeprowadzone dla 10 użytkowników. Pokazują one, że serwer bez problemu radzi sobie z takim
obciążeniem, a wygenerowanie 100 tokenów zajmuje mniej więcej tyle czasu, ile przeczytanie podobnego tekstu na głos
w umiarkowanym tempie (ok. 7–8 sekund). Widać więc, że dopiero model o wielkości około 3 miliardów parametrów osiąga
tempo zbliżone do subiektywnie odczuwalnej „prędkości czytania na głos”, czyli podobnej do GPT-4.

## Zwiększanie obciążenia

W ramach kolejnego testu sprawdzię jak zwiększanie obiążenia do 100 użytkowników wpłynie na te parametry. Ma to istotny
wpływ na możliwości skalowania takiej aplikacji.

| Model        | Typ maszyny   | vCPU | RAM  | Czas 100 tokenów | Maksymalny czas w kolejce | Użycie CPU | Użycie pamięci | Koszt miesięczny |
|--------------|---------------|------|------|------------------|---------------------------|------------|----------------|------------------|
| Qwen2.5-14B  | c4-highcpu-32 | 32   | 64GB | 53.2 s           | ~0.4s                     | 51.5%      | 90.48%         | 1094 USD         |
| Qwen2.5-7B   | c4-highcpu-32 | 32   | 64GB | 23.7 s           | ~0.3s                     | 50.97      | 47.51%         | 1094 USD         |
| Qwen2.5-1.5B | c4-highcpu-32 | 32   | 64GB | 7.1 s            | ~0.3s                     | 51.98%     | 12.98%         | 1094 USD         |

Zwiększenie obciążenia spowodowało ok. dwukrotne zwiększenie się czasu generowania odpowiedzi i w niewielkim stopniu wpłynęło
na czas requestów w kolejce. Nie wpłynęło to natomiast znacząco na użycie CPU i pamięci.

### Zwiększenie obciążenia do 1000 użytkowników

Zwiększenie obciążenia do 1000 użytkowników spowodowało zwiększenie kolejki requestów do 2 i odrzucanie requestów. Nie wpłynęło
na CPU i pamięć, więc niestety trudno jest znaleźć tutaj metrykę do autoskalowania (mimo, że Huggging Face rekomenduje
użycie metryki związanej z kolejką requestów).

## Kwantyzacja

Powszechnie używaną techniką pozwalającą zmniejszyć zapotrzebowanie na zasoby i przyspieszyć inferencję jest kwantyzacja.
Polega ona na zmniejszeniu precyzji liczb, w których są przechowywane wagi modelu, dzięki czemu:
- zmniejsza się rozmiar modelu
- maleją wymagania dotyczące pamięci
- przyspiesza się inferencja
  Kosztem jest jednak utrata jakości odpowiedzi.

Przeprowadzę kilka testów kwantyzacji używając TGI i CPU aby sprawdzyć czy ta technika pomoże nam uzyskać lepsze rezultaty
lub obniżyć koszt.

### Przygotowanie TGI

W chwili pisania tego artykułu wszystkie typy kwantyzacji "on-the-fly" w TGI są zaprojektowane pod backend GPU.
Jedyna teoretycznie działająca opcja to zastosowanie backendu [Llama.cpp](https://github.com/ggml-org/llama.cpp), która
wspiera format [GGUF](https://huggingface.co/docs/hub/gguf) umożliwiający uruchomienie pre-kwantyzowanych modeli.

Do tego jednak trzeba zbudować dedykowaną wersję TGI, ponieważ Hugging Face nie udostępnia gotowych obrazów.
[`Dockerfile` przygotowany w repozytorium TGI](https://github.com/huggingface/text-generation-inference/blob/main/Dockerfile_llamacpp)
niestety nie działał i zajmuje bardzo dużo miejsca więc stworzyłem customowy:

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

Następnie zbudowałem go, używając następujących parametrów:

```shell
docker build \
    -t tgi-llamacpp \
    --build-arg llamacpp_native=OFF \
    --build-arg llamacpp_cuda=OFF \
    --build-arg llamacpp_cpu_arm_arch=x86-64 \
    https://github.com/huggingface/text-generation-inference.git \
    -f Dockerfile_llamacpp_custom
```

### Uruchomienie obrazu

Poniżej przedstawiam pełny Kubernetes deskryptor służący do uruchomienia tak zbudowanego obrazu. Warto zwrócić uwagę
na parametry związane z modelem GGUF, które się pojawiły (`model-gguf`, `tokenizer-config-path`):

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

### Pomiary

Poniżej przedstawiam pomiary jakie uzyskałem dla modeli skwantyzowanych w formacie Q4_K_M. Testy zostały
przeprowadzone dla 10 równoległych użytkowników. Pokusiłem się też o pomiary na słabszych maszynach z uwagi na dobre
wyniki czasu generowania 100 tokenów.

| Model                  | Typ maszyny   | vCPU | RAM  | Czas 100 tokenów | Maksymalny czas w kolejce | Użycie CPU | Użycie pamięci | Koszt miesięczny |
| ---------------------- | ------------- | ---- | ---- | ---------------- |---------------------------|------------|----------------| ---------------- |
| Qwen2.5-14B (q4\_k\_m) | c4-highcpu-32 | 32   | 64GB | 10,65 s          | 81s                       | 100%       | 34.29%         | 1094 USD         |
| Qwen2.5-7B  (q4\_k\_m) | c4-highcpu-32 | 32   | 64GB | 4,73 s           | 35s                       | 80%        | 14,44%         | 1094 USD         |
| Qwen2.5-3B  (q4\_k\_m) | c4-highcpu-32 | 32   | 64GB | 3,16 s           | 22.37s                    | 66.59%     | 7.9%           | 1094 USD         |
| Qwen2.5-14B (q4\_k\_m) | c4-highcpu-16 | 16   | 32GB | 18 s             | 145s                      | 96.74%     | 41.06%         | 547 USD          |
| Qwen2.5-7B  (q4\_k\_m) | c4-highcpu-16 | 16   | 32GB | 6,9 s            | 53s                       | 68.99%     | 24.07%         | 547 USD          |
| Qwen2.5-14B (q4\_k\_m) | n2-standard-8 | 8    | 32GB | 35,95 s          | 145s                      | 56.36%     | 34.29%         | 312 USD          |

Widać znaczącą poprawę w czasie generowania 100 tokenów i w użyciu RAM. Jednak _Max Queue Duration_, czyli czas w kolejce
na przetworzenie drastycznie wzrósł w porównaniu do modeli nieskwantyzowanych. Niestety, format GGUF jest zoptymalizowany
pod jednego klienta, co oznacza, że przepustowość dla takiej konfiguracji jest nieakceptowalna dla produkcyjnych rozwiązań.

## Podsumowanie

Testy pokazały, że uruchamianie LLM na CPU ma w praktyce tylko jedną realną zaletę – znacznie większą dostępność maszyn
w chmurze w porównaniu do GPU i TPU. Czas inferencji szybko staje się barierą, szczególnie dla modeli powyżej 7 miliardów
parametrów. Kwantyzacja w TGI nie jest obecnie użyteczna w scenariuszach produkcyjnych z uwagi na problemy ze skalowaniem
i obsługą wielu równoległych użytkowników. CPU może więc sprawdzić się wyłącznie w zastosowaniach prototypowych,
jako rozwiązanie awaryjne (fallback) oraz przy lekkich modelach do ok. 7 miliardów parametrów – wszędzie indziej GPU pozostaje
praktycznym wyborem.

## Linki warte przejrzenia

- [Wyjaśnienie koncepcji kwantyzacji](https://www.digitalocean.com/community/tutorials/model-quantization-large-language-models)
- [Szczegóły dotyczące poszczególnych modeli kwantyzacji](https://huggingface.co/docs/transformers/main/en//quantization)
- [Backend llamacpp](https://huggingface.co/docs/text-generation-inference/en/backends/llamacpp)