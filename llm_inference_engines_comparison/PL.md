# Porównanie podejść do deploymentu modeli językowych

## Wstęp

W erze sztucznej inteligencji i dużych modeli językowych (LLM), wybór odpowiedniego podejścia do deploymentu modeli
stał się kluczowy dla sukcesu projektów. Różne rozwiązania oferują odmienne zalety i wady, co czyni wybór zależnym
od specyficznych wymagań projektu, dostępnych zasobów oraz poziomu technicznej ekspertyzy zespołu.

W tym poście przedstawiam szczegółowe porównanie czterech głównych podejść do wdrażania modeli
językowych: **vLLM**, **Text Generation Inference (TGI)** firmy [Hugging Face](https://huggingface.co/), **Ollama** oraz podejście naiwne wykorzystujące
bezpośrednio frameworki deeplearningowe. Każde z tych rozwiązań ma swoje miejsce w spektrum zastosowań - od prostych
prototypów lokalnych po zaawansowane systemy produkcyjne obsługujące tysiące użytkowników.

## Wyjaśnienie kluczowych pojęć

Zanim przejdziemy do szczegółowego porównania, warto wyjaśnić kilka kluczowych pojęć, które będą pojawiać się w tym artykule:

- **Wdrożenie (ang. deployment)** - proces uruchomienia modelu w środowisku produkcyjnym, gdzie może on obsługiwać zapytania od użytkowników końcowych.

- **Inferencja** - proces generowania odpowiedzi przez model na podstawie otrzymanego zapytania.

- **GPU (Graphics Processing Unit)** - procesor graficzny, który dzięki swojej wielowątkowej architekturze doskonale nadaje się do obliczeń związanych z uczeniem maszynowym.

- **Continuous batching** - technika grupowania wielu zapytań w jedno, celem optymalizacji wykorzystania zasobów sprzętowych.

- **[Tool calling](https://platform.openai.com/docs/guides/function-calling)** - możliwość wywoływania zewnętrznych funkcji przez model w trakcie generowania odpowiedzi.

- **KV-cache** - pamięć przechowująca wcześniej obliczone wartości klucz-wartość dla historii chatu, co pozwala na przyspieszenie generowania kolejnych słów (tokenów).

- **PagedAttention** - wariant mechanizmu uwagi (attention), który optymalizuje wykorzystanie pamięci poprzez przetwarzanie danych w mniejszych fragmentach. Redukuje narzut obliczeniowy poprzez skupienie się na istotnych częściach danych wejściowych.

## Silniki inferencji

### vLLM

vLLM to silnik inferencji zaprojektowany z myślą o efektywnym wykorzystaniu procesorów graficznych. Jego kluczową cechą jest implementacja techniki PagedAttention, która modyfikuje sposób zarządzania pamięcią KV-cache. To rozwiązanie pozwala na bardziej efektywne wykorzystywanie dostępnej pamięci GPU i obsługę większej liczby równoczesnych zapytań.

Silnik implementuje techniki optymalizacyjne, takie jak continuous batching oraz offloading do CPU. vLLM został zaprojektowany z myślą o skalowalności i wysokiej przepustowości, co sprawia, że nadaje się do aplikacji wymagających obsługi dużej liczby użytkowników jednocześnie.

**Zalety:**
- Znacząco wyższa przepustowość (2-5x) w porównaniu do standardowych implementacji
- Efektywne zarządzanie pamięcią, co pozwala na obsługę większej liczby równoczesnych zapytań
- Wsparcie dla różnych modeli, w tym Llama, Mistral, Gemma i innych
- Możliwość deploymentu modelu na wiele GPU (tensor parallelism)
- Natywnie wspiera tool calling
- OpenAI-kompatybilne API umożliwiające łatwą integrację z istniejącymi aplikacjami

**Wady:**
- Wymaga większej konfiguracji i wiedzy technicznej
- Specyficzne wymagania sprzętowe (głównie NVIDIA GPU, choć dostępne też AMD i Intel)
- Ograniczone wsparcie dla deploymentu na CPU
- Może nie wspierać wszystkich najnowszych modeli natychmiast po ich wydaniu

### Text Generation Inference (TGI) od [Hugging Face](https://www.huggingface.co)

Text Generation Inference to oficjalne rozwiązanie firmy Hugging Face do wdrażania modeli generatywnych w środowiskach produkcyjnych. TGI charakteryzuje się integracją z ekosystemem Hugging Face, co sprawia, że jest odpowiednim wyborem dla zespołów już korzystających z tej platformy. Silnik wykorzystuje optymalizacje, takie jak [token streaming](https://huggingface.co/docs/text-generation-inference/conceptual/streaming) i continuous batching do obsługi zapytań.

TGI jest dostępny w formie kontenera Docker, co upraszcza proces wdrożenia. Silnik można skonfigurować za pomocą zmiennych środowiskowych, co czyni go dostępnym dla zespołów z różnym poziomem wiedzy DevOps.

**Zalety:**
- Dobrze zintegrowany z biblioteką [Transformers](https://huggingface.co/docs/transformers/index) i modelami z Huggingface Hub
- Oferuje REST API i gRPC do komunikacji
- Posiada wbudowane narzędzia monitorowania i profilowania
- Łatwa konfiguracja przez zmienne środowiskowe
- Wspiera zarówno CPU jak i GPU (NVIDIA, AMD, Intel Gaudi, Inferentia)
- Obsługuje modele prywatne i z ograniczonym dostępem

**Wady:**
- Mniej zaawansowane techniki zarządzania pamięcią w porównaniu do vLLM
- Ograniczone opcje dostrajania dla specyficznych przypadków użycia
- Może nie wspierać wszystkich najnowszych modeli natychmiast po ich wydaniu
- Słabo radzi sobie z tool callingiem (na ten moment)

### Ollama

Ollama to aplikacja przeznaczona do uruchamiania modeli językowych na komputerach lokalnych. Narzędzie zostało zaprojektowane z naciskiem na prostotę użycia, umożliwiając użytkownikom o różnym poziomie technicznej wiedzy uruchamianie modeli językowych na własnych komputerach.

Ollama oferuje system pobierania i zarządzania modelami podobny do menedżerów pakietów. Aplikacja jest dostępna na systemach operacyjnych: Windows, macOS oraz Linux.

**Zalety:**
- Niezwykle prosta instalacja i konfiguracja (często single-command)
- Gotowe do użycia modele poprzez system `ollama pull [model]`
- Wbudowane REST API
- Niskie wymagania techniczne dla użytkownika

**Wady:**
- Ograniczona skalowalność do zastosowań produkcyjnych
- Mniejsza wydajność w porównaniu do vLLM czy TGI
- Ograniczone możliwości dostosowania parametrów
- Mniej zaawansowane optymalizacje wykorzystania GPU

### Podejście naiwne (samodzielny serwis uruchamiający model [PyTorch](https://pytorch.org/)/[TensorFlow](https://www.tensorflow.org/)/Transformers)

Podejście naiwne polega na bezpośrednim wykorzystaniu frameworków uczenia maszynowego, takich jak PyTorch, TensorFlow czy Transformers, do wczytania i uruchomienia modelu w ramach własnej aplikacji. Ten sposób daje programistom kontrolę nad każdym aspektem działania systemu, ale wymaga samodzielnej implementacji optymalizacji i funkcjonalności dostępnych w specjalistycznych silnikach inferencyjnych.

Takie podejście może być uzasadnione w specyficznych przypadkach, kiedy standardowe rozwiązania nie spełniają wymagań projektu lub gdy potrzebna jest integracja z określoną architekturą aplikacji.

**Zalety:**
- Dowolna architektura modelu
- Możliwość dostosowania do specyficznych wymagań projektu
- Brak zależności od zewnętrznych silników inferencyjnych
- Łatwość integracji z istniejącą infrastrukturą

**Wady:**
- Znacznie niższa wydajność (5-10x wolniejsze niż zoptymalizowane silniki)
- Konieczność samodzielnej implementacji wielu optymalizacji
- Problemy z zarządzaniem pamięcią przy dużych modelach
- Wysokie ryzyko wycieków pamięci i innych problemów wydajnościowych
- Trudności w implementacji funkcji takich jak continuous batching

### Porównanie kluczowych cech

| Cecha                      | vLLM                                      | TGI                          | Ollama | Podejście naiwne |
|----------------------------|-------------------------------------------|------------------------------|----|------------------|
| Wydajność GPU              | ⭐⭐⭐⭐⭐                                     | ⭐⭐⭐⭐                         | ⭐⭐ | ⭐ |
| Wydajność CPU              | ⭐⭐                                        | ⭐⭐⭐                          | ⭐⭐⭐ | ⭐⭐ |
| Łatwość wdrożenia          | ⭐⭐⭐                                       | ⭐⭐⭐⭐                        | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Skalowalność               | ⭐⭐⭐⭐⭐                                     | ⭐⭐⭐⭐                         | ⭐⭐ | ⭐⭐ |
| Optymalizacja pamięci      | ⭐⭐⭐⭐⭐                                     | ⭐⭐⭐⭐                         | ⭐⭐ | ⭐ |
| Obsługa dużych modeli      | ⭐⭐⭐⭐⭐                                     | ⭐⭐⭐⭐                         | ⭐⭐ | ⭐⭐ |
| Szybkość integracji        | ⭐⭐⭐⭐                                      | ⭐⭐⭐⭐                        | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Wsparcie dla akceleratorów | NVIDIA, AMD, Intel Gaudi, CPU (częściowo) | NVIDIA, AMD, Intel Gaudi, CPU, Inferentia | NVIDIA, CPU | Dowolne |
| Dojrzałość projektu        | ⭐⭐⭐⭐                                      | ⭐⭐⭐⭐⭐                        | ⭐⭐⭐ | n/a |
| Najlepsze zastosowanie     | Produkcja, duża skala                     | Produkcja, ekosystem Hugging Face      | Rozwój, testowanie | Specjalistyczne przypadki |

## Podsumowanie

Wybór odpowiedniego podejścia do deploymentu modeli językowych zależy od wielu czynników, w tym wymagań wydajnościowych,
dostępnej infrastruktury, poziomu technicznego zespołu oraz budżetu projektu.

**vLLM** stanowi najlepszy wybór dla projektów produkcyjnych wymagających maksymalnej wydajności i skalowalności,
szczególnie gdy dysponujemy odpowiednim sprzętem GPU i zespołem technicznym.

Dla organizacji już korzystających z ekosystemu Hugging Face doskonałym rozwiązaniem jest **Text Generation Inference**. Oferuje on dobry
kompromis między wydajnością a łatwością wdrożenia.

**Ollama** sprawdzi się idealnie w fazie prototypowania, rozwoju aplikacji oraz w scenariuszach, gdzie priorytetem
jest prostota użycia nad maksymalną wydajnością.

**Podejście naiwne** powinno być rozważane jedynie w bardzo specyficznych przypadkach, gdzie standardowe rozwiązania
nie spełniają unikalnych wymagań projektu i dysponujemy zasobami na własną implementację optymalizacji.

Niezależnie od wybranego rozwiązania, kluczowe jest przeprowadzenie testów wydajnościowych w środowisku zbliżonym do
produkcyjnego, aby zweryfikować, czy wybrane podejście spełnia oczekiwania dotyczące przepustowości, opóźnień i
wykorzystania zasobów.
