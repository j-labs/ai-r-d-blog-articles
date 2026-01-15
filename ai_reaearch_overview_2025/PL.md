# Podsumowanie Rozwoju AI

## Architektury modeli sekwencyjnych i mechanizmy pamięci

### Wysokopoziomowe podsumowanie sekcji

Ta grupa prezentuje fundamentalne postępy w tym, jak sieci neuronowe przetwarzają informacje sekwencyjne i utrzymują pamięć w długich kontekstach. Badania łączą się w kluczowym spostrzeżeniu: efektywne modelowanie sekwencji wymaga zaawansowanych mechanizmów pamięci wykraczających poza tradycyjną atencję. Te publikacje wprowadzają nowe architektury, które ujednolicają istniejące podejścia poprzez teoretyczne frameworki (Miras), implementują uczenie w trakcie inferencji dla adaptacyjnej pamięci (Titans) i odblokowują nowe zdolności modeli rekurencyjnych poprzez modyfikacje algorytmiczne (ujemne wartości własne w LRNN). Wspólnie przybliżają one do rozwiązania kluczowego wyzwania równoważenia efektywności obliczeniowej z możliwością śledzenia stanu i utrzymywania kontekstu w ekstremalnie długich sekwencjach, z implikacjami obejmującymi modelowanie języka, szeregi czasowe, genomikę i generowanie kodu.

### Publikacje

#### [It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention and Online Optimization](https://arxiv.org/pdf/2504.13173)

Publikacja prezentuje Miras, framework abstrahujący nowoczesne modele sekwencyjne (Transformery, Titans, liniowe RNN, etc.) jako moduły pamięci z attentional bias optymalizujące wewnętrzne cele. Framework ujawnia, że większość istniejących architektur używa dot-product similarity lub regresji L2, i jest oparty na czterech wyborach projektowych: architektura pamięci asocjacyjnej, równanie attentional bias, retention gates (reinterpretujące forget gates jako regularyzację retencji) i algorytmy uczenia pamięci. Opierając się na tych spostrzeżeniach, autorzy proponują trzy nowe modele sekwencyjne: Moneta, Yaad i Memora, które używają alternatywnych attentional bias (odległość Minkowskiego, funkcja straty Hubera) i mechanizmów retencji (w istocie forget gate; odległość Minkowskiego, dywergencja KL), demonstrując obiecującą* wydajność w modelowaniu języka, rozumowaniu zdroworozsądkowym i zadaniach intensywnych pod względem przypominania z ulepszonymi wzorcami skalowania.

*Uwaga: Występują znaczące poprawy wydajności względem baseline'ów, ale prezentowane w wąskim reżimie skalowania.

##### Kluczowe kontrybucje
1. Teoretyczne ujednolicenie architektur modelowania sekwencji przez pryzmat pamięci asocjacyjnej i optymalizacji online
2. Framework projektowy czteroskładnikowy (architektura pamięci, attentional bias, mechanizm retencji, algorytm uczenia) umożliwiający systematyczną eksplorację architektur
3. Trzy nowe architektury (Moneta, Yaad, Memora**) z alternatywnymi celami przewyższające istniejące baseline'y
  
**Uwaga: Memora osiąga najgorsze wyniki w eksplorowanych zakresach parametrów, ale ma najsilniejsze gwarancje stabilności, więc potencjalnie może być łatwiejsza do trenowania w dużej skali.

#### [Titans: Learning to Memorize at Test Time](https://arxiv.org/pdf/2501.00663)

Titans wprowadza rodzinę architektur z nowym modułem neuronowej pamięci długoterminowej, który uczy się zapamiętywać w czasie testowania, używając aktualizacji opartych na gradiencie z bezwładnością i weight decay. Architektura traktuje zaskakujące wydarzenia jako warte zapamiętania i proponuje trzy warianty: Memory as Context (MAC)*, Memory as Gate (MAG) i Memory as Layer (MAL). Łączą one pamięć neuronową dla kontekstu długoterminowego z atencją dla zależności krótkoterminowych i pamięć trwałą dla wiedzy o zadaniu, adresując kwadratową złożoność Transformerów i słabą kompresję liniowych modeli rekurencyjnych.

*Uwaga: Wariant MAC osiąga najlepszą ogólną wydajność w porównaniu z innymi wariantami, ale jest obliczeniowo cięższy.

##### Kluczowe kontrybucje
1. Nowy moduł pamięci uczący się w czasie inferencji z aktualizacjami opartymi na bezwładności i adaptacyjnymi mechanizmami zapominania
2. Trzy warianty architektoniczne (MAC, MAG, MAL) integrujące pamięć długoterminową z atencją poprzez różne mechanizmy
3. Efektywne skalowanie do okien kontekstu 2M+ z atrakcyjnymi wynikami względem konkurencyjnych architektur

#### [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/pdf/2411.12537)

Publikacja identyfikuje i rozwiązuje fundamentalne ograniczenie w nowoczesnych Linear Recurrent Neural Networks (LRNN) jak Mamba i DeltaNet: ich niezdolność do wykonywania zadań śledzenia stanu z powodu restrykcji wartości własnych. Autorzy dowodzą, że LRNN z wartościami własnymi macierzy przejścia* ograniczonymi do [0, 1] nie mogą rozwiązać zadań takich jak parzystość i liczenie modularne w skończonej precyzji, oraz że macierze nietrójkątne są potrzebne do ogólnego liczenia modularnego. Co kluczowe, demonstrują, że rozszerzenie zakresu wartości własnych do [−1, 1] dramatycznie zwiększa moc ekspresyjną, umożliwiając LRNN rozwiązanie wszystkich języków regularnych** poprzez iloczyny uogólnionych macierzy Householdera***.

*Uwaga: Macierz przejścia jest macierzą opisującą ewolucję hidden state w czasie.  
**Uwaga: Języki regularne to klasa języków formalnych rozpoznawanych przez automaty skończone.  
***Uwaga: ta generalizacja pozwala także na rotację oprócz odbicia wektora.

##### Kluczowe kontrybucje
1. Teoretyczny dowód fundamentalnych ograniczeń ekspresyjności w LRNN z tylko pozytywnymi wartościami własnymi macierzy przejścia dla zadań śledzenia stanu
2. Matematyczna demonstracja, że ujemne wartości własne umożliwiają rozwiązanie wszystkich języków regularnych
3. Praktyczna modyfikacja architektoniczna pozwalająca na zakres wartości własnych [−1, 1] macierzy przejścia bez utraty stabilności lub efektywności

---

## Efektywna atencja i wnioskowanie LLM

### Wysokopoziomowe podsumowanie sekcji

W miarę jak modele językowe skalują się do miliardów parametrów i milionów tokenów kontekstu, efektywność obliczeniowa stała się krytycznym wąskim gardłem. Ta grupa rozwiązuje kryzys efektywności poprzez komplementarne innowacje: mechanizmy sparse attention dramatycznie redukujące złożoność obliczeniową dla długich kontekstów (DeepSeek-V3.2-Exp), augmentacje architektoniczne poprawiające ekspresyjność i eliminujące patologiczne zachowania jak attention sinks (Gated Attention), alternatywne paradygmaty generowania umożliwiające równoległość (REFUSION) i optymalizacje treningu redukujące koszty fine-tuningu (SparseLoRA). Te postępy wspólnie umożliwiają bardziej dostępne, szybsze i bardziej opłacalne wdrażanie dużych modeli językowych bez poświęcania jakości modelu - a w niektórych przypadkach poprawiając ją. Praktyczny wpływ obejmuje przyspieszenie wnioskowania, efektywność treningu i poprawę obsługi długiego kontekstu.

### Publikacje

#### [DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf)

DeepSeek-V3.2-Exp wprowadza DeepSeek Sparse Attention (DSA), która używa lightning indexer i fine-grained token selection* do implementacji efektywnej sparse attention. Model jest tworzony poprzez kontynuację treningu DeepSeek-V3.1-Terminus, zgodnie z pipeline'm continual pre-trainingu (dense warm-up**, sparse adaptation) i dwustopniowego post-trainingu. Proponowana architektura atencji znacząco redukuje koszty obliczeń (każdy token query zwraca uwagę na mały, stały podzbiór keys), szczególnie dla bardzo długich kontekstów, zachowując porównywalną wydajność w ogólnych, programistycznych, matematycznych i agentycznych zadaniach wyszukiwania.

*Uwaga: indexer zasadniczo oblicza ważony wynik iloczynu skalarnego między projekcjami fp8 queries i keys, podczas gdy selektor wybiera top-k z niego.  
**Uwaga: warm-up jest używany do inicjalizacji wag indexera.

##### Kluczowe kontrybucje
1. Nowy mechanizm DSA z lightning indexer dla efektywnego obliczania sparse attention
2. Kompleksowy pipeline treningu umożliwiający zmianę wcześniej wytrenowanego modelu dense na sparse
3. Ogromna redukcja kosztów wnioskowania dla scenariuszy długiego kontekstu bez degradacji dokładności

#### [Gated Attention for Large Language Models: Non-linearity, Sparsity and Attention-Sink-Free](https://arxiv.org/pdf/2505.06708)

Publikacja systematycznie bada integrację mechanizmów bramkowych w warstwy atencji dużych modeli językowych poprzez eksperymenty na wielu wariantach modeli (w tym 15B MoE i 1.7B dense) trenowanych do 3.5 biliona tokenów. Badanie wprowadza prostą bramkę sigmoidalną specyficzną dla konkretnych attention-head po Scaled Dot-Product Attention, ujawniając dwa kluczowe mechanizmy poprawy: zwiększoną ekspresyjność poprzez nieliniowość między atencją a warstwami wyjściowymi, oraz attention sparsity zależną od wejścia* eliminującą efekt "attention sink".

*Uwaga: autorzy twierdząc "attention sparsity zależną od wejścia" naprawdę mają na myśli "zależna od wejścia, prawie zerowa attention sparsity" z powodu asymptotycznego zachowania sigmoidy. Z alternatywnej perspektywy niewspomnianej w pracy, ta operacja może być postrzegana jako adaptacyjny forget gate.

##### Kluczowe kontrybucje
1. Kompleksowa analiza empiryczna mechanizmów bramkowych w atencji w dwóch skalach parametrów, wielu poziomach obliczeń i architekturach
2. Identyfikacja dwóch komplementarnych mechanizmów poprawy: nieliniowości i attention sparsity zależnej od wejścia
3. Eliminacja attention sink, umożliwiająca lepszą generalizację długiego kontekstu**
  
**Uwaga: dzięki oszczędnościom w precyzji attention scores wcześniej zmarnowanych na wysoki score attention sink.

#### [REFUSION: A Diffusion Large Language Model with Parallel Autoregressive Decoding](https://arxiv.org/pdf/2512.13586)

REFUSION wprowadza nowy framework LLM łączący paralelizm maskujących modeli dyfuzyjnych (MDM) z autoregresywnym wypełnianiem na poziomie slotów*. Architektura dzieli sekwencje na sloty o stałej długości i stosuje dwustopniowe dekodowanie "plan-and-infill": planowanie globalne oparte na dyfuzji identyfikuje słabo zależne sloty do równoległego przetwarzania**, następnie autoregresywne wypełnianie generuje tokeny w każdym slocie sekwencyjnie. To umożliwia pełne ponowne użycie key-value cache, unikając niespójności na poziomie tokenów, trenowane z hybrydową funckją straty optymalizującą zarówno planowanie globalne, jak i wypełnianie lokalne.

*Uwaga: prostszymi słowami: to hybrydowe podejście łączące autoregresywne LLM z modelami dyfuzyjnymi.  
**Uwaga: zastosowana tutaj empirycznie uzasadniona heurystyka polega na tym, że słabo zależne sloty mają niską ocenę pewności (globalnie), więc mogą potencjalnie "ignorować" się nawzajem podczas dekodowania.

##### Kluczowe kontrybucje
1. Nowa hybrydowa architektura łącząca równoległe generowanie dyfuzyjne ze spójnością generowania autoregresywnego
2. Dwustopniowy algorytm dekodowania plan-and-infill umożliwiający efektywną równoległość bez utraty jakości
3. Pełna możliwość ponownego użycia cache'a KV zwiększająca efektywność o 2.33× względem czysto autoregresywnego baseline'u

#### [SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity](https://arxiv.org/pdf/2506.16500)

SparseLoRA przyspiesza fine-tuning LLM wykorzystując contextual sparsity, dynamicznie wybierając rzadkie podzbiory wag dla obliczeń gradientu i straty używając opartego na SVD estymatora rzadkości bez treningu. Metoda aplikuje rzadkość do macierzy wag selektywnie poprzez warstwy, tokeny i kroki treningu* z minimalnym narzutem. W przeciwieństwie do wcześniejszych podejść PEFT, które głównie redukują pamięć, SparseLoRA bezpośrednio poprawia efektywność obliczeń.

*Uwaga: Początkowe warstwy zyskują z bycia gęstymi macierzami, podczas gdy głębsze warstwy są bardziej redundantne, odblokowując możliwość agresywnej selekcji wektorów. Tokeny kontekstowe mogą używać obliczeń na rzadkich macierzach, ale dekodowane tokeny powinny używać obliczeń na gęstych. Trening zyskuje na progresywnie zwiększającej się rzadkości macierzy dla tokenów i warstw po początkowych w pełni gęstych krokach.

##### Kluczowe kontrybucje
1. Estymacja contextual sparsity bez treningu umożliwiająca dynamiczny wybór rzadkich wag podczas fine-tuningu
2. Strategie rzadkości macierzy względem warstwow, tokenów i czasu, zoptymalizowane dla różnych wzorców obliczeniowych
3. Do 2.2× redukcja kosztów obliczeniowych i 1.6× przyspieszenie względem istniejących metod PEFT przy zachowaniu dokładności
4. Kompatybilność i synergia z metodami opartymi na kwantyzacji (np. QLoRA) dla połączonej efektywności pamięci i obliczeń

---

## Modele podstawowe audiowizualne

### Wysokopoziomowe podsumowanie sekcji

Ta grupa demonstruje dojrzałość modeli podstawowych innych niż tekstowe, ustanawiając nowe paradygmaty dla rozumienia wizualnego i audio. Te prace dzielą wspólny temat: przemyślenie gdzie i jak ekstrahować lub konstruować reprezentacje dla maksymalnej efektywności w różnorodnych zadaniach downstream. SAM Audio osiąga bezprecedensową unifikację w domenach audio poprzez multimodalne promptowanie. Perception Encoder kwestionuje założenie, że warstwy wyjściowe zawierają najlepsze ficzery, a "One Layer Is Enough" pokazuje, że minimalne warstwy adaptacji mogą dostosować potężne pre-trenowane enkodery do zadań generatywnych. Wspólnie reprezentują dążenie w kierunku bardziej efektywnych, wszechstronnych i teoretycznie ugruntowanych podejść do multimodalnego AI, z praktycznymi implikacjami dla systemów produkcyjnych obejmujących edycję audio, computer vision, detekcję, rozumienie wideo i aplikacje generatywne.

### Publikacje

#### [SAM Audio: Segment Anything in Audio](https://arxiv.org/pdf/2512.18099)

SAM Audio jest modelem podstawowym dla generycznej separacji ścieżek audio, który ujednolica promptowanie tekstowe, wizualne i zakresów czasowych w ramach pojedynczej architektury diffusion transformer. Zbudowany na flow matching* i trenowany na dużych ilościach danych audio obejmujących mowę, muzykę i inne dźwięki, osiąga najlepszą w swej klasie wydajność w różnorodnych benchmarkach. Publikacja wprowadza span prompting jako nowy mechanizm warunkowania temporalnego i wydaje SAM Audio-Bench (kompleksowy benchmark separacji multimodalnej z promptami oznaczonymi przez ludzi) i SAM Audio Judge (model ewaluacji bez referencji silnie skorelowany z oceną ludzką).

*Uwaga: Flow matching używa tej samej architektury co dyfuzja, ale z innym celem – zamiast optymalizować stopniowe odszumianie, optymalizuje najkrótszą trajektorię od szumu do danych.

##### Kluczowe kontrybucje
1. Pierwszy model podstawowy osiągający SOTA w wielu domenach audio (mowa, muzyka, ogólne dźwięki) z ujednoliconym promptowaniem multimodalnym
2. Nowy mechanizm span prompting dla warunkowania temporalnego w zadaniach separacji audio

#### [Perception Encoder: The best visual embeddings are not at the output of the network](https://arxiv.org/pdf/2504.13181)

Perception Encoder wprowadza, najlepszą w swojej klasie, rodzinę enkoderów wizualnych odkrywając, że silne ogólne ficzery dla różnorodnych zadań downstream istnieją w warstwach pośrednich modeli trenowanych kontrastowo, a nie w warstwie wyjściowej. Praca rozwija PEcore (solidny, wizualny pretraining), PElang (wariant dostosowany językowo) i PEspatial (wariant dostosowany przestrzennie).

##### Kluczowe kontrybucje
1. Zmieniające paradygmat spostrzeżenie, że optymalne ficzery dla różnorodnych zadań znajdują się w warstwach pośrednich, a nie wyjściach modelu
2. Kompleksowe instrukcje dla pretreningu (PEcore) przewyższające modele trenowane na prywatnych zbiorach danych (JFT-3B/WebLI)
3. Strategie dostosowania specyficzne dla zadań (językowe i przestrzenne) osiągające SOTA w detekcji, VQA i rozumieniu wideo
4. Wydanie modelu o 2B parametrów, kodu i PE Video Dataset (1M filmów, 120K adnotacji udoskonalonych przez ludzi)

#### [One Layer Is Enough: Adapting Pretrained Visual Encoders for Image Generation](https://arxiv.org/pdf/2512.07829)

Ta praca wprowadza FAE (Feature Auto-Encoder), minimalistyczny framework adaptujący wytrenowane self-supervised reprezentacje wizualne (DINOv2, SigLIP)* do low-dimensional latents dla modeli generatywnych. Kluczową innowacją jest użycie pojedynczej warstwy self-attention do kompresji embeddingów o wysokiej wymiarowości, po której następuje architektura double-decoder oddzielająca rekonstrukcję ficzerów od syntezy obrazu. To podejście przezwycięża niekompatybilność między przestrzeniami ficzerów zorientowanymi na rozumienie a ficzerami przyjaznymi dla generowania bez złożonych funkcji straty lub znaczących zmian architektonicznych.

*Uwaga: To dwie różne rodziny modeli, DINOv2 będący self-supervised enkoderem wizji, podczas gdy SigLIP pochodzi z rodziny modeli CLIP (contrastive learning).

##### Kluczowe kontrybucje
1. Minimalna jednowarstwowa architektura kompresji łącząca rozumienie i generowanie z zachowaną jakością semantyczną
2. Architektura double-decoder umożliwiająca efektywne rozdzielenie celów rekonstrukcji ficzerów i syntezy obrazu
3. SOTA lub prawie-SOTA wydajność ze znacznie szybszą konwergencją niż wcześniejsze modele
4. Uniwersalna kompatybilność z różnymi bazowymi enkoderami i rodzinami modeli generatywnych

---

## Reasoning i systemy agentyczne

### Wysokopoziomowe podsumowanie sekcji

Ta grupa reprezentuje fundamentalne przeformułowanie tego, jak systemy AI podchodzą do złożonego rozumowania i rozwiązywania problemów. Zamiast polegać wyłącznie na skali, te publikacje demonstrują, że wybory architektoniczne, strategie orkiestracji i cele treningu mogą umożliwić istotne poprawienie możliwości rozumowania. ToolOrchestra pokazuje, że małe modele mogą koordynować większe modele i narzędzia bardziej efektywnie niż monolityczne giganty, osiągając lepsze wyniki za ułamek kosztu. "Less is More" dowodzi, że małe sieci rekurencyjne mogą przewyższać duże modele językowe w trudnych łamigłówkach poprzez głębokie, iteracyjne rozumowanie. LeJEPA dostarcza teoretyczne ugruntowanie dla self-supervised learning, które usuwa kruche heurystyki, jednocześnie poprawiając solidność modeli. Te postępy wspólnie nakreślają przyszłość, w której możliwości modelowania pochodzą nie tylko z rozmiaru modelu, ale z zasadniczego projektu architektonicznego, efektywnej orkiestracji zasobów i matematycznie dopracowanych procedur treningu.

### Publikacje

#### [ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration](https://arxiv.org/pdf/2511.21689)

ToolOrchestra wprowadza metodologię treningu małych modeli językowych do pełnienia funkcji agentów orkiestracji zarządzających zarówno tradycyjnymi narzędziami (wyszukiwanie w sieci, interpretery kodu), jak i różnorodnymi modelami językowymi jako narzędziami zewnętrznymi. Model Orchestrator-8B jest trenowany end-to-end poprzez reinforcement learning, kierowany przez poprawność wyników, efektywność (koszt i opóźnienie) oraz dopasowanie do preferencji użytkownika. Podejście obejmuje użycie ToolScale - dużego, syntetycznego benchmarku dla wieloetapowych zadań agenta narzędziowego, osiągając lepszą wydajność i efektywność kosztową względem monolitycznych LLM, w tym GPT-5.

##### Kluczowe kontrybucje
1. Zmiana paradygmatu z pojedynczego modelu do zorkiestrowanych systemów multi-tool/multi-model dla rozumowania
2. End-to-end trening RL ze zróżnicowanymi nagrodami (poprawność, efektywność, dopasowanie do użytkownika)
3. Ogromne zyski efektywności: Orchestrator-8B przewyższa znacznie większe modele za ułamek kosztu obliczeń
4. Doskonała generalizacja do nieznanych narzędzi, zadań i preferencji użytkowników, testowana z użyciem benchmarku ToolScale

#### [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/pdf/2510.04871)

Ta publikacja wprowadza Tiny Recursive Model (TRM), uproszczoną architekturę rozumowania rekurencyjnego używającą pojedynczej, małej (2-warstwowej, 7M parametrów) sieci neuronowej, która przewyższa zarówno Hierarchical Reasoning Model (HRM), jak i duże modele językowe w trudnych zadaniach jak Sudoku, Maze i ARC-AGI z minimalną ilością danych treningowych (~1,000 przykładów). W przeciwieństwie do złożonej, dualnej hierarchii sieciowej HRM, TRM używa eleganckiego projektu pojedynczej sieci, która przełącza się między aktualizacjami stanu ukrytego a udoskonalaniem rozwiązania z głęboką superwizją i prostymi kryteriami stopu.

##### Kluczowe kontrybucje
1. State-of-the-art wyniki w ekstremalnych benchmarkach rozumowania (87% Sudoku, 45% ARC-AGI-1) z 7M parametrów
2. Ogromne uproszczenie rozumowania rekurencyjnego: pojedyncza sieć vs. dualna hierarchia sieciowa
3. Dowód, że głęboka rekurencja z małymi sieciami unika nadmiernego dopasowania i maksymalizuje generalizację w reżimach małych ilości danych
4. Wyzwanie dla paradygmatu zorientowanego na skalę: projekt architektoniczny ważniejszy niż liczba parametrów dla strukturalnego rozumowania

#### [LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics](https://arxiv.org/pdf/2511.08544)

LeJEPA wprowadza teoretycznie zasadniczy framework dla self-supervised learning w ramach paradygmatu Joint-Embedding Predictive Architecture. Kluczowym spostrzeżeniem jest to, że izotropowe embeddingi Gaussowskie* wyjątkowo dobrze minimalizują ryzyko predykcji w szerokich rodzinach zadań downstream. LeJEPA łączy stratę predykcyjną z Sketched Isotropic Gaussian Regularization (SIGReg), która (jak udowodniono) wymusza izotropowość Gaussowską używając skalowalnych, lekkich w hiperparametrach, różniczkowalnych metod opartych na projekcji, eliminując kruche heurystyki jak stop-gradient i modele teacher-student.

*Uwaga: Izotropowe embeddingi Gaussowskie mają tę samą wariancję we wszystkich wymiarach, zapewniając optymalną eksploatację pojemności informacyjnej każdego wymiaru.

##### Kluczowe kontrybucje
1. Matematyczny dowód, że izotropowe embeddingi Gaussowskie minimalizują ryzyko downstream
2. SIGReg: skalowalny, różniczkowalny regularyzator wymuszający izotropowość Gaussowską z pojedynczym hiperparametrem i liniową złożonością
3. Walidacja na 60+ architekturach i 10 zbiorach danych pokazująca SOTA lub lepszą wydajność z wyjątkową stabilnością
4. Demonstracja, że domenowe SSL może przewyższać transfer z masywnych modeli podstawowych, kwestionując dominujące założenia

---

## Systemy retrieval i rankingowe

### Wysokopoziomowe podsumowanie sekcji

Ta grupa rozwiązuje praktyczne wyzwania wdrażania systemów AI na skalę przemysłową, gdzie efektywność, dopasowanie i odporność są niezbędne. Te prace niwelują lukę między postępami badawczymi a systemami produkcyjnymi oraz demonstrują, jak techniki z modeli językowych (context engineering, reasoning) mogą transformować zadania dyskryminatywne, jak wyszukiwanie i rekomendacje. OnePiece przenosi rozumowanie w stylu LLM do rankingu e-commerce z wymiernym wpływem na wyniki biznesowe, podczas gdy Late Chunking rozwiązuje fundamentalny problem w systemach retrieval poprzez zachowanie kontekstu na poziomie całego dokumentu. Włączenie Imperceptible Jailbreaking służy jako krytyczne przypomnienie, że w miarę jak te systemy stają się lepsze i szerzej wdrażane, zrozumienie ich podatności staje się niezbędne dla bezpiecznego wdrożenia produkcyjnego. Wspólnie te prace reprezentują dojrzałość AI od prototypów badawczych do odpornych, skalowalnych systemów obsługujących miliardy użytkowników.

### Publikacje

#### [OnePiece: Bringing Context Engineering and Reasoning to Industrial Cascade Ranking System](https://arxiv.org/pdf/2509.18091)

OnePiece wprowadza ujednolicony framework wzmacniający przemysłowe systemy rankingowe poprzez integrację context engineering i rozumowania w stylu LLM. System wzbogaca reprezentacje wejściowe poprzez strukturalny context engineering (historia użytkownika, kotwice preferencji z wiedzy eksperckiej, deskryptory sytuacyjne, zestawy przedmiotów kandydujących), implementuje block-wise latent reasoning dla wieloetapowego rozumowania skalowalnego pod względem przepustowości* i przyjmuje progresywny trening multi-task** używając naturalnych sygnałów feedbacku (kliknięcie, dodaj-do-koszyka, zakup) jako superwizji dla etapów rozumowania.

*Uwaga: Szerszy kanał informacyjny między krokami rozumowania poprzez użycie wielu tokenów zamiast 1 jak wcześniej proponowano.  
**Uwaga: Progresywność zapobiega konkurowaniu gradientów z wielu sygnałów feedbacku.

##### Kluczowe kontrybucje
1. Systematyczna adaptacja mechanizmów paradygmatu LLM (context engineering, wieloetapowe rozumowanie) do dyskryminatywnego rankingu przemysłowego
2. Architektura block-wise latent reasoning umożliwiająca skalowalne, wieloetapowe rozumowanie nad bogatymi reprezentacjami wejściowymi
3. Wdrożenie produkcyjne na skalę Shopee pokazujące wyższe przychody reklamowe i wartość towarów użytkownika wraz z poprawioną efektywnością
4. Lepsza efektywność parametrów/danych i wykorzystanie sprzętu w porównaniu do wysoce zoptymalizowanych i ugruntowanych baseline'ów (DLRM, HSTU)***
  
***Uwaga: DLRM to produkcyjny baseline model rekomendacji Shopee, podczas gdy HSTU to state-of-the-art framework rekomendacji od Meta.

#### [Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models](https://arxiv.org/pdf/2409.04701)

Late Chunking wprowadza nowe podejście do generowania embeddingów chunków tekstu, które zachowuje szersze informacje kontekstowe poprzez przetwarzanie całych dokumentów z modelami embeddingowymi o długim kontekście, aby wytworzyć embeddingi na poziomie tokenów, następnie dzieląc je na chunki. W przeciwieństwie do tradycyjnego chunkowania, które dzieli dokumenty przed embeddingiem, to zapewnia, że każda reprezentacja chunku korzysta z pełnego kontekstu dokumentu. Praca proponuje skalowalne "long late chunking" dla ogromnych dokumentów i wprowadza span pooling fine-tuning dla dalszych ulepszeń.

##### Kluczowe kontrybucje
1. Zmiana paradygmatu z chunkowania przed embeddingiem na chunkowanie po embeddingu zachowującego zależności kontekstowe między chunkami
2. Podejście model-agnostyczne nie wymagające dodatkowego treningu dla podstawowych korzyści z demonstrowanych korzyści dla retrievalu
3. Skalowalne rozwiązanie (long late chunking) dla ogromnych dokumentów przekraczających okna kontekstu modelu
4. Obliczeniowo bardziej efektywne niż alternatywy augmentacji kontekstowej oparte na LLM z natychmiastową stosowalnością w praktyce

#### [Imperceptible Jailbreaking against Large Language Models](https://arxiv.org/pdf/2510.05025)

Ta praca wprowadza imperceptible jailbreaks wykorzystujące niewidoczne selektory wariacji Unicode do dołączania wrogich sufiksów do promptów. Ataki tworzą niewidoczne zmiany, które wpływają na wejście tokenizera, pozostając niewidzialne dla ludzkich czytelników, efektywnie omijając uzgadnianie* bezpieczeństwa różnych open-source LLM. Pipeline optymalizacji chain-of-search wydajnie generuje udane, niewidoczne sufiksy dla różnych promptów i modeli, z wysoką szansą sukcesu w generowaniu szkodliwych outputów i prompt injection.

*Uwaga: Uzgadnianie/dopasowanie modelu językowego jest (z reguły bazującym na RL) procesem uczenia modelu specyficznego stylu odpowiadania na konkretną klasę promptów.

##### Kluczowe kontrybucje
1. Odkrycie nowej klasy podatności opartej na niewidocznych znakach Unicode omijających obecne uzgodnienia bezpieczeństwa
2. Demonstracja transferowalności ataku między wieloma architekturami LLM (Vicuna, Llama-2, Llama-3, Mistral)
3. Optymalizacja chain-of-search umożliwiająca wydajne generowanie wrogich sufiksów
4. Krytyczne ujawnienie podatności na poziomie tokenizera wymagające rewizji filtrowania wejścia i strategii uzgadniania bezpieczeństwa
