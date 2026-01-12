# BootstrapFinetune i paradygmat Teacher–Student w praktyce LLM

## Wstęp

W poprzednim artykule pokazaliśmy, jak **DSPy zastępuje ręczne promptowanie deklaratywnym programowaniem zachowania
modeli językowych** oraz automatyczną optymalizacją pipeline’ów. Sygnatury, moduły i optymalizery pozwalają znaleźć
skuteczne strategie bez manualnego dostrajania promptów.

Kolejnym krokiem jest pytanie: **co zrobić z tym zoptymalizowanym zachowaniem dalej?**
Jak przenieść je do środowiska produkcyjnego, gdzie kluczowe są koszt inferencji, latency i skalowalność?

Odpowiedzią jest paradygmat **Teacher–Student** oraz **BootstrapFinetune** w DSPy - mechanizm, który wykorzystuje duży
model jako nauczyciela do automatycznego generowania danych i fine-tuningu mniejszych, tańszych modeli.

W tym artykule omówię, jak BootstrapFinetune realizuje destylację wiedzy w praktyce, jakie daje korzyści oraz jakie
kompromisy projektowe warto wziąć pod uwagę.

---

## Paradygmat Teacher–Student (Knowledge Distillation)

Paradygmat **Teacher–Student** wywodzi się z klasycznego knowledge distillation, którego celem było przeniesienie wiedzy
z dużych, kosztownych modeli do mniejszych i tańszych odpowiedników. W kontekście **Large Language Models** sens tego
podejścia ulega jednak istotnemu przesunięciu. Nie chodzi już wyłącznie o aproksymację funkcji czy rozkładu
prawdopodobieństwa, lecz o przejęcie **zachowania modelu** - sposobu, w jaki odpowiada, strukturyzuje wyjście i radzi
sobie z niejednoznacznością.

W praktyce teacherem jest zazwyczaj duży model bazowy, działający w trybie zero‑shot lub few‑shot, który wykorzystuje
wiedzę wyniesioną z pretreningu. To właśnie jego odpowiedzi, formaty wyjściowe i heurystyki stają się materiałem uczącym.
Student nie przejmuje pełnej wiedzy teacher'a, lecz uczy się wzorca zachowania - czyli jak odpowiadać na określone typy
zadań, zachowaniem określonego stylu i struktury, a nie co dokładnie wie teacher.

Takie podejście ma istotne konsekwencje praktyczne. Po pierwsze, eliminuje konieczność ręcznego etykietowania danych -
teacher generuje pseudo-etykiety, czyli automatycznie wytworzone etykiety i odpowiedzi (wraz z krokami pośrednimi), 
które są następnie traktowane jak dane treningowe dla modelu studenta. Po drugie, umożliwia znaczną redukcję kosztów 
inferencji, ponieważ zachowanie drogiego modelu może zostać skompresowane do wag mniejszego studenta. Wreszcie, pozwala 
budować modele wyspecjalizowane w wąskich zadaniach, takich jak Text-to-SQL czy klasyfikacja, bez trenowania ich od zera.

Jednocześnie paradygmat Teacher–Student nie jest pozbawiony ograniczeń. Student może przejąć błędy lub biasy teacher'a,
a destylacja zachowania często prowadzi do utraty zaawansowanych zdolności obecnych dużych modelach takich jak złożone
rozumowanie, elastyczne rozumienie instrukcji, spójność przy długim kontekście. Istnieje również ryzyko nadmiernego 
dopasowania do stylu odpowiedzi teacher'a kosztem generalizacji. Z tego powodu skuteczność distillation zależy nie tylko
od architektury studenta, lecz także od jakości i spójności zachowania modelu nauczyciela.

W dalszej części artykułu ten paradygmat zostanie pokazany w praktyce - jako fundament działania **BootstrapFinetune**,
który wykorzystuje teacher'a do automatycznego budowania danych uczących i destyluje ich efekt bezpośrednio do wag
modelu studenta.

---

## Typy optimizerów w DSPy

DSPy traktuje **optymalizację programu** jako problem algorytmiczny. Różne optymizery ingerują w różne warstwy programu:

* **Optymalizacja few‑shotów**
  Automatyczne budowanie demonstracji w promptach (np. `BootstrapFewShot`, `KNNFewShot`).

* **Optymalizacja instrukcji**
  Generowanie i rafinacja promptów (np. `MIPROv2`, `GEPA`).

* **Optymalizacja wag (fine‑tuning)**
  Destylacja programu promptowego do wag modelu - dokładnie to robi **`BootstrapFinetune`**.

W praktyce często zaczyna się od few‑shotów i promptów, a **BootstrapFinetune** może być ostatnim krokiem prowadzącym do
modelu produkcyjnego.


---

## Czym jest BootstrapFinetune

**BootstrapFinetune** to optimizer DSPy, którego celem jest przekształcenie programu opartego na promptach w program
oparty na **wytrenowanych wagach modelu**.

Efektem nie jest pojedynczy fine‑tuned model, lecz **pełny program DSPy**, który:

* zachowuje tę samą strukturę,
* używa tych samych sygnatur,
* ale zamiast promptowania korzysta z **fine‑tuned student LMs**.

Oznacza to więc, że każdy moduł po BootstrapFinetune posiada własnego studenta, co umożliwia selektywną aktualizację
tylko wybranych komponentów pipeline’u.

BootstrapFinetune jest więc praktyczną implementacją paradygmatu Teacher–Student.

---

## Workflow BootstrapFinetune

Proces działania **BootstrapFinetune** można opisać jako sekwencję kilku logicznych kroków, które razem tworzą most
pomiędzy programem opartym na promptach a programem wykorzystującym wytrenowane wagi modelu.

### 1. Setup: student i teacher

Workflow rozpoczyna się od przygotowania **dwóch programów o identycznej strukturze**: studenta oraz teacher'a. Oba
programy mają te same predyktory i sygnatury, różnią się jedynie przypisanym modelem językowym. Teacher jest zwykle
dużym, silnym LLM, natomiast student - mniejszym modelem, który docelowo ma przejąć jego zachowanie.

```python
import dspy


class Text2SQL(dspy.Signature):
    question: str = dspy.InputField()
    db_id: str = dspy.InputField()
    db_schema: str = dspy.InputField(desc="SQLite DDL for the database.")
    query: str = dspy.OutputField(desc="SQL query in SQLite dialect.")


generate_sql_program = dspy.Predict(Text2SQL)

student_program = generate_sql_program.deepcopy()
student_program.set_lm(student_lm)

teacher_program = generate_sql_program.deepcopy()
teacher_program.set_lm(teacher_lm)
```

### 2. Przygotowanie zbioru treningowego (trainset)

Zanim BootstrapFinetune zostanie uruchomiony, konieczne jest przygotowanie **zbioru danych wejściowych**, na których
wykonywany będzie program teacher'a. W DSPy taki zbiór ma postać listy obiektów `dspy.Example` i definiuje *kontrakt
programu*: jakie pola są wejściem oraz jakie informacje są dostępne w trakcie predykcji.

Przykładowo, w zadaniu Text‑to‑SQL zbiór treningowy może zostać zbudowany na bazie Spidera w następujący sposób:

```python
def load_spider_data(schema_path: str = "data/spider_data/"):
    schema_loader = SpiderSchemaLoader(schema_path)
    return [
        dspy.Example(
            question=x["question"],
            db_id=x["db_id"],
            db_schema=schema_loader.load_schema(x["db_id"]),
            query=x["query"],
        ).with_inputs("question", "db_id", "db_schema")
    for x in DataLoader().from_huggingface(
            "xlangai/spider",
            split=split,
            fields=("question", "db_id", "query"),
            input_keys=("question",),
    )
    ]
```

Tak przygotowany `trainset` **nie jest jeszcze danymi do treningu wag**. Służy on wyłącznie jako wejście do uruchomienia
programu teacher'a w kolejnym kroku workflow.

### 3. Bootstrapping i zbieranie śladów

Następnie DSPy uruchamia program teacher'a na zbiorze treningowym. W trakcie tego etapu rejestrowane są nie tylko
końcowe odpowiedzi, lecz także **pełne ślady wykonania programu**. To właśnie one - obejmujące wejścia, wyjścia i
kontekst poszczególnych predyktorów - stanowią bazę danych uczących dla studenta.

Jeżeli dostępna jest metryka jakości (np. porównująca rezultat wykonania Gold SQL i wygenerowanego SQL), może ona zostać
użyta do odfiltrowania nieudanych przykładów. W praktyce często okazuje się, że mniejszy, ale spójny zbiór danych daje
lepsze rezultaty niż duża liczba przykładów o zmiennej jakości.
Metryka ta **nie jest używana do optymalizacji gradientowej**, lecz wyłącznie do **selekcji i filtrowania przykładów**,
które trafiają do zbioru danych uczących dla studenta.

```python
optimizer = dspy.BootstrapFinetune(
    metric=metric,
    num_threads=4,
    train_kwargs=TRAIN_KWARGS,
)
```

### 3. Przygotowanie danych do fine-tuningu

Ten etap dotyczy przetworzenia **śladu wykonania programu teacher'a** na dane akceptowane przez backend fine-tuningu.

Dopiero w trakcie bootstrappingu DSPy wykonuje **program teacher'a** na tych przykładach i rejestruje **ślady
wykonania (execution traces)**. Taki ślad zawiera m.in.:

- faktyczne prompty wysłane do modelu,

- odpowiedzi wygenerowane przez teacher'a,

- kontekst predyktorów i ewentualne reasoning.

Na tym etapie BootstrapFinetune konwertuje te ślady do **formatu treningowego**, który rozumie backend fine-tuningu.
Może to być na przykład **format czatu** (listy wiadomości `user` / `assistant`):

```json
{
  "messages": [
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ]
}
```

### 4. Fine-tuning i aktualizacja programu

W ostatnim kroku **BootstrapFinetune** wywołuje metodę `finetune()` na modelu przypisanym do studenta, delegując
faktyczny trening wag do wybranego providera. Po zakończeniu treningu nowy model jest automatycznie podpinany do
programu studenta, a - opcjonalnie - demonstracje promptowe mogą zostać usunięte.

```python
compiled_student = optimizer.compile(
    student=student_program,
    teacher=teacher_program,
    trainset=train_data,
)

compiled_student.save(
    "./artifacts/text2sql_student/",
    save_program=True,
)
```

Efektem całego procesu jest **gotowy program produkcyjny**, który zachowuje strukturę oryginalnego programu DSPy, ale
nie wymaga już uruchamiania teacher'a w czasie inferencji.

---

## Jak to działa wewnętrznie: abstrakcja fine‑tuningu w DSPy

Jednym z istotnych elementów projektu **BootstrapFinetune** jest to, że sam optimizer **nie implementuje treningu modelu
bezpośrednio**. Zamiast tego opiera się na abstrakcji metody `finetune()` dostępnej na obiekcie `LM`. Dzięki temu logika
destylacji (zbieranie śladów, filtrowanie, przygotowanie danych) jest oddzielona od **konkretnego backendu treningowego**.

Z perspektywy kodu DSPy:

- `BootstrapFinetune`:
    - buduje dane treningowe,
    - grupuje je per model językowy,
    - wywołuje `lm.finetune(**kwargs)`.
- szczegóły treningu są delegowane do **providera** przypisanego do danego `LM`.

Ta separacja pozwala stosować ten sam optimizer zarówno dla modeli lokalnych, jak i zdalnych API.

### Interfejs `LM.finetune()`

Dla `BootstrapFinetune` kluczowe jest jedynie to, że obiekt `LM` udostępnia metodę:

```python
finetuned_lm = lm.finetune(train_data=..., **train_kwargs)
```

Optimizer nie zakłada:

- jakiego frameworka używa trening (PyTorch, TRL, API),
- czy trenowane są pełne wagi czy adaptery (PEFT),
- czy trening jest lokalny czy zdalny.

To sprawia, że **BootstrapFinetune jest backend‑agnostyczny**.

### LocalProvider

`LocalProvider` udostępnia najprostszą możliwą implementację metody `LM.finetune()`, opartą bezpośrednio o *
*transformers** i **TRL**. W praktyce oznacza to uruchomienie lokalnego treningu SFT na GPU, bez dodatkowej warstwy
orkiestracji czy optymalizacji.

Na poziomie koncepcyjnym pipeline jest prosty: dane wygenerowane przez teacher'a są zapisywane lokalnie w formacie
czatu, ładowany jest model bazowy wraz z tokenizerem, a następnie uruchamiany jest trening **Supervised Fine-Tuning**,
opcjonalnie z użyciem technik **PEFT/LoRA**. Wynikiem jest nowy checkpoint modelu, który DSPy traktuje jak kolejny `LM`.

Choć takie podejście działa technicznie, w praktyce szybko ujawniają się jego ograniczenia. Trening oparty bezpośrednio
o gołe `transformers` i `trl` wymaga ręcznego dostrajania hiperparametrów pod konkretny model i GPU, co łatwo prowadzi
do problemów z pamięcią (OOM), niskiej wydajności, problemami z fine-tuningiem modeli zkwantyzowanych lub niestabilnej
jakości. Jednocześnie traci się część abstrakcji, którą oferuje DSPy - zamiast pracować na poziomie programu i
destylacji zachowania, użytkownik zaczyna debugować szczegóły treningu wag.

Dodatkowym problemem jest ograniczona kontrola operacyjna: brak wygodnego wznawiania treningu, przerwań czy
jednoznacznej odpowiedzi na pytanie, *co dokładnie* i w jakiej konfiguracji jest trenowane. Z tego powodu
`LocalProvider` warto traktować raczej jako ciekawostkę implementacyjną lub narzędzie demonstracyjne, pozwalające
zrozumieć przepływ BootstrapFinetune end‑to‑end, a nie jako rozwiązanie do sensownego treningu wag modeli open‑weight.

Dla kompletności, minimalne użycie `LocalProvider` wygląda następująco:

```python
import dspy
from dspy.clients.lm_local import LocalProvider

# Włączenie eksperymentalnego fine-tuningu
dspy.settings.experimental = True

# Definicja lokalnego modelu studenta
student_lm = dspy.LM(
    model="openai/local:Qwen/Qwen3-0.6B",
    provider=LocalProvider(),
    max_tokens=3000,
)

# Podpięcie LM do programu
student_program = program.deepcopy()
student_program.set_lm(student_lm)

# BootstrapFinetune wywoła student_lm.finetune(...) pod spodem
optimizer = dspy.BootstrapFinetune(train_kwargs={"use_peft": True})
compiled = optimizer.compile(
    student=student_program,
    teacher=teacher__program,
    trainset=trainset,
)

```

W praktycznych scenariuszach fine‑tuningu modeli open‑source znacznie lepiej sprawdzają się wyspecjalizowane narzędzia,
takie jak **Unsloth**, **Axolotl** czy **MLX**, które oferują lepszą kontrolę nad pamięcią, wydajnością i reżimem
treningu. W tym ujęciu `LocalProvider` pozostaje wygodnym punktem integracji w DSPy, ale niekoniecznie docelowym
backendem treningowym.

### OpenAIProvider

W przypadku `OpenAIProvider` ten sam interfejs `LM.finetune()` mapuje się na zdalny proces fine‑tuningu zarządzany przez
API. Dane treningowe są wysyłane do usługi zewnętrznej, trening odbywa się asynchronicznie, a jego wynikiem jest nowy
identyfikator modelu, który może być używany w DSPy tak samo jak każdy inny `LM`.

Z punktu widzenia `BootstrapFinetune` różnice implementacyjne między providerami są w dużej mierze ukryte. Optimizer
otrzymuje po prostu nowy model do podpięcia w programie, niezależnie od tego, czy trening odbył się lokalnie, czy w
infrastrukturze dostawcy API.

Dla kompletności, minimalne użycie fine‑tuningu z wykorzystaniem `OpenAIProvider` wygląda następująco:

```python
import dspy

# Model studenta – będzie fine‑tuningowany przez API
student_lm = dspy.LM(
    model="openai/gpt-5-nano",
    max_tokens=2000,
)

# Model teacher'a
teacher_lm = dspy.LM(
    model="openai/gpt-5.1-codex-mini",
    max_tokens=16000,
)

student_program = program.deepcopy()
student_program.set_lm(student_lm)

teacher_program = program.deepcopy()
teacher_program.set_lm(teacher_lm)

# BootstrapFinetune uruchomi zdalny job fine‑tuningu
optimizer = dspy.BootstrapFinetune(
    metric=metric,  # opcjonalnie, jeśli mamy etykiety
)

compiled = optimizer.compile(
    student=student_program,
    teacher=teacher_program,
    trainset=trainset,
)
```

W tym wariancie cała złożoność treningu (dobór batchy, harmonogramy, stabilność) jest obsługiwana przez dostawcę API, a
DSPy zachowuje swoją rolę warstwy orkiestracji.

---

## Eksperyment: Text-to-SQL na zbiorze Spider

Aby zweryfikować omawiany workflow w praktyce, przeprowadziłem eksperyment na zadaniu **Text-to-SQL** z wykorzystaniem
popularnego zbioru danych **Spider**. Celem było sprawdzenie, w jakim stopniu destylacja zachowania teacher'a do wag
mniejszego modelu pozwala poprawić jakość generowanych zapytań SQL oraz jakie kompromisy pojawiają się przy różnych
konfiguracjach treningu.

### Konfiguracja eksperymentu

- **Zadanie**: Text-to-SQL (Spider)
- **Metryka**: procent zapytań SQL wygenerowanych przez model, które po wykonaniu dały **identyczny rezultat** jak
  zapytanie Gold SQL (execution‑based accuracy)
- **Teacher**: GPT-5.1-codex-mini
- **Studenci**: modele GPT-4.1-nano oraz Qwen3-0.6B
- **Metoda**: BootstrapFinetune (pełny fine-tuning lub LoRA)

Przedstawiony eksperyment ma charakter eksploracyjny, a jego celem jest ilustracja wpływu destylacji zachowania na
jakość modeli, a nie pełna porównawcza ewaluacja benchmarkowa.

### Rezultaty bez fine tuningu

| Student Model           | Moduł/Program  | Rezultat |
|-------------------------|----------------|----------|
| GPT-5.1-codex-mini      | ChainOfThought | 77%      |
| GPT-5.1-codex-mini      | Predict        | 78%      |
| gpt-4.1-nano-2025-04-14 | ChainOfThought | 76%      |
| gpt-4.1-nano-2025-04-14 | Predict        | 72%      |
| Qwen3-0.6B              | ChainOfThought | 27%      |
| Qwen3-0.6B              | Predict        | 23%      |

### Wyniki

| Student Model                               | Moduł/Program  | Metoda fine-tuningu | Rezultat |
|---------------------------------------------|----------------|---------------------|----------|
| ft:gpt-4.1-nano-2025-04-14:j-labs::Cw77bexu | ChainOfThought | LoRA                | 70%      |
| ft:gpt-4.1-nano-2025-04-14:j-labs::Cw7RBixD | Predict        | LoRA                | 68%      |
| Qwen3-0.6B                                  | ChainOfThought | Full Training       | 44%      |
| Qwen3-0.6B                                  | Predict        | Full Training       | 42%      | 
| Qwen3-0.6B                                  | Predict        | LoRA                | 27%      |

### Interpretacja i wnioski

Wyniki potwierdzają, że **BootstrapFinetune skutecznie kompresuje zachowanie silnego teacher'a do wag
mniejszych modeli**, przy czym skala poprawy zależy od relacji pomiędzy modelem bazowym a zadaniem.

Najbardziej jednoznaczne efekty widoczne są w przypadku **Qwen3-0.6B**. Model bazowy osiągał 23–27% execution accuracy,
natomiast po pełnym fine-tuningu z wykorzystaniem odpowiedzi teacher'a wynik wzrósł do **42–44%**. Pokazuje to, że nawet
bardzo mały model jest w stanie przejąć część heurystyk i struktury rozwiązań większego LLM, co czyni destylację
użytecznym narzędziem redukcji kosztu inferencji.

W przypadku **GPT-nano**, który już bez fine-tuningu osiągał wyniki zbliżone do teacher'a (72–76%), zastosowanie
BootstrapFinetune z LoRA nie przyniosło poprawy, a wręcz doprowadziło do **spadku jakości** (68–70%). Sugeruje to, że
destylacja zachowania ma ograniczoną wartość dla już mocnych modeli i może prowadzić do degradacji, jeśli student
jest już dobrze dopasowany do zadania.

Porównanie modułów `ChainOfThought` i `Predict` nie wykazało istotnych różnic jakościowych. W zadaniu Text-to-SQL
kluczowa jest poprawność końcowego zapytania, a nie jawna reprezentacja kroków pośrednich, co czyni destylację
reasoning'u opcjonalną decyzją projektową.

Istotne różnice pojawiają się natomiast pomiędzy metodami fine-tuningu. Dla Qwen3-0.6B **pełny trening wag** okazał się
wyraźnie skuteczniejszy niż LoRA, co sugeruje, że przy bardzo małych modelach i zadaniach strukturalnych ekspresyjność
adapterów może być niewystarczająca.

Podsumowując, BootstrapFinetune najlepiej sprawdza się jako narzędzie **świadomej kompresji zachowania**: szczególnie
tam, gdzie student jest wyraźnie słabszy od teacher'a, a celem jest obniżenie kosztów inferencji przy kontrolowanej
utracie jakości. Nie jest to jednak uniwersalna metoda poprawy modeli i wymaga dopasowania do charakteru zadania oraz
architektury studenta.

---

## Podsumowanie

Paradygmat **Teacher–Student** w połączeniu z **BootstrapFinetune** stanowi spójny sposób przejścia od prototypu
opartego na promptach do systemu wykorzystującego wytrenowane wagi modeli.

BootstrapFinetune nie jest narzędziem do samego treningu wag. Jego rolą jest **orkiestracja procesu destylacji** na
poziomie programu DSPy: określenie, *jakie zachowanie* ma zostać skompresowane i *w jakiej strukturze* ma działać model
po fine-tuningu. To podejście najlepiej sprawdza się w scenariuszach, w których student jest wyraźnie słabszy od
teacher'a, a celem jest redukcja kosztu inferencji przy kontrolowanej utracie jakości.

Jednocześnie artykuł pokazuje, że skuteczność takiego workflow jest silnie **zależna od kontekstu**: od wyboru
teacher'a, charakteru zadania oraz technologii użytej do fine-tuningu wag. W szczególności destylacja zachowania nie
jest uniwersalnym sposobem poprawy modeli i może przynosić ograniczone lub negatywne efekty w przypadku modeli już
dobrze dopasowanych do zadania.

W tym ujęciu BootstrapFinetune nie "rozwiązuje problemu fine-tuningu", lecz go porządkuje - wyznaczając klarowną
granicę między projektowaniem zachowania modelu a inżynierią jego wag.

---

## Źródła
- [Teacher-Student Architecture for Knowledge Distillation: A Survey - arXiv:2308.04268v1](https://arxiv.org/pdf/2308.04268)
- [Tutorial DSPy - classification finetuning](https://dspy.ai/tutorials/classification_finetuning/)
- [Dokumentacja BootstrapFinetune](https://dspy.ai/api/optimizers/BootstrapFinetune/)
