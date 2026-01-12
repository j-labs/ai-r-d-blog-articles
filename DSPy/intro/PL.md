# DSPy: Programowanie Modeli Językowych Zamiast Promptowania

Prompt engineering stał się standardową praktyką w pracy z modelami językowymi. Jednak ręczne tworzenie promptów jest czasochłonne, 
zawodne i trudne w utrzymaniu w sytuacji zmiennych wymagań. DSPy, opracowany przez badaczy NLP ze Stanford, oferuje alternatywne podejście: 
traktowanie interakcji z modelami językowymi jako programowalnych modułów, które można automatycznie optymalizować, zamiast 
ręcznie dostrajać prompty.

## Wprowadzenie

DSPy (Declarative Self-improving Python) to framework, który strukturyzuje pipeline'y modeli językowych jako deklaracje transformacji 
tekstu. Zamiast mozolnie tworzyć prompty, deweloperzy deklarują pożądane zachowanie, używając sygnatur i modułów.
Framework następnie używa optimizerów (nazywanych "teleprompters" w pierwszej wersji) do automatycznego 
ulepszania promptów lub wag modelu (fine-tuning również uwzględniono ;)) na podstawie określone metryki (często bardzo customizowej).

Główna filozofia: opisz, co ma zrobić AI, a nie jak je do tego nakłonić. Poniższy przykład pokazuje tworzenie 
podstawowego agenta ReAct (Reason and Act), dla którego zdefiniowano sygnaturę (format wejścia-wyjścia i opcjonalną 
początkową instrukcję) oraz narzędzie do wyszukiwania dokumentów (w tym przykładnie jest to serwer modelu ColBERT, jednak 
mogłoby być zastosowany jakikolwiek inny dostawca wyszukiwania semantycznego).

```python
import dspy


def search(query: str, k: int) -> list[str]:
    results = dspy.ColBERTv2(url='http://127.0.0.1:8000/search')(query, k=k)  # client to local semantic search server
    return [x['text'] for x in results]


def semantic_search(query: str) -> list[str]:
    """Returns top-5 results and then the titles of the top-5 to top-30 results."""

    topK = search(query, 30)
    titles, topK = [f"`{x.split(' | ')[0]}`" for x in topK[5:30]], topK[:5]
    return topK + [f"Other retrieved pages have titles: {', '.join(titles)}."]

# Define the task signature
signature = dspy.Signature(
    "question -> titles: list[str]",
    instructions="Based on user's prompt create a semantic search query to find all document titles needed to answer the question."
)

# Create a module with a strategy
react = dspy.ReAct(signature, tools=[semantic_search], max_iters=20)

print(react("Who flew over the cuckoo nest in 1975?"))
# Prediction(
#   titles=["Article Title 1", "Article Title 2", "Article Title 3"]
# )
```

W przeciwieństwie do tradycyjnego prompt engineeringu, gdzie można spędzić godziny na dostrajaniu detali promptu, DSPy 
deleguje swojemu optimizerowi automatyczne rozbudowywanie instrukcji, z zachowaniem wymaganego formatu wejścia-wyjścia.

## Podstawowe Komponenty

DSPy składa się z trzech głównych elementów:

### Sygnatury

`Signatures` definiują charakterystykę wejścia-wyjścia wywołania modelu językowego wraz z opcjonalnymi instrukcjami, 
które mają wyrażać wysokopoziomowe intencje programisty i nie muszą być bardzo rozbudowane.

```python
# Simple signature: question in, answer out
import dspy

signature = "question -> answer"

# Signature with type hints
signature = "question -> titles: list[str]"

# Full signature with instructions
signature = dspy.Signature(
    "question, articles: list[str] -> answer",
    instructions="Generate an answer based on the provided articles."  # very simple, high-level instructions
)

# class based approach to signatures
class QuestionAnswering(dspy.Signature):
    """Generates an answer based on the provided articles."""
    question: str = dspy.InputField(desc="Question to answer.")  # InputField declares what comes in
    articles: list[str] = dspy.InputField(desc="List of articles to use for answering.")
    answer: str = dspy.OutputField(desc="The generated answer.")  # OutputField declares what comes out
```

### Moduły

`Modules` implementują różne strategie wywoływania modeli językowych. Popularne moduły to:

- `dspy.Predict`: Podstawowa predykcja
- `dspy.ChainOfThought`: Dodaje ścieżkę rozumowania przed rezultatem
- `dspy.ReAct`: Rozumowanie z użyciem narzędzi

W poniższym przykładzie używane są dwa oddzielne sub-moduły do implementacji pipeline'u retrieval-augmented generation (RAG). 
Użycie osobnych modułów pozwala na kompilację (w rozumieniu DSPy: stopniowe, ewolucyjne budowanie) wyspecjalizowanych 
promptów dla każdego z nich:

```python
class RAGAgent(dspy.Module):
    def __init__(self, retrieval_prompt_path: str):
        super().__init__()
        # ReAct module for document retrieval
        self.retrieval = dspy.ReAct(
            tools=[semantic_search],
            signature="question -> titles: list[str]",
            max_iters=20,
        )
        self.retrieval.load(retrieval_prompt_path)

        # ChainOfThought module for answer generation -> this will be optimized
        self.answer = dspy.ChainOfThought(
            signature="question, articles: list[str] -> answer"
        )

    def forward(self, question: str) -> tuple[Prediction, list[str]]:
        """
        This is the main entry point for the module that defines the pipeline and can be an arbitrary Python code.
        """
        titles = self.retrieval(question=question)
        articles = [get_article(title) for title in titles]
        answer = self.answer(question=question, articles=articles)
        return answer, articles
```

### Optimizers

'Optimizers' automatycznie poprawiają wydajność modułów poprzez:
- Generowanie demonstracji (kilka przykładów)
- Ulepszanie instrukcji
- Wybieranie efektywnych strategii promptowania
- Bootstrapping z udanych wykonań

Najbardziej zaawansowanym optimizerem jest obecnie GEPA (Generalized Evolutionary Prompt Adapter), który używa refleksji 
i strategii ewolucyjnych do ulepszania promptów.

## Proces Optymalizacji

Workflow optymalizacji składa się z następujących kroków:

1. **Zdefiniuj metrykę**, która mierzy wydajność modułu
2. **Przygotuj datasety** (treningowy i walidacyjny)
3. **Uruchom optimizer**, aby znaleźć lepsze prompty/wagi
4. **Ewaluuj** zoptymalizowany moduł

### Własne metryki

Metryki mogą zwracać proste score'y lub dostarczać szczegółowy feedback dla optymalizacji na poziomie predykcji:

```python
def top5_recall(
    example, pred, trace=None, pred_name=None, pred_trace=None, *args, **kwargs
) -> float | ScoreWithFeedback:
    """Compute top-5 recall for document retrieval."""
    gold_titles = example.titles
    recall = sum(
        x in getattr(pred, 'titles', [])[:5]
        for x in gold_titles
    ) / max(1, len(gold_titles))

    # Predictor-level feedback for GEPA optimization
    if pred_name is not None or pred_trace is not None:
        feedback_text = (
            f"Predictor '{pred_name}' achieved top-5 recall {recall:.3f}. "
            f"Gold titles: {gold_titles}. "
            f"Predicted titles (top-5): {getattr(pred, 'titles', [])[:5]}"
        )
        return ScoreWithFeedback(score=recall, feedback=feedback_text)

    return recall
```

W bardziej złożonych scenariuszach deweloperzy mogą pisać mocno customizowane metryki:

```python
class TeacherGuidedJudgedMetric:
    """
    Metric using a teacher model to dynamically generate reference answers,
    then judging the prediction quality.
    """
    def __init__(
        self,
        teacher_lm: dspy.LM,
        judge_lm: dspy.LM,
        teacher_instruction: str = (
                "Generate a reference answer for a question based on the context. If impossible to answer with this "
                "context, state 'Impossible to answer with this context'."
        ),
        judge_instruction: str = (
                "Score the correctness of the predicted answer as a float between 0 and 1, where 0 is total nonsense and 1 is perfect match."
        )
    ):
        super().__init__()
        self.teacher_lm = teacher_lm
        teacher_signature = dspy.Signature(
            "question, context: list[str] -> answer",
            instructions=teacher_instruction
        )
        self.teacher = dspy.ChainOfThought(teacher_signature)

        self.judge_lm = judge_lm
        judge_signature = dspy.Signature(
            "gold, pred, context: list[str] -> score: float",
            instructions=judge_instruction
        )
        self.judge = dspy.Predict(judge_signature)

    def __call__(self, gold: Example, pred: tuple[Prediction, list[str]],
                 trace=None, pred_name=None, pred_trace=None):
        with dspy.context(lm=self.teacher_lm):
            teacher_answer = self.teacher(
                question=gold.question,
                context=pred[1]
            ).answer

        with dspy.context(lm=self.judge_lm):
            score = self.judge(
                gold=teacher_answer,
                pred=pred[0].answer,
                context=pred[1]
            ).score

        if pred_name is not None or pred_trace is not None:
            feedback_text = (
                f"Predictor '{pred_name}' achieved score {score}. "
                f"Gold answer: {teacher_answer}. "
                f"Predicted answer: {pred[0].answer}. "
                f"Context: {pred[1]}"
            )
            return ScoreWithFeedback(score=score, feedback=feedback_text)

        return score
```

### Uruchamianie Optymalizacji

Gdy metryka jest zdefiniowana, optymalizacja jest prosta:

```python
# Configure language model
gpt = dspy.LM("openai/gpt-4.1-nano", max_tokens=8000)

# Prepare evaluation
evaluate = dspy.Evaluate(
    devset=testset,
    metric=top5_recall,
    num_threads=2,
    display_progress=True
)

# Optimize with GEPA
with dspy.context(lm=gpt):
    gepa_optimizer = dspy.GEPA(
        metric=top5_recall,
        auto="medium",
        num_threads=2,
        reflection_lm=dspy.LM("openai/gpt-4.1", temperature=1.0, max_tokens=8000)
    )
    optimized_react = gepa_optimizer.compile(
        react,  # the ReAct module from the beginning of this blogpost
        trainset=trainset,
        valset=devset
    )

# Save the optimized module
optimized_react.save("optimized_react.json")
```

Optimizer eksploruje przestrzeń możliwych instrukcji i demonstracji (przykładów z zestawu treningowego), 
używając walidacyjnego zestawu do wyboru najlepszej konfiguracji. Można realizować w ten sposób optymalizację złożonych 
przypadków, jak np. wielopoziomowych modułów. Produktem optymalizacji jest nowy prompt w formacie JSON, który może być 
łatwo załadowany do kompatybilnej instancji modułu DSPy.

## Komponowanie Pipeline'ów

Moduły DSPy łatwo się komponują. Kompletny pipeline RAG (Retrieval-Augmented Generation) łączy retrieval i generację. 
Cały pipeline może być optymalizowany end-to-end. Kontrastuje to z optymalizacją "płaskiego" modułu, zaprezentowaną w jednym 
ze wcześniejszych przykładów.

```python
agent = RAGAgent("optimized_react.json", "optimized_cot.json")

# Optimize the complete RAG pipeline
with dspy.context(lm=gpt):
    tp = dspy.GEPA(
        metric=TeacherGuidedJudgedMetric(
            teacher_lm=gpt_teacher,
            judge_lm=gpt_teacher  # but might be a different model
        ),
        auto="medium",
        num_threads=12,
        reflection_lm=gpt_teacher
    )
    optimized_rag_agent = tp.compile(
        agent,
        trainset=trainset,
        valset=devset
    )
```

## Przewagi nad ręcznym promptowaniem

DSPy oferuje szereg korzyści w porównaniu z tradycyjnym prompt engineeringiem:

**Utrzymywalność**: Zmiany w pipeline nie wymagają przepisywania promptów. Modyfikujesz sygnaturę lub strukturę modułu, 
ponownie optimizujesz i wdrażasz.

**Przenoszalność**: Ten sam kod działa z różnymi modelami językowymi. Optymalizacja adaptuje prompty do charakterystyk każdego modelu.

**Systematyczne ulepszanie**: Optymalizacja jest data-driven i powtarzalna, nie zależy od manualnych prób i błędów.

**Separacja odpowiedzialności**: Logika systemu (co zrobić) jest oddzielona od strategii promptowania (jak zapytać model).

**Optymalizacja kosztów**: DSPy, dzięki modularności i automatycznej optymalizacji, pozwala testować różne modele i 
warianty promptów w sposób systematyczny. Ułatwia to znalezienie najkorzystniejszego punktu równowagi między jakością a kosztem.

## Charakterystyki wydajnościowe

Na podstawie oryginalnej pracy o DSPy i praktycznych eksperymentów:

- Czas kompilacji: od kilku minut do kilku godzin w zależności od rozmiaru datasetu, ustawień optimizera i rate limitów dostawcy LLM 
(wpływa to na możliwość użycia współbieżności)
- Wzrost wydajności: znacząca poprawa w stosunku do baselinów few-shot prompting (dla naszego RAGAgent score wzrósł z 68% do 77%!)

Koszt optymalizacji płaci się raz; wynikające zoptymalizowane moduły mogą być zapisane, wersjonowane i wdrażane jak każdy inny artefakt kodu.

## Przykłady promptów

Niezoptymalizowany:
```
Based on user's prompt create a semantic search query to find all document titles needed to answer the
question.\n\nYou are an Agent. In each episode, you will be given the fields `question` as input. And you can
see your past trajectory so far.\nYour goal is to use one or more of the supplied tools to collect any necessary
information for producing `titles`.\n\nTo do this, you will interleave next_thought, next_tool_name,
and next_tool_args in each turn, and also when finishing the task.\nAfter each tool call, you receive a
resulting observation, which gets appended to your trajectory.\n\nWhen writing next_thought, you may reason
about the current situation and plan for future steps.\nWhen selecting the next_tool_name and its
next_tool_args, the tool must be one of:\n\n(1) semantic_search, whose description is <desc>Returns top-5
results and then the titles of the top-5 to top-30 results.</desc>. It takes arguments {'query': {'type':
'string'}}.\n(2) finish, whose description is <desc>Marks the task as complete. That is, signals that all
information for producing the outputs, i.e. `titles`, are now available to be extracted.</desc>. It takes
arguments {}.\nWhen providing `next_tool_args`, the value inside the field must be in JSON format
```

Zoptymalizowany:
```
You are an Agent tasked with generating precise semantic search queries to retrieve all relevant document titles
needed to answer user-supplied factual questions, particularly those related to specific entities (such as TV
episodes, geographic features, or music releases).\n\nTask Format and Workflow:\n- Each episode consists of a
single input: a `question` field with a natural language factual query requiring one or more document titles to
answer.\n- You will reason through your approach by writing a `next_thought` to plan your query.\n- You will
then select and call a tool using `next_tool_name` and `next_tool_args`. The tools available are:\n  1.
`semantic_search`: Takes a JSON argument `{'query': <string>}` and returns the top-5 results and the titles of
results ranked 6–30. This is used to retrieve documents whose titles are relevant to your semantic query.\n  2.
`finish`: Takes `{}` as its argument and marks the task as complete when you have gathered enough information to
produce the list of required titles.\n- After each tool call, read the resulting observation, then reason and
proceed to the next step or finish.\n\nBest Practices and Domain-Specific Guidelines:\n1. Carefully parse the
user question for specific entities, proper nouns, qualifiers (e.g., \"season finale\", \"distinct summit\",
\"debut single\") and relationships.\n2. When formulating the semantic search query:\n   - Explicitly include
all crucial context and identifiers (e.g., show names, season/episode numbers, artist/song names, geographic
names).\n   - Rephrase or expand abbreviations, and include both key terms and synonyms where relevant to
maximize coverage.\n   - For ambiguous queries (such as when the answer involves a two-step process,
like identifying an artist and then their debut year), create a query that brings together both aspects (e.g.,
\"<song> singer debut single release year\").\n3. The goal of your query is not to immediately answer the
question, but to retrieve document titles that, collectively, contain or point to the answer.\n4. For questions
requesting a comparison or relationship (e.g., \"Which mountain ...\"), ensure both entities are named in the
query.\n5. Never invent facts or titles; your role is only to retrieve all potentially relevant titles.\n6. Use
the `finish` tool only when you are confident all necessary titles have been retrieved via semantic search.\n7.
Throughout, make your `next_thought` explicit and strategic: describe your reasoning for entity extraction,
and justify your query construction.\n8. Do not output answers to factual questions—your focus is on
constructing optimal semantic search queries and managing tool use.\n\nExample Workflow (Given Past
Performance):\n- For a TV episode finale, search for \"<show name> season <number> finale episode title\".\n-
For comparing properties of two geographic features, search for \"<Entity1> <Entity2> <comparison keyword("
s)>\".\n- For music release chronology, include both the song and artist and reference debut
information.\n\nAlways optimize for recall: your query should maximize the chance of retrieving every document
whose title may plausibly answer the user's question.\n\nYour outputs should always strictly adhere to the
required interleaving: next_thought, next_tool_name, next_tool_args (with valid JSON values inside
`next_tool_args`).\n\nProceed step by step, reflecting and reasoning about your choices, and always ensure the
query captures all important aspects of the user's question.
```

## Podsumowanie

DSPy reprezentuje krok naprzód od ręcznego tworzenia promptów do programatycznego prompt engineeringu. Poprzez 
definiowanie sygnatur tasków, komponowanie modułów i używanie automatycznej optymalizacji, deweloperzy mogą budować bardziej 
utrzymywalne i efektywne pipeline'y modeli językowych. Deklaratywne podejście frameworku oddziela specyfikację pożądanego 
zachowania od szczegółów implementacyjnych promptowania, czyniąc systemy AI bardziej odpornymi na zmiany w bazowych modelach i wymaganiach.

Podczas gdy tradycyjne promptowanie pozostaje użyteczne dla szybkich prototypów i prostych zadań, DSPy zapewnia ustrukturyzowaną 
ścieżkę do aplikacji modeli językowych produkcyjnej jakości. W miarę ewolucji modeli, posiadanie frameworku, który może 
automatycznie adaptować prompty i strategie, staje się coraz bardziej wartościowe.

## Źródła

[^1]: [Stanford NLP DSPy Repository](https://github.com/stanfordnlp/dspy)
[^2]: [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714)
[^3]: [DSPy Official Documentation](https://dspy.ai/)
