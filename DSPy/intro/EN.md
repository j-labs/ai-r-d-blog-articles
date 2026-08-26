# DSPy: Programming Language Models Instead of Prompting Them

Prompt engineering has become a standard practice in working with language models. However, manually crafting prompts 
is time-consuming, fragile and difficult to maintain as requirements evolve. DSPy, developed by Stanford NLP researchers, 
offers an alternative approach: treating language model interactions as programmatic modules that can be automatically 
optimized rather than manually tuned strings (prompts).

## Intro

DSPy (Declarative Self-improving Python) is a framework that abstracts language model pipelines as text transformation 
declarations. Instead of carefully crafting prompt strings, developers define the desired behavior declaratively using signatures 
and modules. The framework then uses optimizers (called "teleprompters" in the very first release) to automatically improve 
prompts or model weights (yes, finetuning included ;)) based on specified metrics (often very customized).

The core philosophy: describe what you want the AI to do, not how to prompt it to do so. The example below shows
the creation of a basic ReAct (Reason and Act) agent, for which a signature (input-output format and optional
initial instruction) and a document search tool are defined (in this example it's a ColBERT model server, but
any other semantic search provider could be used).

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

print(react(question="Who flew over the cuckoo nest in 1975?"))
# Prediction(
#   titles=["Article Title 1", "Article Title 2", "Article Title 3"]
# )
```

Unlike traditional prompt engineering where you might spend hours tuning a prompt string, DSPy
delegates to its optimizer the automatic expansion of instructions, while maintaining the required input-output format.

## Core Components

DSPy consists of three main building blocks:

### Signatures

Signatures define the input-output specification of a language model call, along with optional instructions
that should express high-level developer intentions and do not need to be very elaborate.

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

### Modules

Modules implement different strategies for invoking language models. Popular modules include:

- `dspy.Predict`: Basic prediction
- `dspy.ChainOfThought`: Adds reasoning steps before output
- `dspy.ReAct`: Reasoning and acting with tool use

In the example below, two separate sub-modules are used to implement a retrieval-augmented generation (RAG) pipeline.
Usage of separate modules allows compiling (in DSPy terminology: gradually, evolutionarily building) specialized
prompts for each one:

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

    def forward(self, question: str) -> tuple[dspy.Prediction, list[str]]:
        """
        This is the main entry point for the module that defines the pipeline and can be an arbitrary Python code.
        """
        titles = self.retrieval(question=question).titles
        articles = [get_article(title) for title in titles]
        answer = self.answer(question=question, articles=articles)
        return answer, articles
```

### Optimizers

Optimizers automatically improve module performance by:
- Generating demonstrations (few-shot examples)
- Refining instructions
- Selecting effective prompting strategies
- Bootstrapping from successful executions

One of DSPy's most capable prompt optimizers is GEPA (Generalized Evolutionary Prompt Adapter), which uses reflection and
evolutionary strategies to improve prompts.

## Optimization Process

The optimization workflow follows these steps:

1. **Define a metric** that measures module performance
2. **Prepare datasets** (training and validation)
3. **Run the optimizer** to find better prompts/weights
4. **Evaluate** the optimized module

### Custom Metrics

Metrics can return simple scores or provide detailed feedback for predictor-level optimization:

```python
def top5_recall(
    example, pred, trace=None, pred_name=None, pred_trace=None, *args, **kwargs
) -> float | dspy.Prediction:
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
        return dspy.Prediction(score=recall, feedback=feedback_text)

    return recall
```

For more complex scenarios developers can write heavily customized metrics:

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

    def __call__(self, gold: dspy.Example, pred: tuple[dspy.Prediction, list[str]],
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
            return dspy.Prediction(score=score, feedback=feedback_text)

        return score
```

### Running Optimization

Once the metric is defined, optimization is straightforward:

```python
# Configure language model
gpt = dspy.LM("openai/gpt-5.6-luna", max_tokens=8000)

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
        reflection_lm=dspy.LM("openai/gpt-5.6-sol", max_tokens=8000)
    )
    optimized_react = gepa_optimizer.compile(
        react,  # the ReAct module from the beginning of this blogpost
        trainset=trainset,
        valset=devset
    )

# Evaluate the optimized module
evaluate(optimized_react)

# Save the optimized module
optimized_react.save("optimized_react.json")
```

The optimizer explores the space of possible instructions and demonstrations (examples from the training set),
using the validation set to select the best configuration. This allows optimization of complex cases, such as
multi-level modules. The product of optimization is compiled module state containing optimized instructions and,
depending on the optimizer, demonstrations. It can be saved as JSON and loaded into a compatible DSPy module instance.

## Composing Pipelines

DSPy modules compose naturally. A complete RAG (Retrieval-Augmented Generation) pipeline combines retrieval and generation.
The entire pipeline can be optimized end-to-end. This contrasts with "flat" module optimization, shown in one
of the earlier examples.

```python
agent = RAGAgent("optimized_react.json")

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

## Advantages over manual prompting

DSPy offers several benefits compared to traditional prompt engineering:

**Maintainability**: Changes to the pipeline don't require re-crafting prompts. Modify the signature or module structure, re-optimize and deploy.

**Portability**: The same code works across different language models. Optimization adapts prompts to each model's characteristics.

**Systematic improvement**: Optimization is data-driven and reproducible, not dependent on manual trial-and-error.

**Separation of concerns**: System logic (what to do) is separated from prompting strategy (how to ask the model).

**Cost optimization**: DSPy, through modularity and automatic optimization, allows systematic testing of different models and
prompt variants. This makes it easier to find the optimal balance between quality and cost.

## Performance characteristics

Based on the original DSPy paper and practical experiments:

- Compilation time: from several minutes to several hours, depending on dataset size, optimizer settings and rate limits of used LLM provider
(impacts how much concurrency to use)
- Performance gains: significant improvement over few-shot prompting baselines (for our RAGAgent score went from 68% to 77%!)

The optimization cost is paid once; the resulting optimized modules can be saved, versioned and deployed like any other code artifact.

## Examples of prompts

Unoptimized:
```
Based on user's prompt create a semantic search query to find all document titles needed to answer the
question.\n\nYou are an Agent. In each episode, you will be given the fields `question` as input. And you can
see your past trajectory so far.\nYour goal is to use one or more of the supplied tools to collect any necessary
information for producing `titles`.\n\nTo do this, you will interleave next_thought, next_tool_name,
and next_tool_args in each turn and also when finishing the task.\nAfter each tool call, you receive a
resulting observation, which gets appended to your trajectory.\n\nWhen writing next_thought, you may reason 
about the current situation and plan for future steps.\nWhen selecting the next_tool_name and its
next_tool_args, the tool must be one of:\n\n(1) semantic_search, whose description is <desc>Returns top-5
results and then the titles of the top-5 to top-30 results.</desc>. It takes arguments {'query': {'type':
'string'}}.\n(2) finish, whose description is <desc>Marks the task as complete. That is, signals that all
information for producing the outputs, i.e. `titles`, are now available to be extracted.</desc>. It takes
arguments {}.\nWhen providing `next_tool_args`, the value inside the field must be in JSON format
```

Optimized:
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
names).\n   - Rephrase or expand abbreviations and include both key terms and synonyms where relevant to
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
`next_tool_args`).\n\nProceed step by step, reflecting and reasoning about your choices and always ensure the
query captures all important aspects of the user's question.
```

## Summary

DSPy represents a step forward from manual prompt crafting to programmatic prompt engineering. By defining task 
signatures, composing modules and using automatic optimization, developers can build more maintainable and effective 
language model pipelines. The framework's declarative approach separates the specification of desired behavior from the 
implementation details of prompting, making AI systems more robust to changes in underlying models and requirements.

While traditional prompting remains useful for quick prototypes and simple tasks, DSPy provides a structured path to 
production-quality language model applications. As models continue to evolve, having a framework that can automatically 
adapt prompts and strategies becomes increasingly valuable.

## Sources

[^1]: [Stanford NLP DSPy Repository](https://github.com/stanfordnlp/dspy)
[^2]: [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714)
[^3]: [DSPy Official Documentation](https://dspy.ai/)
