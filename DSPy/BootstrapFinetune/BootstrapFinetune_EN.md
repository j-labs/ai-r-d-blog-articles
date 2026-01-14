# BootstrapFinetune and the Teacher–Student Paradigm in Practical LLM Workflows

## Introduction

In [the previous article](https://github.com/j-labs/ai-r-d-blog-articles/blob/master/DSPy/intro/EN.md), we showed how
**DSPy replaces manual prompting with declarative programming of language-model behavior** and automatic pipeline
optimization. Signatures, modules, and optimizers make it possible to discover effective strategies without manually
tweaking prompts.

The next step is the question: **what do we do with this optimized behavior next?**
How do we move it into a production environment, where inference cost, latency, and scalability are key?

The answer is the **Teacher–Student** paradigm and **BootstrapFinetune** in DSPy-a mechanism that uses a large model as
a teacher to automatically generate data and fine-tune smaller, cheaper models.

In this article, I discuss how BootstrapFinetune implements knowledge distillation in practice, what benefits it brings,
and what design trade-offs are worth considering.

---

## The Teacher–Student Paradigm (Knowledge Distillation)

The **Teacher–Student** paradigm originates from classic knowledge distillation, whose goal was to transfer knowledge
from large, expensive models to smaller and cheaper counterparts. In the context of **Large Language Models**, however,
the meaning of this approach shifts significantly. It is no longer only about approximating a function or probability
distribution, but about inheriting the model’s **behavior** - how it responds, how it structures outputs, and how it
handles ambiguity.

In practice, the teacher is usually a large base model operating in zero-shot or few-shot mode, leveraging knowledge
acquired during pretraining. Its answers, output formats, and heuristics become the training material. The student does
not absorb the teacher’s full knowledge; instead, it learns a behavioral pattern - how to respond to specific task types,
with a particular style and structure, rather than what the teacher specifically "knows."

This approach has important practical consequences. First, it eliminates the need for manual labeling - the teacher
generates pseudo-labels, i.e., automatically produced labels and answers (including intermediate steps), which are then
treated as training data for the student model. Second, it enables a significant reduction in inference costs, because
the expensive model’s behavior can be compressed into the weights of a smaller student. Finally, it allows you to build
models specialized for narrow tasks, such as Text-to-SQL or classification, without training them from scratch.

At the same time, the Teacher–Student paradigm is not without limitations. The student can inherit the teacher’s errors
or biases, and behavior distillation often leads to the loss of advanced capabilities present in large models - such as
complex reasoning, flexible instruction-following, or long-context coherence. There is also a risk of overfitting to the
teacher’s response style at the expense of generalization. For this reason, the effectiveness of distillation depends
not only on the student architecture, but also on the quality and consistency of the teacher model’s behavior.

In the next part of the article, this paradigm is demonstrated in practice - as the foundation of **BootstrapFinetune**,
which uses a teacher to automatically build training data and distills the effect directly into the student’s weights.

---

## Types of Optimizers in DSPy

DSPy treats **program optimization** as an algorithmic problem. Different optimizers intervene at different layers of a
program:

* **Few-shot optimization**
  Automatic construction of demonstrations in prompts (e.g., `BootstrapFewShot`, `KNNFewShot`).

* **Instruction optimization**
  Prompt generation and refinement (e.g., `MIPROv2`, `GEPA`).

* **Weight optimization (fine-tuning)**
  Distilling a prompt-based program into model weights - this is exactly what **`BootstrapFinetune`** does.

In practice, you often start with few-shots and prompts, and **BootstrapFinetune** can be an additional step that leads
to a smaller and cheaper production model.

---

## What BootstrapFinetune Is

**BootstrapFinetune** is a DSPy optimizer whose goal is to transform a prompt-based program into a program based on *
*trained model weights**.

The output is not a single fine-tuned model, but a **full DSPy program** that:

* preserves the same structure,
* uses the same signatures,
* but uses **fine-tuned student LMs** instead of prompting.

This means that after BootstrapFinetune, each module has its own student, enabling selective updates of only chosen
components of the pipeline.

BootstrapFinetune is therefore a practical implementation of the Teacher–Student paradigm.

---

## The BootstrapFinetune Workflow

The **BootstrapFinetune** process can be described as a sequence of several logical steps, which together form a bridge
between a prompt-based program and a program that uses trained model weights.

### 1. Setup: student and teacher

The workflow begins with preparing **two programs with an identical structure**: the student and the teacher. Both
programs have the same predictors and signatures; they differ only in the assigned language model. The teacher is
usually a large, strong LLM, while the student is a smaller model that is intended to adopt the teacher’s behavior.

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

### 2. Preparing the training set (trainset)

Before BootstrapFinetune is run, you must prepare an **input dataset** on which the teacher program will be executed. In
DSPy, such a set is a list of `dspy.Example` objects and defines the program’s *contract*: which fields are inputs and
what information is available during prediction.

For example, for the Text-to-SQL task, the training set can be built from Spider as follows:

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

This `trainset` **is not yet the data used for training weights**. It only serves as input to execute the teacher
program in the next step of the workflow.

### 3. Bootstrapping and trace collection

Next, DSPy runs the teacher program on the training set. During this stage, it records not only the final answers but
also **full execution traces** of the program. These traces - covering inputs, outputs, and the context of individual
predictors - become the training-data base for the student.

If a quality metric is available (e.g., comparing the result of executing the Gold SQL versus the generated SQL), it can
be used to filter out failed examples. In practice, a smaller but consistent dataset often yields better results than a
large number of examples with variable quality. The metric is **not used for gradient-based optimization**, but only for
**selection and filtering** of examples that go into the student’s training dataset.

```python
optimizer = dspy.BootstrapFinetune(
    metric=metric,
    num_threads=4,
    train_kwargs=TRAIN_KWARGS,
)
```

### 4. Preparing data for fine-tuning

This stage is about processing the **teacher program’s execution trace** into data accepted by the fine-tuning backend.

Only during bootstrapping does DSPy execute the **teacher program** on those examples and record **execution traces**.
Such a trace includes, among other things:

* the actual prompts sent to the model,
* the answers generated by the teacher,
* predictor context and optional reasoning.

At this stage, BootstrapFinetune converts these traces into a **training format** understood by the fine-tuning backend.
This can be, for example, a **chat format** (a list of `user` / `assistant` messages):

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

### 5. Fine-tuning and program update

In the final step, **BootstrapFinetune** calls `finetune()` on the model assigned to the student, delegating the actual
weight training to the selected provider. Once training finishes, the new model is automatically attached to the student
program, and - optionally - prompt demonstrations can be removed.

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

The result of the entire process is a **ready-to-serve production program** that preserves the original DSPy program
structure but no longer requires running the teacher during inference.

---

## How It Works Internally: Fine-Tuning Abstraction in DSPy

One important design element of **BootstrapFinetune** is that the optimizer **does not implement model training directly
**. Instead, it relies on the `finetune()` method available on the `LM` object. This separates distillation logic (trace
collection, filtering, data preparation) from the **concrete training backend**.

From the DSPy code perspective:

* `BootstrapFinetune`:

    * builds the training data,
    * groups it per language model,
    * calls `lm.finetune(**kwargs)`.
* training details are delegated to the **provider** assigned to the given `LM`.

This separation allows the same optimizer to be used for both local models and remote APIs.

### The `LM.finetune()` Interface

For `BootstrapFinetune`, the only requirement is that the `LM` object exposes a method:

```python
finetuned_lm = lm.finetune(train_data=..., **train_kwargs)
```

The optimizer does not assume:

* which framework is used for training (PyTorch, TRL, API),
* whether full weights or adapters (PEFT) are trained,
* whether training is local or remote.

This makes **BootstrapFinetune backend-agnostic**.

### LocalProvider

`LocalProvider` provides the simplest possible implementation of `LM.finetune()`, based directly on **transformers** and
**TRL**. In practice, this means running local SFT training on GPU without an additional orchestration or optimization
layer.

Conceptually, the pipeline is straightforward: the data generated by the teacher is saved locally in chat format, a base
model and tokenizer are loaded, and then **Supervised Fine-Tuning** is run - optionally with **PEFT/LoRA** techniques. 
The result is a new model checkpoint that DSPy treats as another `LM`.

While this approach works technically, its limitations show up quickly in practice. Training directly on raw
`transformers` and `trl` requires manual hyperparameter tuning for a specific model and GPU, which easily leads to
memory issues (OOM), low performance, difficulty fine-tuning quantized models, or unstable quality. At the same time,
you lose part of the abstraction that DSPy provides - rather than working at the level of program behavior and
distillation, you start debugging details of weight training.

Another issue is limited operational control: no convenient resume/retry, interruptions, or a clear answer to the
question of *what exactly* is being trained and with what configuration. For this reason, `LocalProvider` is best
treated as an implementation curiosity or a demonstration tool that helps you understand the end-to-end
BootstrapFinetune flow, rather than as a solution for serious open-weight fine-tuning.

For completeness, minimal usage of `LocalProvider` looks like this:

```python
import dspy
from dspy.clients.lm_local import LocalProvider

# Enable experimental fine-tuning
dspy.settings.experimental = True

# Define a local student model
student_lm = dspy.LM(
    model="openai/local:Qwen/Qwen3-0.6B",
    provider=LocalProvider(),
    max_tokens=3000,
)

# Attach the LM to the program
student_program = program.deepcopy()
student_program.set_lm(student_lm)

# BootstrapFinetune will call student_lm.finetune(...) under the hood
optimizer = dspy.BootstrapFinetune(train_kwargs={"use_peft": True})
compiled = optimizer.compile(
    student=student_program,
    teacher=teacher__program,
    trainset=trainset,
)

```

In practical open-source fine-tuning scenarios, specialized tools such as **Unsloth**, **Axolotl**, or **MLX** tend to
work much better, offering improved control over memory, performance, and training regimes. In this view,
`LocalProvider` remains a convenient integration point in DSPy, but not necessarily the target training backend.

### OpenAIProvider

With `OpenAIProvider`, the same `LM.finetune()` interface maps to a remote fine-tuning process managed by an API.
Training data is sent to an external service, training happens asynchronously, and the result is a new model identifier
that can be used in DSPy like any other `LM`.

From BootstrapFinetune’s perspective, implementation differences between providers are largely hidden. The optimizer
simply receives a new model to attach to the program - regardless of whether training happened locally or in an API
provider’s infrastructure.

For completeness, minimal usage of fine-tuning with `OpenAIProvider` looks like this:

```python
import dspy

# Student model – will be fine-tuned via the API
student_lm = dspy.LM(
    model="openai/gpt-5-nano",
    max_tokens=2000,
)

# Teacher model
teacher_lm = dspy.LM(
    model="openai/gpt-5.1-codex-mini",
    max_tokens=16000,
)

student_program = program.deepcopy()
student_program.set_lm(student_lm)

teacher_program = program.deepcopy()
teacher_program.set_lm(teacher_lm)

# BootstrapFinetune will launch a remote fine-tuning job
optimizer = dspy.BootstrapFinetune(
    metric=metric,  # optional, if we have labels
)

compiled = optimizer.compile(
    student=student_program,
    teacher=teacher_program,
    trainset=trainset,
)
```

In this variant, all training complexity (batching, schedules, stability) is handled by the API provider, and DSPy
retains its role as an orchestration layer.

---

## Experiment: Text-to-SQL on the Spider Dataset

To validate the workflow in practice, I ran an experiment on **Text-to-SQL** using the popular **Spider** dataset. The
goal was to evaluate to what extent distilling teacher behavior into the weights of a smaller model improves SQL
generation quality, and what trade-offs appear under different training configurations.

### Experiment Setup

* **Task**: Text-to-SQL (Spider)
* **Metric**: percentage of SQL queries generated by the model that, when executed, produced the **same result** as the
  Gold SQL (execution-based accuracy)
* **Teacher**: GPT-5.1-codex-mini
* **Students**: GPT-4.1-nano and Qwen3-0.6B
* **Method**: BootstrapFinetune (full fine-tuning or LoRA)

This experiment is exploratory; its purpose is to illustrate the impact of behavioral distillation on model quality
rather than to provide a full benchmark-style comparative evaluation.

### Results Without Fine-Tuning

| Student Model           | Module/Program | Result |
|-------------------------|----------------|--------|
| GPT-5.1-codex-mini      | ChainOfThought | 77%    |
| GPT-5.1-codex-mini      | Predict        | 78%    |
| gpt-4.1-nano-2025-04-14 | ChainOfThought | 76%    |
| gpt-4.1-nano-2025-04-14 | Predict        | 72%    |
| Qwen3-0.6B              | ChainOfThought | 27%    |
| Qwen3-0.6B              | Predict        | 23%    |

### Fine-Tuned Results

| Student Model                               | Module/Program | Fine-tuning method | Result |
|---------------------------------------------|----------------|--------------------|--------|
| ft:gpt-4.1-nano-2025-04-14:j-labs::Cw77bexu | ChainOfThought | LoRA               | 70%    |
| ft:gpt-4.1-nano-2025-04-14:j-labs::Cw7RBixD | Predict        | LoRA               | 68%    |
| Qwen3-0.6B                                  | ChainOfThought | Full Training      | 44%    |
| Qwen3-0.6B                                  | Predict        | Full Training      | 42%    |
| Qwen3-0.6B                                  | Predict        | LoRA               | 27%    |

### Interpretation and Takeaways

The results confirm that **BootstrapFinetune can effectively compress the behavior of a strong teacher into the weights
of smaller models**, but the magnitude of improvement depends on the relationship between the base model and the task.

The most clear-cut effect is for **Qwen3-0.6B**. The base model achieved 23–27% execution accuracy, while after full
fine-tuning using teacher outputs the score increased to **42–44%**. This shows that even a very small model can pick up
some heuristics and solution structure from a larger LLM, making distillation a useful tool for reducing inference cost.

For **GPT-nano**, which already performed close to the teacher without fine-tuning (72–76%), applying BootstrapFinetune
with LoRA did not improve results and even caused a **quality drop** (68–70%). This suggests that behavior distillation
has limited value for already strong models and can degrade performance if the student is already well matched to the
task.

The comparison between `ChainOfThought` and `Predict` modules did not show meaningful quality differences. In
Text-to-SQL, the correctness of the final query is what matters, not an explicit representation of intermediate
steps-making reasoning distillation an optional design decision.

More significant differences appear between fine-tuning methods. For Qwen3-0.6B, **full weight training** was clearly
more effective than LoRA, suggesting that for very small models and structural tasks, adapter expressiveness may be
insufficient.

In summary, BootstrapFinetune works best as a tool for **intentional behavior compression**: especially when the student
is clearly weaker than the teacher and the goal is to reduce inference cost with a controlled quality loss. It is not a
universal method for improving models, and it requires fitting the approach to the task characteristics and student
architecture.

---

## Summary

The **Teacher–Student** paradigm combined with **BootstrapFinetune** provides a coherent path from a prompt-based
prototype to a system that uses trained model weights.

BootstrapFinetune is not a tool for weight training itself. Its role is **orchestrating the distillation process** at
the DSPy program level: defining *which behavior* should be compressed and *in what structure* the model should operate
after fine-tuning. This approach works best when the student is clearly weaker than the teacher and the goal is to
reduce inference cost with a controlled loss of quality.

At the same time, this article shows that the effectiveness of such a workflow is strongly **context-dependent**: on the
choice of teacher, the nature of the task, and the technology used for weight fine-tuning. In particular, behavior
distillation is not a universal way to improve models and may yield limited or negative effects when models are already
well matched to the task.

In this view, BootstrapFinetune does not "solve fine-tuning" - it organizes it, drawing a clear boundary between
designing model behavior and engineering its weights.

---

## Sources

* [Teacher-Student Architecture for Knowledge Distillation: A Survey - arXiv:2308.04268v1](https://arxiv.org/pdf/2308.04268)
* [DSPy Tutorial - classification finetuning](https://dspy.ai/tutorials/classification_finetuning/)
* [BootstrapFinetune documentation](https://dspy.ai/api/optimizers/BootstrapFinetune/)
