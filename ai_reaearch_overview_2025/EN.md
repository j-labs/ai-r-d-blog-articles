# AI Advancements Summary

## Sequence model architectures & memory mechanisms

### Section executive summary

This group represents fundamental advances in how neural networks handle sequential information and maintain memory over long contexts. The research converges on a critical insight: effective sequence modeling requires sophisticated memory mechanisms that go beyond traditional attention. These papers introduce novel architectures that unify existing approaches through theoretical frameworks (Miras), implement test-time learning for adaptive memory (Titans), and unlock previously impossible capabilities through algorithmic modifications (negative eigenvalues in LRNNs). Together, they address the core challenge of balancing computational efficiency with the ability to track state and maintain context across extremely long sequences, with implications spanning language modeling, time series, genomics and code generation.

### Papers

#### It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention and Online Optimization

This paper presents Miras, a framework that abstracts modern sequence models (Transformers, Titans, linear RNNs, etc.) as memory modules with attentional bias that optimize internal objectives. The framework reveals that most existing architectures use either dot-product similarity or L2 regression, and is built on four design choices: associative memory architecture, attentional bias objective, retention gates (reinterpreting forget gates as retention regularization) and memory learning algorithms. Building on these insights, the authors propose three novel sequence models: Moneta, Yaad and Memora, that use alternative attentional biases (Lp-norm, Huber loss) and retention mechanisms (in essence a forget gate; Lp-norm, KL-divergence), demonstrating promising* performance on language modeling, common-sense reasoning and recall-intensive tasks with improved scaling patterns.

*Note: There are some significant performance improvements over the baselines, but presented in a narrow scaling regime.

##### Key contributions
1. Theoretical unification of sequence modeling architectures through the lens of associative memory and online optimization
2. Four-component design framework (memory architecture, attentional bias, retention mechanism, learning algorithm) enabling systematic architecture exploration
3. Three novel architectures (Moneta, Yaad, Memora**) with alternative objectives that outperform existing baselines

**Note: Memora performs the worst in explored parameter ranges, but has the strongest stability guarantees, so potentially might be easier to train at scale.

#### Titans: Learning to Memorize at Test Time

Titans introduces a family of architectures featuring a novel neural long-term memory module that learns to memorize at test time using gradient-based updates with momentum and weight decay. The architecture treats surprising events as more memorable and proposes three variants: Memory as Context (MAC)*, Memory as Gate (MAG), and Memory as Layer (MAL). These combine neural memory for long-term context with attention for short-term dependencies and persistent memory for task knowledge, addressing the quadratic complexity of Transformers and poor compression of linear recurrent models.

*Note: MAC variant achieves the best overall performance compared to other variants, but is computationally havier.

##### Key contributions
1. Novel test-time learning memory module with momentum-based updates and adaptive forgetting mechanisms
2. Three architectural variants (MAC, MAG, MAL) that integrate long-term memory with attention via different mechanisms
3. Effective scaling to 2M+ context windows with competitive results against competing architectures

#### Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues

This paper identifies and resolves a fundamental limitation in modern Linear Recurrent Neural Networks (LRNNs) like Mamba and DeltaNet: their inability to perform state-tracking tasks due to eigenvalue restrictions. The authors prove that LRNNs with state-transition matrix* eigenvalues restricted to [0, 1] cannot solve tasks like parity and modular counting in finite precision, and that non-triangular matrices are needed for general modular counting. Critically, they demonstrate that expanding the eigenvalue range to [−1, 1] dramatically enhances expressive power, enabling LRNNs to solve all regular languages** through products of generalized Householder matrices***.

*Note: State-transition matrix is a matrix that describes how the hidden state evolves over time.
**Note: Regular languages are a class of formal languages that can be recognized by finite state automata.
***Note: this generalization allows rotation in addition to vector reflection.

##### Key contributions
1. Theoretical proof of fundamental expressivity limitations in positive-eigenvalue-only state-transition matrices LRNNs for state-tracking tasks
2. Mathematical demonstration that negative eigenvalues enable solving all regular languages
3. Practical architectural modification allowing eigenvalue range [−1, 1] in state-transition matrices without stability or efficiency loss

---

## Efficient Attention & LLM Inference

### Section executive summary

As language models scale to billions of parameters and multi-million token contexts, computational efficiency has become a critical bottleneck. This group addresses the efficiency crisis through complementary innovations: sparse attention mechanisms that dramatically reduce computation for long contexts (DeepSeek-V3.2-Exp), architectural augmentations that improve expressiveness and eliminate pathological behaviors like attention sinks (Gated Attention), alternative generation paradigms that enable parallelism (REFUSION) and training optimizations that reduce fine-tuning costs (SparseLoRA). These advances collectively enable more accessible, faster and cost-effective deployment of large language models without sacrificing – and in some cases improving—model quality. The practical impact spans inference acceleration, training efficiency and improved long-context handling.

### Papers

#### DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention

DeepSeek-V3.2-Exp introduces DeepSeek Sparse Attention (DSA), which uses a lightning indexer and fine-grained token selection* to implement efficient sparse attention. The model is created through continued training of DeepSeek-V3.1-Terminus, following a pipeline of continual pre-training (dense warm-up**, sparse adaptation) and two stage post-training. Proposed attention architecture significantly reduces computation costs (each query token attends to a small, fixed subset of keys), especially for very long contexts, while maintaining comparable performance across general, code, math and agentic search tasks.

*Note: indexer essentially computes a weighted dot product score between fp8 projections of queries and keys, while selector retrieves top-k from it.
**Note: warm-up is used to initialize indexer weights.

##### Key contributions
1. Novel DSA mechanism with lightning indexer for efficient sparse attention computation
2. Comprehensive training pipeline that allows transitioning previously trained model to sparsity
3. Huge reduction in inference costs for long-context scenarios without accuracy degradation

#### Gated Attention for Large Language Models: Non-linearity, Sparsity and Attention-Sink-Free

This work systematically investigates integrating gating mechanisms into attention layers of large language models through experiments across many model variants (including 15B MoE and 1.7B dense models) trained on up to 3.5 trillion tokens. The study introduces a simple head-specific sigmoid gate after Scaled Dot-Product Attention, revealing two key improvement mechanisms: increased expressiveness via non-linearity between attention and output layers, and input-dependent sparsity* that eliminates the "attention sink" effect.

*Note: authors claiming "input-dependent sparsity" really mean "input-dependent near-zero approximation of sparsity" because of sigmoid's asymptotic behavior. From an alternative angle not mentioned explicitly in the paper, this operation can be viewed as an adaptive forget gate.

##### Key contributions
1. Comprehensive empirical analysis of gating mechanisms in attention across two parameters scales, multiple compute levels and architectures
2. Identification of two complementary improvement mechanisms: non-linearity and input-dependent sparsity
3. Elimination of attention sink, enabling better long-context generalization**

**Note: thanks to savings in precision of attention scores previously wasted on massive sink score.

#### REFUSION: A Diffusion Large Language Model with Parallel Autoregressive Decoding

REFUSION introduces a novel LLM framework combining masked diffusion model (MDM) parallelism with slot-level autoregressive infilling*. The architecture partitions sequences into fixed-length slots and employs a two-stage "plan-and-infill" decoding: diffusion-based global planning identifies weakly dependent slots for parallel processing**, then autoregressive infilling generates tokens within each slot sequentially. This enables full reuse of key-value caches while avoiding token-level incoherence, trained with a hybrid objective optimizing both global planning and local infilling.

*Note: in simpler words: it is a hybrid approach that marries autoregressive LLMs with diffusion models.
**Note: empirically grounded heuristic here is that weakly dependent slots have low confidence score (globally), thus can potentially "ignore" each other during decoding.

##### Key contributions
1. Novel hybrid architecture bridging parallel diffusion generation with autoregressive coherence
2. Two-stage plan-and-infill decoding algorithm enabling efficient parallelization without quality loss
3. Full KV cache reusability increasing efficiency by 2.33× over a purely autoregressive baseline

#### SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity

SparseLoRA accelerates LLM fine-tuning by leveraging contextual sparsity, dynamically selecting sparse weight subsets for gradient and loss computations using a training-free SVD-based sparsity estimator. The method applies sparsity selectively across layers, tokens and training steps* with minimal overhead. Unlike prior PEFT approaches that primarily reduce memory, SparseLoRA directly improves compute efficiency.

*Note: Initial layers benefit from being dense, while deeper layers are more redundant, thus unlocking aggressive sparsification. Context tokens can use sparse computation, but decoded tokens shall use dense computation. Training benefits from progressively increasing token and layer sparsity after initial fully dense steps.

##### Key contributions
1. Training-free contextual sparsity estimation enabling dynamic sparse weight selection during fine-tuning
2. Layer-wise, token-wise and temporal sparsity strategies optimized for different computational patterns
3. Up to 2.2× reduction in computational cost and 1.6× speedup over existing PEFT methods while maintaining accuracy
4. Compatibility and synergy with quantization-based methods (e.g., QLoRA) for combined memory and compute efficiency

---

## Vision & Audio Foundation Models

### Section executive summary

This group demonstrates the maturation of foundation models beyond text, establishing new paradigms for visual and audio understanding. These papers share a common theme: rethinking where and how to extract or construct representations for maximum effectiveness across diverse downstream tasks. SAM Audio achieves unprecedented unification across audio domains through multimodal prompting. Perception Encoder challenges the assumption that output layers contain the best features and "One Layer Is Enough" shows that minimal adaptation layers can bridge powerful pretrained encoders to generative tasks. Together, they represent a shift toward more efficient, versatile and theoretically grounded approaches to multimodal AI, with practical implications for production systems spanning audio editing, computer vision, detection, video understanding and generative applications.

### Papers

#### SAM Audio: Segment Anything in Audio

SAM Audio is a foundation model for general audio separation that unifies text, visual, and temporal span prompting within a single diffusion transformer architecture. Built on flow matching* and trained on large-scale audio data spanning speech, music and general sounds, it achieves state-of-the-art performance across diverse benchmarks. The paper introduces span prompting as a novel temporal conditioning mechanism and releases SAM Audio-Bench (comprehensive multimodal separation benchmark with human-labeled prompts) and SAM Audio Judge (reference-free evaluation model strongly correlated with human judgment).

*Note: Flow matching uses the same architecture as diffussion, but with a different objective – instead of optimizing stepwise denoising it optimizes the shortest trajectory from noise to data.

##### Key contributions
1. First foundation model achieving SOTA across multiple audio domains (speech, music, general sounds) with unified multimodal prompting
2. Novel span prompting mechanism for temporal conditioning in audio separation tasks

#### Perception Encoder: The best visual embeddings are not at the output of the network

Perception Encoder introduces a state-of-the-art vision encoder family discovering that strong general features for diverse downstream tasks exist in intermediate layers of contrastively-trained models, not the output layer. The work develops PEcore (robust image pretraining), PElang (language-aligned variant), and PEspatial (spatially-aligned variant ).

##### Key contributions
1. Paradigm-shifting insight that optimal features for diverse tasks reside in intermediate layers, not model outputs
2. Comprehensive pretraining recipes (PEcore) outperforming models trained on proprietary datasets (JFT-3B/WebLI)
3. Task-specific alignment strategies (language and spatial) achieving SOTA on detection, VQA and video understanding
4. Release of 2B parameter model, code, and PE Video Dataset (1M videos, 120K human-refined annotations)

#### One Layer Is Enough: Adapting Pretrained Visual Encoders for Image Generation

This paper introduces FAE (Feature Auto-Encoder), a minimalist framework adapting pretrained self-supervised visual representations (DINOv2, SigLIP)* into low-dimensional latents for generative models. The key innovation is using a single self-attention layer to compress high-dimensional embeddings, followed by a double-decoder architecture separating feature reconstruction from image synthesis. This approach overcomes incompatibility between understanding-oriented feature spaces and generation-friendly latents without complex objectives or substantial architectural changes.

*Note: Those are two different model families, DINOv2 being a self-supervised vision encoder, while SigLIP is from the CLIP (contrastive learning) family.

##### Key contributions
1. Minimal single-layer compression architecture bridging understanding and generation with preserved semantic quality
2. Double-decoder design enabling effective separation of feature reconstruction and image synthesis objectives
3. State-of-the-art or near-SOTA performance with significantly faster convergence than previous models
4. Universal compatibility with various backbone encoders and generative model families

---

## Reasoning & Agentic Systems

### Section executive summary

This group represents a fundamental rethinking of how AI systems approach complex reasoning and problem-solving. Rather than relying solely on scale, these works demonstrate that architectural choices, orchestration strategies and training objectives can unlock dramatically improved reasoning capabilities. ToolOrchestra shows that small models can coordinate larger ones and tools more effectively than monolithic giants, achieving superior results at a fraction of the cost. "Less is More" proves that tiny recursive networks can outperform large language models on challenging puzzles through deep iterative reasoning. LeJEPA provides theoretical grounding for self-supervised learning that removes brittle heuristics while improving robustness. These advances collectively suggest a future where modeling capability comes not just from model size, but from principled architectural design, efficient resource orchestration and mathematically sound training procedures.

### Papers

#### ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration

ToolOrchestra introduces a methodology for training small language models to serve as orchestration agents managing both traditional tools (web search, code interpreters) and diverse domain-specialized and general-purpose LLMs as external tools. The Orchestrator-8B model is trained end-to-end via reinforcement learning, guided by outcome correctness, efficiency (cost and latency) and user preference alignment. The approach includes ToolScale, a large synthetic benchmark for multi-turn tool-use agent tasks, achieving superior performance and cost efficiency over monolithic LLMs including GPT-5.

##### Key contributions
1. Paradigm shift from single-model to orchestrated multi-tool/multi-model systems for reasoning
2. End-to-end RL training with multi-objective rewards (correctness, efficiency, user alignment)
3. Dramatic efficiency gains: 8B orchestrator outperforms substantially larger models at fraction of compute cost
4. Strong generalization to unseen tools, tasks and user preferences tested with ToolScale benchmark

#### Less is More: Recursive Reasoning with Tiny Networks

This paper introduces Tiny Recursive Model (TRM), a simplified recursive reasoning architecture using a single tiny (2-layer, 7M parameter) neural network that outperforms both Hierarchical Reasoning Model (HRM) and large language models on challenging tasks like Sudoku, Maze, and ARC-AGI with minimal training data (~1,000 examples). Unlike HRM's complex dual-network hierarchy, TRM uses an elegant single-network design alternating between latent state updates and solution refinement with deep supervision and simple halting criteria.

##### Key contributions
1. State-of-the-art results on extreme reasoning benchmarks (87% Sudoku, 45% ARC-AGI-1) with 7M parameters
2. Dramatic simplification of recursive reasoning: single network vs. dual-network hierarchy
3. Evidence that deep recursion with small networks avoids overfitting and maximizes generalization in small-sample regimes
4. Challenge to scale-centric paradigm: architectural design over parameter count for structured reasoning

#### LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics

LeJEPA introduces a theoretically principled framework for self-supervised learning within the Joint-Embedding Predictive Architecture paradigm. The core insight is that isotropic Gaussian embeddings* uniquely minimize downstream prediction risk across broad task families. LeJEPA combines predictive loss with Sketched Isotropic Gaussian Regularization (SIGReg), which provably enforces isotropic Gaussianity using scalable, hyperparameter-light, differentiable projection-based methods, eliminating brittle heuristics like stop-gradient and teacher-student models.

*Note: Isotropic Gaussian embeddings have the same variance across all dimensions, ensuring optimal information capacity exploitation of every dimension.

##### Key contributions
1. Mathematical proof that isotropic Gaussian embeddings minimize downstream risk across task families
2. SIGReg: scalable, differentiable regularizer enforcing isotropic Gaussianity with single hyperparameter and linear complexity
3. Validation across 60+ architectures and 10 datasets showing SOTA or superior performance with exceptional stability
4. Demonstration that in-domain SSL can outperform transfer from massive foundation models, challenging prevailing assumptions

---

## Retrieval & Ranking Systems

### Section executive summary

This group addresses the practical challenges of deploying AI systems at an industrial scale, where efficiency, relevance and robustness are essential. These papers bridge the gap between research advances and production systems, demonstrating how techniques from language models (context engineering, reasoning) can transform discriminative tasks like search and recommendation. OnePiece brings LLM-style reasoning to e-commerce ranking with measurable business impact, while Late Chunking solves a fundamental problem in retrieval systems by preserving document-level context. The inclusion of Imperceptible Jailbreaking serves as a critical reminder that as these systems become more capable and widely deployed, understanding their vulnerabilities becomes essential for safe production deployment. Together, these works represent the maturation of AI from research prototypes to robust, scalable systems handling billions of users.

### Papers

#### OnePiece: Bringing Context Engineering and Reasoning to Industrial Cascade Ranking System

OnePiece introduces a unified framework enhancing industrial ranking systems by integrating LLM-style context engineering and reasoning. The system enriches input representations via structured context engineering (user history, preference anchors from expert knowledge, situational descriptors, candidate item sets), implements block-wise latent reasoning for multi-step bandwidth-scalable reasoning* and adopts progressive multi-task training** using natural feedback signals (click, add-to-cart, purchase) as supervision for reasoning stages.

*Note: Wider information channel between reasoning steps by using multiple tokens instead of 1 as previously proposed.
**Note: Progressiveness prevents competing gradients from multiple feedback signals.


##### Key contributions
1. Systematic adaptation of LLM paradigm mechanisms (context engineering, multi-step reasoning) to discriminative industrial ranking
2. Block-wise latent reasoning architecture enabling scalable multi-step reasoning over rich input representations
3. Production deployment at Shopee's scale showing higher advertising revenue and user's merchandise value with improved efficiency
4. Superior parameter/data efficiency and hardware utilization compared to highly optimized entrenched baselines (DLRM, HSTU)***

***Note: DLRM is Shopee's production baseline recommendation model, while HSTU is a state-of-the-art recommendation framework from Meta.

#### Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models

Late Chunking introduces a novel approach to the generation of text chunk embeddings that preserves broader contextual information by processing entire documents with long-context embedding models to produce token-level embeddings, then splitting these into chunks. Unlike traditional chunking that divides documents before embedding, this ensures each chunk representation benefits from the full document context. The paper proposes scalable "long late chunking" for huge documents and introduces span pooling fine-tuning for further improvements.

##### Key contributions
1. Paradigm shift from pre-embedding chunking to post-embedding chunking preserving cross-chunk contextual dependencies
2. Model-agnostic approach requires no additional training for core benefits with demonstrated retrieval improvements
3. Scalable solution (long late chunking) for huge documents exceeding model context windows
4. Computationally more efficient than LLM-based contextual augmentation alternatives with immediate practical applicability

#### Imperceptible Jailbreaking against Large Language Models

This paper introduces imperceptible jailbreaks exploiting invisible Unicode variation selectors to append adversarial suffixes to prompts. The attacks create invisible changes that alter tokenizer input while remaining invisible to human readers, effectively bypassing safety alignment of various open-source LLMs. A chain-of-search optimization pipeline efficiently generates successful invisible suffixes across prompts and models with high success rates in harmful output generation and prompt injection.

##### Key contributions
1. Discovery of a new vulnerability class based on invisible Unicode characters bypassing current safety mechanisms
2. Demonstration of attack transferability across multiple LLM architectures (Vicuna, Llama-2, Llama-3, Mistral)
3. Chain-of-search optimization enabling efficient generation of adversarial suffixes
4. Critical revelation of tokenizer-level vulnerabilities requiring revision of input filtering and safety alignment strategies
