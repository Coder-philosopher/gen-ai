# ML / DL / Transformers — Detailed Interview Notes

> Written assuming first-time reading. Each topic: **what it is → why it matters → example → how it connects to the next piece.** Ends with a case study that walks through picking an architecture for a real problem, plus a cram checklist.

---

# PART 1: Classical ML (The "Must-Know" Basics)

## 1. Regression vs. Classification

These are the two fundamental types of **supervised learning** problems — meaning the model learns from labeled examples (input → correct output pairs).

**Regression:** The target you're predicting is a **continuous number** — it can take any value on a range. 
**Example:** Predicting house price ($245,000, $310,500, $1.2M — infinite possible values).
**Metrics:**
- **MSE (Mean Squared Error):** Average of the squared differences between predicted and actual values. Squaring penalizes larger errors much more heavily than small ones.
- **RMSE (Root Mean Squared Error):** Just the square root of MSE — brings the error back into the *same units* as your original target (e.g., dollars instead of dollars-squared), making it more interpretable.
- **MAE (Mean Absolute Error):** Average of the *absolute* differences — treats all errors linearly, so it's less sensitive to big outliers than MSE/RMSE.

**Classification:** The target is a **discrete category** from a fixed set of possible labels. 
**Example:** Spam / Not Spam, or Cat / Dog / Bird.
**Metrics:**
- **Accuracy:** % of predictions that were correct overall.
- **Precision:** Of everything the model *predicted* as positive, how many were actually positive? (Answers: "when it says yes, can I trust it?")
- **Recall:** Of everything that was *actually* positive, how many did the model catch? (Answers: "did it miss any?")
- **F1-Score:** The harmonic mean of Precision and Recall — a single balanced number when you care about both.
- **ROC-AUC:** Measures how well the model separates the two classes across all possible decision thresholds, not just one fixed cutoff.

**Why Accuracy can be misleading (very common interview follow-up):** Imagine a fraud detection dataset where only 1% of transactions are actually fraud. A lazy model that predicts "not fraud" for *every single transaction* would score **99% accuracy** — while being completely useless, since it catches zero actual fraud. This is why, for **imbalanced datasets**, you look at Precision/Recall/F1 instead — they specifically expose how well the model handles the minority class.

**Interview soundbite (ready to use):** *"I choose regression for continuous targets and classification for discrete ones. In production, I prefer F1-Score over Accuracy for imbalanced datasets because Accuracy can be misleading."*

---

## 2. Bias-Variance Tradeoff (Overfitting vs. Underfitting)

This is one of the most fundamental concepts in ML — almost guaranteed to come up in some form.

**Underfitting (High Bias):** The model is **too simple** to capture the real patterns in the data — think of trying to fit a straight line through data that's actually curved. It performs poorly on *both* the training data and new/unseen data, because it never learned the underlying relationship in the first place.
**Fix:** Use a more complex model, add more/better features, train longer.

**Overfitting (High Variance):** The model is **too complex** relative to the amount/complexity of the data — it starts memorizing the specific training examples (including their noise and quirks) rather than learning the general pattern. It performs *very well* on training data but *poorly* on new data, because it learned "this specific dataset" rather than "the underlying rule."
**Fix:** Get more training data, apply **Regularization** (L1/L2 — covered below), use **Dropout** (covered in Part 2), simplify the model.

**How to detect which one you have:** Plot training loss vs. validation loss over training epochs.
- Both losses high and close together → **underfitting**.
- Training loss low, but validation loss much higher → **overfitting** (the gap between them is the signal).
- Both losses low and close together → good fit, this is what you want.

**What L1 vs L2 regularization actually do (common follow-up):** Both work by adding a penalty term to the loss function based on the size of the model's weights, discouraging the model from relying too heavily on any single feature.
- **L1 (Lasso):** Can shrink some weights all the way to **exactly zero** — effectively performs automatic feature selection by "turning off" less useful features entirely.
- **L2 (Ridge):** Shrinks weights **smaller but rarely to exactly zero** — spreads the influence more evenly across features rather than eliminating any completely.

**Interview soundbite (ready to use):** *"In my projects, I always monitor the gap between training and validation loss. If the gap is large, I know I'm overfitting and will introduce Dropout or L2 regularization."*

---

## 3. Data Preprocessing

Real-world data is almost never ready to feed directly into a model — this stage prepares it.

### Normalization / Standardization
Scaling all your features so they're on a **comparable range**, instead of one feature ranging 0–1 and another ranging 0–1,000,000.

**Why this is crucial specifically for Neural Networks and SVMs:** These models are sensitive to the *scale* of inputs — a feature with naturally huge numeric values (like "annual salary") could dominate the learning process purely because of its scale, not because it's actually more important than a feature like "age." Scaling puts every feature on equal footing so the model learns based on actual signal, not arbitrary units.

**Two common approaches:**
- **Normalization (Min-Max scaling):** Rescales values into a fixed range, usually 0 to 1.
- **Standardization (Z-score scaling):** Rescales values to have mean = 0 and standard deviation = 1.

### Handling Missing Data
- **Imputation:** Fill in missing values using a reasonable estimate — commonly the **mean** or **median** of that column (median is more robust when there are outliers).
- **Dropping rows:** Simply remove rows with missing data — only sensible when you have plenty of data and missing values are rare, otherwise you lose valuable information.

### Class Imbalance
When one class vastly outnumbers another (like the 1% fraud example above), the model tends to just learn to predict the majority class, since that alone already minimizes average error.

**Fixes:**
- **SMOTE (Synthetic Minority Over-sampling Technique):** Instead of just duplicating existing minority-class examples, SMOTE generates **new, synthetic** minority-class examples by interpolating between existing real ones — giving the model more varied examples of the rare class to learn from.
- **Class weights:** Tell the model, during training, to treat mistakes on the minority class as more "costly" than mistakes on the majority class — this changes what the loss function penalizes without touching the data itself.

---

# PART 2: Deep Learning Foundations (ANNs)

## 1. The Perceptron & Backpropagation

A **neural network** is built from layers of simple units (neurons), each doing a small computation, stacked together to model complex patterns.

### Forward Pass
This is how the network produces a prediction from an input:
**Input → Weights → Activation → Output → Loss Calculation**

- Each input is multiplied by a learned **weight** (representing how important that input is) and summed up.
- That sum is passed through an **activation function** (Section 2 below) — this is what allows the network to learn *non-linear* patterns, not just straight lines.
- This repeats layer by layer until you get a final **output**.
- The output is compared to the true/correct answer using a **loss function**, which produces a single number representing "how wrong was this prediction."

### Backward Pass (Backpropagation)
This is how the network **learns** — how it adjusts its weights to make the loss smaller next time.

**How it works:** Using the **Chain Rule** from calculus, the network calculates the **gradient** of the loss with respect to *every single weight* in the network — meaning, "if I nudge this specific weight slightly, how much would the final loss change, and in which direction?" It does this working *backward* from the output layer toward the input layer (hence "back"-propagation), because each layer's gradient depends on the layer after it.

Once you know each weight's gradient, you update every weight slightly in the direction that *reduces* the loss — this update step is called **Gradient Descent**.

**Analogy:** Imagine you're blindfolded on a hilly landscape and want to reach the lowest point. You can't see the whole landscape, but you can feel which direction is downhill *right where you're standing* — so you take a small step downhill, then reassess, then step again. That repeated "feel the slope, take a small step" process is gradient descent; the "slope you feel" at every point is the gradient computed via backpropagation.

**Interview soundbite (ready to use):** *"Backpropagation is just the chain rule applied to find how much each weight contributed to the final error, allowing us to minimize the loss."*

---

## 2. Activation Functions

Without activation functions, stacking multiple layers would be mathematically equivalent to just **one** linear layer — no matter how many layers you add, you'd still only be able to model straight-line relationships. Activation functions introduce **non-linearity**, which is what lets deep networks model complex, curved, real-world patterns.

| Function | Formula/Range | Where used | Key trait |
|---|---|---|---|
| **ReLU** | `max(0, x)` → outputs 0 or positive | Default choice for hidden layers | Fast to compute, helps avoid the vanishing gradient problem (its gradient is either 0 or exactly 1, no shrinking) |
| **Sigmoid** | `1 / (1 + e^-x)` → range 0 to 1 | Binary classification **output layer** | Squashes any input into a probability-like 0-1 range — perfect for "yes/no" style final answers |
| **Softmax** | Outputs a vector that sums to 1 | Multi-class classification **output layer** | Converts raw scores across multiple classes into a proper probability distribution (e.g., "70% cat, 20% dog, 10% bird") |
| **Tanh** | Range -1 to 1 | Sometimes used in hidden layers | Zero-centered (unlike Sigmoid), but still suffers from vanishing gradient for large positive/negative inputs |

**Why ReLU became the default for hidden layers:** Both Sigmoid and Tanh "saturate" — for very large or very negative inputs, their curve goes nearly flat, meaning the gradient (slope) there is nearly zero. During backpropagation, near-zero gradients multiplied across many layers shrink toward nothing — this is exactly the **vanishing gradient problem**. ReLU's gradient stays exactly 1 for any positive input, no matter how large, so it doesn't shrink the same way, letting gradients flow better through deep networks.

---

## 3. Optimizers & Regularization

### SGD vs. Adam
Both are algorithms for *how* to actually update the weights once you have their gradients.

- **SGD (Stochastic Gradient Descent):** The simplest approach — update weights by moving a fixed step size in the direction of the gradient. Simple and well understood, but can be slow to converge and treats every parameter the same way.
- **Adam (Adaptive Moment Estimation):** The **industry-standard default** for most deep learning today. It **adapts the learning rate individually for each parameter**, based on the history of that parameter's recent gradients — parameters that have been getting large, consistent gradients get smaller step sizes (to avoid overshooting), while parameters with small/sparse gradients get relatively larger steps. This generally leads to faster, more stable convergence with less manual tuning than plain SGD.

### Dropout
During training, **randomly "turn off" (zero out) a fraction of neurons** in a layer on each training pass (e.g., 20-50% of neurons, chosen randomly each time).

**Why this works:** It prevents the network from becoming overly reliant on any specific neuron or narrow combination of neurons — since any given neuron might be "switched off" at any time, the network is forced to learn **robust, redundant features** spread across many neurons, rather than a fragile pattern that only works if a specific exact set of neurons cooperates. This is one of the most effective, widely-used techniques against **overfitting** (connects directly back to Part 1, Section 2).

### Batch Normalization
Normalizes the *inputs to each layer* (not just the original input data) during training, so that as data flows deeper into the network, each layer receives inputs with a stable, consistent distribution (mean ~0, consistent scale).

**Why it helps:** As earlier layers' weights update during training, the *distribution* of values flowing into later layers keeps shifting around (sometimes called "internal covariate shift"), which makes training unstable and slow — layers are constantly having to adapt to a "moving target." Batch Normalization stabilizes this, which generally allows for faster training and lets you use higher learning rates safely.

---

# PART 3: Computer Vision (CNNs)

## 1. Core Concepts

CNNs (Convolutional Neural Networks) are specifically designed for grid-like data — most commonly images.

### Convolution
Instead of connecting every input pixel to every neuron (which is what a normal dense/fully-connected layer does — extremely wasteful for images), a CNN slides a small **filter (kernel)** — e.g., a 3×3 grid of learnable weights — across the image, computing a small local calculation at each position.

**What this achieves:** Each filter learns to detect a specific **local feature** — early layers might learn to detect simple things like edges or color gradients; deeper layers combine those into more complex features like textures, shapes, and eventually whole object parts.

### Pooling
After convolution, **pooling** (most commonly **Max Pooling**) reduces the spatial size of the data — e.g., taking a 2×2 block of values and keeping only the maximum one, shrinking the data to a quarter of its size.

**Why:** Reduces the amount of computation needed in later layers, and makes the learned features slightly more robust to small shifts/distortions in the image (a feature detected in slightly different exact pixel positions still gets picked up).

### Padding & Stride
- **Padding:** Adding extra (usually zero-value) pixels around the border of the image before applying a filter. Without padding, pixels at the edges of the image get "seen" by the filter far fewer times than pixels in the center, so edge information gets under-represented — padding fixes this.
- **Stride:** How many pixels the filter moves each step as it slides across the image. Stride 1 = moves one pixel at a time (dense, more overlap, more computation). Larger stride = moves further each step (coarser, faster, more downsampling).

### Parameter Sharing
A single filter's weights are **reused across the entire image** — the same 3×3 edge-detector filter slides across every position, rather than learning a completely separate set of weights for every single pixel location (which is what a dense/fully-connected network would effectively require).

**Why this matters:** This drastically reduces the total number of parameters the network needs to learn compared to a fully-connected network processing the same image, *and* it makes sense conceptually — an edge is an edge whether it appears in the top-left or bottom-right of an image, so it's reasonable (and efficient) to detect it the same way everywhere.

---

## 2. Evolution of Architectures

Understanding *why* each architecture was introduced (what problem it solved) is what interviewers are actually testing — not just being able to name them.

| Architecture | What it introduced / proved | Problem solved |
|---|---|---|
| **LeNet / AlexNet** | Early, relatively shallow CNNs | Proved that deep learning could actually work well for vision tasks (this was a big deal historically — AlexNet's 2012 ImageNet win is often cited as the moment deep learning "took off") |
| **VGG** | Simply used **more layers** (deeper network) with small, consistent 3×3 filters throughout | Showed that increasing **depth** (more layers) generally improves performance, given enough data |
| **ResNet** | Introduced **Skip Connections** (also called residual connections): `output = x + f(x)`, where the original input `x` is added directly to the transformed output `f(x)` | Solved the **Vanishing Gradient Problem** in very deep networks — without skip connections, extremely deep networks (100+ layers) actually got *worse*, because gradients had to flow backward through so many layers that they vanished before reaching the earliest layers. The skip connection gives gradients a direct "shortcut path" backward, bypassing that problem, and enabled reliably training networks with 100+ layers |

**Why skip connections work (good follow-up detail):** During backpropagation, the gradient can flow through the `+ x` shortcut path essentially unchanged, in addition to flowing through the transformed `f(x)` path — so even if the `f(x)` path's gradient shrinks toward zero through many layers, the shortcut path still delivers a usable gradient signal back to earlier layers.

**Transfer Learning connection:** Because architectures like ResNet were trained on huge datasets (like ImageNet, millions of labeled images), they've already learned rich, general-purpose visual features (edges → textures → shapes → object parts). Rather than training a new CNN from scratch on your smaller dataset, you can take a **pre-trained ResNet** and fine-tune it on your specific task — saving enormous amounts of time, compute, and data.

**Interview soundbite (ready to use):** *"I would use a pre-trained ResNet for a computer vision task via Transfer Learning, as it has already learned rich feature representations from ImageNet, saving me time and data."*

---

# PART 4: Sequential Data (RNN, LSTM, GRU)

## 1. The Problem with Standard RNNs

**RNNs (Recurrent Neural Networks)** are designed for sequential data (text, time series, audio) — they process input step by step, carrying forward a **hidden state** that acts as a summary of everything seen so far.

### Vanishing / Exploding Gradients
When backpropagating through time (i.e., backward through *every* time step of the sequence, not just through layers), the gradient gets **multiplied repeatedly**, once per time step. If those repeated multiplications involve numbers less than 1, the gradient shrinks toward zero the further back you go (**vanishing**); if they involve numbers greater than 1, it grows uncontrollably toward infinity (**exploding**).

### Short-term Memory
As a direct practical consequence of vanishing gradients, standard RNNs **struggle to learn dependencies that span many steps back** — e.g., in a long sentence or document, information from 50 words ago has essentially no influence on the current prediction, because its gradient signal vanished long before training could reinforce it. This is why standard RNNs are described as having only "short-term" memory despite technically processing the whole sequence.

---

## 2. LSTM (Long Short-Term Memory) — *Highly Asked!*

**The solution:** LSTM adds a separate **Cell State** — think of it as a dedicated "long-term memory conveyor belt" that runs through the entire sequence, largely undisturbed, alongside the regular hidden state. Information can be added to or removed from this cell state in a very controlled way, via **Gates**.

**The Three Gates** (each gate is a small neural network layer that outputs values between 0 and 1, acting like a dimmer switch/valve):

1. **Forget Gate:** Looks at the current input and previous hidden state, and decides **what old information in the cell state to discard** — outputting values close to 0 for information that should be forgotten, close to 1 for information that should be kept.
2. **Input Gate:** Decides **what new information from the current input should be added** to the cell state.
3. **Output Gate:** Decides **what part of the (now-updated) cell state should be exposed** as the hidden state output for this time step — the cell state itself isn't fully revealed, only a filtered portion of it.

**Why this solves vanishing gradients:** The cell state pathway is designed so that, when the forget gate says "keep this," information (and the corresponding gradient during backpropagation) can flow through many time steps **largely unchanged** — much like the skip connections in ResNet, this gives gradients a more direct path that avoids repeated shrinking multiplications.

**Interview soundbite (ready to use):** *"LSTMs solved the vanishing gradient problem of RNNs by using a gated mechanism that allows gradients to flow unchanged through the cell state, enabling the network to learn long-term dependencies."*

---

## 3. GRU (Gated Recurrent Unit)

A **simplified version of LSTM**: it merges the Forget Gate and Input Gate into a single **"Update Gate"**, and doesn't maintain a fully separate cell state the way LSTM does.

**Trade-offs:**
- **Fewer parameters** → faster to train, less memory required.
- **Usually similar performance** to LSTM on many tasks, though LSTM's extra complexity can occasionally give it an edge on tasks requiring very long, nuanced memory.

**When to mention GRU over LSTM in an interview:** If asked about efficiency/speed trade-offs, GRU is the answer — "fewer gates, simpler, faster, similar results in practice for many tasks."

---

# PART 5: The Transformer Architecture (The Absolute MVP)

*Since the JD explicitly mentions LLMs and RAG, this section is likely the single most important one — you must understand it deeply, not just recite definitions.*

## 1. Why Transformers Replaced RNNs/LSTMs

### Parallelization
RNNs/LSTMs process sequences **one step at a time, in order** (you can't compute step 5 until step 4 is done, because step 5 needs step 4's hidden state) — this is inherently sequential and slow, especially on modern GPUs which are built for massive *parallel* computation.

Transformers instead process the **entire sequence at once** — every token's representation is computed in parallel, not step by step. This makes training dramatically faster on modern hardware, which is a huge part of why Transformer-based models could scale up to the massive sizes we see in today's LLMs.

### Long-Range Dependencies
Because RNNs pass information step-by-step, information from far back in a sequence has to survive being passed through *every* intermediate step to influence a later prediction (connecting back to the vanishing gradient problem in Part 4). **Self-attention**, by contrast, lets **any token directly look at any other token**, regardless of how far apart they are in the sequence — there's no "information decay" over distance the way there is in RNNs.

---

## 2. Self-Attention Mechanism (The Core)

This is the single most important mechanism to be able to explain clearly and confidently.

### Query, Key, Value (QKV)
For every token in the input, the model learns to produce three separate vectors:
- **Query (Q):** Represents "what information am I, this token, looking for from other tokens?"
- **Key (K):** Represents "what information do I, this token, contain/offer, that other tokens might be looking for?"
- **Value (V):** The actual content/information this token will contribute if it's deemed relevant.

**Analogy:** Think of it like a search engine. Your search text is the **Query**. Every webpage has **Keys** (like tags/titles describing what that page is about) that get matched against your query. Once a match is found, you get back the actual **Value** (the page's content) — weighted by how good a match it was.

### The Process (Scaled Dot-Product Attention)
1. Take the **dot product** of a token's Query vector with every other token's Key vector — this produces a raw **attention score** for each pair, representing how relevant that other token is to this one.
2. Scale these scores down (dividing by the square root of the key vector's dimension) to keep the numbers in a stable range for training.
3. Apply a **Softmax** (from Part 2, Section 2!) to turn these scores into a clean probability distribution that sums to 1.
4. Use these softmax weights to compute a **weighted sum of all the Value vectors** — tokens that got a high attention score contribute more strongly to the final output for this token.

**In plain terms:** Every token asks "who among all the other tokens is relevant to me right now?", gets back a relevance score for each of them, and then builds its updated representation as a weighted blend of everyone's information — weighted by how relevant they were.

### Multi-Head Attention
Rather than doing this self-attention process just once, the Transformer runs **several attention operations in parallel** (multiple "heads"), each with its own separately learned Q/K/V weight matrices.

**Why:** Different heads can learn to specialize in capturing **different types of relationships** — e.g., one head might learn to track grammatical structure (subject-verb agreement), another might track semantic/topical relationships (which words relate to the same concept), another might track coreference (resolving "it" to what it refers to). Combining multiple heads gives the model a much richer, multi-faceted understanding than a single attention pass could provide.

---

## 3. Positional Encoding

Since Transformers process the entire sequence **in parallel** (Section 1) rather than step-by-step like RNNs, they have **no inherent sense of word order** — mathematically, without any extra info, "dog bites man" and "man bites dog" would look identical to a pure self-attention mechanism, since it's just comparing tokens to each other regardless of position.

**The fix:** Before feeding tokens into the Transformer, add a **Positional Encoding** vector to each token's embedding — these are typically generated using sine and cosine waves of different frequencies, producing a unique pattern for each position in the sequence. This injects positional/order information directly into the input representations, so the model can distinguish "dog bites man" from "man bites dog" even though attention itself is order-agnostic.

---

## 4. Encoder vs. Decoder

Transformers come in three general shapes, each suited to different types of tasks:

| Type | Example models | Attention style | Best for |
|---|---|---|---|
| **Encoder-only** | BERT | **Bidirectional** — each token can attend to *every* other token in the sequence, both before and after it | **Understanding** tasks: classification, Named Entity Recognition (NER), generating embeddings for search/retrieval |
| **Decoder-only** | GPT, Claude, Llama | **Causal (masked)** — each token can only attend to tokens *before* it, never future ones | **Generation** tasks: producing text one token at a time, since at generation time you genuinely don't have access to "future" tokens yet |
| **Encoder-Decoder** | T5, BART | Encoder reads the full input bidirectionally, then a decoder generates output causally, attending back to the encoder's representations | Tasks that transform one full sequence into another: **translation**, **summarization** |

**Why decoder-only models use causal/masked attention (important connection to make in interviews):** Because when the model is actually *generating* text one token at a time (like when Claude or GPT is writing a response), it can only ever see the tokens it has already produced — it cannot see tokens that don't exist yet. So during training, the model is deliberately **masked** from looking at future tokens too, forcing it to learn to predict the next token using only past context — exactly matching how it will actually be used at generation/inference time.

**Why this matters for your RAG/Agents knowledge (tying it all together):** Every LLM you've been discussing in your Agents and RAG notes — the one doing function calling, the one generating grounded answers from retrieved context — is a **decoder-only Transformer**. Understanding causal attention explains *why* these models generate text left-to-right, one token at a time, and why the context window (from your RAG/Agents notes) is really just "how many tokens of causal attention history the model can look back on."

---

# CASE STUDY — Choosing the Right Architecture

**Prompt:** *"Walk me through how you'd approach three different problems: (1) predicting customer churn from tabular data, (2) classifying product images, and (3) building a document Q&A chatbot."*

This is a great way to demonstrate you don't just memorize architectures — you know **when** to reach for each one.

### Problem 1: Predicting Customer Churn (Tabular Data)
- This is a **classification** problem (churn / no churn) — binary output. (Part 1, Section 1)
- Start with **data preprocessing**: handle missing values (imputation), check for **class imbalance** (churn is usually the minority class — most customers don't churn) → apply **class weights or SMOTE**. (Part 1, Section 3)
- If the dataset is imbalanced, evaluate using **F1-Score / ROC-AUC**, not plain Accuracy. (Part 1, Section 1)
- Monitor training vs. validation loss to check for **overfitting** — apply **L2 regularization** if the model is memorizing rather than generalizing. (Part 1, Section 2)
- A deep neural network isn't necessarily even required here — classical ML models (e.g., gradient-boosted trees) often perform excellently on structured tabular data — but if using a neural net, normalize/standardize the numeric features first, since NNs are sensitive to feature scale. (Part 1, Section 3)

### Problem 2: Classifying Product Images
- This calls for a **CNN**, since images are grid-structured spatial data where **parameter sharing** and **convolution** are far more efficient than a plain dense network. (Part 3, Section 1)
- Rather than training a CNN from scratch (which needs huge amounts of labeled image data), use **Transfer Learning** with a **pre-trained ResNet** — leveraging features it already learned from ImageNet. (Part 3, Section 2)
- Use **ReLU** activations in hidden layers (fast, avoids vanishing gradients) and **Softmax** on the output layer if classifying into multiple product categories. (Part 2, Section 2)
- Use **Dropout** and/or data augmentation if the model starts overfitting to the specific training images. (Part 2, Section 3)

### Problem 3: Document Q&A Chatbot
- This is fundamentally a **language generation** problem — you need a **decoder-only Transformer** (like the LLMs discussed in your RAG notes) to generate fluent, grounded answers. (Part 5, Section 4)
- The model uses **self-attention** to understand relationships between words in both the user's question and the retrieved document context — including **long-range dependencies** that an RNN/LSTM would struggle to capture reliably. (Part 5, Sections 1–2)
- Because this is generation (not just classification/understanding), a **decoder-only** architecture with **causal attention** is the right shape — not an encoder-only model like BERT, which is better suited for *retrieving/embedding* the documents in the first place (connecting directly back to the embeddings step in your RAG notes) but not for generating the final free-form answer.
- **Positional Encoding** ensures the model understands word order in both the question and the retrieved context chunks. (Part 5, Section 3)

**The key takeaway to say out loud in an interview:** *"The architecture choice always follows from the shape and nature of the data and the task — tabular/structured data often doesn't even need deep learning, grid-structured image data calls for CNNs with transfer learning where possible, and sequential language generation tasks call for decoder-only Transformers, which is exactly the backbone of the LLMs used in RAG and agent systems."*

---

## 🎯 Final Interview Cram Checklist

- [ ] Can I explain why Accuracy is misleading on imbalanced data, with a concrete example?
- [ ] Can I explain overfitting vs underfitting and how to detect each from a loss curve?
- [ ] Can I explain the difference between L1 and L2 regularization?
- [ ] Can I walk through the forward pass and backward pass of a neural network in my own words?
- [ ] Can I explain why ReLU solves the vanishing gradient problem better than Sigmoid/Tanh?
- [ ] Can I explain what Dropout and Batch Normalization each do, and why they help?
- [ ] Can I explain convolution, pooling, and parameter sharing in one sentence each?
- [ ] Can I explain what problem ResNet's skip connections solved, and why it worked?
- [ ] Can I explain the vanishing gradient problem in RNNs, and how LSTM's three gates fix it?
- [ ] Can I explain GRU vs LSTM as a simplicity/speed trade-off?
- [ ] Can I explain Query/Key/Value in self-attention using the "search engine" analogy?
- [ ] Can I explain why Transformers need Positional Encoding but RNNs don't?
- [ ] Can I explain the difference between Encoder-only, Decoder-only, and Encoder-Decoder models, with an example of each?
- [ ] Can I explain why LLMs like GPT/Claude are decoder-only and use causal attention?
- [ ] Can I pick the right architecture (tabular / CNN / Transformer) for a new problem and justify why, unprompted?
