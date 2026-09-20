# RAG (Retrieval-Augmented Generation) — Detailed Interview Notes

> Written assuming first-time reading. Each topic: **what it is → why it matters → example → how it connects to the next piece.** Mapped where relevant to a "DocuMind"-style project (Hybrid retrieval, RRF, Cohere Rerank, Langfuse, Grounding check).

---

# PART 1: RAG Fundamentals

## 1. What is RAG & Why It Exists

**Definition:** RAG is an architecture that combines an LLM's ability to *generate* fluent language with an **external retrieval system** that fetches relevant, factual information before the LLM answers. Instead of relying purely on what the model "memorized" during training, it looks things up first — like an open-book exam instead of a closed-book one.

**The problem it solves — three specific LLM weaknesses:**
- **Knowledge cutoff:** The model only knows what existed up to its training date. It has no idea about anything that happened after, or anything that was never in its training data.
- **No access to private/enterprise data:** A company's internal policy docs, contracts, or product manuals were never part of any public LLM's training data — the model simply cannot know them unless you feed that data in.
- **Hallucination:** As covered in Part 1 of your Agents notes — when the model doesn't know something, it doesn't say "I don't know," it *makes something up* that sounds plausible.

**Why RAG instead of fine-tuning?** Fine-tuning a model on your company data is expensive, slow to update (every time your docs change, you'd need to retrain), and doesn't actually guarantee the model won't hallucinate. RAG instead **grounds** the model at *query time* by handing it the actual relevant text to read from — cheaper, instantly updatable (just re-index new documents), and far more controllable.

**Analogy:** Fine-tuning is like trying to make someone *memorize* an entire textbook. RAG is like handing them the *right page* of the textbook right when they need it, open-book style.

### The Three Pillars of RAG

1. **Ingestion** — preparing your data (loading, chunking, embedding, storing) *before* any query happens.
2. **Retrieval** — at query time, finding the most relevant pieces of that stored data.
3. **Generation** — the LLM writes the final answer using the retrieved pieces as its source of truth.

Everything below in Parts 1–4 is essentially a deep dive into these three pillars, in order.

---

## 2. Ingestion & Data Preparation (Pre-Retrieval)

This is everything that happens **before** a user ever asks a question — you're building your searchable knowledge base.

### 2.1 Data Loading

Real-world data isn't clean plain text — it comes in messy formats: PDFs (often with tables, images, multi-column layouts), Markdown, SQL databases, HTML pages, Word docs, etc. You need tools that can extract clean text (and ideally structure like tables/headers) from these formats.

- **`Unstructured`** — a general-purpose library that handles many file types and tries to preserve structure (titles, tables, lists).
- **`LlamaParse`** — specialized for complex documents, particularly good at parsing tricky PDF layouts (tables, multi-column text) accurately.
- **`PyPDF`** — a simpler, lighter-weight PDF text extractor — good for straightforward PDFs but weaker on complex layouts.

**Why this matters for interviews:** If you say "I just used `open()` and read raw text," that signals you don't understand that document parsing is a real engineering challenge — messy extraction = messy, garbled chunks = bad retrieval later, no matter how good your embeddings are. Garbage in, garbage out.

### 2.2 Chunking Strategies (Critical — commonly asked)

You can't embed and search an entire 100-page PDF as one unit — it's too large, and too much irrelevant content would get bundled with the relevant part. So you split documents into smaller **chunks**. *How* you split matters a lot.

| Strategy | How it works | Trade-off |
|---|---|---|
| **Fixed-size** | Split every N tokens (e.g., 500 tokens with 50-token overlap between chunks) | Simple and fast, but can cut a sentence or idea right in half, splitting relevant info across two chunks |
| **Recursive** | Try splitting by paragraph first; if a paragraph is still too big, split by sentence; if still too big, split by word. (LangChain's default approach) | Much more likely to keep whole ideas together than pure fixed-size, while still respecting a max size limit |
| **Semantic** | Use embeddings to detect where the *meaning* naturally shifts, and split there | Produces the most coherent chunks (each chunk = one complete idea), but more computationally expensive to generate |
| **Document-specific** | Markdown-aware (split by `#` headers), Code-aware (split by function/class boundaries) | Preserves the document's own logical structure — e.g., a whole function stays in one chunk, not sliced in half |

**Why overlap matters:** If chunk 1 ends mid-sentence and chunk 2 begins with the rest of that sentence, adding a small overlap (e.g., last 50 tokens of chunk 1 repeated at the start of chunk 2) ensures neither chunk loses context at its boundary.

**Interview tip:** If asked "how would you chunk X," always tie it to the *nature of the content* — e.g., legal contracts should use semantic/clause-based chunking (each clause is a self-contained legal idea), while FAQ-style support docs should use small, fixed or document-specific chunks (each Q&A pair is naturally short and self-contained).

### 2.3 Metadata

Along with the actual text, you attach **tags** to each chunk — like date, author, department, document type, source file, access permissions.

**Why this is crucial:** Metadata lets you **filter** before or during search. 
**Example:** A user asks *"What's our 2024 refund policy?"* — with metadata, you can filter to only search chunks tagged `document_type: policy` AND `year: 2024`, instead of semantically searching your *entire* knowledge base and hoping the right year comes up in the top results. This is essential in enterprise settings where you often need row-level security too (e.g., "only search HR documents this specific user has permission to see").

---

## 3. Embeddings & Vector Stores

### 3.1 Embeddings

**What it is:** A process that converts a piece of text into a **vector** — an array of floating-point numbers (e.g., 1536 numbers for OpenAI's embedding model) that represents the text's *meaning* in a mathematical space.

**The key property:** Text with **similar meaning ends up with vectors that are mathematically close together** — measured using **Cosine Similarity** (essentially: how similar is the *direction* these two vectors point in, regardless of their length).

**Example:** The sentence "The dog ran in the park" and "A canine sprinted across the field" would produce vectors that are *close* to each other, even though they share almost no exact words — because the embedding model captures the underlying meaning, not just keyword overlap.

**Common embedding models:**
- **OpenAI `text-embedding-3`** — widely used, strong general-purpose quality, paid API.
- **Cohere `embed-english-v3`** — another strong commercial option, good multilingual support in other variants.
- **Open-source: `BGE`, `E5`** — you can self-host these, no per-call API cost, good if you need data privacy or want to avoid external API dependency.

### 3.2 Vector Databases

Once you've converted all your chunks into vectors, you need somewhere to **store and search** them efficiently — searching millions of vectors by brute-force comparison would be far too slow.

**Common options:** Pinecone (fully managed cloud service), Milvus (open-source, scalable), Qdrant (open-source, fast), Chroma (lightweight, great for prototyping/local dev), **`pgvector`** (a PostgreSQL extension — great if you already use Postgres and want vector search without adding a whole new database system).

**Indexing Algorithm — HNSW (Hierarchical Navigable Small World):**
This is the algorithm most vector DBs use under the hood to make search fast. Instead of comparing your query vector to *every single* stored vector (which would be extremely slow at scale), HNSW builds a **layered graph structure** that lets it quickly navigate to the "neighborhood" of similar vectors, skipping most of the search space. It's called **Approximate Nearest Neighbor (ANN)** search — "approximate" because it trades a tiny bit of accuracy for a massive speed gain, which is almost always the right trade-off at scale.

### 3.3 Sparse vs. Dense Retrieval

This is a very common interview question, so be precise:

| Type | How it works | Strength | Weakness |
|---|---|---|---|
| **Dense (Vector)** | Compares meaning via embeddings | Great at *semantic* matches — "Canine" retrieves documents about "Dog" even without the exact word | Can miss exact technical terms, IDs, or rare words because it's focused on general meaning, not exact text |
| **Sparse (Keyword / BM25)** | Classic keyword-matching algorithm (like an advanced version of Ctrl+F, scoring based on term frequency) | Great at *exact* matches — acronyms, product codes, error codes (e.g., "Error Code 404" needs an *exact* match, not a "similar meaning" match) | Misses semantic matches — won't connect "Canine" to "Dog" if the exact word isn't present |

**Why this split matters:** Neither approach alone is complete — dense retrieval is "fuzzy but smart," sparse retrieval is "exact but rigid." This naturally leads to the next big topic: combining both.

---

# PART 2: Advanced Retrieval & Reranking

## 4. Hybrid Search

**The concept:** Run **both** Dense (vector/semantic) search AND Sparse (BM25/keyword) search on the same query, getting two separate ranked lists of results — then merge them into a single, better-ranked list.

**Why:** This gets you the best of both worlds — you catch semantic nuance ("canine" → "dog") *and* exact technical terms ("Error Code 404") in the same system, instead of picking one approach and accepting its blind spot.

### Reciprocal Rank Fusion (RRF)

This is the algorithm used to **merge** the two separate ranked lists (dense results + sparse results) into one unified ranking.

**How it conceptually works:** Instead of trying to compare the raw scores from two very different scoring systems (which aren't on the same scale — a BM25 score and a cosine similarity score mean different things), RRF looks at each document's **rank/position** in each list, and combines those rank positions using a formula that rewards documents that rank highly in *either* (or ideally *both*) lists. A document ranked #1 in the vector search and #3 in the keyword search gets a strong combined score, even if it wasn't literally the top result in either single method.

**Why rank-based instead of score-based:** Because dense similarity scores and sparse BM25 scores are fundamentally different units/scales — you can't directly average "0.87 cosine similarity" with "12.4 BM25 score" and have it mean anything. Rank position, on the other hand, is a comparable, unit-less measure across both systems.

**Interview soundbite (ready to use):** *"In my DocuMind project, I used hybrid retrieval with RRF to ensure we caught both semantic nuances and exact technical terms — because relying on vector search alone would miss exact matches like specific product codes, while keyword search alone would miss paraphrased or conceptually similar questions."*

---

## 5. Reranking

**The problem this solves:** Vector search is **fast but "fuzzy."** When you retrieve, say, the top 50 candidate chunks by vector similarity, the single *most* relevant chunk for the user's actual question might not be ranked #1 — it could be sitting at rank #15, because pure embedding similarity isn't perfectly precise.

**How reranking works:** You take those top-50 (or top-20) initial candidates, and run them through a **Cross-Encoder** model — this is different from the embedding model used earlier. Instead of embedding the query and the document *separately* and comparing vectors (what the initial retrieval does — called a "bi-encoder" approach), a cross-encoder looks at the **query and document together, at the same time**, and directly outputs a much more accurate relevance score for that specific pair.

**Common rerankers:** Cohere Rerank (commercial API), BGE Reranker (open-source).

**Why cross-encoders are more accurate but can't be used for the *initial* search:** Because a cross-encoder has to process the query + every single document *together* — this is far too slow to run against your entire database of millions of chunks. So the typical pattern is: **fast, fuzzy dense/sparse search first (narrows millions down to ~50) → slow, precise cross-encoder reranking second (narrows those 50 down to the best 3-5)**. This two-stage pattern (fast recall stage, then precise reranking stage) is a very standard production RAG design.

**Trade-off to explicitly mention in interviews:** Reranking **adds latency** (it's an extra model call on every query) but **drastically improves precision** (the final top chunks handed to the LLM are much more likely to be genuinely relevant). Mentioning this trade-off explicitly signals you understand production systems aren't just about maximizing accuracy — you have to balance it against speed and cost.

---

## 6. Query Transformation & Routing

Sometimes the *user's raw question* isn't the best possible search query. These techniques modify or route the query before retrieval happens.

### Multi-Query

The LLM generates **several different phrasings** of the same underlying question (e.g., 3 variations), and you run retrieval for *all* of them, then combine/deduplicate the results.

**Why:** Improves **recall** — a single query might miss a relevant chunk simply because of word choice, but one of the 3 rephrased versions might happen to match it. It's a way of "casting a wider net" without relying on the user to phrase things perfectly.

### HyDE (Hypothetical Document Embeddings)

This one is a bit counter-intuitive and often comes up as a "explain this to me" interview question:

**How it works:** Instead of embedding the user's *question* and searching for documents that match it, the LLM first **generates a fake, hypothetical answer** to the question — even though it might not be fully accurate — and then *that hypothetical answer* gets embedded and used to search the vector DB.

**Why this works:** A short question and a long, detailed answer often don't "look" very similar in vector space, even when the answer is exactly what you need — questions and answers have different linguistic structure. But a **hypothetical (even imperfect) answer** and a **real, correct answer** tend to be much more similar in vector space, because they're the same *type* of text (both are answers, not questions). So searching with a fake answer's embedding often retrieves the *real* answer's chunk more effectively than searching with the original question.

**When to use it:** Works especially well for **vague, underspecified, or complex questions** where the user's exact wording doesn't closely match how the answer is written in your source documents.

### Query Routing

Using an LLM as a **traffic controller** — before doing any retrieval, it first decides **which** knowledge base/database to even search.

**Example:** *"Is this an HR question?"* → route to the HR vector DB. *"Is this a code/technical question?"* → route to the code documentation DB. This avoids polluting search results with irrelevant domains, and can also save cost/latency by not searching every database for every query.

---

# PART 3: Generation & Context Engineering

## 7. Context Engineering

The JD explicitly calls this out, so understand it deeply — this is about how you assemble what actually gets sent to the LLM once you've retrieved your chunks.

### The "Lost in the Middle" Phenomenon

**What it is:** Research has shown that LLMs are noticeably better at using information placed at the **very beginning** or **very end** of their context window, and tend to under-utilize (effectively "forget" or ignore) information buried in the **middle** of a long prompt — even though technically it's all within the context window and the model "can see" it.

**Practical implication:** If you dump 20 retrieved chunks into the prompt in arbitrary order, the most important chunk might land in the middle and get under-weighted by the model, hurting answer quality — even if retrieval itself was perfect.

**How to handle it:**
- Don't just dump everything — **curate**. Select only the top 3-5 most relevant chunks (this is exactly why reranking, Section 5, matters so much — better precision means you need fewer chunks).
- **Order intelligently** — e.g., put your single most relevant chunk either first or last, not buried in the middle.
- Include useful **metadata** alongside each chunk (like its source, date) so the model has context about *where* this information came from, which also helps it cite sources.

### Grounding

**Definition:** Explicitly constraining the LLM to **only** use the information you gave it in the retrieved context — not its own general "memorized" knowledge — when answering.

**Why:** This is your main defense against hallucination in a RAG system. Even with perfect retrieval, if you don't explicitly instruct the model to stick to the provided context, it may blend in outside "knowledge" (which could be outdated, wrong, or simply not what the user wanted grounded answers from).

**Example prompt template (very commonly asked to write out in interviews):**
```
You are a helpful assistant. Answer ONLY using the provided context.
If the answer is not in the context, say "I don't know."
Cite your sources.

Context:
{retrieved_chunks}

Question: {user_question}
```

Notice three distinct instructions bundled together here: (1) restrict to context only, (2) explicit fallback behavior when the answer isn't present (rather than letting the model guess), (3) require citations (which both increases user trust and makes hallucinations easier to spot/verify).

---

## 8. Evaluation — The RAGAS Framework

**Interview soundbite (ready to use):** *"You can't improve what you don't measure. I use RAGAS for offline evaluation."*

RAGAS is a framework specifically built to evaluate RAG systems (as opposed to generic LLM evaluation), because RAG has two separate things that can go wrong — **retrieval** and **generation** — and you need to measure both independently to know *where* a problem is coming from.

| Metric | What it measures | What a low score tells you |
|---|---|---|
| **Faithfulness** | Is the generated answer derived *only* from the retrieved context, with no invented facts? | Your generation step is hallucinating, even if retrieval was fine — a generation-stage problem |
| **Answer Relevance** | Does the answer actually address what the user asked? | The answer might be factually grounded but off-topic or incomplete relative to the actual question |
| **Context Precision** | Of the chunks you retrieved, how many were actually relevant/useful? | Your retrieval step is pulling in noise/irrelevant chunks — a retrieval-stage problem |
| **Context Recall** | Did you retrieve *all* the necessary information needed to fully answer the question? | Your retrieval step is *missing* relevant chunks entirely — also a retrieval-stage problem, but the opposite failure mode from precision |

**Why splitting these four matters:** A RAG system can fail in very different ways, and these metrics let you diagnose *which* stage broke. Example: if Context Precision and Recall are both high (retrieval is doing its job) but Faithfulness is low, you know the *LLM itself* is the problem — it's ignoring good context and hallucinating anyway, meaning you need to fix your prompt/grounding instructions, not your retrieval pipeline.

**LLM-as-a-Judge:** Since these metrics are about *quality* (not something you can check with simple string matching), RAGAS typically uses a powerful LLM (e.g., GPT-4) to actually read the question, context, and answer, and score them against these criteria — essentially using AI to grade AI, which is now a standard evaluation pattern across the industry.

---

# PART 4: Production & Agentic RAG

## 9. Scaling & Production Ops

### Latency
Total response time = Retrieval time + Reranking time + LLM Generation time — these stack up sequentially, and each stage adds real, noticeable delay. **Streaming** (sending the LLM's response token-by-token to the user as it's generated, rather than waiting for the full answer) doesn't reduce *total* latency, but it drastically improves **perceived** latency — the user sees something happening immediately instead of staring at a blank loading spinner.

### Cost
- **Semantic Caching:** If many users ask very similar questions (common in customer support), you can cache the embeddings/answers for previously-seen semantically-similar queries and skip the full retrieval+generation pipeline entirely for repeat-style questions — saving both cost and latency.
- **Model sizing:** Use **smaller, cheaper models** for simpler sub-tasks like query routing or reranking, and reserve your **most powerful (and expensive) model** for the final answer generation step, where quality matters most. Not every step in the pipeline needs your biggest model.

### Incremental Indexing
When new documents arrive, you don't want to re-embed and re-index your *entire* database from scratch (extremely wasteful at scale) — you need a system that can **add new chunks/vectors** to the existing index without disrupting or duplicating what's already there. This is an important production consideration that's easy to overlook when you're just prototyping locally.

### Observability
Using tools like **Langfuse** or **OpenTelemetry** (same concept from your AgentOps notes) to **trace** the exact path of a query through your RAG pipeline — how long did retrieval take, what chunks got retrieved, how long did reranking take, what was the final prompt sent to the LLM, how long did generation take. This is essential for finding bottlenecks (e.g., "oh, reranking is actually our slowest step, not generation") and for debugging bad answers after the fact.

---

## 10. Agentic RAG — The Frontier

**Standard RAG is a fixed, straight-line pipeline:** Query → Retrieve → Generate. Every single query goes through the exact same steps, regardless of whether it actually needed retrieval at all, or whether the first retrieval attempt actually got good results.

**Agentic RAG puts an AI Agent in charge of the process itself**, making dynamic decisions at each step instead of blindly following a fixed pipeline:

- **"Do I even need to retrieve?"** — A simple greeting like "hi" or a general knowledge question doesn't need a knowledge-base lookup at all. An agent can recognize this and skip retrieval, saving time and cost.
- **"Is the retrieved info good enough?"** — This is the idea behind **Self-RAG** and **Corrective RAG**: after retrieving, the agent evaluates its own retrieved chunks (e.g., "are these actually relevant to the question?") *before* generating an answer, rather than blindly trusting whatever came back.
- **"Do I need to re-write the query and try again?"** — This is **multi-hop retrieval**: if the first retrieval attempt didn't return good results, the agent can reformulate the query (similar to Multi-Query/HyDE from Section 6, but applied *dynamically* based on evaluating the first attempt) and search again — potentially multiple times, chaining together information from different searches to answer complex, multi-part questions.
- **"Do I need a different tool entirely?"** — Sometimes the answer isn't in the document knowledge base at all — it needs a calculator, a live web search, or an API call instead. An agentic RAG system can recognize this and route to the *right* tool rather than being locked into "always search the vector DB."

**Why this matters for the interview:** This directly ties your RAG knowledge back to your Agents/AgentOps knowledge (Part 1 & 2 of your Agents notes) — Agentic RAG is literally "put an agent loop (Perceive → Plan → Act) around a RAG pipeline." It shows the interviewer you see these as connected systems, not separate silos — which is exactly what a JD asking for both "RAG pipelines" *and* "Agentic workflows" is testing for.

---

# PART 5: The Universal RAG System Design Framework

If they hand you an open-ended case study — *"Design a RAG system for a legal firm"* or *"Design a RAG system for a customer support bot"* — don't freeze. Walk through this **6-step framework** out loud. This structure alone demonstrates senior-level thinking, even before you fill in the specific details.

### Step 1 — Understand the Data
Ask/state: What format is the data in (PDFs, SQL, emails, Slack messages)? How often does it change (static archive vs. constantly updated)? How technical/specialized is the language (general text vs. dense legal/medical jargon)?

### Step 2 — Define the Chunking Strategy
Tie this back to Section 2.2. **Example answer for the legal firm case:** "Legal documents should use semantic or document-aware chunking by clause — each clause is a legally self-contained unit, and splitting mid-clause could change its meaning entirely." **Example for customer support:** "FAQ-style content works best with small, fixed-size chunks since each Q&A pair is naturally short and self-contained."

### Step 3 — Choose the Retrieval Strategy
Tie back to Sections 3.3 and 4. Ask: Do they need exact keyword/ID matches (→ Hybrid Search with BM25, e.g., searching for a specific contract clause number or case citation) or mostly conceptual/semantic understanding (→ pure Dense retrieval)? Do they need metadata filtering (e.g., "only search contracts from 2024," or "only documents this user's role has access to")?

### Step 4 — Design the Context Window
Tie back to Sections 5 and 7. How many chunks realistically fit without hitting "Lost in the Middle" issues? Do you need **Reranking** to confidently narrow down to the best 3-5 chunks rather than dumping in 20 loosely-relevant ones?

### Step 5 — Set Up Guardrails & Evaluation
Tie back to Sections 7 and 8. How do you prevent hallucination? (Grounding prompts, explicit "say I don't know" instructions, RAGAS Faithfulness scoring). How do you handle the "I don't know" case gracefully instead of forcing an answer?

### Step 6 — Plan for Production
Tie back to Section 9. How will you monitor cost and latency over time (OpenTelemetry/Langfuse traces)? How will the index be updated as new documents arrive (incremental indexing) without downtime or full re-processing?

**Why this framework is valuable beyond just this one interview:** It's genuinely how you'd *actually* approach any new RAG project in real work — which is exactly why walking through it out loud (even briefly) shows you're not just reciting definitions, you understand the actual engineering decision-making process.

---

# CASE STUDY — Full Walkthrough Example

**Prompt:** *"Design a RAG system for a legal firm that needs to answer questions about client contracts."*

**Your structured answer, step by step:**

1. **Data understanding:** Contracts are PDFs, often scanned/complex layouts (multi-column, tables of terms), fairly static once signed but the overall corpus grows as new contracts are added. Language is dense, technical legal jargon. → Use **LlamaParse** for robust extraction of complex PDF structure.

2. **Chunking:** Use **semantic/document-aware chunking by clause** — each clause (e.g., "Termination Clause," "Indemnification Clause") is a legally self-contained unit and should not be split mid-clause, or its meaning could be misrepresented. Attach **metadata**: client name, contract date, contract type, clause type.

3. **Retrieval:** Legal queries often reference specific clause types or exact terms ("What's the termination notice period in the Acme contract?") — so use **Hybrid Search** (Dense + Sparse/BM25) merged with **RRF**, plus **metadata filtering** (e.g., filter to `client: Acme` before searching, so you're not accidentally retrieving another client's contract).

4. **Reranking:** Given the high stakes of legal accuracy, add a **Cohere Rerank** step after initial hybrid retrieval — precision matters enormously here (citing the wrong clause could have real legal consequences), and the added latency is an acceptable trade-off for that precision.

5. **Context & Generation:** Assemble the top 3-5 reranked chunks, ordered with the most relevant first (mitigating "Lost in the Middle"), and use a strict **grounding prompt**: *"Answer ONLY using the provided contract text. If the answer isn't in the provided clauses, say so explicitly — do not guess on legal matters. Cite the specific clause and contract."*

6. **Evaluation:** Run **RAGAS** offline before shipping — especially prioritize **Faithfulness** (zero tolerance for hallucinated legal terms) and **Context Precision** (make sure retrieved clauses are actually the right ones).

7. **Production:** Use **Langfuse** to trace every query end-to-end (which clauses were retrieved, rerank scores, final prompt, latency per stage) so you can debug any incorrect answer after the fact. Set up **incremental indexing** so new contracts can be added to the vector DB the moment they're signed, without re-processing the entire archive.

8. **(Bonus — Agentic layer):** Could extend this to **Agentic RAG** — e.g., if a query spans multiple contracts ("compare the termination clauses across all our vendor contracts"), an agent could recognize this needs **multi-hop retrieval** (search multiple times, once per relevant contract) rather than a single-shot retrieval, then synthesize a combined answer.

---

## 🎯 Final Interview Cram Checklist

- [ ] Can I explain why RAG exists (3 LLM weaknesses it solves) without notes?
- [ ] Can I list 4 chunking strategies and when to use each?
- [ ] Can I explain Dense vs. Sparse retrieval with one example each?
- [ ] Can I explain Reciprocal Rank Fusion (RRF) — *why rank-based, not score-based*?
- [ ] Can I explain why Reranking exists, and the two-stage retrieval pattern (fast recall → precise rerank)?
- [ ] Can I explain HyDE and *why* it works (question vs. answer vector-space mismatch)?
- [ ] Can I explain "Lost in the Middle" and how to design around it?
- [ ] Can I write out a grounding prompt template from memory?
- [ ] Can I name all 4 RAGAS metrics and what a low score in each one tells you?
- [ ] Can I explain how Agentic RAG differs from standard RAG (the 4 dynamic decisions an agent makes)?
- [ ] Can I walk through the 6-step system design framework unprompted, given any new case study?
