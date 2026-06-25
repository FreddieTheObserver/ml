# Week 1: Neural Networks

> Welcome to **Course 2 — Advanced Learning Algorithms**. Course 1 covered linear and logistic regression; now we build the algorithm behind most of modern AI: the **neural network** (a.k.a. **deep learning**). This first lesson is the *intuition* — where neural networks came from (the brain), the very simplified math model we'll actually use, and **why** they suddenly took off in the last decade.

## Neurons and the Brain

### Original Motivation — Mimic the Brain

Neural networks were originally invented to **write software that mimics how the human (biological) brain learns and thinks**. The brain demonstrates a far higher level of intelligence than anything else we've built, so it was a natural thing to try to copy.

> **Reality check:** today's **artificial neural networks** have become *very different* from how the brain actually works — but some of the biological vocabulary and intuition still lingers in how we describe them.

### A Brief History — Out of Favor, Then Back, Twice

Neural networks have had a bumpy ride. The ideas are **many decades old**:

| Era | What happened |
|---|---|
| **1950s** | Work on neural networks begins… then **falls out of favor** for a while. |
| **1980s – early 1990s** | **Resurgence** — real traction on applications like **handwritten digit recognition**, used to read **postal codes** for routing mail and **dollar amounts** on handwritten checks. |
| **Late 1990s** | Falls out of favor **again**. |
| **~2005 onward** | **Resurgence**, and rebranded as **deep learning**. |

> The rebrand mattered more than you'd think. "Deep learning" and "neural networks" mean almost the same thing — but the term *deep learning* simply **sounds better** ("deep" + "learning"), and that's the brand that took off over the last decade-plus.

### Where Neural Networks Are Used Today

Since ~2005, neural networks have revolutionized **one application area after another**, roughly in this order:

```
speech recognition  →  computer vision  →  text / NLP  →  everywhere
   (early win)         (ImageNet, 2012)                 (the last few years)
```

- **Speech recognition** — probably the first area modern deep learning transformed (researchers like **Li Deng** and **Geoff Hinton** were instrumental).
- **Computer vision** — the **"ImageNet moment" in 2012** was the bigger public splash that caught broad imagination.
- **Text / natural language processing** — over the next few years.
- **Today: nearly everywhere** — climate change, medical imaging, online advertising, product recommendations, and far more.

### How a Biological Neuron Works

All human thought comes from **neurons** sending **electrical impulses** to each other and occasionally forming **new connections**.

```
   other neurons ──► [ dendrites ]──►( CELL BODY )──► [ axon ]──► other neurons
      (inputs)        input wires       nucleus      output wire     (outputs)
                                     does computation
```

A single neuron:
1. **Receives** electrical impulses from many other neurons through its **input wires (dendrites)**.
2. Carries out **some computation** in its **cell body** (the nucleus is here).
3. **Sends its output** to yet other neurons along its **output wire (the axon)** — that output becomes another neuron's input.

> You **don't** need to memorize the biology terms (dendrite, axon, nucleus). They're shown only to anchor the analogy.

### The Artificial Neuron — A Simplified Model

An artificial neural network uses a **very simplified mathematical model** of that neuron. Draw a neuron as a little circle:

```
   inputs (numbers)            outputs (numbers)
        x ──►(  neuron  )──►  a   ──►( neuron )──► …
              takes inputs,
              does a computation,
              outputs a number
```

- A neuron takes **one or more inputs — just numbers**, does a computation, and **outputs another number**.
- That output can become the **input to the next neuron**.

When building a real network you don't build **one neuron at a time** — you simulate **many neurons at once**. A group of them collectively takes in a few numbers, computes, and outputs other numbers:

```
        ┌──►( neuron )──┐
 inputs ├──►( neuron )──┼──► outputs
        └──►( neuron )──┘
        (several neurons computing in parallel)
```

### Big Caveat — We Don't Really Know How the Brain Works

> Even though the analogy between biological and artificial neurons is appealing, **we have almost no idea how the human brain actually works.** Neuroscientists make fundamental breakthroughs every few years, which signals how much is still undiscovered. Blindly mimicking our (very limited) current understanding of the brain probably won't get us to real intelligence.

In practice:

- Deep-learning researchers have **largely shifted away** from biological motivation.
- Instead, they use **engineering principles** to figure out what actually builds more *effective* algorithms.
- Yet even these **extremely simplified** neuron models are enough to build **very powerful** deep-learning algorithms.

So: enjoy the brain analogy, but **don't take the biological motivation too seriously.**

### Why Now? — Data and Compute

The ideas are decades old, so *why did neural networks only take off recently?* The classic answer is one picture — **amount of data** (horizontal) vs **performance** (vertical):

```
performance
   ▲
   │                                   ____________  large neural network
   │                              ____/
   │                         ____/ ________________  medium neural network
   │                    ___ /_____/
   │               ____/____/  ____________________  small neural network
   │          ____/____/  ____/
   │      ___/____/  ____/______________________      traditional ML
   │   __/___/  ___/______/                           (linear / logistic regression) — plateaus
   │ _/__/__/__/
   └──────────────────────────────────────────────►  amount of data
```

Two trends combined:

1. **Data exploded.** With the internet, mobile phones, and the digitalization of society, records that used to be on paper (purchases, health records) became **digital**. The amount of data marched steadily to the right.
2. **Traditional algorithms plateau.** Feeding more data to **linear regression** or **logistic regression** eventually stops helping — they **can't scale** to take advantage of all that data.

What researchers observed about neural networks on the *same* big datasets:

| Network size | Behavior as data grows |
|---|---|
| **Small** neural network | Performance rises somewhat |
| **Medium** neural network (more neurons) | Rises more |
| **Large** neural network (lots of neurons) | Performance **keeps going up** for some applications |

> The lesson: when you **have a lot of data ("big data")** *and* train a **large enough neural network**, you unlock performance on speech recognition, image recognition, NLP, and more that was **simply not possible** with earlier algorithms. That's what made deep learning take off.

**And the hardware mattered too.** Faster processors — especially **GPUs (graphics processing units)**, hardware originally built to render computer graphics — turned out to be **really powerful for deep learning**. Cheap, fast compute was a major force in getting neural networks to where they are today.

### Takeaway

| Idea | Detail |
|---|---|
| Original motivation | Build software that **mimics the brain** (1950s origins) |
| History | 1950s start → out of favor → 1980s–90s resurgence (digit recognition) → out again → ~2005 rebrand as **deep learning** |
| Impact today | Speech → vision (**ImageNet 2012**) → NLP → nearly everything |
| Biological neuron | **dendrites** (inputs) → **cell body** (computes) → **axon** (output) → next neuron |
| Artificial neuron | Takes **numbers** in, computes, outputs a **number**; simulate **many at once** |
| Big caveat | We barely understand the real brain — **don't lean on the biology**; use engineering principles |
| Why now? | **Big data** + **large neural networks** + **fast compute (GPUs)** — traditional ML plateaus, large NNs keep improving |
| What's next | How a neural network **actually works** — the math of a layer of neurons |

## Demand Prediction — A First Neural Network

Time to see a neural network actually wired up. The running example is **demand prediction**: look at a product and predict **"will this be a top seller — yes or no?"**

> Retailers use exactly this kind of model to **plan inventory and marketing** — if you know something will sell, you buy more stock in advance.

### A Single Neuron *Is* Logistic Regression

Sell **T-shirts**; predict top-seller (yes/no) from the **price** `x`. Fit logistic regression — a sigmoid — to the data:

```
a = 1 / (1 + e^−(w·x + b))
```

This is the same model as Course 1, with **one notation change**: we now call the output **`a`** instead of `f(x)`.

- **`a` stands for "activation"** — a term borrowed from neuroscience, describing how much a neuron sends a **high output** to the neurons downstream of it.
- This little logistic-regression unit is a **very simplified model of a single neuron**: it takes the input `x` (price), computes the formula, and outputs the number `a` (the probability the T-shirt is a top seller).

> Think of a neuron as a **tiny computer** whose only job is: take in **one or a few numbers** → output **one or a few numbers**. (Reminder from the last lesson: this is *vastly* simpler than a real biological neuron — but it works very well in practice.)

Build a neural network by taking a bunch of these neurons and **wiring them together**.

### Wiring Neurons Into a Layer — The Four-Feature Example

![Hand-drawn slide titled 'Demand Prediction'. On the left, an input layer labelled x⃗ (vector x, with (x, y) noted as a training example) listing four features stacked vertically: price, shipping cost, marketing, material — annotated '4 numbers'. An orange arrow feeds them into a rounded box (the 'hidden layer', annotated 'layer ← can have multiple neurons') containing three blue circles (neurons). The three neurons are labelled with their activations: affordability (top), awareness (middle), perceived quality (bottom) — annotated '3 numbers' and 'activation values'. An arrow labelled a⃗ leads to a single magenta neuron on the right inside its own box, labelled 'layer ← output layer', annotated '1 number'. Its output arrow points to the text 'probability of being a top seller'. The word 'activations' is written above the hidden layer.](images/demand-prediction-neural-network.png)

Now use **four features** to predict a top seller:

| Feature | |
|---|---|
| **price** | how much the shirt costs |
| **shipping cost** | cost to ship it |
| **marketing** | how much it's promoted |
| **material quality** | thick high-quality cotton vs cheaper material |

You might suspect top-seller status really depends on a few **underlying factors**, so create one neuron to estimate each:

| Neuron (hidden) | Estimates | Driven mainly by |
|---|---|---|
| 1 | **affordability** | price + shipping cost (the total you pay) |
| 2 | **awareness** | marketing |
| 3 | **perceived quality** | price + material (a high price can *signal* quality) |

Group these three neurons into a **layer**. Their three outputs — affordability, awareness, perceived quality — are **activations**. Wire all three into a **final neuron** (another logistic-regression unit) that takes those three numbers and outputs the **probability of being a top seller**.

### Layers and Terminology

A **layer** is a group of neurons that take the **same or similar inputs** and together output a few numbers. A layer can have **many neurons or just one**.

| Layer | Role in this example |
|---|---|
| **Input layer** | The vector of **4 features** `x` (just a list of four numbers) |
| **Hidden layer** | The middle layer of **3 neurons** → outputs 3 activation values |
| **Output layer** | The final **single neuron** → outputs the predicted probability |

> **Why "hidden"?** In a training set you observe both the inputs `x` and the label `y` — but you **never see** the "correct" values of affordability, awareness, or perceived quality. Those intermediate values are **hidden** from the data, hence *hidden layer*.

### Simplification — Every Neuron Sees Every Feature

Manually deciding *which* features feed *which* neuron (affordability ← price+shipping, awareness ← marketing, …) would be a huge amount of work for a large network. So in practice:

> **Each neuron in a layer has access to *every* value from the previous layer.** Draw an arrow from **every** input feature to **every** hidden neuron.

The network then **learns the parameters** itself — e.g. the "affordability" neuron can learn to set the weights on marketing and material near zero, focusing only on the features that actually matter for affordability.

### Vector Notation

Bundle the inputs and activations into **vectors**:

```
x  =  [price, shipping, marketing, material]      ← input vector (4 numbers)
          │
          ▼   hidden layer
a  =  [affordability, awareness, perceived quality]   ← activation vector (3 numbers)
          │
          ▼   output layer
output  =  probability of being a top seller          ← 1 number
```

So a neural network is just **a few layers, where each layer takes in a vector and outputs another vector** of numbers.

### The Key Insight — A Network *Learns Its Own Features*

Cover up the left half of the diagram. What's left — the output neuron taking affordability, awareness, perceived quality → top-seller probability — is **just logistic regression**.

The clever part: it's logistic regression running on a **new, better set of features** (affordability, awareness, perceived quality) instead of the raw ones (price, shipping, …). These learned features are hopefully **more predictive**.

> **One way to see a neural network: logistic regression that learns its own features.**

Contrast with Course 1's housing example, where you **manually** engineered a feature — multiplying lot frontage `x1` by depth `x2` to get lot size `x1·x2`. That was **hand-done feature engineering**. A neural network instead **learns its own features automatically**, making the learning problem easier for itself — which is what makes it one of the most powerful learning algorithms in the world.

> **Nice property:** even though we *described* the hidden units as affordability/awareness/quality, you **don't** have to specify them. Train the network on data and it **decides for itself** what features the hidden layer should compute.

### Multiple Hidden Layers

![Slide titled 'Multiple hidden layers' showing two neural network architectures. Left network: an orange input x⃗ feeds a 1st hidden layer of 3 blue neurons; its output vector a⃗ feeds a 2nd hidden layer of 2 blue neurons; that a⃗ feeds a single magenta output-layer neuron with a pink output arrow. Annotated '1st hidden layer', '2nd hidden layer', 'output layer', and '"multilayer perceptron"'. Right network: input x⃗ feeds a 1st hidden layer of 4 neurons, then a 2nd hidden layer of 5 neurons, then a 3rd hidden layer of 3 neurons, then a single output neuron. Annotated '1st / 2nd / 3rd hidden layer'. Bottom caption: 'neural network architecture'.](images/multiple-hidden-layers.jpg)

Networks can stack **more than one** hidden layer. Each hidden layer takes the previous layer's activation vector `a⃗` and produces its own:

- **Example (left):** `x` → hidden layer 1 (3 neurons) → hidden layer 2 (2 neurons) → output layer (1 neuron).
- **Example (right):** `x` → hidden layer 1 (4) → hidden layer 2 (5) → hidden layer 3 (3) → output.

When building your own network you must choose **how many hidden layers** and **how many neurons per layer**. This is the network's **architecture**, and it affects performance — you'll learn tips for choosing a good one later in the course.

> **Jargon:** a multi-layer network like this is sometimes called a **multilayer perceptron** in the literature — same thing as the neural networks shown here.

### Takeaway

| Idea | Detail |
|---|---|
| A neuron | A logistic-regression unit: numbers in → number out; output is the **activation `a`** |
| Layer | A group of neurons sharing inputs and outputting numbers together (can be 1 or many neurons) |
| Three layer types | **Input** (feature vector `x`) → **hidden** (activations) → **output** (final prediction) |
| Why "hidden" | The training data never shows the correct intermediate activation values |
| Full connectivity | In practice every neuron sees **all** values from the previous layer; weights decide what matters |
| Vectors | Each layer takes a **vector** in and outputs a **vector** (`x` → `a` → output) |
| Big idea | A neural network is **logistic regression that learns its own features** — no manual feature engineering |
| Architecture | How many hidden layers + neurons per layer; chosen by you, affects performance ("multilayer perceptron") |
| What's next | Applying these ideas elsewhere — e.g. **face recognition** in computer vision |

---

*Course 1 — Supervised Machine Learning: [Week 1 — Introduction](../../../supervised-machine-learning/introduction-to-machinelearning/week1/week1.md) · [Week 2 — Multiple Input Variables](../../../supervised-machine-learning/regression-with-multiple-input-variables/week2/week2.md) · [Week 3 — Classification](../../../supervised-machine-learning/classification/week3/week3.md)*
