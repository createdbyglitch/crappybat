# CrappyBat 🦇

> A toy neural reasoning engine that nobody asked for, written in a single C++23 file.

CrappyBat is an interactive REPL that combines a **symbolic knowledge base** with a **from-scratch neural network** — all in one self-contained C++ source file. You can teach it facts, write inference rules, make it reason, train its neural net, and inspect word embeddings, all from a terminal prompt.

Is it production-ready? No. Is it crappy? Yes. Is it kind of fun? Absolutely.

---

## Features

- **Knowledge base** — add first-order logic style facts with arity-N terms, confidence scores, and source tracking
- **Rule engine** — forward-chaining inference with pattern matching and variable unification (`?var` syntax)
- **Explainability** — full reasoning chain traces for any derived conclusion
- **Analogy detection** — finds subjects that share structural properties in the KB
- **Neural network** — hand-rolled backprop MLP (8→32→16→4) trained incrementally on facts you add
- **Word embeddings** — lightweight 32-dim vectors updated via co-occurrence signals, with cosine similarity queries
- **Memory store** — weighted key-value cache with LRU-style eviction
- **Confidence decay** — derived facts lose confidence over time; prune them with `forget`
- **Model persistence** — save and load the neural network weights to/from disk
- **Script mode** — pipe a file of commands as a batch script
- **ANSI colors** — because life is too short for monochrome terminals

---

## Installation

Requires a C++23-capable compiler (GCC 13+ or Clang 17+) and `g++` on your `PATH`.

```bash
curl -o crappybat.cpp https://raw.githubusercontent.com/createdbyglitch/crappybat/refs/heads/main/crappybat.cpp && \
g++ -std=c++23 -O2 -o crappybat crappybat.cpp && \
./crappybat
```

That's it. No dependencies, no build system, no package manager. One file, one binary.

---

## Usage

### Interactive REPL

```
./crappybat
```

You'll land at the `>>>` prompt. Type `help` to see all commands.

### Script mode

```
./crappybat my_script.bat
```

Each line of the file is executed as a command, echoed to stdout in gray. Useful for reproducible sessions.

---

## Commands

### Knowledge Base

| Command | Description |
|---|---|
| `add_fact <term>` | Add a fact, e.g. `sky(color, blue)` or just `mortal` |
| `add_rule "name" if <cond> [and <cond>...] then <conclusion>` | Add an inference rule |
| `think` | Run forward-chaining inference over all rules |
| `query <statement>` | Check if a statement is true (direct or derived) |
| `explain <statement>` | Show the full reasoning chain for a statement |
| `facts` | List all known facts with source and confidence |
| `rules` | List all rules with condition/conclusion structure |
| `derivations` | List all conclusions derived by `think` |
| `analogy` | Find concept pairs that share structural properties |
| `concepts` | Show concept frequency across the knowledge base |
| `forget [threshold]` | Drop facts below a confidence threshold (default 0.3) |
| `history` | Show the last 20 session events |
| `memory` | Show the top-10 weighted memory entries |

### Neural Network

| Command | Description |
|---|---|
| `nn_info` | Print architecture, layer shapes, weight norms |
| `nn_train <epochs> <lr>` | Train the MLP on current facts for N epochs |
| `nn_predict f0 f1 f2 f3 f4 f5 f6 f7` | Run a forward pass with 8 raw float inputs |

### Embeddings & Session

| Command | Description |
|---|---|
| `embed_sim word1 word2` | Cosine similarity between two word vectors |
| `save [file]` | Save NN weights (default: `crappybat.model`) |
| `load [file]` | Load NN weights from disk |
| `setname <name>` | Set the session name (default: Pipsqueak) |
| `help` | Print this command list |
| `exit` / `quit` | Quit |

---

## Example Session

```
>>> add_fact human(socrates)
✔ Added: human(socrates)

>>> add_fact human(plato)
✔ Added: human(plato)

>>> add_rule "humans-are-mortal" if human(?x) then mortal ?x
✔ Added rule: humans-are-mortal

>>> think
Reasoning...
  ✦ Derived via [humans-are-mortal]: mortal socrates
  ✦ Derived via [humans-are-mortal]: mortal plato

>>> query mortal socrates
✔ TRUE
  Rule "humans-are-mortal" derived: mortal socrates
  Binding: ?x=socrates

>>> explain mortal plato
Chain of reasoning:
  1. Rule "humans-are-mortal" derived: mortal plato
  2.   Binding: ?x=plato
  3.   Fact used: human(plato) (source: user)

>>> embed_sim socrates plato
Similarity(socrates, plato) = 0.3821

>>> nn_train 100 0.01
✔ Trained 100 epochs | avg loss: 0.043217

>>> save
✔ Saved model to: crappybat.model
```

---

## Architecture

CrappyBat is structured as a set of cooperating components, all living in one `.cpp` file:

**`NeuralNet`** — a fully connected MLP with configurable activation functions (ReLU, Sigmoid, Tanh, Linear), He initialization, MSE loss, and standard backpropagation. Automatically trained on every `add_fact` call and on demand via `nn_train`.

**`KnowledgeBase`** — stores typed `Fact` and `Rule` structs. Rules use a pattern-matching unifier that resolves `?variable` bindings across multi-condition antecedents. Forward chaining runs to fixpoint. Facts carry confidence scores that decay over time for derived (non-user) entries.

**`SimpleEmbedding`** — a 32-dimensional word vector table initialized with random values and updated via co-occurrence signals whenever facts are added. Queried with cosine similarity.

**`MemoryStore`** — a fixed-capacity weighted key-value cache. On overflow it evicts the lowest-weight entry. Stores every fact as it's added.

---

## Requirements

- C++23 or later
- GCC 13+ or Clang 17+ (MSVC untested)
- A terminal with ANSI escape code support (virtually every modern terminal)

---

## License

GNU General Public License v3.0 — see [https://www.gnu.org/licenses/gpl-3.0.html](https://www.gnu.org/licenses/gpl-3.0.html)

You are free to use, modify, and distribute this software under the terms of the GPLv3. Any derivative work must also be licensed under the GPLv3.