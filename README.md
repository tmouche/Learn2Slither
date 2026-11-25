# Learn2Slither

A snake that teaches itself to play with reinforcement learning. The project has two agents: a tabular **Q-Learning** agent and a **Deep Q-Learning** agent. The DQL agent runs on a multilayer perceptron written from scratch, with no PyTorch or TensorFlow.

## Table of Contents

- [🚀 Introduction](#introduction)
- [🏗️ Architecture](#architecture)
- [⚙️ Features](#features)
- [🛠 Tech Stack](#tech-stack)
- [🤖 AI Utilisation](#ai-utilisation)
- [📖 Sources](#sources)

---

<a id="introduction"></a>
## 🚀 Introduction

Learn2Slither is a reinforcement learning project. A snake moves on a 10×10 board and has to learn from its own experience how to grow as long as possible.

### Game rules

| Element | Start | Effect |
|---|---|---|
| 🐍 Snake | length 3, centre of the board | Dies if it hits a wall or its own body |
| 🟢 Green apple | 2 on the board | Length +1, then a new apple spawns |
| 🔴 Red apple | 1 on the board | Length −1, then a new apple spawns. Length 0 means game over |

The snake wins if it fills the whole board.

### What the snake sees

The agent does **not** see the whole board. It only sees the row and the column that pass through its head. From this view it builds a 12-value state, with 3 values for each of the 4 directions:

- the distance to the nearest danger (wall or body)
- the distance to a green apple
- the distance to a red apple

Each distance `d` is encoded as `1/d`, so closer objects give stronger signals. A value of `0` means "nothing in sight".

### Quick start

```bash
./setup.sh                    # creates .venv and installs the dependencies (including the local mlp package)
source .venv/bin/activate

python main.py --visual                                        # play yourself with the arrow keys
python main.py --load agent_ql_100000 --visual                 # watch a trained Q-Learning agent
python main.py --load agent_dql_3500 --agent dql --visual      # watch a trained Deep Q-Learning agent
```

### Command line

| Flag | Description |
|---|---|
| `--train <name>` | Train an agent and save it as `models/<name>…pkl` |
| `--load <name>` | Load a saved agent from `models/` (with `--train`, training continues from it) |
| `--session <n>` | Number of training episodes, or number of evaluation games (default `1`) |
| `--agent ql\|dql` | Agent type (default `ql`) |
| `--visual` | Open the pygame window (human play, visual training or visual replay) |

Examples:

```bash
python main.py --train my_agent --session 50000                 # headless Q-Learning training
python main.py --train my_dqn --agent dql --session 3000        # headless DQL training
python main.py --train my_agent --session 100 --visual          # watch the agent learn live
python main.py --load agent_ql_100000 --session 100             # evaluate: median and max length over 100 games
```

### In-game controls

| Key | Action |
|---|---|
| `↑ ↓ ← →` | Move the snake (human mode) |
| `Space` | Pause / resume |
| `P` / `M` | Speed up / slow down |
| `R` | Reset speed |
| `Shift` | Toggle step-by-step mode (AI play) |
| `N` | Next step in step-by-step mode |
| `Esc` | Quit |

---

<a id="architecture"></a>
## 🏗️ Architecture

```
Learn2Slither/
├── main.py                 # Entry point: hyperparameters, rewards, CLI dispatch
├── setup.sh                # venv + dependency installation
├── requirements.txt
├── history.csv             # One row of metrics per training run
│
├── app/                    # pygame front-end
│   ├── application.py      # Game loop, modes (human / AI play / AI train), key handling
│   ├── grid.py             # Board rendering
│   └── square.py           # One cell of the board
│
├── core/                   # Reinforcement learning
│   ├── environment.py      # Snake, apples, collisions, snake vision
│   ├── agent.py            # Abstract Agent: state extraction, rewards, epsilon decay, metrics, plots
│   ├── agent_ql.py         # Tabular Q-Learning (dict-based Q-table)
│   └── agent_dql.py        # Double DQN + prioritized replay buffer
│
├── utils/                  # Settings, enums, dataclasses, custom exceptions, logger
├── models/                 # Saved agents (Q-tables and network weights/biases)
├── functional_tests/       # Manual scripts for each mode
│
└── mlp_package/            # Homemade multilayer perceptron library ("mlp")
    └── mlp/
        ├── core/           # Network, Layer, Model
        ├── ml_tools/       # Activations, losses, optimizers, initialisations, forward pass
        └── utils/
```

### How the parts interact

```
            ┌──────────────┐  change_move / update   ┌──────────────┐
            │    Agent     │ ──────────────────────▶ │ Environment  │
            │  (QL / DQL)  │ ◀────────────────────── │              │
            └──────┬───────┘  snake_view + game      └──────▲───────┘
                   │          events (exceptions)           │
   train() headless│                                        │ renders map
     or generator  │          ┌──────────────┐              │
                   └────────▶ │ Application  │ ─────────────┘
                              │   (pygame)   │
                              └──────────────┘
```

- **Environment** keeps the board as a flat list (`W` wall, `S` body, `H` head, `G`/`R` apples, `O` empty). It reports game events (green apple, red apple, wall hit, tail hit, win) by raising exceptions. The agent catches them and turns them into rewards.
- **Agent** is the shared base class. It handles state extraction, the reward loop, the linear epsilon decay, episode statistics, saving to `history.csv` and Plotly charts. Subclasses implement `train`, `train_sequentially`, `take_action`, `save` and `load`.
- **Application** drives the environment for a human or an agent. For visual training it consumes the agent's `train_sequentially()` generator one step at a time, so the training is drawn as it runs.

### Agents

**Q-Learning (`AgentQL`)**
- The Q-table is a `dict` that maps a hashed state to 4 action values. New states are added on the fly.
- Standard Bellman update: `Q(s,a) ← Q(s,a) + α · (r + γ · max Q(s',·) − Q(s,a))`.

**Deep Q-Learning (`AgentDQL`)**
- Network: `12 → 32 (LeakyReLU, He init) → 4 (linear, Xavier init)`, trained with MSE loss and the Adam optimizer.
- **Double DQN**: the online network picks the next action and the target network scores it. The target network is synchronised every 150 steps.
- **Prioritized experience replay**: a buffer of 6,000 transitions, sampled in proportion to `|TD error|^0.6`. TD errors are clipped to `[-1, 1]`.

**Shared mechanisms**
- **Action masking**: the move straight back into the body is never chosen.
- **Linear epsilon decay**: exploration goes from `1.0` down to `0.05` over the first 30% of the episodes, whatever the total number of episodes.
- **Anti-loop guard**: after 75 steps with no positive reward, the agent is forced to take a random move. The Q-Learning agent also gets a penalty in that case.

---

<a id="features"></a>
## ⚙️ Features

- 🎮 **Human mode**: play Snake yourself in a pygame window.
- 🧠 **Two RL agents**: tabular Q-Learning and Deep Q-Learning (Double DQN with prioritized replay).
- 🏋️ **Headless training**: fast training with no rendering, for many episodes.
- 👀 **Visual training**: watch the agent learn live, with adjustable speed.
- ⏯️ **Step-by-step replay**: freeze a trained agent and step through its decisions one move at a time.
- 💾 **Save and load**: Q-tables and network weights are pickled into `models/`. A loaded agent can be trained further.
- 📊 **Evaluation**: `--load <name> --session <n>` plays `n` games (up to 700 steps each) and logs the median and maximum snake length.
- 📈 **Training history**: every run adds its hyperparameters, rewards and results to `history.csv`. Plotly charts can be produced (deaths, apples eaten, reward, length per episode).
- 🧩 **Homemade neural network library** (`mlp`): layers, 6 activations (identity, Sigmoid, Tanh, ReLU, LeakyReLU, Softmax), 6 losses, 5 optimizers (SGD, RMSProp, Nesterov, Adam, and an Adam/Nesterov hybrid), and Xavier/He initialisations.

### Pre-trained models

The `models/` folder contains checkpoints saved at increasing numbers of training episodes. These are the results recorded in `history.csv`. Lengths are measured *during training*, so exploration is still active:

| Agent | Episodes | Mean length | Max length |
|---|---:|---:|---:|
| `agent_ql_1000` | 1,000 | 3.5 | 7 |
| `agent_ql_10000` | 10,000 | 5.2 | 19 |
| `agent_ql_45000` | 45,000 | 8.6 | 42 |
| `agent_ql_100000` | 100,000 | 10.4 | **49** |
| `agent_dql_100` | 100 | 3.1 | 9 |
| `agent_dql_1000` | 1,000 | 9.8 | 35 |
| `agent_dql_3000` | 3,000 | 10.3 | **44** |
| `agent_dql_3500` | 3,500 | 10.2 | 42 |

The DQL agent reaches the same level as the Q-Learning agent with about **30× fewer episodes**. Its network generalises across states it has never seen, while the Q-table has to visit every state.

---

<a id="tech-stack"></a>
## 🛠 Tech Stack

| Tool | Use |
|---|---|
| **Python ≥ 3.12** | Language (uses `typing.override` and `match`) |
| **NumPy** | Vector and matrix maths, replay buffer sampling |
| **pandas** | Training history (`history.csv`) |
| **Plotly** | Interactive training charts (HTML) |
| **pygame-ce** | Game window, rendering and keyboard input |
| **mlp** (homemade) | Multilayer perceptron framework, shipped as a local package in `mlp_package/dist/` |
| **flake8** | Linting / norm compliance |
| **pickle** | Saving and loading models |

---

<a id="ai-utilisation"></a>
## 🤖 AI Utilisation

An AI assistant (Claude, through Claude Code) was used during development as a **debugging partner**. The design and the core implementation were written by hand: the environment, the agents, the RL algorithms and the `mlp` library. The assistant was used for:

- **Checking the replay buffer**: a temporary measurement script to check the spread of prioritized samples.
- **Documentation**: drafting this README.

---

<a id="sources"></a>
## 📖 Sources

- H. van Hasselt, A. Guez & D. Silver, "Deep Reinforcement Learning with Double Q-learning", 2016 [arXiv](https://arxiv.org/pdf/1509.06461).
- T. Schaul, J. Quan, I. Antonoglou & D. Silver, "Prioritized Experience Replay", 2016 [arXiv](https://arxiv.org/pdf/1511.05952)
- D. P. Kingma & J. Ba, "Adam: A Method for Stochastic Optimization", 2015 [arXiv](https://arxiv.org/pdf/1412.6980).
- [pygame-ce documentation](https://pyga.me/docs/)
- [Plotly Python documentation](https://plotly.com/python/)
