# RUN_GUIDE: How to run `sdn-qos-ppo`

This file is a step-by-step runbook. It is written so that a person or an AI agent with no prior context can set up, train, evaluate, and run the live experiment.

## 0. What this project is

An SDN controller (Ryu) that chooses a route for each flow using a PPO reinforcement-learning agent. It is tested on an emulated six-switch Mininet network with redundant paths.

- The agent observes link utilization, delay, packet-loss estimate, and the flow's bandwidth demand.
- The reward favors throughput and penalizes congestion, delay, loss, and route churn.
- The live experiment compares **PPO routing** against a fixed **shortest-path baseline** for one flow, `h1` to `h4`.

There are three stages: **(A) train** the PPO model in simulation, **(B) evaluate** it in simulation, **(C) run it live** in Mininet with the Ryu controller.

## 1. Repository layout

```
sdn-qos-ppo/
├── README.md
├── RUN_GUIDE.md             <- this file
├── requirements.txt         <- training/evaluation deps (Python 3.12)
├── requirements-ryu.txt     <- extra deps for the Ryu controller
├── mininet/
│   └── topology.py          <- six-switch Mininet topology
├── src/sdn_rl/
│   ├── __init__.py
│   ├── config.py            <- shared constants
│   ├── traffic_env.py       <- Gymnasium training environment
│   ├── train.py             <- trains PPO, saves models/ppo_qos_router.zip
│   ├── evaluate.py          <- evaluates the model, can compare to shortest path
│   └── ryu_app.py           <- Ryu controller app (loads the trained model)
└── models/                  <- created by train.py (may not exist in a fresh clone)
    └── ppo_qos_router.zip
```

## 2. Requirements

| Item | Needed for | Notes |
|---|---|---|
| Linux, or Windows with WSL2 (Ubuntu) | Everything live | Mininet and Open vSwitch need Linux. Training also works on plain Windows. |
| Python 3.12 | Training and evaluation (`.venv`) | Used with `requirements.txt` |
| Python 3.10 or 3.11 | Ryu controller (`.venv-ryu`) | Ryu 4.34 does not install on Python 3.12 |
| Mininet | Live run | `sudo apt install mininet` |
| Open vSwitch | Live run | `sudo apt install openvswitch-switch` |
| iperf3 | Live run | `sudo apt install iperf3` |
| sudo access | Live run | Mininet must run as root |

Install the system packages once (Ubuntu or WSL):

```bash
sudo apt update
sudo apt install -y mininet openvswitch-switch iperf3 python3-venv
```

If Python 3.11 is missing, install it (for example via the `deadsnakes` PPA on Ubuntu) before creating `.venv-ryu`.

## 3. One-time setup

Run these from the repo root, inside WSL or Linux. **Always create the virtual environments inside the Linux filesystem or WSL, not from Windows.** Virtual environments created on Windows contain symlinks that break when copied.

```bash
cd /path/to/sdn-qos-ppo

# Environment 1: training and evaluation (Python 3.12)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
deactivate

# Environment 2: Ryu controller (Python 3.10 or 3.11)
python3.11 -m venv .venv-ryu
source .venv-ryu/bin/activate
pip install -r requirements.txt -r requirements-ryu.txt
deactivate
```

In WSL, a Windows folder is reached as `/mnt/c/Users/<name>/...`. Replace `/path/to/sdn-qos-ppo` with the real path.

## 4. Stage A: Train the PPO model

The trained model is **not stored in git** (`*.zip` is excluded). Create it once. You do not need to retrain before every experiment.

```bash
cd /path/to/sdn-qos-ppo
source .venv/bin/activate          # the main env, NOT .venv-ryu
export PYTHONPATH=src
python -m sdn_rl.train --timesteps 100000
```

**Expected result:** the file `models/ppo_qos_router.zip` exists afterwards.

```bash
ls -l models/ppo_qos_router.zip
```

If the file is not at that path, find where `train.py` saves it with `grep -n "models\|save" src/sdn_rl/train.py`.

## 5. Stage B: Evaluate in simulation (optional but recommended)

```bash
source .venv/bin/activate
export PYTHONPATH=src

# Evaluate the trained model
python -m sdn_rl.evaluate --model models/ppo_qos_router.zip

# Compare PPO with the fixed shortest-path baseline on identical scenarios
python -m sdn_rl.evaluate --model models/ppo_qos_router.zip --compare-shortest
```

## 6. Stage C: Live experiment (Ryu + Mininet)

Use **two terminals**, both inside WSL or Linux. **Start the controller first, then Mininet.**

### 6.1 Terminal 1: start the controller (PPO mode)

```bash
wsl                                   # only if you are starting from Windows
cd /path/to/sdn-qos-ppo
sudo service openvswitch-switch start
source .venv-ryu/bin/activate
export PYTHONPATH=src
ROUTING_MODE=ppo FLOW_DEMAND_MBPS=20 ryu-manager sdn_rl.ryu_app
```

Leave it running. It should log that it loaded the model and, later, that switches connected.

> Check that `FLOW_DEMAND_MBPS` is actually read:
> `grep -n "FLOW_DEMAND_MBPS" src/sdn_rl/ryu_app.py`.
> If nothing is found, the controller ignores it and uses its built-in default (the original README mentions 10 Mbps).

### 6.2 Terminal 2: start the emulated network

```bash
wsl                                   # only if you are starting from Windows
cd /path/to/sdn-qos-ppo
sudo mn -c                            # clean up old Mininet state
sudo python3 mininet/topology.py
```

When the `mininet>` prompt appears, run:

```
h1 ping -c 2 h4
h4 iperf3 -s -1 &
h1 iperf3 -c 10.0.0.4 -u -b 20M -t 30
```

What the commands do:

1. `ping` checks connectivity and triggers the controller to install the route.
2. `iperf3 -s -1 &` starts a one-shot UDP server on `h4` in the background.
3. `iperf3 -c ... -u -b 20M -t 30` sends 20 Mbps of UDP traffic from `h1` to `h4` for 30 seconds.

### 6.3 Run the shortest-path baseline

1. In Terminal 2, type `exit` at the `mininet>` prompt.
2. In Terminal 1, press `Ctrl+C` to stop Ryu.
3. In Terminal 2, run `sudo mn -c`.
4. Restart the controller in Terminal 1 with the baseline mode:

   ```bash
   ROUTING_MODE=shortest FLOW_DEMAND_MBPS=20 ryu-manager sdn_rl.ryu_app
   ```

5. In Terminal 2, run `sudo python3 mininet/topology.py` and then the same three commands from 6.2.

### 6.4 Expected behavior

- **Shortest mode** always picks candidate 2 (`s1 -> s2 -> s3 -> s4`). The `s2-s3` link is limited to 15 Mbps, so with 20 Mbps offered you should see received bitrate near 15 Mbps and noticeable packet loss.
- **PPO mode** should select a route that avoids the 15 Mbps bottleneck and achieve higher throughput and lower loss.
- The controller log shows which path was selected.

### 6.5 What to record

For each mode, repeat the run **3 times** and record:

- received bitrate (from the iperf3 server summary printed in Mininet)
- jitter
- packet loss
- the path the controller selected (from the Ryu log)

Report the average per mode.

## 7. Quick command reference

| Goal | Command |
|---|---|
| Train | `source .venv/bin/activate && export PYTHONPATH=src && python -m sdn_rl.train --timesteps 100000` |
| Evaluate | `python -m sdn_rl.evaluate --model models/ppo_qos_router.zip` |
| Compare in simulation | `python -m sdn_rl.evaluate --model models/ppo_qos_router.zip --compare-shortest` |
| Controller, PPO | `ROUTING_MODE=ppo FLOW_DEMAND_MBPS=20 ryu-manager sdn_rl.ryu_app` |
| Controller, baseline | `ROUTING_MODE=shortest FLOW_DEMAND_MBPS=20 ryu-manager sdn_rl.ryu_app` |
| Start network | `sudo mn -c && sudo python3 mininet/topology.py` |
| Clean Mininet | `sudo mn -c` |

## 8. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `ModuleNotFoundError: sdn_rl` | `PYTHONPATH=src` is not set in this shell. Run `export PYTHONPATH=src` from the repo root. |
| `FileNotFoundError` for `ppo_qos_router.zip` | The model has not been trained on this machine. Do Stage A. |
| Ryu fails to install or import | Wrong Python. `.venv-ryu` must use Python 3.10 or 3.11, not 3.12. |
| `ryu-manager: command not found` | `.venv-ryu` is not activated. Run `source .venv-ryu/bin/activate`. |
| Mininet cannot connect to the controller, or pings fail | Ryu was not started first, or Open vSwitch is not running. Start Ryu, run `sudo service openvswitch-switch start`, then `sudo mn -c` and restart Mininet. |
| `Exception: Error creating interface pair` or stale state | Run `sudo mn -c` and retry. |
| Port already in use (6653) | An old Ryu process is still running. Stop it (`pkill -f ryu-manager`) and retry. |
| `iperf3: command not found` | `sudo apt install iperf3`. |
| `iperf3` says the server is busy | Another server is still running on `h4`. Restart Mininet. |
| First `ping` shows packet loss | Normal. The controller installs flows on the first packets. Repeat the ping. |
| Copying the repo from Windows throws errors on `.venv-ryu` | Virtual environments must not be copied. Exclude `.venv*` and recreate them in WSL (Section 3). |

## 9. Notes for an AI agent running this project

- The project cannot run on native Windows for the live stage. Use WSL2 or Linux with root access.
- Two separate Python environments are required: `.venv` (Python 3.12, training and evaluation) and `.venv-ryu` (Python 3.10 or 3.11, controller). Do not mix them.
- Always set `PYTHONPATH=src` and run commands from the repo root, because modules live in `src/sdn_rl/` and are launched as `python -m sdn_rl.<module>` or `ryu-manager sdn_rl.ryu_app`.
- Run order for the live stage: train the model, start Ryu, then start Mininet. Between runs, always `exit` Mininet, stop Ryu, and run `sudo mn -c`.
- Live mode is selected only by the `ROUTING_MODE` environment variable (`ppo` or `shortest`). The default is PPO.
- The live proof of concept only installs paths for the `h1` to `h4` flow on the fixed topology. Other host pairs are not yet supported.
- A 100000-timestep training run is the documented default. More timesteps can be passed with `--timesteps`.
- Do not commit `.venv*`, `__pycache__`, logs, or capture files. The `.gitignore` already covers them.
