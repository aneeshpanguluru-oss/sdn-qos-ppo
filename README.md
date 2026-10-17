# Predictive QoS-Aware PPO Routing for SDN

This starter project implements the recommended CN project: an SDN controller that chooses a route for each flow using a PPO agent. The agent observes current link utilization, delay, packet-loss estimate, and the flow bandwidth demand. Its reward favors admitted throughput while penalizing congestion, delay, loss, and route churn.

## What is included

- A Gymnasium environment for training a PPO policy before attaching it to a live network.
- A six-switch Mininet topology with redundant paths and bandwidth/delay controls.
- A Ryu application skeleton that reads the trained PPO model and maintains live port-utilization measurements.
- A reproducible training and evaluation command.

## Quick start (Linux or WSL)

Install Mininet, Open vSwitch, and Ryu using your Linux distribution's packages, then create a Python virtual environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is enough to train and evaluate the PPO policy on Python
3.12. Ryu 4.34 must be installed separately in a Python 3.10 or 3.11 virtual
environment because its packaging does not work with Python 3.12:

```bash
python3.11 -m venv .venv-ryu
source .venv-ryu/bin/activate
pip install -r requirements.txt -r requirements-ryu.txt
```

Train the simulated policy:

```bash
PYTHONPATH=src python -m sdn_rl.train --timesteps 100000
```

Evaluate it:

```bash
PYTHONPATH=src python -m sdn_rl.evaluate --model models/ppo_qos_router.zip
```

Compare PPO to the fixed shortest-path baseline on identical simulated
traffic scenarios:

```bash
PYTHONPATH=src python -m sdn_rl.evaluate --model models/ppo_qos_router.zip --compare-shortest
```

In separate terminals, start the controller and emulated network:

```bash
source .venv-ryu/bin/activate
PYTHONPATH=src ryu-manager sdn_rl.ryu_app
sudo python mininet/topology.py
```

Then generate traffic, for example:

```bash
mininet> h1 iperf3 -s &
mininet> h4 iperf3 -c 10.0.0.4 -b 15M -t 30
```

The first live demonstration routes traffic from `h1` to `h4`; use a 10 Mbps
stream to match the controller's initial demand estimate:

```bash
mininet> h4 iperf3 -s &
mininet> h1 iperf3 -c 10.0.0.4 -u -b 10M -t 30
```

## Live PPO versus shortest-path experiment

Use the same Mininet topology and a 20 Mbps UDP offered load for each run.
PPO uses the default controller mode:

```bash
PYTHONPATH=src ROUTING_MODE=ppo ryu-manager sdn_rl.ryu_app
```

For the baseline, restart the controller with the fixed shortest path. It
always selects candidate 2 (`s1-s2-s3-s4`), whose `s2-s3` link is limited to
15 Mbps:

```bash
PYTHONPATH=src ROUTING_MODE=shortest ryu-manager sdn_rl.ryu_app
```

In each Mininet run, use:

```bash
mininet> h4 iperf3 -s &
mininet> h1 iperf3 -c 10.0.0.4 -u -b 20M -t 30
```

Record received bitrate, jitter, packet loss, and the selected controller path.

## Experiment plan

Compare these routing choices under the same flows and link-failure scenario:

1. shortest path (baseline)
2. PPO policy using current QoS state
3. PPO policy with the prediction feature enabled (the next milestone)

Record throughput, end-to-end delay, packet loss, maximum link utilization, and controller decision time.

## Important integration note

The included Ryu app uses the fixed, explicit switch IDs and port map in the
provided Mininet topology to install a policy-selected OpenFlow path for the
`h1` to `h4` test flow. Expanding this initial proof of concept to arbitrary
host pairs is the next controller milestone.
