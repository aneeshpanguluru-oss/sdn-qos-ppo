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
PYTHONPATH=src python -m sdn_rl.evaluate --model models/ppo_qos_router.zip
```

## Predictive QoS and fault-tolerance experiment

The controller uses actual configured port capacities, not a fixed 100 Mbps
divider. Train the predictive policy with one-step utilization momentum and a
mid-episode `s2-s4` link failure:

```bash
PYTHONPATH=src python -m sdn_rl.train --failure-scenario --timesteps 100000
PYTHONPATH=src python -m sdn_rl.train --predictive --failure-scenario --timesteps 100000
PYTHONPATH=src python -m sdn_rl.evaluate --compare-all --episodes 50
PYTHONPATH=src pytest -q
```

`--compare-all` reports throughput, maximum utilization, delay, estimated
packet-loss rate, feasibility, decision time, and route churn for shortest
path, min-hop, standard PPO, and predictive PPO under normal traffic and the
controlled failure scenario. It prints the two simulation result tables:
**Normal traffic scenario** and **Fault tolerance: s2-s4 fails at step 60**.

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

## Live routing-policy experiment

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

For the min-hop baseline, restart the controller with:

```bash
PYTHONPATH=src ROUTING_MODE=minhop ryu-manager sdn_rl.ryu_app
```

Min-hop selects candidate 0 (`s1-s2-s4`). Its route has a 25 Mbps bottleneck,
but it fails when `s2-s4` is taken down because fixed baselines do not perform
automatic recovery.

In each Mininet run, use:

```bash
mininet> h4 iperf3 -s &
mininet> h1 iperf3 -c 10.0.0.4 -u -b 20M -t 30
```

Record received bitrate, jitter, packet loss, and the selected controller path.

## Experiment plan

Compare these routing choices under the same flows and link-failure scenario:

1. shortest path (baseline)
2. min-hop (baseline)
3. PPO policy using current QoS state
4. PPO policy with the prediction feature enabled

Record throughput, end-to-end delay, packet loss, maximum link utilization, and controller decision time.

## Important integration note

The included Ryu app uses the fixed, explicit switch IDs and port map in the
provided Mininet topology to install a policy-selected OpenFlow path for the
`h1` to `h4` test flow. Expanding this initial proof of concept to arbitrary
host pairs is the next controller milestone.

Use `ROUTING_MODE=predictive` to load the predictive model in Ryu after it has
been trained. The controller then includes a one-interval utilization forecast
in the policy state.

## Live link-failure recovery

The Ryu controller listens for Mininet/OpenFlow port-down events. When the
`s2-s4` link fails, it removes the managed `h1` to `h4` rules and installs the
best viable candidate path immediately. Run a 60-second test in the Mininet
CLI, then break the link after about ten seconds:

```bash
mininet> h4 iperf3 -s &
mininet> h1 iperf3 -c 10.0.0.4 -u -b 20M -t 60 &
mininet> link s2 s4 down
```

The controller log should report `Link failure detected` followed by
`Recovery selected candidate 1: [1, 3, 4]`. Restore the test topology with
`link s2 s4 up`.

For a fixed-baseline failure test, recovery is disabled automatically for the
`minhop` and `shortest` modes. This intentionally leaves traffic on the
chosen route; `RECOVERY_ENABLED=0` can also disable recovery for a PPO
ablation:

```bash
RECOVERY_ENABLED=0 ROUTING_MODE=minhop FLOW_DEMAND_MBPS=20 ryu-manager sdn_rl.ryu_app
```
