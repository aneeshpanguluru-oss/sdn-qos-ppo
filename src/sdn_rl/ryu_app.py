"""Ryu controller foundation for live QoS measurements and PPO routing decisions.

Run after training: PYTHONPATH=src ryu-manager sdn_rl.ryu_app

ROUTING_MODE: ppo (default) | minhop (fewest switches) | shortest (lowest total delay)
"""

from __future__ import annotations

import os
import time
from collections import defaultdict

import numpy as np
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.lib.packet import arp, ethernet, ether_types, ipv4, packet
from ryu.ofproto import ofproto_v1_3

from .config import CANDIDATE_PATHS, LINKS, MAX_DEMAND_MBPS, MODEL_PATH


# These values follow the link-creation order in mininet/topology.py.  Switch
# DPIDs are explicitly assigned there, making this map stable across launches.
HOST_PORT = {1: 1, 2: 1, 3: 1, 4: 1}
LINK_PORT = {
    (1, 2): 2, (2, 1): 2,
    (2, 4): 3, (4, 2): 2,
    (1, 3): 3, (3, 1): 2,
    (3, 4): 3, (4, 3): 3,
    (2, 3): 4, (3, 2): 4,
}
SWITCH_ID = {f"s{number}": number for number in range(1, 7)}


class QosPpoController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datapaths = {}
        self.port_bytes = defaultdict(int)
        self.port_utilization = defaultdict(float)
        self.last_sample_time = {}
        self.routing_mode = os.environ.get("ROUTING_MODE", "ppo").lower()
        if self.routing_mode not in {"ppo", "shortest", "minhop"}:
            raise ValueError("ROUTING_MODE must be 'ppo', 'shortest' or 'minhop'")
        self.flow_demand_mbps = float(os.environ.get("FLOW_DEMAND_MBPS", "10"))
        if not 0.0 < self.flow_demand_mbps <= MAX_DEMAND_MBPS:
            raise ValueError(f"FLOW_DEMAND_MBPS must be in (0, {MAX_DEMAND_MBPS}]")
        self.logger.info("Routing mode: %s", self.routing_mode)
        self.policy = self._load_policy() if self.routing_mode == "ppo" else None
        if self.routing_mode == "shortest":
            self.logger.info("Lowest-delay baseline selected; PPO model loading skipped")
        if self.routing_mode == "minhop":
            self.logger.info("Min-hop baseline selected; PPO model loading skipped")
        self.monitor_thread = hub.spawn(self._monitor)

    def _load_policy(self):
        try:
            from stable_baselines3 import PPO
            model = PPO.load(str(MODEL_PATH))
            self.logger.info("Loaded PPO policy from %s.zip", MODEL_PATH)
            return model
        except Exception as error:
            raise RuntimeError(f"PPO policy failed to load from {MODEL_PATH}: {error}") from error

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, event):
        datapath = event.msg.datapath
        parser, ofproto = datapath.ofproto_parser, datapath.ofproto
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        datapath.send_msg(parser.OFPFlowMod(datapath=datapath, priority=0, match=match, instructions=[parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]))

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER])
    def state_change_handler(self, event):
        self.datapaths[event.datapath.id] = event.datapath

    def _monitor(self):
        while True:
            for datapath in list(self.datapaths.values()):
                datapath.send_msg(datapath.ofproto_parser.OFPPortStatsRequest(datapath, 0, datapath.ofproto.OFPP_ANY))
            hub.sleep(2)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, event):
        dpid = event.msg.datapath.id
        now = time.monotonic()
        previous = self.last_sample_time.get(dpid)
        for stat in event.msg.body:
            if stat.port_no > event.msg.datapath.ofproto.OFPP_MAX:
                continue
            key = (dpid, stat.port_no)
            # Skip the first sample per switch so the raw counter is not read as a spike.
            if previous is not None:
                byte_delta = max(stat.tx_bytes - self.port_bytes[key], 0)
                interval = max(now - previous, 0.001)
                # Normalized utilization requires link capacity from the topology map.
                # Until that map is connected, retain a safe bounded activity score.
                self.port_utilization[key] = min((byte_delta * 8 / interval) / 100_000_000, 1.0)
            self.port_bytes[key] = stat.tx_bytes
        self.last_sample_time[dpid] = now

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, event):
        """Route h1 <-> h4 packets through the policy-selected candidate path."""
        message = event.msg
        datapath = message.datapath
        in_port = message.match["in_port"]
        parsed = packet.Packet(message.data)
        ethernet_frame = parsed.get_protocol(ethernet.ethernet)
        if ethernet_frame is None or ethernet_frame.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        ip_packet = parsed.get_protocol(ipv4.ipv4)
        arp_packet = parsed.get_protocol(arp.arp)
        source_ip = ip_packet.src if ip_packet else (arp_packet.src_ip if arp_packet else None)
        destination_ip = ip_packet.dst if ip_packet else (arp_packet.dst_ip if arp_packet else None)
        if {source_ip, destination_ip} != {"10.0.0.1", "10.0.0.4"}:
            return

        # Route creation occurs only at the ingress host-facing switch.
        expected_ingress = 1 if source_ip == "10.0.0.1" else 4
        if datapath.id != expected_ingress or in_port != HOST_PORT[expected_ingress]:
            return
        candidate = self.choose_path(self._policy_observation())
        forward = source_ip == "10.0.0.1"
        path = [SWITCH_ID[name] for name in CANDIDATE_PATHS[candidate]]
        if not forward:
            path.reverse()
        self.logger.info("%s -> %s selected candidate %s: %s", source_ip, destination_ip, candidate, path)
        self._install_path(path, source_ip, destination_ip, ethernet_frame.ethertype)

        # Forward the triggering packet immediately; later packets use the
        # installed OpenFlow rules.
        output_port = self._next_port(path, 0)
        parser = datapath.ofproto_parser
        actions = [parser.OFPActionOutput(output_port)]
        datapath.send_msg(parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=message.buffer_id,
            in_port=in_port,
            actions=actions,
            data=None if message.buffer_id != datapath.ofproto.OFP_NO_BUFFER else message.data,
        ))

    def _install_path(self, path: list[int], source_ip: str, destination_ip: str, ethertype: int) -> None:
        for index, switch_id in enumerate(path):
            datapath = self.datapaths.get(switch_id)
            if datapath is None:
                self.logger.warning("Switch %s is not connected yet", switch_id)
                return
            parser = datapath.ofproto_parser
            ofproto = datapath.ofproto
            output_port = self._next_port(path, index)
            if ethertype == ether_types.ETH_TYPE_IP:
                match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, ipv4_src=source_ip, ipv4_dst=destination_ip)
            elif ethertype == ether_types.ETH_TYPE_ARP:
                match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_ARP, arp_spa=source_ip, arp_tpa=destination_ip)
            else:
                return
            instructions = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, [parser.OFPActionOutput(output_port)])]
            datapath.send_msg(parser.OFPFlowMod(datapath=datapath, priority=100, idle_timeout=15, match=match, instructions=instructions))

    @staticmethod
    def _next_port(path: list[int], index: int) -> int:
        if index == len(path) - 1:
            return HOST_PORT[path[index]]
        return LINK_PORT[(path[index], path[index + 1])]

    def _policy_observation(self) -> np.ndarray:
        """Translate live port activity into the PPO observation format."""
        features: list[float] = []
        for path_names in CANDIDATE_PATHS:
            switch_path = [SWITCH_ID[name] for name in path_names]
            link_utilizations = [
                self.port_utilization[(switch_path[index], LINK_PORT[(switch_path[index], switch_path[index + 1])])]
                for index in range(len(switch_path) - 1)
            ]
            static = [LINKS[frozenset((left, right))] for left, right in zip(path_names, path_names[1:])]
            max_utilization = max(link_utilizations, default=0.0)
            delay_ms = sum(link[1] for link in static)
            loss_rate = sum(link[2] for link in static)
            residual_mbps = min(capacity * (1.0 - utilization) for utilization, (capacity, _, _) in zip(link_utilizations, static))
            features.extend((min(max_utilization, 1.0), min(delay_ms / 30.0, 1.0), min(loss_rate / 0.15, 1.0), min(residual_mbps / MAX_DEMAND_MBPS, 1.0)))
        features.append(self.flow_demand_mbps / MAX_DEMAND_MBPS)
        return np.asarray(features, dtype=np.float32)

    def choose_path(self, observation: np.ndarray) -> int:
        """Return candidate-path index; call this from the topology-aware flow installer."""
        if self.routing_mode == "minhop":
            return int(min(range(len(CANDIDATE_PATHS)), key=lambda i: len(CANDIDATE_PATHS[i])))
        if self.routing_mode == "shortest":
            delays = [
                sum(LINKS[frozenset((left, right))][1] for left, right in zip(path, path[1:]))
                for path in CANDIDATE_PATHS
            ]
            return int(min(range(len(delays)), key=delays.__getitem__))
        action, _ = self.policy.predict(observation.astype(np.float32), deterministic=True)
        return int(action)
