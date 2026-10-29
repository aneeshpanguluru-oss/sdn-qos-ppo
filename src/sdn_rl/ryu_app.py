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

from .config import CANDIDATE_PATHS, LINKS, MAX_DEMAND_MBPS, MODEL_PATH, PREDICTIVE_MODEL_PATH, get_port_capacity_bps


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
# Maps a physical switch port to its inter-switch link.  This lets OpenFlow
# port-status events from Mininet become policy-visible link failures.
PORT_TO_LINK = {
    (1, 2): frozenset(("s1", "s2")), (2, 2): frozenset(("s1", "s2")),
    (2, 3): frozenset(("s2", "s4")), (4, 2): frozenset(("s2", "s4")),
    (1, 3): frozenset(("s1", "s3")), (3, 2): frozenset(("s1", "s3")),
    (3, 3): frozenset(("s3", "s4")), (4, 3): frozenset(("s3", "s4")),
    (2, 4): frozenset(("s2", "s3")), (3, 4): frozenset(("s2", "s3")),
}


class QosPpoController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datapaths = {}
        self.port_bytes = defaultdict(int)
        self.port_utilization = defaultdict(float)
        self.previous_port_utilization = defaultdict(float)
        self.last_sample_time = {}
        self.failed_links: set[frozenset[str]] = set()
        self.last_logged_candidate: dict[tuple[str, str], int] = {}
        self.routing_mode = os.environ.get("ROUTING_MODE", "ppo").lower()
        if self.routing_mode not in {"ppo", "predictive", "shortest", "minhop"}:
            raise ValueError("ROUTING_MODE must be 'ppo', 'predictive', 'shortest' or 'minhop'")
        self.flow_demand_mbps = float(os.environ.get("FLOW_DEMAND_MBPS", "10"))
        if not 0.0 < self.flow_demand_mbps <= MAX_DEMAND_MBPS:
            raise ValueError(f"FLOW_DEMAND_MBPS must be in (0, {MAX_DEMAND_MBPS}]")
        # A fixed baseline must not be given the PPO controller's recovery
        # mechanism; otherwise the comparison is unfair. PPO modes recover
        # by default, and can still be deliberately disabled for an ablation.
        requested_recovery = os.environ.get("RECOVERY_ENABLED", "1") not in {"0", "false", "False"}
        self.recovery_enabled = self.routing_mode in {"ppo", "predictive"} and requested_recovery
        self.logger.info("Routing mode: %s", self.routing_mode)
        self.policy = self._load_policy() if self.routing_mode in {"ppo", "predictive"} else None
        if self.routing_mode == "shortest":
            self.logger.info("Lowest-delay baseline selected; PPO model loading skipped")
        if self.routing_mode == "minhop":
            self.logger.info("Min-hop baseline selected; PPO model loading skipped")
        self.monitor_thread = hub.spawn(self._monitor)

    def _load_policy(self):
        try:
            from stable_baselines3 import PPO
            model_path = PREDICTIVE_MODEL_PATH if self.routing_mode == "predictive" else MODEL_PATH
            model = PPO.load(str(model_path))
            self.logger.info("Loaded PPO policy from %s.zip", model_path)
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
                self.previous_port_utilization[key] = self.port_utilization[key]
                capacity_bps = get_port_capacity_bps(dpid, stat.port_no)
                self.port_utilization[key] = min((byte_delta * 8 / interval) / capacity_bps, 1.0)
            self.port_bytes[key] = stat.tx_bytes
        self.last_sample_time[dpid] = now

    @set_ev_cls(ofp_event.EventOFPPortStatus, MAIN_DISPATCHER)
    def port_status_handler(self, event):
        """Let PPO modes recover when Mininet reports an inter-switch port down."""
        datapath = event.msg.datapath
        descriptor = event.msg.desc
        link = PORT_TO_LINK.get((datapath.id, descriptor.port_no))
        if link is None:
            return
        is_down = bool(descriptor.state & datapath.ofproto.OFPPS_LINK_DOWN)
        if is_down:
            if link not in self.failed_links:
                self.failed_links.add(link)
                if self.recovery_enabled:
                    self.logger.warning("Link failure detected: %s", sorted(link))
                    self._remove_managed_flows()
                    self._install_recovery_routes()
        elif link in self.failed_links:
            self.failed_links.remove(link)

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
        flow_key = (source_ip, destination_ip)
        if self.last_logged_candidate.get(flow_key) != candidate:
            self.logger.info("%s -> %s selected candidate %s: %s", source_ip, destination_ip, candidate, path)
            self.last_logged_candidate[flow_key] = candidate
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

    def _remove_managed_flows(self) -> None:
        """Remove only this app's h1-to-h4 IP and ARP flows, not table-miss rules."""
        managed_pairs = (("10.0.0.1", "10.0.0.4"), ("10.0.0.4", "10.0.0.1"))
        for datapath in self.datapaths.values():
            parser, ofproto = datapath.ofproto_parser, datapath.ofproto
            for source_ip, destination_ip in managed_pairs:
                for ethertype, fields in (
                    (ether_types.ETH_TYPE_IP, {"ipv4_src": source_ip, "ipv4_dst": destination_ip}),
                    (ether_types.ETH_TYPE_ARP, {"arp_spa": source_ip, "arp_tpa": destination_ip}),
                ):
                    match = parser.OFPMatch(eth_type=ethertype, **fields)
                    datapath.send_msg(parser.OFPFlowMod(
                        datapath=datapath,
                        command=ofproto.OFPFC_DELETE,
                        out_port=ofproto.OFPP_ANY,
                        out_group=ofproto.OFPG_ANY,
                        priority=100,
                        match=match,
                    ))

    def _install_recovery_routes(self) -> None:
        """Install a fresh bidirectional route immediately after a link failure."""
        if not all(switch_id in self.datapaths for switch_id in (1, 2, 3, 4)):
            self.logger.warning("Recovery deferred because not all experiment switches are connected")
            return
        candidate = self.choose_path(self._policy_observation())
        forward_path = [SWITCH_ID[name] for name in CANDIDATE_PATHS[candidate]]
        reverse_path = list(reversed(forward_path))
        self.logger.warning("Recovery selected candidate %s: %s", candidate, forward_path)
        for ethertype in (ether_types.ETH_TYPE_IP, ether_types.ETH_TYPE_ARP):
            self._install_path(forward_path, "10.0.0.1", "10.0.0.4", ethertype)
            self._install_path(reverse_path, "10.0.0.4", "10.0.0.1", ethertype)

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
            path_links = [frozenset((left, right)) for left, right in zip(path_names, path_names[1:])]
            link_utilizations = [
                1.0 if path_links[index] in self.failed_links else self.port_utilization[(switch_path[index], LINK_PORT[(switch_path[index], switch_path[index + 1])])]
                for index in range(len(switch_path) - 1)
            ]
            static = [LINKS[frozenset((left, right))] for left, right in zip(path_names, path_names[1:])]
            max_utilization = max(link_utilizations, default=0.0)
            delay_ms = sum(link[1] for link in static)
            loss_rate = sum(1.0 if path_links[index] in self.failed_links else base_loss + max(utilization - 0.75, 0.0) * 0.12 for index, (utilization, (_, _, base_loss)) in enumerate(zip(link_utilizations, static)))
            residual_mbps = min(capacity * (1.0 - utilization) for utilization, (capacity, _, _) in zip(link_utilizations, static))
            features.extend((min(max_utilization, 1.0), min(delay_ms / 30.0, 1.0), min(loss_rate / 0.15, 1.0), min(residual_mbps / MAX_DEMAND_MBPS, 1.0)))
            if self.routing_mode == "predictive":
                previous = max(
                    (self.previous_port_utilization[(switch_path[index], LINK_PORT[(switch_path[index], switch_path[index + 1])])] for index in range(len(switch_path) - 1)),
                    default=0.0,
                )
                features.append(float(np.clip(max_utilization + (max_utilization - previous), 0.0, 1.0)))
        features.append(self.flow_demand_mbps / MAX_DEMAND_MBPS)
        return np.asarray(features, dtype=np.float32)

    def choose_path(self, observation: np.ndarray) -> int:
        """Return candidate-path index; call this from the topology-aware flow installer."""
        # Fixed baselines intentionally remain fixed after a fault. They are
        # therefore true non-recovering references for the PPO comparison.
        if self.routing_mode == "minhop":
            return int(min(range(len(CANDIDATE_PATHS)), key=lambda i: len(CANDIDATE_PATHS[i])))
        if self.routing_mode == "shortest":
            delays = [
                sum(LINKS[frozenset((left, right))][1] for left, right in zip(path, path[1:]))
                for path in CANDIDATE_PATHS
            ]
            return int(min(range(len(delays)), key=delays.__getitem__))

        # PPO modes receive the failed-link mask and may choose a safe route.
        available = [index for index, path in enumerate(CANDIDATE_PATHS) if all(frozenset((left, right)) not in self.failed_links for left, right in zip(path, path[1:]))]
        if not available:
            raise RuntimeError("No viable candidate path remains after link failure")
        action, _ = self.policy.predict(observation.astype(np.float32), deterministic=True)
        if int(action) in available:
            return int(action)
        # Safety mask: a trained policy may still select a failed candidate;
        # select the least-utilized viable path rather than black-holing traffic.
        width = 5 if self.routing_mode == "predictive" else 4
        return int(min(available, key=lambda index: (float(observation[index * width]), index)))
