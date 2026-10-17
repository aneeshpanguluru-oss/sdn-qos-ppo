"""Six-switch redundant-path topology for the QoS-aware routing experiment."""

from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.topo import Topo


class QosMeshTopo(Topo):
    def build(self):
        switches = {
            name: self.addSwitch(name, dpid=f"{index:016x}", protocols="OpenFlow13")
            for index, name in enumerate(("s1", "s2", "s3", "s4", "s5", "s6"), start=1)
        }
        for index, switch in enumerate(("s1", "s2", "s3", "s4"), start=1):
            host = self.addHost(f"h{index}", ip=f"10.0.0.{index}/24")
            self.addLink(host, switches[switch], cls=TCLink, bw=100, delay="1ms")
        # Two primary routes from s1 to s4 plus cross links for rerouting.
        for left, right, bw, delay in (
            ("s1", "s2", 30, "4ms"), ("s2", "s4", 25, "6ms"),
            ("s1", "s3", 20, "8ms"), ("s3", "s4", 30, "3ms"),
            ("s2", "s3", 15, "2ms"), ("s2", "s5", 20, "5ms"),
            ("s5", "s6", 20, "3ms"), ("s6", "s4", 20, "5ms"),
        ):
            self.addLink(switches[left], switches[right], cls=TCLink, bw=bw, delay=delay, loss=0.2)


def run():
    net = Mininet(topo=QosMeshTopo(), controller=None, switch=OVSSwitch, link=TCLink, autoSetMacs=True)
    net.addController(RemoteController("ryu", ip="127.0.0.1", port=6633))
    net.start()
    print("Topology started. Use 'links', 'pingall', or iperf3 from the Mininet CLI.")
    CLI(net)
    net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    run()
