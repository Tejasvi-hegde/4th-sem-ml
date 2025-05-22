import subprocess
import time
import numpy as np
import logging
import re
from typing import Dict
from utils.influx_client import InfluxDBWriter
from typing import Optional, Dict  # Add Optional if not already present
class NetworkMonitor:
    def __init__(self):
        self.writer = InfluxDBWriter()
        self._init_logging()

    def _init_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("network_monitor.log"),
                logging.StreamHandler()
            ]
        )

    def _run_command(self, command: str) -> Optional[str]:
        """Execute shell command and return output on success, None on failure"""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                check=True, 
                capture_output=True, 
                text=True
            )
            return result.stdout.strip()  # Return the command output
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed: {e.stderr}")
            return None
 
    def _check_connectivity(self) -> bool:
        """Verify connectivity between ns-client and ns-server"""
        ping_cmd = "ip netns exec ns-client ping -c 1 10.0.0.2"
        output = self._run_command(ping_cmd)
        if output and "1 received" in output:
            return True
        logging.warning("No connectivity between ns-client and ns-server")
        return False

    def _clean_numeric(self, value: str) -> int:
        """Extract numeric value from string, removing non-digits"""
        try:
            # Extract digits using regex
            numeric = re.sub(r'[^0-9]', '', value)
            return int(numeric) if numeric else 0
        except ValueError:
            logging.warning(f"Could not parse numeric value: {value}")
            return 0

    def get_rtt(self) -> float:
        """Get Round-Trip Time using ping"""
        cmd = "ip netns exec ns-client ping -c 4 10.0.0.2 | grep 'rtt' | awk -F'/' '{print $5}'"
        output = self._run_command(cmd)
        if output is None:
            return 0.0
        try:
            return float(output)
        except ValueError:
            logging.warning(f"Could not parse RTT: {output}")
            return 0.0

    def get_jitter(self) -> float:
        """Get jitter using iperf3"""
        if not self._check_connectivity():
            return 0.0

        cmd = "ip netns exec ns-client iperf3 -c 10.0.0.2 -u -b 1M -t 5 -J"
        output = self._run_command(cmd)
        if output is None:
            return 0.0
        try:
            import json
            data = json.loads(output)
            jitter_ms = data['end']['sum']['jitter_ms']
            return float(jitter_ms) if jitter_ms else 0.0
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Failed to parse iperf3 jitter output: {e}")
            return 0.0


    def get_throughput(self) -> float:
        """Get throughput using iperf3"""
        if not self._check_connectivity():
            return 0.0

        cmd = "ip netns exec ns-client iperf3 -c 10.0.0.2 -t 5 -J"
        output = self._run_command(cmd)
        if output is None:
            return 0.0
        try:
            import json
            data = json.loads(output)
            throughput = data['end']['sum_received']['bits_per_second'] / 1e6
            return float(throughput) if throughput else 0.0
        except (json.JSONDecodeError, KeyError):
            logging.warning("Failed to parse iperf3 throughput output")
            return 0.0


    def get_goodput(self) -> float:
        """Get goodput (application-level throughput)"""
        throughput = self.get_throughput()
        loss = self.get_loss()
        return throughput * (1 - loss / 100) if throughput  else 0.0

    def get_loss(self) -> float:
        """More reliable loss measurement using ping"""
        cmd = "ip netns exec ns-client ping -c 10 10.0.0.2 | grep 'packet loss' | awk '{print $6}'"
        output = self._run_command(cmd)
        return float(output.strip('%')) if output else 0.0

    def get_bdp(self) -> float:
        """Get Bandwidth-Delay Product"""
        throughput = self.get_throughput()  # Mbps
        rtt = self.get_rtt()  # ms
        return (throughput * rtt) / 8 if throughput and rtt else 0.0

    def get_queue_length(self) -> int:
        """Get queue length from tc"""
        cmd = "ip netns exec ns-client tc -s qdisc show dev veth-client | grep 'backlog' | awk '{print $2}'"
        output = self._run_command(cmd)
        return self._clean_numeric(output)  # Handles '66b' -> 66

    def get_path_mtu(self) -> int:
        """Get Path MTU"""
        cmd = "ip netns exec ns-client ip route get 10.0.0.2 | grep -o 'mtu [0-9]*' | awk '{print $2}'"
        output = self._run_command(cmd)
        return self._clean_numeric(output) or 1500  # Default to 1500 if empty
    
    def get_bufferbloat(self) -> float:
        """Get bufferbloat (additional latency under load)"""
        baseline_rtt = self.get_rtt()
        if not self._check_connectivity():
            return 0.0
        cmd = "ip netns exec ns-client ping -c 10 10.0.0.2 -s 1400 | grep 'rtt' | awk -F'/' '{print $5}'"
        output = self._run_command(cmd)
        loaded_rtt = float(output) if output else 0.0
        bufferbloat = max(0, loaded_rtt - baseline_rtt) if loaded_rtt and baseline_rtt else 0.0
        return bufferbloat



    def get_tcp_stats(self) -> Dict:
        """Get detailed TCP statistics including retransmissions and congestion window"""
        if not self._check_connectivity():
            return {'retransmits': 0, 'tcp_window_size': 0}

        cmd = "ip netns exec ns-client ss -tin | grep -o 'retrans:[0-9]*,cwnd:[0-9]*' | head -1"
        output = self._run_command(cmd)

        retr = 0
        cwnd = 0

        if output:
            try:
                retr = int(re.search(r'retrans:(\d+)', output).group(1))
                cwnd = int(re.search(r'cwnd:(\d+)', output).group(1))
            except (AttributeError, ValueError):
                logging.warning("Failed to parse TCP stats")

        return {
            'retransmits': retr,
            'tcp_window_size': cwnd
        }



    

    def collect_all_metrics(self,cca:str) -> Dict:
        """Collect all network metrics"""
        try:
            metrics = {
                'rtt': self.get_rtt(),
                'jitter': self.get_jitter(),
                'throughput': self.get_throughput(),
                'goodput': self.get_goodput(),
                'loss': self.get_loss(),
                'bdp': self.get_bdp(),
                'queue_length': self.get_queue_length(),
                'path_mtu': self.get_path_mtu(),
                'bufferbloat': self.get_bufferbloat(),
                'current_cca': cca,
                'timestamp': time.time_ns()  # Returns nanoseconds (int) # Returns seconds (float)
            }
            return {**metrics, **self.get_tcp_stats()}
        except Exception as e:
            logging.error(f"Error collecting metrics: {e}")
            return {}

    def collect_metrics_stable(self, cca:str, iterations: int = 1, delay: float = 1.0) -> Dict:
        """Collect metrics multiple times and return averages for stability"""
        metrics_list = []
        for _ in range(iterations):
            metrics = self.collect_all_metrics(cca)
            if all(metrics.get(key) is not None for key in [
                'rtt', 'jitter', 'throughput', 'goodput', 'loss', 'bdp',
                'queue_length', 'path_mtu', 'tcp_window_size', 'bufferbloat', 'retransmits'
            ]):
                metrics_list.append(metrics)
            time.sleep(delay)
        
        if not metrics_list:
            logging.warning("No valid metrics collected")
            return {}
        
        # Compute averages for numeric fields
        avg_metrics = {
            'rtt': np.mean([m['rtt'] for m in metrics_list]),
            'jitter': np.mean([m['jitter'] for m in metrics_list]),
            'throughput': np.mean([m['throughput'] for m in metrics_list]),
            'goodput': np.mean([m['goodput'] for m in metrics_list]),
            'loss': np.mean([m['loss'] for m in metrics_list]),
            'bdp': np.mean([m['bdp'] for m in metrics_list]),
            'queue_length': np.mean([m['queue_length'] for m in metrics_list]),
            'path_mtu': np.mean([m['path_mtu'] for m in metrics_list]),
            'tcp_window_size': np.mean([m['tcp_window_size'] for m in metrics_list]),
            'bufferbloat': np.mean([m['bufferbloat'] for m in metrics_list]),
            'retransmits': np.mean([m['retransmits'] for m in metrics_list]),
            'current_cca': cca,
            'timestamp': time.time_ns()  # Returns nanoseconds (int)
        }
        self.writer.write(avg_metrics)
        return avg_metrics

if __name__ == "__main__":
    monitor = NetworkMonitor()
    print("Testing Stable Network Monitoring:")
    metrics = monitor.collect_metrics_stable(cca="cubic")
    print(metrics)