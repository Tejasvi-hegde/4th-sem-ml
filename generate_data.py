import subprocess
import time
import logging
import sys
import random
from typing import Dict, Optional
sys.path.append("/home/tejasvi/Desktop/CN complete working copy/")
from network.monitor import NetworkMonitor
from utils.influx_client import InfluxDBWriter

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("data_generation.log"),
            logging.StreamHandler()
        ]
    )

def run_command(command: str, silent: bool = False) -> bool:
    """Execute shell command with error handling"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if not silent:
            logging.debug(f"Command succeeded: {command}")
        return True
    except subprocess.CalledProcessError as e:
        if not silent:
            logging.error(f"Command failed: {command}\nError: {e.stderr}")
        return False

def check_network_setup(interface: str = "veth-client", namespace: str = "ns-client") -> bool:
    """Verify network namespace and interface exist"""
    if not run_command(f"ip netns list | grep {namespace}", silent=True):
        logging.error(f"Network namespace {namespace} does not exist")
        return False
    if not run_command(f"ip netns exec {namespace} ip link show {interface}", silent=True):
        logging.error(f"Interface {interface} does not exist in namespace {namespace}")
        return False
    return True

def set_congestion_control(cca: str, namespace: str = "ns-client",server_ns: str = "ns-server") -> bool:
    """Set TCP congestion control algorithm"""
    client_cmd = f"ip netns exec {namespace} sysctl -w net.ipv4.tcp_congestion_control={cca}"
    server_cmd = f"ip netns exec {server_ns} sysctl -w net.ipv4.tcp_congestion_control={cca}"
    client_success = run_command(client_cmd)
    server_success = run_command(server_cmd)
    if client_success and server_success:
        logging.info(f"Set CCA to {cca} on both client and server")
        return True
    else:
        logging.error(f"Failed to set CCA on both endpoints (client: {client_success}, server: {server_success})")
        return False

def configure_network(interface: str, delay_ms: int, jitter_ms: int, loss_pct: float, bandwidth_mbit: int, queue_limit: int,mtu: int = 1500) -> bool:
    """Configure network conditions using tc and netem"""
    # Reset existing configuration
    run_command(f"ip netns exec ns-client tc qdisc del dev {interface} root", silent=True)
    run_command(f"ip netns exec ns-client ip link set dev {interface} mtu {mtu}")
    # Apply netem for delay, jitter, loss
    netem_cmd = f"ip netns exec ns-client tc qdisc add dev {interface} root handle 1:0 netem delay {delay_ms}ms {jitter_ms}ms loss {loss_pct}% limit {queue_limit}"
    if not run_command(netem_cmd):
        logging.error("Failed to set netem configuration")
        return False
    
    # Configure queue length, loss, delay, etc.
    
    # Apply tbf for bandwidth
    tbf_cmd = f"ip netns exec ns-client tc qdisc add dev {interface} parent 1:1 handle 10: tbf rate {bandwidth_mbit}mbit burst 32kbit latency 50ms"
    if not run_command(tbf_cmd):
        logging.error("Failed to set tbf configuration")
        return False
    
    # Set queue limit
    queue_cmd = f"ip netns exec ns-client tc qdisc add dev {interface} parent 10:1 handle 20: netem limit {queue_limit}"
    if not run_command(queue_cmd):
        logging.error("Failed to set queue limit")
        return False
    
    # Verify configuration
    if run_command(f"ip netns exec ns-client tc qdisc show dev {interface}", silent=True):
        logging.info(f"Configured network: delay={delay_ms}ms, jitter={jitter_ms}ms, loss={loss_pct}%, bandwidth={bandwidth_mbit}mbit, queue_limit={queue_limit}")
        return True
    logging.error("Network configuration verification failed")
    return False
  

def generate_data(max_conditions: int = 100, iterations_per_condition: int = 1):
    """Generate data for all CC algorithms under various network conditions"""
    setup_logging()
    writer = InfluxDBWriter()
    
    # Test InfluxDB connection
    if not writer.test_connection():
        logging.error("Cannot connect to InfluxDB. Exiting.")
        return
    
    # Verify network setup
    if not check_network_setup():
        logging.error("Network setup incomplete. Please create ns-client, ns-server, and veth-client interfaces.")
        return
    
    monitor = NetworkMonitor()
    
    # Define CC algorithms
    ccas = [ 'westwood',  'vegas', 'hybla']
    
    # Define network condition ranges (expanded for diversity)
    conditions = [{
        'delay_ms': random.choice([10, 50, 100, 200, 500]),
        'jitter_ms': random.choice([0, 5, 10, 20, 50]),
        'loss_pct': random.choice([0, 0.5, 1, 2, 5, 10]),
        'bandwidth_mbit': random.choice([1, 5, 10, 50, 100, 500]),
        'queue_limit': random.choice([10, 50, 100, 500, 1000]),
        'mtu': random.choice([1200, 1500, 9000])
    } for _ in range(max_conditions)]
    
    try:
        
            
            for cond in conditions:
                # Configure network in ns-client
                if not configure_network(
                    interface="veth-client",
                    delay_ms=cond['delay_ms'],
                    jitter_ms=cond['jitter_ms'],
                    loss_pct=cond['loss_pct'],
                    bandwidth_mbit=cond['bandwidth_mbit'],
                    queue_limit=cond['queue_limit']
                ):
                    logging.error(f"Skipping condition {cond} due to network setup failure")
                    continue

                for cca in ccas:
                    if not set_congestion_control(cca):
                        logging.error(f"Skipping CCA {cca} due to setup failure")
                        continue
                    
                    # Wait for network to stabilize
                    time.sleep(2)
                    
                    # Collect metrics
                    for _ in range(iterations_per_condition):
                        try:
                            metrics = monitor.collect_metrics_stable(cca=cca,iterations=1, delay=1.0)
                            if metrics:
                                # Add condition metadata to metrics
                                metrics.update({
                                    'condition_delay_ms': cond['delay_ms'],
                                    'condition_jitter_ms': cond['jitter_ms'],
                                    'condition_loss_pct': cond['loss_pct'],
                                    'condition_bandwidth_mbit': cond['bandwidth_mbit'],
                                    'condition_queue_limit': cond['queue_limit']
                                })
                                # Write to InfluxDB with retry
                                for attempt in range(1):
                                    try:
                                        writer.write(metrics)
                                        logging.info(f"Stored metrics for CCA {cca} under condition {cond}")
                                        break
                                    except Exception as e:
                                        logging.warning(f"Failed to write to InfluxDB (attempt {attempt + 1}): {e}")
                                        time.sleep(0.5)
                                else:
                                    logging.error(f"Failed to write metrics for CCA {cca} after 3 attempts")
                            else:
                                logging.warning(f"No valid metrics for CCA {cca} under condition {cond}")
                        except Exception as e:
                            logging.error(f"Error collecting metrics for CCA {cca}: {e}")
    except KeyboardInterrupt:
        logging.info("Data generation interrupted by user")
    finally:
        # Reset network configuration
        run_command("tc qdisc del dev veth-client root", silent=True)
        logging.info("Data generation completed")

if __name__ == "__main__":
    generate_data(max_conditions=500, iterations_per_condition=1)