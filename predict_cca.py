import subprocess
import re
import time
import logging
import sys
sys.path.append("/home/tejasvi/Desktop/CN complete working copy/")
import numpy as np
import pandas as pd
from cca_selector import CCASelector
from typing import Dict, Optional

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("predict_cca.log"),
            logging.StreamHandler()
        ]
    )

def run_command(command: str) -> Optional[str]:
    """Execute shell command and return output on success, None on failure"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else "No error message available"
        logging.error(f"Command failed: {command}\nError: {error_msg}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error running command: {command}\nError: {e}")
        return None

def get_rtt_and_loss() -> tuple[float, float]:
    """Get RTT and packet loss using ping from ns-client to ns-server"""
    cmd = "ip netns exec ns-client ping -c 4 10.0.0.2"
    output = run_command(cmd)
    if not output:
        return 0.0, 0.0
    
    rtt = 0.0
    loss = 0.0
    for line in output.splitlines():
        if "rtt min/avg/max/mdev" in line:
            rtt = float(line.split(" = ")[1].split("/")[1])  # avg RTT in ms
        if "packet loss" in line:
            loss = float(line.split("%")[0].split()[-1])
    
    return rtt, loss

def get_throughput_and_jitter(max_retries: int = 3) -> tuple[float, float]:
    """Get throughput and jitter using iperf3 with retries"""
    throughput = 0.0
    jitter = 0.0
    
    # Ensure no existing iperf3 processes
    
    # Start iperf3 server in background
    server_process = subprocess.Popen(
        "ip netns exec ns-server iperf3 -s",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(2)  # Give server time to start

    # Retry iperf3 client
    for attempt in range(max_retries):
        cmd = "ip netns exec ns-client iperf3 -c 10.0.0.2 -t 2 -J"
        output = run_command(cmd)
        if output:
            try:
                import json
                data = json.loads(output)
                throughput = data['end']['sum_received']['bits_per_second'] / 1e6  # Mbps
                jitter = data['end']['sum']['jitter_ms']  # ms
                break
            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Attempt {attempt + 1}: Failed to parse iperf3 output: {e}")
                time.sleep(1)
        else:
            logging.warning(f"Attempt {attempt + 1}: iperf3 command failed")
            time.sleep(1)
    
    # Clean up server process

    
    return throughput, jitter

def collect_metrics() -> Dict:
    """Collect network metrics and estimate network conditions"""
    rtt, loss = get_rtt_and_loss()
    throughput, jitter = get_throughput_and_jitter()
    
    # Use defaults if metrics are invalid
    rtt = rtt if rtt > 0 else 1.0  # Avoid zero RTT
    throughput = throughput if throughput > 0 else 10.0  # Default bandwidth
    jitter = jitter if jitter >= 0 else 0.0
    loss = loss if loss >= 0 else 0.0
    
    metrics = {
        'rtt': rtt,
        'jitter': jitter,
        'throughput': throughput,
        'loss': loss,
        'queue_length': 100.0,  # Default value
        'path_mtu': 1500.0,  # Default MTU
        'tcp_window_size': 65535.0,  # Default TCP window size
        'bufferbloat': 0.0,  # Default
        'retransmits': 0.0,  # Default
        'condition_delay_ms': rtt,
        'condition_jitter_ms': jitter,
        'condition_loss_pct': loss,
        'condition_bandwidth_mbit': throughput,
        'condition_queue_limit': 100
    }
    logging.info(f"Collected metrics: {metrics}")
    return metrics

def set_congestion_control(cca: str, namespace: str = "ns-client", server_ns: str = "ns-server") -> bool:
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

def main():
    """Main function to collect metrics, predict, and apply CCA"""
    setup_logging()
    
    # Collect metrics
    network_conditions = collect_metrics()
    if not network_conditions:
        logging.error("Failed to collect metrics. Exiting.")
        return
    
    # Initialize CCASelector and load the model
    selector = CCASelector()
    try:
        selector.load_model()
    except Exception as e:
        logging.error("Model not found or failed to load. Please run cca_selector.py to train the model first.")
        return
    
    # Predict the best CCA
    try:
        recommended_cca = selector.predict_cca(network_conditions)
        print(f"Recommended CCA for the current network conditions: {recommended_cca}")
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return
    
    # Apply the recommended CCA
    if set_congestion_control(recommended_cca):
        print(f"Successfully applied CCA: {recommended_cca}")
    else:
        print("Failed to apply the recommended CCA")

if __name__ == "__main__":
    main()