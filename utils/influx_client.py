from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import logging
import time

class InfluxDBWriter:
    def __init__(self):
        self.client = InfluxDBClient(
            url="http://localhost:8086",
            token="JU8UDuL3viuNAVfv6tRI_HpCXH-oO4RaF5JyOAYv7qzGuOpPNt8EWSy5blc5YuCYwEjxAI50rotWYErz2LzjyA==",
            org="network_monitor"
        )
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self._init_logging()
        self.test_connection()

    def _init_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("influx_client.log"),
                logging.StreamHandler()
            ]
        )

    def test_connection(self):
        """Test InfluxDB connection"""
        try:
            buckets_api = self.client.buckets_api()
            buckets = buckets_api.find_buckets().buckets
            logging.info("Connected to InfluxDB. Available buckets:")
            for bucket in buckets:
                logging.info(f"- {bucket.name}")
            return True
        except Exception as e:
            logging.error(f"Connection failed: {e}")
            return False

    def write(self, metrics):
        """Write metrics to InfluxDB"""
        try:
            timestamp = metrics.get("timestamp")
            logging.info(f"Metrics to be written: {metrics}") 
            if timestamp is not None:
                if isinstance(timestamp, float):
                    timestamp = int(timestamp * 1e9)
                elif isinstance(timestamp, int) and timestamp < 1e18:
                    timestamp = timestamp * 1e9

            point = Point("network_metrics") \
                .tag("cca", metrics.get("current_cca", "unknown")) \
                .field("rtt", float(metrics.get("rtt", 0.0))) \
                .field("jitter", float(metrics.get("jitter", 0.0))) \
                .field("throughput", float(metrics.get("throughput", 0.0))) \
                .field("goodput", float(metrics.get("goodput", 0.0))) \
                .field("loss", float(metrics.get("loss", 0.0))) \
                .field("bdp", float(metrics.get("bdp", 0.0))) \
                .field("queue_length", int(metrics.get("queue_length", 0))) \
                .field("retransmits", int(metrics.get("retransmits", 0))) \
                .field("tcp_window_size", int(metrics.get("tcp_window_size", 0))) \
                .field("path_mtu", int(metrics.get("path_mtu", 0))) \
                .field("bufferbloat", float(metrics.get("bufferbloat", 0.0))) \
                .field("condition_delay_ms", float(metrics.get("condition_delay_ms", 0.0))) \
                .field("condition_jitter_ms", float(metrics.get("condition_jitter_ms", 0.0))) \
                .field("condition_loss_pct", float(metrics.get("condition_loss_pct", 0.0))) \
                .field("condition_bandwidth_mbit", float(metrics.get("condition_bandwidth_mbit", 0.0))) \
                .field("condition_queue_limit", int(metrics.get("condition_queue_limit", 0))) \
                .time(timestamp if timestamp is not None else time.time_ns())

            self.write_api.write(
                bucket="new bucket",
                org="network_monitor",
                record=point
            )
            logging.info("Successfully wrote metrics to InfluxDB")
        except Exception as e:
            logging.error(f"Error writing to InfluxDB: {e}")

    def __del__(self):
        """Cleanup InfluxDB client"""
        try:
            self.client.close()
            logging.info("InfluxDB client closed")
        except Exception as e:
            logging.error(f"Error closing InfluxDB client: {e}")

if __name__ == "__main__":
    writer = InfluxDBWriter()
    if writer.test_connection():
        test_metrics = {
            'rtt': 45.6,
            'jitter': 2.5,
            'throughput': 88.2,
            'goodput': 85.0,
            'loss': 1.7,
            'bdp': 4000.0,
            'queue_length': 10,
            'path_mtu': 1500,
            'tcp_window_size': 65535,
            'bufferbloat': 20.0,
            'retransmits': 2,
            'current_cca': 'cubic',
            'timestamp': time.time()
        }
        writer.write(test_metrics)