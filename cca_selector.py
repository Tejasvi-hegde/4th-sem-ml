import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from influxdb_client import InfluxDBClient
from typing import Dict, List, Tuple
import joblib  # Added for model persistence

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("cca_selector.log"),
            logging.StreamHandler()
        ]
    )

class CCADataset:
    def __init__(self):
        self.client = InfluxDBClient(
            url="http://localhost:8086",
            token="JU8UDuL3viuNAVfv6tRI_HpCXH-oO4RaF5JyOAYv7qzGuOpPNt8EWSy5blc5YuCYwEjxAI50rotWYErz2LzjyA==",
            org="network_monitor"
        )
        self.query_api = self.client.query_api()

    def fetch_data(self) -> pd.DataFrame:
        """Fetch network metrics from InfluxDB"""
        query = '''
        from(bucket: "new bucket")
          |> range(start: -30d)
          |> filter(fn: (r) => r._measurement == "network_metrics")
          |> filter(fn: (r) => r._field == "rtt" or r._field == "jitter" or
                               r._field == "throughput" or r._field == "goodput" or
                               r._field == "loss" or r._field == "bdp" or
                               r._field == "queue_length" or r._field == "path_mtu" or
                               r._field == "tcp_window_size" or r._field == "bufferbloat" or
                               r._field == "retransmits" or r._field == "condition_delay_ms" or
                               r._field == "condition_jitter_ms" or r._field == "condition_loss_pct" or
                               r._field == "condition_bandwidth_mbit" or r._field == "condition_queue_limit")
          |> pivot(rowKey:["_time", "cca"], columnKey: ["_field"], valueColumn: "_value")
          |> keep(columns: ["_time", "cca", "rtt", "jitter", "throughput", "goodput", "loss",
                           "bdp", "queue_length", "path_mtu", "tcp_window_size", "bufferbloat",
                           "retransmits", "condition_delay_ms", "condition_jitter_ms",
                           "condition_loss_pct", "condition_bandwidth_mbit", "condition_queue_limit"])
        '''
        try:
            tables = self.query_api.query(query=query)
            records = []
            for table in tables:
                for record in table.records:
                    records.append(record.values)
            df = pd.DataFrame(records)
            logging.info(f"Fetched {len(df)} records from InfluxDB")
            return df
        except Exception as e:
            logging.error(f"Error fetching data from InfluxDB: {e}")
            return pd.DataFrame()

class CCAModel:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ccas = ['bbr', 'westwood', 'reno', 'vegas', 'hybla', 'cubic']
        self.max_values = {}  # For normalization

    def normalize_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize key metrics for fair comparison"""
        metrics_to_normalize = ['throughput', 'rtt', 'jitter', 'loss', 'retransmits', 'bufferbloat']
        for metric in metrics_to_normalize:
            max_val = df[metric].max() if not df[metric].empty else 1
            self.max_values[metric] = max_val if max_val != 0 else 1
            df[metric] = df[metric] / self.max_values[metric]
        return df

    def calculate_efficiency(self, row: pd.Series, cca: str) -> float:
        """Calculate efficiency score for a CCA using a tailored equation"""
        # Extract normalized metrics
        throughput = row['throughput']
        rtt = row['rtt'] if row['rtt'] > 0 else 0.001  # Avoid division by zero
        jitter = row['jitter']
        loss = row['loss']
        retransmits = row['retransmits']
        bufferbloat = row['bufferbloat']

        # Extract condition metadata
        condition_delay = row['condition_delay_ms']
        condition_loss = row['condition_loss_pct']
        condition_bandwidth = row['condition_bandwidth_mbit']

        # Adjust weights based on network conditions
        # High latency: > 200ms
        # High loss: > 5%
        # High bandwidth: > 100 Mbps
        is_high_latency = condition_delay > 200
        is_high_loss = condition_loss > 5
        is_high_bandwidth = condition_bandwidth > 100

        # Define CCA-specific efficiency equations
        if cca == 'bbr':
            # BBR: Prioritize throughput in high-bandwidth, high-latency networks
            # Less penalty for RTT in high-latency conditions
            rtt_weight = 0.5 if is_high_latency else 1.0
            throughput_weight = 1.5 if is_high_bandwidth else 1.0
            efficiency = (throughput_weight * throughput * (1 - loss / 100)) / \
                         (rtt * rtt_weight * (1 + retransmits) * (1 + bufferbloat))
        
        elif cca == 'cubic':
            # Cubic: Focus on throughput in high-speed networks
            # Moderate penalty for RTT
            throughput_weight = 1.5 if is_high_bandwidth else 1.0
            efficiency = (throughput_weight * throughput * (1 - loss / 100)) / \
                         (rtt * (1 + retransmits) * (1 + bufferbloat))
        
        elif cca == 'reno':
            # Reno: Performs well in low-loss environments
            # Heavy penalty for loss
            loss_weight = 2.0 if is_high_loss else 1.0
            efficiency = (throughput * (1 - loss_weight * loss / 100)) / \
                         (rtt * (1 + retransmits) * (1 + bufferbloat))
        
        elif cca == 'westwood':
            # Westwood: Optimized for high-loss wireless networks
            # Reduce loss penalty in high-loss conditions
            loss_weight = 0.5 if is_high_loss else 1.0
            efficiency = (throughput * (1 - loss_weight * loss / 100)) / \
                         (rtt * (1 + retransmits) * (1 + bufferbloat))
        
        elif cca == 'vegas':
            # Vegas: Delay-based, prioritize low RTT
            # Heavy penalty for RTT
            rtt_weight = 2.0
            efficiency = (throughput * (1 - loss / 100)) / \
                         (rtt * rtt_weight * (1 + retransmits) * (1 + bufferbloat))
        
        elif cca == 'hybla':
            # Hybla: Designed for satellite links (high latency)
            # Reduce RTT penalty in high-latency conditions
            rtt_weight = 0.5 if is_high_latency else 1.0
            efficiency = (throughput * (1 - loss / 100)) / \
                         (rtt * rtt_weight * (1 + retransmits) * (1 + bufferbloat))
        
        else:
            # Default equation for unknown CCAs
            efficiency = (throughput * (1 - loss / 100)) / \
                         (rtt * (1 + retransmits) * (1 + bufferbloat))
        
        return efficiency if efficiency > 0 else 0.0

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare the dataset by selecting the optimal CCA for each condition"""
        if df.empty:
            logging.error("No data to process")
            return pd.DataFrame(), pd.Series()

        # Normalize metrics
        df = self.normalize_metrics(df)

        # Group by network condition
        condition_columns = ['condition_delay_ms', 'condition_jitter_ms', 'condition_loss_pct',
                            'condition_bandwidth_mbit', 'condition_queue_limit']
        grouped = df.groupby(condition_columns)

        # Select the optimal CCA for each condition
        optimal_records = []
        for condition, group in grouped:
            efficiencies = {}
            for cca in self.ccas:
                cca_group = group[group['cca'] == cca]
                if not cca_group.empty:
                    efficiency = self.calculate_efficiency(cca_group.iloc[0], cca)
                    efficiencies[cca] = efficiency
            if efficiencies:
                optimal_cca = max(efficiencies, key=efficiencies.get)
                optimal_row = group[group['cca'] == optimal_cca].iloc[0].copy()
                optimal_row['optimal_cca'] = optimal_cca
                optimal_records.append(optimal_row)

        if not optimal_records:
            logging.error("No optimal CCAs selected")
            return pd.DataFrame(), pd.Series()

        optimal_df = pd.DataFrame(optimal_records)
        logging.info(f"Optimal CCA distribution:\n{optimal_df['optimal_cca'].value_counts()}")

        # Prepare features and labels
        feature_columns = ['rtt', 'jitter', 'throughput', 'loss', 'queue_length',
                          'path_mtu', 'tcp_window_size', 'bufferbloat', 'retransmits',
                          'condition_delay_ms', 'condition_jitter_ms', 'condition_loss_pct',
                          'condition_bandwidth_mbit', 'condition_queue_limit']
        X = optimal_df[feature_columns]
        y = optimal_df['optimal_cca']
        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the CCA selection model"""
        if X.empty or y.empty:
            logging.error("Cannot train model: Empty dataset")
            return

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        report = classification_report(
            y_test, y_pred, target_names=self.label_encoder.classes_, zero_division=0
        )
        logging.info(f"Model Evaluation:\n{report}")

    def predict(self, metrics: Dict) -> str:
        """Predict the optimal CCA for given metrics"""
        feature_columns = ['rtt', 'jitter', 'throughput', 'loss', 'queue_length',
                          'path_mtu', 'tcp_window_size', 'bufferbloat', 'retransmits',
                          'condition_delay_ms', 'condition_jitter_ms', 'condition_loss_pct',
                          'condition_bandwidth_mbit', 'condition_queue_limit']
        input_df = pd.DataFrame([metrics], columns=feature_columns)
        
        # Normalize input metrics
        for metric in ['throughput', 'rtt', 'jitter', 'loss', 'retransmits', 'bufferbloat']:
            input_df[metric] = input_df[metric] / self.max_values.get(metric, 1)
        
        # Scale features
        input_scaled = self.scaler.transform(input_df)
        
        # Predict
        prediction = self.model.predict(input_scaled)
        return self.label_encoder.inverse_transform(prediction)[0]

class CCASelector:
    def __init__(self):
        self.model = CCAModel()
        self.dataset = CCADataset()
        self.model_file = "cca_model.joblib"
        self.scaler_file = "cca_scaler.joblib"
        self.encoder_file = "cca_encoder.joblib"
        self.max_values_file = "cca_max_values.joblib"

    def train_and_save(self):
        """Fetch data, train the model, and save it"""
        df = self.dataset.fetch_data()
        X, y = self.model.prepare_data(df)
        self.model.train(X, y)
        
        # Save the model, scaler, encoder, and max_values
        joblib.dump(self.model.model, self.model_file)
        joblib.dump(self.model.scaler, self.scaler_file)
        joblib.dump(self.model.label_encoder, self.encoder_file)
        joblib.dump(self.model.max_values, self.max_values_file)
        logging.info("Model and components saved successfully")

    def load_model(self):
        """Load the trained model and components"""
        try:
            self.model.model = joblib.load(self.model_file)
            self.model.scaler = joblib.load(self.scaler_file)
            self.model.label_encoder = joblib.load(self.encoder_file)
            self.model.max_values = joblib.load(self.max_values_file)
            logging.info("Model and components loaded successfully")
        except FileNotFoundError as e:
            logging.error(f"Model file not found: {e}")
            raise
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def predict_cca(self, metrics: Dict) -> str:
        """Predict the optimal CCA for given metrics"""
        return self.model.predict(metrics)

def main():
    setup_logging()
    selector = CCASelector()
    selector.train_and_save()

if __name__ == "__main__":
    main()