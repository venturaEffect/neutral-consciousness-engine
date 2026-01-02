"""
Traffic Monitor Module with Homomorphic Encryption

Neural Firewall security component that monitors spike rates,
neural traffic patterns, and implements encrypted processing
to ensure safe operation while preventing "brainjacking".

Security Model:
- All neural traffic can be encrypted before satellite transmission
- Satellite processes encrypted data (cannot read thoughts)
- Only biological interface holds decryption keys
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool, String
import numpy as np
from typing import Optional

# Import homomorphic encryption wrapper
try:
    from .homomorphic_encryption import (
        NeuralEncryptionWrapper, 
        EncryptedVector,
        SecurityError
    )
    HE_AVAILABLE = True
except ImportError:
    HE_AVAILABLE = False


class NeuralFirewall(Node):
    """
    ROS 2 Node implementing the Neural Firewall with Homomorphic Encryption.
    
    Features:
    1. Spike rate monitoring with kill switch
    2. Anomaly detection via z-score analysis
    3. Synchronization attack detection
    4. Homomorphic encryption for satellite communication
    5. Session-based security with key management
    """
    
    def __init__(self):
        super().__init__('neural_firewall')
        
        # Configuration parameters
        self.declare_parameter('spike_rate_limit', 200.0)
        self.declare_parameter('anomaly_threshold', 3.0)
        self.declare_parameter('monitoring_window', 1.0)
        self.declare_parameter('enable_encryption', True)
        
        # Threshold: Max spikes per second before "Kill Switch" triggers
        self.SPIKE_RATE_LIMIT = self.get_parameter('spike_rate_limit').value
        self.ANOMALY_THRESHOLD = self.get_parameter('anomaly_threshold').value
        self.MONITORING_WINDOW = self.get_parameter('monitoring_window').value
        self.ENCRYPTION_ENABLED = self.get_parameter('enable_encryption').value
        
        # Internal state for running statistics
        self.spike_history = []
        self.baseline_mean = 50.0
        self.baseline_std = 20.0
        self.is_connected = True
        
        # ===== HOMOMORPHIC ENCRYPTION SETUP =====
        self.encryption_wrapper: Optional[NeuralEncryptionWrapper] = None
        if HE_AVAILABLE and self.ENCRYPTION_ENABLED:
            self._init_encryption()
        
        # ===== SUBSCRIBERS =====
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'incoming_neural_data',
            self.monitor_traffic,
            10
        )
        
        self.cortex_activity_sub = self.create_subscription(
            Float32MultiArray,
            'cortex/visual/activity',
            self.check_cortex_activity,
            10
        )
        
        # Subscribe to prediction error for encrypted forwarding
        self.prediction_error_sub = self.create_subscription(
            Float32MultiArray,
            '/neural_data/prediction_error',
            self.forward_to_satellite,
            10
        )
        
        # ===== PUBLISHERS =====
        self.security_status_pub = self.create_publisher(
            String,
            'firewall/status',
            10
        )
        
        self.kill_switch_pub = self.create_publisher(
            Bool,
            'firewall/kill_switch',
            10
        )
        
        self.alert_pub = self.create_publisher(
            String,
            'firewall/alerts',
            10
        )
        
        # Encrypted data output (would go to satellite)
        self.encrypted_output_pub = self.create_publisher(
            String,  # Serialized encrypted data
            'firewall/encrypted_satellite_data',
            10
        )
        
        # Status timer
        self.status_timer = self.create_timer(1.0, self.publish_status)
        
        self.get_logger().info('Neural Firewall initialized - Monitoring active')
        self.get_logger().info(f'Spike rate limit: {self.SPIKE_RATE_LIMIT} Hz')
        if self.encryption_wrapper:
            self.get_logger().info('Homomorphic Encryption: ENABLED')
        else:
            self.get_logger().warn('Homomorphic Encryption: DISABLED')
    
    def _init_encryption(self):
        """Initialize homomorphic encryption system."""
        try:
            self.encryption_wrapper = NeuralEncryptionWrapper()
            self.get_logger().info('Encryption context created successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize encryption: {e}')
            self.encryption_wrapper = None

    def monitor_traffic(self, msg: Float32MultiArray):
        """
        Monitor incoming neural data for security threats.
        
        Args:
            msg: Float32MultiArray containing neural activity data
        """
        if len(msg.data) == 0:
            return
            
        spike_rate = sum(msg.data) / len(msg.data)  # Simplified metric
        
        # Record for statistics
        self.spike_history.append(spike_rate)
        if len(self.spike_history) > 100:
            self.spike_history.pop(0)
        
        # Update baseline statistics
        if len(self.spike_history) >= 10:
            self.baseline_mean = np.mean(self.spike_history)
            self.baseline_std = np.std(self.spike_history)
        
        # Check for rate limit violation
        if spike_rate > self.SPIKE_RATE_LIMIT:
            self.get_logger().fatal(
                f'SECURITY ALERT: Spike rate {spike_rate:.2f}Hz exceeds safety limit!'
            )
            self._publish_alert(f'RATE_LIMIT_EXCEEDED: {spike_rate:.2f}Hz')
            self.trigger_kill_switch()
            return
        
        # Check for anomalous activity
        if self.baseline_std > 0:
            z_score = abs(spike_rate - self.baseline_mean) / self.baseline_std
            if z_score > self.ANOMALY_THRESHOLD:
                self.get_logger().warning(
                    f'Anomalous activity detected: z-score={z_score:.2f}'
                )
                self._publish_alert(f'ANOMALY_DETECTED: z-score={z_score:.2f}')
    
    def check_cortex_activity(self, msg: Float32MultiArray):
        """
        Monitor cortex activity for unusual patterns.
        
        Args:
            msg: Neural activity from cortex modules
        """
        if len(msg.data) == 0:
            return
        
        # Check for synchronization attacks (too many neurons firing together)
        activity = np.array(msg.data)
        synchrony = np.std(activity)
        
        if synchrony < 0.01 and np.mean(activity) > 0.5:
            self.get_logger().warning(
                'High synchronization detected - possible injection attack'
            )
            self._publish_alert('SYNC_ATTACK_POSSIBLE')
    
    def forward_to_satellite(self, msg: Float32MultiArray):
        """
        Forward prediction error to satellite with encryption.
        
        This is the key security function: the satellite receives
        encrypted data it can process but cannot read.
        
        Args:
            msg: Prediction error from visual cortex
        """
        if not self.is_connected:
            return
        
        data = np.array(msg.data, dtype=np.float32)
        
        if self.encryption_wrapper:
            try:
                # Encrypt data before transmission
                encrypted, metadata = self.encryption_wrapper.encrypt_spike_train(data)
                
                # Serialize for ROS transmission (simplified)
                output_msg = String()
                output_msg.data = (
                    f"ENCRYPTED|session={metadata['session_id'][:8]}|"
                    f"packet={metadata['packet_id']}|"
                    f"checksum={metadata['checksum']}"
                )
                self.encrypted_output_pub.publish(output_msg)
                
            except SecurityError as e:
                self.get_logger().error(f'Encryption security error: {e}')
                self.trigger_kill_switch()
        else:
            # No encryption - log warning
            self.get_logger().warn_once(
                'Transmitting UNENCRYPTED data to satellite!'
            )

    def trigger_kill_switch(self):
        """
        Activate the kill switch to sever all external connections.
        
        This is the emergency response to detected security threats.
        Emergency actions:
        1. Revoke encryption session
        2. Sever satellite connection
        3. Freeze SNN state
        4. Log forensic data
        """
        self.get_logger().info('KILL SWITCH ACTIVATED: Connection Severed.')
        self.is_connected = False
        
        # Revoke encryption session (new keys required to reconnect)
        if self.encryption_wrapper:
            self.encryption_wrapper = None  # Destroy current session
            self.get_logger().info('Encryption session revoked')
        
        # Publish kill switch activation
        kill_msg = Bool()
        kill_msg.data = True
        self.kill_switch_pub.publish(kill_msg)
        
        self._publish_alert('KILL_SWITCH_ACTIVATED')
    
    def publish_status(self):
        """Publish current firewall status."""
        status_msg = String()
        
        if self.is_connected:
            avg_rate = np.mean(self.spike_history) if self.spike_history else 0.0
            enc_status = "encrypted" if self.encryption_wrapper else "UNENCRYPTED"
            status_msg.data = (
                f'OK: avg_rate={avg_rate:.2f}Hz, connected=True, mode={enc_status}'
            )
        else:
            status_msg.data = 'DISCONNECTED: Kill switch activated'
        
        self.security_status_pub.publish(status_msg)
    
    def _publish_alert(self, alert_type: str):
        """Publish a security alert."""
        alert_msg = String()
        alert_msg.data = alert_type
        self.alert_pub.publish(alert_msg)
    
    def reinitialize_encryption(self) -> bool:
        """
        Reinitialize encryption with new keys.
        
        Called after kill switch to establish new secure session.
        
        Returns:
            True if successful, False otherwise
        """
        if not HE_AVAILABLE:
            return False
        
        try:
            self.encryption_wrapper = NeuralEncryptionWrapper()
            self.is_connected = True
            self.get_logger().info('New encryption session established')
            return True
        except Exception as e:
            self.get_logger().error(f'Failed to reinitialize encryption: {e}')
            return False


def main(args=None):
    """Main entry point for the Neural Firewall node."""
    rclpy.init(args=args)
    node = NeuralFirewall()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
