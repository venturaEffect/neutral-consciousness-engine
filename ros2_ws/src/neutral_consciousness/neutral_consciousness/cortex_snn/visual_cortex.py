"""
Visual Cortex Module - Predictive Coding Implementation

Implements the Generative Model using Nengo Spiking Neural Network.
This module demonstrates that an SNN can perform Predictive Coding,
the core of Watanabe's consciousness transfer theory.

Scientific Context:
This simulates the brain's energy-efficient coding. If the prediction error
is zero, the 'conscious' system perfectly understands reality, minimizing
bandwidth for the satellite link.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
import numpy as np

try:
    import nengo
    NENGO_AVAILABLE = True
except ImportError:
    NENGO_AVAILABLE = False
    print("Warning: Nengo not installed. Running in stub mode.")


class VisualCortexNode(Node):
    """
    ROS 2 Node implementing the Visual Cortex with Predictive Coding.
    
    Architecture (Generative Loop):
        Input -> Ensemble -> Prediction
                     ^           |
                     |___________|  (Feedback)
        
        Error = Input - Prediction
    
    The prediction error is published for downstream processing and
    represents the "surprise" signal - what the brain didn't expect.
    """
    
    # Dimensions for the encoded representation
    INPUT_DIM = 64
    
    def __init__(self):
        super().__init__('visual_cortex')
        
        # Configuration parameters
        self.declare_parameter('n_neurons', 1000)
        self.declare_parameter('simulation_dt', 0.001)
        self.declare_parameter('learning_rate', 1e-4)
        
        self.n_neurons = self.get_parameter('n_neurons').value
        self.simulation_dt = self.get_parameter('simulation_dt').value
        self.learning_rate = self.get_parameter('learning_rate').value
        
        # State variables for predictive coding
        self.current_input = np.zeros(self.INPUT_DIM, dtype=np.float32)
        self.prediction = np.zeros(self.INPUT_DIM, dtype=np.float32)
        self.prediction_error = np.zeros(self.INPUT_DIM, dtype=np.float32)
        
        # ===== SUBSCRIBERS =====
        self.visual_input_sub = self.create_subscription(
            Image,
            'unity/camera/raw',
            self.visual_input_callback,
            10
        )
        
        # Also accept direct float array input for testing
        self.direct_input_sub = self.create_subscription(
            Float32MultiArray,
            'cortex/visual/input',
            self.direct_input_callback,
            10
        )
        
        # ===== PUBLISHERS =====
        # Prediction Error - the key output for consciousness bandwidth
        self.prediction_error_pub = self.create_publisher(
            Float32MultiArray,
            '/neural_data/prediction_error',  # As specified in requirements
            10
        )
        
        # Current prediction for monitoring
        self.prediction_pub = self.create_publisher(
            Float32MultiArray,
            'cortex/visual/prediction',
            10
        )
        
        # Neural activity for downstream processing
        self.neural_activity_pub = self.create_publisher(
            Float32MultiArray,
            'cortex/visual/activity',
            10
        )
        
        # Initialize the SNN if Nengo is available
        if NENGO_AVAILABLE:
            self._build_predictive_snn()
            self.get_logger().info(
                f'Visual Cortex initialized with {self.n_neurons} LIF neurons (Nengo)'
            )
        else:
            self._init_stub_model()
            self.get_logger().warn(
                f'Visual Cortex running in STUB mode (Nengo not available)'
            )
        
        # Processing timer - runs the generative loop
        self.process_timer = self.create_timer(0.033, self.run_generative_loop)  # ~30Hz
        
        self.get_logger().info('Predictive Coding architecture active')
    
    def _build_predictive_snn(self):
        """
        Build the Nengo Spiking Neural Network with Predictive Coding.
        
        Architecture:
        1. Input Node - receives sensory data
        2. V1 Ensemble - 1000 LIF neurons, encodes the input
        3. Prediction Ensemble - generates predictions via feedback
        4. Error Node - computes Input - Prediction
        """
        self.model = nengo.Network(label='Visual Cortex - Predictive Coding')
        
        with self.model:
            # ===== INPUT LAYER =====
            # Sensory input from Unity/external source
            self.input_node = nengo.Node(
                output=lambda t: self.current_input,
                size_out=self.INPUT_DIM,
                label='Sensory Input'
            )
            
            # ===== PRIMARY VISUAL CORTEX (V1) =====
            # Main processing ensemble with 1000 LIF neurons
            self.v1_ensemble = nengo.Ensemble(
                n_neurons=self.n_neurons,
                dimensions=self.INPUT_DIM,
                neuron_type=nengo.LIF(
                    tau_rc=0.02,    # Membrane time constant (20ms)
                    tau_ref=0.002   # Refractory period (2ms)
                ),
                max_rates=nengo.dists.Uniform(100, 200),  # Firing rates
                label='V1 Cortex'
            )
            
            # ===== PREDICTION LAYER =====
            # Generates predictions based on internal state
            self.prediction_ensemble = nengo.Ensemble(
                n_neurons=self.n_neurons // 2,  # 500 neurons for prediction
                dimensions=self.INPUT_DIM,
                neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0.002),
                label='Prediction Generator'
            )
            
            # ===== CONNECTIONS =====
            
            # Input -> V1 (feedforward)
            nengo.Connection(
                self.input_node, 
                self.v1_ensemble,
                synapse=0.005,  # 5ms synaptic delay
                label='Sensory -> V1'
            )
            
            # V1 -> Prediction (the generative path)
            nengo.Connection(
                self.v1_ensemble, 
                self.prediction_ensemble,
                synapse=0.01,  # 10ms delay
                label='V1 -> Prediction'
            )
            
            # FEEDBACK: Prediction -> V1 (recurrent connection)
            # This is the key to predictive coding - the brain predicts its own input
            self.feedback_conn = nengo.Connection(
                self.prediction_ensemble,
                self.v1_ensemble,
                transform=-0.5,  # Inhibitory feedback (subtract prediction)
                synapse=0.02,    # 20ms delay for feedback
                label='Prediction Feedback'
            )
            
            # ===== PROBES =====
            
            # Probe V1 neural activity (spike rates)
            self.v1_activity_probe = nengo.Probe(
                self.v1_ensemble.neurons,
                sample_every=self.simulation_dt * 10
            )
            
            # Probe decoded prediction output
            self.prediction_probe = nengo.Probe(
                self.prediction_ensemble,
                synapse=0.01,
                sample_every=self.simulation_dt * 10
            )
            
            # Probe V1 decoded output
            self.v1_output_probe = nengo.Probe(
                self.v1_ensemble,
                synapse=0.01,
                sample_every=self.simulation_dt * 10
            )
        
        # Create simulator
        self.sim = nengo.Simulator(self.model, dt=self.simulation_dt)
        self.get_logger().info('Nengo Predictive Coding model built successfully')
    
    def _init_stub_model(self):
        """Initialize stub model for testing without Nengo."""
        # Simple linear prediction model as fallback
        self.stub_weights = np.random.randn(
            self.INPUT_DIM, self.INPUT_DIM
        ).astype(np.float32) * 0.1
    
    def visual_input_callback(self, msg: Image):
        """Process incoming visual data from Unity camera."""
        self.current_input = self._preprocess_image(msg)
    
    def direct_input_callback(self, msg: Float32MultiArray):
        """Accept direct float array input for testing."""
        if len(msg.data) >= self.INPUT_DIM:
            self.current_input = np.array(
                msg.data[:self.INPUT_DIM], 
                dtype=np.float32
            )
    
    def run_generative_loop(self):
        """
        Execute the Generative Loop - the core of Predictive Coding.
        
        1. Run SNN with current input
        2. Get prediction from the network
        3. Compute prediction error: Error = Input - Prediction
        4. Publish error (this is what gets transmitted to satellite)
        
        If error is near zero, the system "understands" reality perfectly,
        enabling minimal bandwidth for consciousness transmission.
        """
        if NENGO_AVAILABLE:
            # Run Nengo simulation steps
            self.sim.run_steps(10)
            
            # Get prediction from probe
            if len(self.sim.data[self.prediction_probe]) > 0:
                self.prediction = self.sim.data[self.prediction_probe][-1].astype(np.float32)
            
            # Get V1 activity for publishing
            if len(self.sim.data[self.v1_activity_probe]) > 0:
                v1_activity = self.sim.data[self.v1_activity_probe][-1]
            else:
                v1_activity = np.zeros(self.n_neurons, dtype=np.float32)
        else:
            # Stub mode: simple linear prediction
            self.prediction = np.tanh(self.stub_weights @ self.prediction) * 0.9
            # Stub learning
            error_outer = np.outer(self.prediction_error, self.current_input)
            self.stub_weights += self.learning_rate * error_outer
            v1_activity = np.random.rand(self.n_neurons).astype(np.float32)
        
        # ===== COMPUTE PREDICTION ERROR =====
        # This is the key output: Error = Input - Prediction
        self.prediction_error = self.current_input - self.prediction
        
        # ===== PUBLISH PREDICTION ERROR =====
        # This is published to /neural_data/prediction_error as required
        error_msg = Float32MultiArray()
        error_msg.data = self.prediction_error.tolist()
        self.prediction_error_pub.publish(error_msg)
        
        # Publish current prediction
        pred_msg = Float32MultiArray()
        pred_msg.data = self.prediction.tolist()
        self.prediction_pub.publish(pred_msg)
        
        # Publish neural activity
        activity_msg = Float32MultiArray()
        activity_msg.data = v1_activity.flatten().tolist()[:100]  # Limit size
        self.neural_activity_pub.publish(activity_msg)
        
        # Log prediction quality periodically
        error_magnitude = np.linalg.norm(self.prediction_error)
        if np.random.random() < 0.01:  # Log 1% of frames
            self.get_logger().info(
                f'Prediction Error Magnitude: {error_magnitude:.4f}'
            )
    
    def _preprocess_image(self, msg: Image) -> np.ndarray:
        """Preprocess ROS Image message for SNN input."""
        if len(msg.data) > 0:
            img_array = np.frombuffer(msg.data, dtype=np.uint8)
            img_array = img_array.reshape((msg.height, msg.width, -1))
            # Downsample to 8x8 grid and flatten for 64-dimensional input
            h_step = max(1, msg.height // 8)
            w_step = max(1, msg.width // 8)
            downsampled = img_array[::h_step, ::w_step, 0][:8, :8]
            return downsampled.flatten().astype(np.float32) / 255.0
        else:
            return np.zeros(self.INPUT_DIM, dtype=np.float32)
    
    def get_prediction_quality(self) -> float:
        """
        Get current prediction quality metric.
        
        Returns:
            Float between 0 and 1, where 1 = perfect prediction
        """
        error_magnitude = np.linalg.norm(self.prediction_error)
        input_magnitude = np.linalg.norm(self.current_input) + 1e-8
        quality = 1.0 - min(1.0, error_magnitude / input_magnitude)
        return float(quality)


# Alias for backward compatibility
VisualCortex = VisualCortexNode


def main(args=None):
    """Main entry point for the Visual Cortex node."""
    rclpy.init(args=args)
    node = VisualCortexNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
