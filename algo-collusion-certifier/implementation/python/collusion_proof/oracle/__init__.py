"""Oracle module for collusion detection via strategic perturbation."""

try:
    from collusion_proof.oracle.passive_oracle import PassiveOracle
    from collusion_proof.oracle.checkpoint_oracle import CheckpointOracle
    from collusion_proof.oracle.rewind_oracle import RewindOracle
except ImportError:
    pass
