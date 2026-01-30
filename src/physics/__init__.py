"""Physics-informed constraints and conservation laws."""

from .symmetries import (
    generate_random_rotation_matrix,
    apply_rotation_to_data,
    verify_rotation_invariance,
    rotation_invariance_test_suite,
    apply_time_reversal_to_data,
    time_reversal_symmetry_loss,
    verify_time_reversal_symmetry,
    TimeReversalSymmetryLoss
)

from .conservation import (
    compute_kinetic_energy,
    compute_total_energy,
    energy_conservation_error,
    energy_conservation_loss,
    EnergyConservationLoss,
    compute_total_momentum,
    momentum_conservation_error,
    momentum_conservation_loss,
    MomentumConservationLoss,
    compute_total_charge,
    charge_conservation_error,
    charge_conservation_loss,
    ChargeConservationLoss,
    combined_conservation_loss,
    CombinedConservationLoss
)

from .known_limits import (
    vlasov_solver_ballistic,
    vlasov_limit_loss,
    VlasovLimitLoss,
    maxwell_boltzmann_distribution,
    compute_momentum_statistics,
    maxwell_boltzmann_limit_loss,
    MaxwellBoltzmannLimitLoss,
    interpolate_stopping_power,
    compute_energy_loss,
    stopping_power_consistency_loss,
    StoppingPowerConsistencyLoss,
    AuxiliaryPhysicsHeads,
    auxiliary_physics_loss,
    CombinedPhysicsLimitLoss,
    STOPPING_POWER_TABLE,
    K_B
)

__all__ = [
    # Symmetries
    'generate_random_rotation_matrix',
    'apply_rotation_to_data',
    'verify_rotation_invariance',
    'rotation_invariance_test_suite',
    'apply_time_reversal_to_data',
    'time_reversal_symmetry_loss',
    'verify_time_reversal_symmetry',
    'TimeReversalSymmetryLoss',
    # Conservation laws
    'compute_kinetic_energy',
    'compute_total_energy',
    'energy_conservation_error',
    'energy_conservation_loss',
    'EnergyConservationLoss',
    'compute_total_momentum',
    'momentum_conservation_error',
    'momentum_conservation_loss',
    'MomentumConservationLoss',
    'compute_total_charge',
    'charge_conservation_error',
    'charge_conservation_loss',
    'ChargeConservationLoss',
    'combined_conservation_loss',
    'CombinedConservationLoss',
    # Known limits
    'vlasov_solver_ballistic',
    'vlasov_limit_loss',
    'VlasovLimitLoss',
    'maxwell_boltzmann_distribution',
    'compute_momentum_statistics',
    'maxwell_boltzmann_limit_loss',
    'MaxwellBoltzmannLimitLoss',
    'interpolate_stopping_power',
    'compute_energy_loss',
    'stopping_power_consistency_loss',
    'StoppingPowerConsistencyLoss',
    'AuxiliaryPhysicsHeads',
    'auxiliary_physics_loss',
    'CombinedPhysicsLimitLoss',
    'STOPPING_POWER_TABLE',
    'K_B'
]
