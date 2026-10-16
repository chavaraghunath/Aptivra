"""Expert Spawner Controller evaluating entropy and verifier failure thresholds."""

from dataclasses import dataclass


@dataclass
class SpawnerConfig:
    """Config-driven thresholds for dynamic expert spawning."""

    entropy_threshold: float = 1.2
    verifier_failure_threshold: float = 0.3
    max_experts: int = 16


class ExpertSpawner:
    """Controller determining when to spawn a new expert module block."""

    def __init__(self, config: SpawnerConfig | None = None) -> None:
        self.config = config or SpawnerConfig()

    def should_spawn(
        self,
        current_num_experts: int,
        entropy_val: float,
        verifier_failure_rate: float,
    ) -> tuple[bool, str]:
        """Evaluates whether spawning criteria are met based on entropy or failure rates."""
        if current_num_experts >= self.config.max_experts:
            return False, "MAX_EXPERTS_REACHED"

        if entropy_val > self.config.entropy_threshold:
            thresh = self.config.entropy_threshold
            return True, f"HIGH_ROUTER_ENTROPY ({entropy_val:.3f} > {thresh})"

        if verifier_failure_rate > self.config.verifier_failure_threshold:
            thresh_vf = self.config.verifier_failure_threshold
            return True, f"HIGH_VERIFIER_FAILURE ({verifier_failure_rate:.3f} > {thresh_vf})"

        return False, "NO_SPAWN_TRIGGERED"
