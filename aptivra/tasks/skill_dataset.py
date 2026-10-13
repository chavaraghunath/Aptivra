# ruff: noqa: E501
"""MCP Skill Dataset Loader for Aptivra Continual Expertise Training."""

import json
from pathlib import Path

from aptivra.tasks.base import (
    MechanicalVerifier,
    TaskFamily,
    TaskInstance,
    VerifierResult,
)


class MCPSkillVerifier(MechanicalVerifier):
    """Mechanical verifier for MCP skill task instances."""

    def verify(self, instance: TaskInstance, model_output: str) -> VerifierResult:
        expected = instance.target_solution
        passed = expected in model_output or (
            len(model_output.strip()) > 0 and "pass" in model_output
        )
        return VerifierResult(
            passed=passed,
            expected=expected,
            actual=model_output,
            error_message=None if passed else "Model output did not match expected skill pattern",
        )


class MCPSkillTaskFamily(TaskFamily):
    """Task Family generated dynamically from an MCP skill definition."""

    def __init__(
        self,
        skill_name: str,
        description: str,
        skill_markdown: str,
        eval_samples: list[dict[str, str]] | None = None,
    ) -> None:
        super().__init__(
            family_id=f"skill.{skill_name}",
            name=f"MCP Skill: {skill_name}",
            description=description,
        )
        self.skill_markdown = skill_markdown
        self.eval_samples = eval_samples or []
        self._verifier = MCPSkillVerifier()

    @property
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    def generate_instances(self, count: int, seed: int = 42) -> list[TaskInstance]:
        """Generates task training instances from skill prompt and eval templates."""
        instances: list[TaskInstance] = []
        base_prompt = f"Skill: {self.name}\nDescription: {self.description}\n"

        if self.eval_samples:
            for idx, item in enumerate(self.eval_samples[:count]):
                inp = item.get("input", f"Execute {self.name} task {idx}")
                out = item.get("expected_output", f"Result for {self.name} {idx}")
                instances.append(
                    TaskInstance(
                        task_id=f"{self.family_id}_{idx}",
                        task_family=self.family_id,
                        prompt=f"{base_prompt}Input: {inp}\nTask:",
                        target_solution=out,
                        metadata={"seed": seed, "index": idx},
                    )
                )

        while len(instances) < count:
            idx = len(instances)
            instances.append(
                TaskInstance(
                    task_id=f"{self.family_id}_{idx}",
                    task_family=self.family_id,
                    prompt=f"{base_prompt}Instruction: Generate solution for step {idx}\nOutput:",
                    target_solution=f"# Solution for {self.family_id} step {idx}\npass",
                    metadata={"seed": seed, "index": idx},
                )
            )

        return instances[:count]


class MCPSkillRepositoryLoader:
    """Scans and loads skills from an mcp-skills directory repository."""

    def __init__(
        self, skills_dir: Path | str = "/Users/raghunath/code/others/skills-mcp/skills"
    ) -> None:
        self.skills_dir = Path(skills_dir)

    def list_available_skills(self) -> list[str]:
        """Lists all valid skill directory names."""
        if not self.skills_dir.exists():
            return []
        return [
            d.name
            for d in self.skills_dir.iterdir()
            if d.is_dir() and (d / "manifest.json").exists()
        ]

    def load_skill_task_family(self, skill_name: str) -> MCPSkillTaskFamily | None:
        """Loads a single skill as an Aptivra MCPSkillTaskFamily."""
        skill_path = self.skills_dir / skill_name
        manifest_file = skill_path / "manifest.json"
        skill_md_file = skill_path / "skill.md"

        if not manifest_file.exists():
            return None

        try:
            manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
            description = manifest_data.get("description", skill_name)
        except Exception:
            description = skill_name

        skill_md = ""
        if skill_md_file.exists():
            try:
                skill_md = skill_md_file.read_text(encoding="utf-8")
            except Exception:
                skill_md = ""

        return MCPSkillTaskFamily(
            skill_name=skill_name,
            description=description,
            skill_markdown=skill_md,
        )
