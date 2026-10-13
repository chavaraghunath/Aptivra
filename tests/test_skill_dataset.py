"""Unit tests for MCP Skill Dataset Loader."""

from aptivra.tasks.skill_dataset import MCPSkillRepositoryLoader, MCPSkillTaskFamily


def test_mcp_skill_task_family() -> None:
    family = MCPSkillTaskFamily(
        skill_name="test.skill",
        description="Test skill description",
        skill_markdown="# Test Skill\nDo something.",
    )
    instances = family.generate_instances(count=3, seed=42)
    assert len(instances) == 3
    assert instances[0].task_id == "skill.test.skill_0"

    sol_text = instances[0].target_solution
    res = family.verifier.verify(instances[0], sol_text)
    assert res.passed is True
    assert family.verifier.verify(instances[0], "pass").passed is False
    prefixed = family.verifier.verify(instances[0], f"Result text containing {sol_text}")
    assert prefixed.passed is False


def test_mcp_skill_repository_loader() -> None:
    loader = MCPSkillRepositoryLoader("/Users/raghunath/code/others/skills-mcp/skills")
    skills = loader.list_available_skills()
    assert isinstance(skills, list)
    assert len(skills) > 0, "Expected at least 1 skill in skills-mcp repository"

    sample_skill = skills[0]
    family = loader.load_skill_task_family(sample_skill)
    assert family is not None
    assert family.family_id == f"skill.{sample_skill}"
