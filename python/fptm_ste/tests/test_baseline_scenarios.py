from pathlib import Path

from fptm_ste.tests.run_mnist_equiv import (
    BASELINE_SCENARIOS,
    build_arg_parser,
    build_baseline_args,
)


def test_baseline_scenarios_have_descriptions():
    assert BASELINE_SCENARIOS, "Expected predefined baseline scenarios."
    for name, payload in BASELINE_SCENARIOS.items():
        assert payload.get("description"), f"Scenario '{name}' is missing a description."
        assert "overrides" in payload and payload["overrides"], f"Scenario '{name}' is missing overrides."


def test_build_baseline_args_overrides(tmp_path: Path):
    parser = build_arg_parser()
    base_args = parser.parse_args([])
    base_args.output_json = str(tmp_path / "summary.json")

    scenario_args = build_baseline_args(base_args, "mnist_vit_patch4", output_dir=str(tmp_path / "reports"))

    assert scenario_args.dataset == "mnist"
    assert scenario_args.models == ["transformer"]
    assert scenario_args.seed == 1337
    assert scenario_args.output_json == str(tmp_path / "reports" / "mnist_vit_patch4.json")
    assert scenario_args.baseline_scenarios is None
    assert scenario_args.baseline_only is False
    assert scenario_args.list_baselines is False


def test_build_baseline_args_default_dir(tmp_path: Path):
    parser = build_arg_parser()
    base_args = parser.parse_args([])
    base_args.output_json = str(tmp_path / "main_summary.json")

    scenario_args = build_baseline_args(base_args, "mnist_swin_window4")
    expected_dir = tmp_path / "baselines"
    assert scenario_args.output_json == str(expected_dir / "mnist_swin_window4.json")
    assert expected_dir.exists()
