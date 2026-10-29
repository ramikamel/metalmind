"""MetalMind's initial runtime diagnostic command."""
import argparse
from dataclasses import asdict
import json
from metalmind.runtime.environment import probe_environment


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="metalmind")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Describe the local PyTorch/Metal environment")
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable diagnostics")
    doctor.add_argument("--smoke", action="store_true", help="Compile and execute a known-good Metal kernel")
    args = parser.parse_args(argv)
    info = probe_environment()
    report = asdict(info)
    ready = info.capabilities.mps_available and info.capabilities.compile_shader
    if args.smoke:
        report["shader_health_passed"] = False
        if ready:
            from metalmind.runtime.health import check_health
            report["shader_health_passed"] = check_health()
        ready = ready and report["shader_health_passed"]
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print(f"Python {info.python_version} | PyTorch {info.torch_version} | {info.architecture}")
        print(f"MPS available: {info.capabilities.mps_available}")
        print(f"Shader API available: {info.capabilities.compile_shader}")
        if args.smoke:
            print(f"Shader health passed: {report['shader_health_passed']}")
        if info.unavailable_reason:
            print(info.unavailable_reason)
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
