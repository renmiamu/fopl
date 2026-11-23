from __future__ import annotations

try:
    from .evaluate_arithmetic import ArithmeticSpec, run
except ImportError:
    from evaluate_arithmetic import ArithmeticSpec, run

SPEC = ArithmeticSpec("svamp", "SVAMP", "ChilleD/SVAMP")

if __name__ == "__main__":
    run(SPEC)
