"""One-step inviscid ONERA M6 example regression check."""
from examples.oneram6_euler import run


def test_oneram6_euler_step():
    run(steps=1, report_every=1)


if __name__ == "__main__":
    test_oneram6_euler_step()