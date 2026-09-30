"""10 Hz GPU telemetry sampler based on NVML (runs until SIGTERM/SIGINT).

Columns: t_ns, energy_mJ (NVML cumulative energy counter), power_mW,
temp_C, sm_clock_MHz, gpu_util_pct, cpu_util_pct (host, from /proc/stat).

GPU energy of a window is taken from the cumulative counter (interpolated at
the window boundaries); the instantaneous power column is kept as a cross-check.

Usage: python -m ww.nvml_sampler --out nvml.csv [--interval 0.1] [--fake]
"""
import argparse
import csv
import signal
import sys
import time

_running = True


def _stop(*_):
    global _running
    _running = False


def read_cpu_times():
    with open("/proc/stat") as f:
        parts = f.readline().split()[1:]
    vals = list(map(int, parts))
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    return sum(vals), idle


class Nvml:
    def __init__(self, index=0):
        import pynvml as n
        self.n = n
        n.nvmlInit()
        self.h = n.nvmlDeviceGetHandleByIndex(index)

    def _try(self, fn, *a):
        try:
            return fn(self.h, *a)
        except Exception:  # noqa: BLE001
            return None

    def sample(self):
        n = self.n
        util = self._try(n.nvmlDeviceGetUtilizationRates)
        return (self._try(n.nvmlDeviceGetTotalEnergyConsumption),
                self._try(n.nvmlDeviceGetPowerUsage),
                self._try(n.nvmlDeviceGetTemperature, n.NVML_TEMPERATURE_GPU),
                self._try(n.nvmlDeviceGetClockInfo, n.NVML_CLOCK_SM),
                util.gpu if util is not None else None)


class Fake:
    """For local tests without a GPU: 50 W constant."""
    def __init__(self):
        self.t0 = time.time()

    def sample(self):
        return (int((time.time() - self.t0) * 50_000), 50_000, 40, 1500, 50)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=0.1)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args(argv)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    dev = Fake() if args.fake else Nvml()
    try:
        prev_total, prev_idle = read_cpu_times()
    except OSError:
        prev_total = prev_idle = None
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ns", "energy_mJ", "power_mW", "temp_C", "sm_clock_MHz", "gpu_util_pct", "cpu_util_pct"])
        next_t = time.time()
        rows = 0
        while _running:
            e, p, temp, clk, util = dev.sample()
            cpu = None
            if prev_total is not None:
                total, idle = read_cpu_times()
                dt, di = total - prev_total, idle - prev_idle
                cpu = round(100.0 * (1 - di / dt), 2) if dt > 0 else None
                prev_total, prev_idle = total, idle
            w.writerow([time.time_ns(), e, p, temp, clk, util, cpu])
            rows += 1
            if rows % 10 == 0:
                f.flush()
            next_t += args.interval
            time.sleep(max(0.0, next_t - time.time()))
        f.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
