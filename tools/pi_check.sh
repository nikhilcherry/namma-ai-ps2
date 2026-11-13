#!/usr/bin/env bash
# M1: make the board run at full speed, before the first benchmark. Safe to run repeatedly.
#   bash tools/pi_check.sh [--perf]     --perf also sets the performance governor (needs sudo)
echo "board:    $((tr -d "\0" </proc/device-tree/model) 2>/dev/null || uname -m)"
echo "os:       $(. /etc/os-release 2>/dev/null; echo "$PRETTY_NAME")  kernel $(uname -r)"
echo "python:   $(${PY:-python} --version 2>&1)"
echo "cores:    $(nproc)   load: $(cut -d' ' -f1-3 /proc/loadavg)  (want idle: first number < 0.5)"
echo "RAM free: $(awk '/MemAvailable/{printf "%d MB",$2/1024}' /proc/meminfo)"
if command -v vcgencmd >/dev/null; then
  t=$(vcgencmd get_throttled | cut -d= -f2)
  echo "throttle: $t   (want 0x0)"
  [ "$t" != "0x0" ] && echo "  bits: 0 undervolt now, 1 freq capped, 2 throttled, 3 soft temp limit; 16-19 = happened since boot"
  echo "temp:     $(vcgencmd measure_temp)   (want well under 80'C during a run)"
  echo "clock:    $(vcgencmd measure_clock arm | awk -F= '{printf "%d MHz",$2/1e6}')"
else
  echo "vcgencmd: not found (not a Pi, or install libraspberrypi-bin)"
fi
g=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)
echo "governor: $g   (want performance)"
if [ "$1" = "--perf" ] && [ "$g" != "n/a" ]; then
  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null && echo "  -> set to performance"
fi
echo "busy procs (close these):"; ps -eo pcpu,comm --sort=-pcpu | sed -n 2,5p | sed 's/^/  /'
echo "tip: watch 'htop' during a run; all $(nproc) cores should be busy. If only 1-2 are, the runtime thread count is the problem."
