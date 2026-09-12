#!/usr/bin/env bash
# hwlog.sh - sample hardware telemetry to CSV during a long unattended run.
#
#   ./hwlog.sh                      # 10s interval, logs to hwlog-<date>.csv
#   ./hwlog.sh 30 mylog.csv         # 30s interval, custom file
#
# Ctrl+C to stop; it prints a max/min summary on the way out.
#
# What this machine actually exposes, checked rather than assumed:
#   - GPU core temp, fan %, power draw, clocks, VRAM used   (nvidia-smi)
#   - CPU package temp                                       (k10temp)
#   - NVMe composite temp                                    (nvme hwmon)
# NOT available here, so not logged:
#   - GDDR6X memory junction temp - the proprietary driver reports N/A for
#     this card and registers no hwmon entry, and nvidia-settings needs X
#     (this box is Wayland). GPU core temp is the proxy.
#   - Chassis/CPU fan RPM - the ASUS board's hwmon exposes no fan inputs
#     without the nct6775 module loaded.
#
# Thresholds below are this card's own reported limits, not guesses:
#   target 83C, slowdown 94C, shutdown 97C  (nvidia-smi -q -d TEMPERATURE)

set -u
INTERVAL="${1:-10}"
OUT="${2:-hwlog-$(date +%Y%m%d-%H%M%S).csv}"

WARN_GPU=85     # above the 83C target spec, well below the 94C slowdown
WARN_CPU=90

max_gpu=0; max_cpu=0; max_pw=0; max_fan=0; samples=0; warns=0

cpu_temp() {
    # k10temp Tctl, in millidegrees
    for d in /sys/class/hwmon/hwmon*; do
        [ "$(cat "$d/name" 2>/dev/null)" = "k10temp" ] || continue
        v=$(cat "$d/temp1_input" 2>/dev/null) && echo $((v / 1000)) && return
    done
    echo ""
}

nvme_temp() {
    for d in /sys/class/hwmon/hwmon*; do
        [ "$(cat "$d/name" 2>/dev/null)" = "nvme" ] || continue
        v=$(cat "$d/temp1_input" 2>/dev/null) && echo $((v / 1000)) && return
    done
    echo ""
}

summary() {
    echo
    echo "--- $samples samples, $warns warning(s) ---"
    echo "  peak GPU temp   : ${max_gpu}C   (target 83, slowdown 94, shutdown 97)"
    echo "  peak GPU fan    : ${max_fan}%"
    echo "  peak GPU power  : ${max_pw}W    (limit 450)"
    echo "  peak CPU temp   : ${max_cpu}C"
    echo "  log written to  : $OUT"
    exit 0
}
trap summary INT TERM

echo "timestamp,gpu_temp_c,gpu_fan_pct,gpu_power_w,gpu_util_pct,vram_used_mb,sm_clock_mhz,cpu_temp_c,nvme_temp_c" > "$OUT"
echo "logging every ${INTERVAL}s to $OUT - Ctrl+C to stop and see a summary"

while true; do
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    read -r gt gf gp gu vu sc < <(
        nvidia-smi --query-gpu=temperature.gpu,fan.speed,power.draw,utilization.gpu,memory.used,clocks.current.sm \
                   --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr ',' ' '
    )
    ct=$(cpu_temp); nt=$(nvme_temp)
    echo "$ts,$gt,$gf,$gp,$gu,$vu,$sc,$ct,$nt" >> "$OUT"

    # integer-compare on the truncated power reading
    pw=${gp%%.*}
    [ -n "${gt:-}" ] && [ "$gt" -gt "$max_gpu" ] 2>/dev/null && max_gpu=$gt
    [ -n "${gf:-}" ] && [ "$gf" -gt "$max_fan" ] 2>/dev/null && max_fan=$gf
    [ -n "${pw:-}" ] && [ "$pw" -gt "$max_pw" ] 2>/dev/null && max_pw=$pw
    [ -n "${ct:-}" ] && [ "$ct" -gt "$max_cpu" ] 2>/dev/null && max_cpu=$ct
    samples=$((samples + 1))

    if [ -n "${gt:-}" ] && [ "$gt" -ge "$WARN_GPU" ] 2>/dev/null; then
        warns=$((warns + 1))
        echo "  [$ts] WARNING gpu ${gt}C fan ${gf}% ${gp}W" >&2
    fi
    if [ -n "${ct:-}" ] && [ "$ct" -ge "$WARN_CPU" ] 2>/dev/null; then
        warns=$((warns + 1))
        echo "  [$ts] WARNING cpu ${ct}C" >&2
    fi

    sleep "$INTERVAL"
done
