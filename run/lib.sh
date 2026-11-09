# pool JOBS_FILE: run each line of JOBS_FILE (a shell command that may use $GPU and $MEM) on the GPUs in $GPUS,
# $SLOTS jobs at a time per GPU; $MEM is each job's share of GPU memory. Returns non-zero if any job failed.
pool() {
  local jobs=$1 next=$1.next fail=$1.failed g s
  echo 0 > "$next"; rm -f "$fail"
  local mem; mem=$(python3 -c "print(round(0.85 / $SLOTS, 3))")
  for g in ${GPUS//,/ }; do for ((s = 0; s < SLOTS; s++)); do
    (
      while true; do
        n=$(flock "$next" bash -c 'n=$(cat "$0"); echo $((n + 1)) > "$0"; echo $n' "$next")
        line=$(sed -n "$((n + 1))p" "$jobs")
        [ -z "$line" ] && break
        echo "$(date -Is) gpu $g: $line"
        # one retry: vLLM's start-up memory check can fail when another job on the same GPU frees memory meanwhile
        GPU=$g MEM=$mem bash -c "$line" || { echo "$(date -Is) retrying: $line"; sleep 60; GPU=$g MEM=$mem bash -c "$line"; } \
          || { echo "$(date -Is) FAILED: $line"; echo "$line" >> "$fail"; }
      done
    ) &
    sleep "${STAGGER:-20}"   # let each vLLM instance take its memory before the next one starts on the same GPU
  done; done
  wait
  [ ! -e "$fail" ]
}

# arms_of SELECTION_JSON KEY: print a JSON list field as words
json_list() { python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))[sys.argv[2]]))" "$1" "$2"; }
