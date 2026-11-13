#!/usr/bin/env bash
base='outputs/teacher_attention_cache/rinalmo_pretrain_10pct_selected_heads'
log="${RHA_ROOT:-$PWD}/logs/teacher_cache/monitor.log"
mkdir -p "$(dirname "$log")"
echo "$(date +%Y-%m-%d\ %H:%M:%S) monitor start" >> "$log"
while true; do
  total=0
  line=""
  for m in nucleotide_transformer_v2_500m rnafm rinalmo_mega splicebert plantrna_fm utrlm_mrl rinalmo_giga mrnafm moderngena_base; do
    n=$(find "${base}/model=${m}" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l)
    total=$((total+n))
    line="${line}${m}=${n} "
  done
  q=$(squeue -u "${USER:-unknown}" -h -o '%i %j %T' | grep -c 'rha_tcache' || true)
  echo "$(date +%Y-%m-%d\ %H:%M:%S) total=${total}/9297 jobs=${q} ${line}" >> "$log"
  sleep 600
done
