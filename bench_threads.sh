#!/bin/bash
# Compare index build time and query latency/recall (Panama vs native, main vs this branch) across
# build thread counts using BenchYAML.
#
# Usage: ./bench_threads.sh [branches="main mybranch"] [threads="2 8 16 32"] [datasets="a b c"] [modes="panama native"]
#                           [bulk="false true"] [reps=3] [xmx=32G] [configs-from=<git ref>|none] [skip-build]
#   e.g. ./bench_threads.sh
#        ./bench_threads.sh skip-build datasets="sift-128-euclidean" reps=1
#
# The working tree must be clean (commit this script first): the script checks out other branches.
#
# The provider is forced with -Djvector.vectorization_provider=panama|native. This property exists on
# all branches and overrides their differing defaults, and a forced provider that fails to load throws
# instead of silently falling back.
#
# Bulk scoring is controlled with -Djvector.bulk_exact_scoring=true|false. Branches whose source does not
# contain that property (e.g. main) are run with bulk=false only, and the property is not passed.
#
# BenchYAML reads its configs from jvector-examples/yaml-configs/ in the working tree. So every branch
# runs the same configs, index-parameters/ is overlaid from configs-from (default: the commit this
# script was started on) after each checkout, and restored before the next. configs-from=none disables this.
#
# For each branch: checkout, build (once), then for each mode, bulk setting, and thread count, run
# BenchYAML (one dataset per JVM) with -Djvector.physical_core_count=<threads>
# (this sizes PhysicalCoreExecutor, the pool used for graph construction).
#
# Output (in bench-threads-results/):
#   <dataset>-<branch>-<mode>-bulk<bulk>-t<threads>-r<rep>.log   raw logs
#   results.csv     dataset,branch,mode,bulk,threads,rep,build_time_s
#   query.csv       dataset,branch,mode,bulk,threads,rep,overquery,qps,mean_latency_ms,recall_at_10
#                   (one row per overquery value of the Recall@10 table)
#   summary.txt     per-dataset pivot table of mean build times
# Plot with: python3 plot_bench_threads.py [bench-threads-results]

set -u

# Bash reads scripts incrementally and this file may not exist on the branches checked out below,
# so run from a private copy.
if [ -z "${BENCH_THREADS_COPY:-}" ]; then
  export BENCH_THREADS_REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
  export BENCH_THREADS_COPY="$(mktemp "${TMPDIR:-/tmp}/bench_threads.XXXXXX.sh")"
  cp "$0" "$BENCH_THREADS_COPY"
  exec bash "$BENCH_THREADS_COPY" "$@"
fi

BRANCHES="panama-threshold-with-bulk-scoring"
THREADS="1 8 16"
DATASETS="lastfm-64-dot nytimes-256-angular sift-128-euclidean e5-small-v2-100k"
REPS=2
MODES="panama native"
BULK="false true"
BUILD=true
XMX="32G"
CONFIGS_FROM=""

for arg in "$@"; do
  case "$arg" in
    branches=*) BRANCHES="${arg#*=}" ;;
    threads=*)  THREADS="${arg#*=}" ;;
    datasets=*) DATASETS="${arg#*=}" ;;
    reps=*)     REPS="${arg#*=}" ;;
    modes=*)    MODES="${arg#*=}" ;;
    bulk=*)     BULK="${arg#*=}" ;;
    bulk)       BULK="true" ;;
    xmx=*)      XMX="${arg#*=}" ;;
    configs-from=*) CONFIGS_FROM="${arg#*=}" ;;
    skip-build) BUILD=false ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

for M in $MODES; do
  case "$M" in panama|native) ;; *) echo "Invalid mode '$M' (use panama and/or native)" >&2; exit 1 ;; esac
done

for B in $BULK; do
  case "$B" in true|false) ;; *) echo "Invalid bulk '$B' (use true and/or false)" >&2; exit 1 ;; esac
done

REPO_ROOT="$BENCH_THREADS_REPO_ROOT"
OUT="${REPO_ROOT}/bench-threads-results"
CSV="${OUT}/results.csv"
QCSV="${OUT}/query.csv"
CFG_DIR="jvector-examples/yaml-configs/index-parameters"
mkdir -p "$OUT"
cd "$REPO_ROOT"

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Working tree has uncommitted changes to tracked files; commit or stash them first:" >&2
  git status --short --untracked-files=no >&2
  exit 1
fi

ORIG_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[ -n "$CONFIGS_FROM" ] || CONFIGS_FROM="$(git rev-parse HEAD)"
OVERLAID=false

restore_configs() {
  if [ "$OVERLAID" = true ]; then
    git reset -q -- "$CFG_DIR"
    git checkout -q -- "$CFG_DIR"
    git clean -fdq -- "$CFG_DIR"
    OVERLAID=false
  fi
}

trap 'restore_configs; git checkout -q "$ORIG_BRANCH"; rm -f "$BENCH_THREADS_COPY"' EXIT
# The JVM handles Ctrl-C itself and exits normally, so bash would otherwise keep looping.
trap 'echo "Interrupted, stopping." >&2; exit 130' INT TERM

HEADER="dataset,branch,mode,bulk,threads,rep,build_time_s"
QHEADER="dataset,branch,mode,bulk,threads,rep,overquery,qps,mean_latency_ms,recall_at_10"
# Set aside results from an older script version with a different column layout
if [ -f "$CSV" ] && [ "$(head -1 "$CSV")" != "$HEADER" ]; then mv "$CSV" "${OUT}/results.old.$(date +%s).csv"; fi
[ -f "$CSV" ] || echo "$HEADER" > "$CSV"
if [ -f "$QCSV" ] && [ "$(head -1 "$QCSV")" != "$QHEADER" ]; then mv "$QCSV" "${OUT}/query.old.$(date +%s).csv"; fi
[ -f "$QCSV" ] || echo "$QHEADER" > "$QCSV"

# Print "overquery,qps,mean_latency_ms,recall" for each row of every Recall@10 table in a BenchYAML log.
# Table columns: Overquery, Avg QPS, Std Dev, CV %, Mean Latency (ms), Avg Visited, Recall@10, ...
parse_query_table() {
  awk '/Overquery/ && /Recall@10[[:space:]]/ { f=1; next }
       f==1 && /^---/                         { f=2; next }
       f==2 && NF==0                          { f=0; next }
       f==2                                   { print $1","$2","$5","$7 }' "$1"
}

for BR in $BRANCHES; do
  echo "################ Branch $BR ################"
  restore_configs
  git checkout -q "$BR" || { echo "checkout $BR failed" >&2; exit 1; }

  # The property is read in jvector-examples (Grid.java); search all Java sources, not this script/docs.
  if git grep -q 'jvector\.bulk_exact_scoring' HEAD -- '*.java'; then
    BR_BULK="$BULK"; BULK_SUPPORTED=true
    echo "Branch $BR has bulk exact scoring; running bulk=true only"
  else
    BR_BULK="false"; BULK_SUPPORTED=false
    echo "Branch $BR has no bulk exact scoring; running bulk=false only"
  fi
  exit 0

  CP_FILE="${OUT}/${BR//\//_}.classpath"
  if [ "$BUILD" = true ] || [ ! -f "$CP_FILE" ]; then
    # Branches use different native build dir layouts; stale output breaks 'meson setup --wipe'.
    rm -rf "${REPO_ROOT}"/jvector-native/target/meson-build*
    # Build jvector-examples and its reactor deps, then dump the runtime classpath
    # (reactor modules resolve to their target/ jars/classes).
    ./mvnw -q -pl jvector-examples -am package -DskipTests \
        dependency:build-classpath -Dmdep.outputFile="$CP_FILE" || { echo "build failed on $BR" >&2; exit 1; }
    # build-classpath omits the module's own output; add it
    echo -n ":${REPO_ROOT}/jvector-examples/target/classes" >> "$CP_FILE"
  fi
  CP="$(cat "$CP_FILE")"

  if [ "$CONFIGS_FROM" != none ]; then
    git checkout -q "$CONFIGS_FROM" -- "$CFG_DIR" || { echo "overlaying configs from $CONFIGS_FROM failed" >&2; exit 1; }
    OVERLAID=true
  fi

  for DATASET in $DATASETS; do
    for MODE in $MODES; do
      for B in $BR_BULK; do
        BULK_OPT=()
        [ "$BULK_SUPPORTED" = true ] && BULK_OPT=(-Djvector.bulk_exact_scoring=${B})
        for T in $THREADS; do
          for R in $(seq 1 "$REPS"); do
            TAG="${DATASET}-${BR//\//_}-${MODE}-bulk${B}-t${T}-r${R}"
            LOG="${OUT}/${TAG}.log"
            echo "--- dataset=$DATASET branch=$BR mode=$MODE bulk=$B threads=$T rep=$R"
            java -Xmx${XMX} -ea \
              --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED --enable-preview \
              -Djvector.vectorization_provider=${MODE} \
              ${BULK_OPT[@]+"${BULK_OPT[@]}"} \
              -Djvector.physical_core_count=${T} \
              -cp "$CP" io.github.jbellis.jvector.example.BenchYAML "$DATASET" > "$LOG" 2>&1
            BT="$(grep -m1 'Index build time:' "$LOG" | sed -E 's/.*Index build time: ([0-9.]+) seconds.*/\1/')"
            echo "    build_time_s=${BT:-FAILED}"
            echo "${DATASET},${BR},${MODE},${B},${T},${R},${BT:-NA}" >> "$CSV"

            ROWS="$(parse_query_table "$LOG")"
            if [ -z "$ROWS" ]; then
              echo "    query results: NONE FOUND"
            else
              while IFS=, read -r OQ QPS LAT REC; do
                printf "    overquery=%-6s mean_latency_ms=%-8s recall@10=%-6s qps=%s\n" "$OQ" "$LAT" "$REC" "$QPS"
                echo "${DATASET},${BR},${MODE},${B},${T},${R},${OQ},${QPS},${LAT},${REC}" >> "$QCSV"
              done <<< "$ROWS"
            fi
          done
        done
      done
    done
  done
done

# Summary: mean build time per dataset, (branch, mode, bulk) x threads
awk -F, 'NR>1 && $7!="NA" { d=$1; k=$2"/"$3"/bulk="$4; c=d","k","$5; s[c]+=$7; n[c]++; keys[d","k]=1; th[$5]=1; ds[d]=1 }
END {
  nt=asorti(th, ts, "@ind_num_asc"); nd=asorti(ds, dss)
  for (j=1;j<=nd;j++) { d=dss[j]; print "== " d " (mean build time, s) =="
    printf "%-48s", "branch/mode/bulk"; for (i=1;i<=nt;i++) printf "%10s", "t="ts[i]; printf "\n"
    m=asorti(keys, ks)
    for (x=1;x<=m;x++) { split(ks[x], a, ","); if (a[1]!=d) continue; k=a[2]; printf "%-48s", k
      for (i=1;i<=nt;i++) { c=d","k","ts[i]; if (n[c]) printf "%10.2f", s[c]/n[c]; else printf "%10s", "-" }
      printf "\n" }
    printf "\n" } }' "$CSV" | tee "${OUT}/summary.txt"

# Summary: mean latency / recall@10 per dataset, (branch, mode, bulk, threads) x overquery
awk -F, 'NR>1 { k=$1","$2"/"$3"/bulk="$4"/t="$5","$7; l[k]+=$9; r[k]+=$10; n[k]++ }
END {
  m=asorti(n, ks)
  print "== mean latency (ms) / recall@10 =="
  printf "%-24s %-56s %10s %12s %10s\n", "dataset", "branch/mode/bulk/threads", "overquery", "latency_ms", "recall@10"
  for (x=1;x<=m;x++) { split(ks[x], a, ","); k=ks[x]
    printf "%-24s %-56s %10s %12.3f %10.3f\n", a[1], a[2], a[3], l[k]/n[k], r[k]/n[k] } }' "$QCSV" | tee -a "${OUT}/summary.txt"
