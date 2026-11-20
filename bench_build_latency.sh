#!/bin/bash
# Measure full-precision index construction time (1 thread) and latency vs recall with BenchYAML, for two branches
# (default: main and panama-threshold) and both vectorization providers (native and panama).
#
# Usage: ./bench_build_latency.sh [branches="main panama-threshold"] [datasets="a b"] [modes="native panama"]
#                                 [reps=1] [xmx=16G] [threads=1] [skip-build]
#   e.g. ./bench_build_latency.sh
#        ./bench_build_latency.sh datasets="lastfm-64-dot" modes=native
#
# Default datasets: lastfm-64-dot sift-128-euclidean e5-small-v2-100k e5-base-v2-100k e5-large-v2-100k
#
# Each BenchYAML run builds a full-precision index (no compression, FP reranking, M=32, efConstruction=100) with
# -Djvector.physical_core_count=<threads> (default 1), so the construction time is single-threaded, and then searches
# it with top-10 overquery 1, 2, 5, 10. One run yields both the build time and the latency/recall table.
#
# The working tree must have no uncommitted changes to tracked files: the script checks out each branch in turn and
# benchmarks the COMMITTED code. Untracked files are left alone. The original branch is checked out again on exit.
# BenchYAML reads index-parameters/<dataset>.yml and run-config.yml from the working tree; this script overwrites
# them (so every branch runs the same thing) and puts the originals back, tracked or not, on exit.
#
# With reps > 1 the branch order alternates between reps to cancel drift in machine state.
#
# Output (in bench-build-latency-results/):
#   <dataset>-<branch>-<mode>-r<rep>.log   raw BenchYAML output
#   runs/<tag>/                            BenchYAML's experiments.csv
#   build.csv   dataset,branch,mode,rep,build_s
#   query.csv   dataset,branch,mode,rep,topk,overquery,qps,mean_latency_ms,recall
# Plot with: python3 plot_bench_build_latency.py bench-build-latency-results

set -u

# Bash reads scripts incrementally and this file may not exist on the branches checked out below,
# so run from a private copy.
if [ -z "${BENCH_BL_COPY:-}" ]; then
  export BENCH_BL_REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
  export BENCH_BL_COPY="$(mktemp "${TMPDIR:-/tmp}/bench_build_latency.XXXXXX.sh")"
  cp "$0" "$BENCH_BL_COPY"
  exec bash "$BENCH_BL_COPY" "$@"
fi

BRANCHES="main panama-threshold"
DATASETS="lastfm-64-dot sift-128-euclidean e5-small-v2-100k e5-base-v2-100k e5-large-v2-100k"
MODES="native panama"
REPS=1
XMX="16G"
THREADS=1
BUILD=true
MVN="${MVN:-./mvnw}"

for arg in "$@"; do
  case "$arg" in
    branches=*) BRANCHES="${arg#*=}" ;;
    datasets=*) DATASETS="${arg#*=}" ;;
    modes=*)    MODES="${arg#*=}" ;;
    reps=*)     REPS="${arg#*=}" ;;
    xmx=*)      XMX="${arg#*=}" ;;
    threads=*)  THREADS="${arg#*=}" ;;
    skip-build) BUILD=false ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

for M in $MODES; do
  case "$M" in panama|native) ;; *) echo "Invalid mode '$M' (use panama and/or native)" >&2; exit 1 ;; esac
done

REPO_ROOT="$BENCH_BL_REPO_ROOT"
OUT="${REPO_ROOT}/bench-build-latency-results"
BCSV="${OUT}/build.csv"
QCSV="${OUT}/query.csv"
CFG_DIR="jvector-examples/yaml-configs/index-parameters"
RUN_CFG="jvector-examples/yaml-configs/run-config.yml"
mkdir -p "$OUT/runs"
cd "$REPO_ROOT"

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Working tree has uncommitted changes to tracked files; commit or stash them first:" >&2
  git status --short --untracked-files=no >&2
  exit 1
fi

ORIG_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$ORIG_BRANCH" = "HEAD" ]; then ORIG_BRANCH="$(git rev-parse HEAD)"; fi
BRANCHES="$(echo $BRANCHES | tr ' ' '\n' | awk '!seen[$0]++' | tr '\n' ' ')"
if [ "$BUILD" = false ] && [ "$(echo $BRANCHES | wc -w)" -gt 1 ]; then
  echo "skip-build reuses target/ from the last build, so it only makes sense with a single branch" >&2
  exit 1
fi
REVERSED="$(echo $BRANCHES | tr ' ' '\n' | tac | tr '\n' ' ')"

# Files this script overwrote. An existing file (tracked or not) is saved in BACKUP_DIR first and put back by
# restore_configs; a file that did not exist is deleted again.
BACKUP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/bench_bl_backup.XXXXXX")"
CLAIMED=()   # "<kind>|<path>|<backup index>"

restore_configs() {
  local e kind f idx
  for e in ${CLAIMED[@]+"${CLAIMED[@]}"}; do
    IFS='|' read -r kind f idx <<< "$e"
    if [ "$kind" = "existing" ]; then cp -p "${BACKUP_DIR}/${idx}" "$f"; else rm -f "$f"; fi
  done
  CLAIMED=()
}

claim() {
  local f="$1" idx="${#CLAIMED[@]}"
  if [ -e "$f" ]; then
    cp -p "$f" "${BACKUP_DIR}/${idx}"
    CLAIMED+=("existing|${f}|${idx}")
  else
    CLAIMED+=("new|${f}|${idx}")
  fi
}

write_configs() {
  local d f
  for d in $DATASETS; do
    f="${CFG_DIR}/${d}.yml"
    claim "$f"
    cat > "$f" <<EOF
yamlSchemaVersion: 1
onDiskIndexVersion: 6

dataset: ${d}
useDatasetPrefetching: false

construction:
  outDegree: [32]
  efConstruction: [100]
  neighborOverflow: [1.2f]
  addHierarchy: [Yes]
  refineFinalGraph: [Yes]
  fusedGraph: [No]
  compression:
    - type: None
  reranking:
    - FP
  useSavedIndexIfExists: No

search:
  topKOverquery:
    10: [1.0, 2.0, 5.0, 10.0]
  useSearchPruning: [Yes]
  compression:
    - type: None
EOF
  done
}

# $1 = directory in which BenchYAML writes experiments.csv
write_run_config() {
  claim "$RUN_CFG"
  cat > "$RUN_CFG" <<EOF
yamlSchemaVersion: 1
onDiskIndexVersion: 6

benchmarks:
  throughput: [ AVG ]
  latency:    [ AVG ]
  count:      [ visited ]
  accuracy:   [ recall ]

console:
  benchmarks:
    throughput: [ AVG ]
    latency:    [ AVG ]
    count:      [ visited ]
    accuracy:   [ recall ]
  metrics:
    construction: [ index_build_time_s ]

logging:
  dir: $1
  jvectorRef: "bench-build-latency"
  runId: "run"
  type: csv
  benchmarks:
    throughput: [ AVG ]
  metrics:
    construction: [ index_build_time_s ]
EOF
}

trap 'restore_configs; git checkout -q "$ORIG_BRANCH"; rm -rf "$BENCH_BL_COPY" "$BACKUP_DIR" "${PARSER:-}"' EXIT
# The JVM handles Ctrl-C itself and exits normally, so bash would otherwise keep looping.
trap 'echo "Interrupted, stopping." >&2; exit 130' INT TERM

BHEADER="dataset,branch,mode,rep,build_s"
QHEADER="dataset,branch,mode,rep,topk,overquery,qps,mean_latency_ms,recall"
if [ -f "$BCSV" ] && [ "$(head -1 "$BCSV")" != "$BHEADER" ]; then mv "$BCSV" "${OUT}/build.old.$(date +%s).csv"; fi
if [ -f "$QCSV" ] && [ "$(head -1 "$QCSV")" != "$QHEADER" ]; then mv "$QCSV" "${OUT}/query.old.$(date +%s).csv"; fi
[ -f "$BCSV" ] || echo "$BHEADER" > "$BCSV"
[ -f "$QCSV" ] || echo "$QHEADER" > "$QCSV"

# Print the index build time (seconds) from a run's experiments.csv, or nothing.
PARSER="$(mktemp "${TMPDIR:-/tmp}/bench_bl_parse.XXXXXX.py")"
cat > "$PARSER" <<'EOF'
import csv, glob, sys
for path in glob.glob(sys.argv[1] + "/**/experiments.csv", recursive=True):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for col, val in (rows[0].items() if rows else []):
        if val and col.endswith("index_build_time_s"):
            print("%.3f" % float(val))
            sys.exit(0)
EOF

# Print "topk,overquery,qps,mean_latency_ms,recall" for each row of every Recall@10 / Recall@100 table in a
# BenchYAML log. Table columns: Overquery, Avg QPS, Std Dev, CV %, Mean Latency (ms), Avg Visited, Recall@K, ...
parse_query_tables() {
  awk '/Overquery/ && /Recall@(10|100)([[:space:]]|$)/ { k = ($0 ~ /Recall@100([[:space:]]|$)/) ? 100 : 10; f=1; next }
       f==1 && /^---/                                  { f=2; next }
       f==2 && NF==0                                   { f=0; next }
       f==2                                            { print k","$1","$2","$5","$7 }' "$1"
}

BUILT=""
for R in $(seq 1 "$REPS"); do
  if [ $((R % 2)) -eq 1 ]; then ORDER="$BRANCHES"; else ORDER="$REVERSED"; fi
  for BR in $ORDER; do
    echo "################ Rep $R, branch $BR ################"
    restore_configs
    git checkout -q "$BR" || { echo "checkout $BR failed" >&2; exit 1; }

    CP_FILE="${OUT}/${BR//\//_}.classpath"
    if { [ "$BUILD" = true ] && [ "$BUILT" != "$BR" ]; } || [ ! -f "$CP_FILE" ]; then
      # Branches use different native build dir layouts; stale output breaks 'meson setup --wipe'.
      rm -rf "${REPO_ROOT}"/jvector-native/target/meson-build*
      $MVN -q -pl jvector-examples -am package -DskipTests \
          dependency:build-classpath -Dmdep.outputFile="$CP_FILE" || { echo "build failed on $BR" >&2; exit 1; }
      # build-classpath omits the module's own output; add it
      echo -n ":${REPO_ROOT}/jvector-examples/target/classes" >> "$CP_FILE"
      BUILT="$BR"
    fi
    CP="$(cat "$CP_FILE")"

    write_configs

    for DATASET in $DATASETS; do
      for MODE in $MODES; do
        TAG="${DATASET}-${BR//\//_}-${MODE}-r${R}"
        LOG="${OUT}/${TAG}.log"
        RUN_DIR="${OUT}/runs/${TAG}"
        rm -rf "$RUN_DIR"
        write_run_config "$RUN_DIR"
        echo "--- dataset=$DATASET branch=$BR mode=$MODE rep=$R threads=$THREADS"
        java -Xmx${XMX} \
          --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED --enable-preview \
          -Djvector.vectorization_provider=${MODE} -Djvector.physical_core_count=${THREADS} \
          -cp "$CP" io.github.jbellis.jvector.example.BenchYAML "$DATASET" > "$LOG" 2>&1
        BUILDT="$(python3 -I "$PARSER" "$RUN_DIR")"
        if [ -z "$BUILDT" ]; then
          echo "    build time: NONE FOUND (see $LOG)"
        else
          echo "    index_build_s=$BUILDT"
          echo "${DATASET},${BR},${MODE},${R},${BUILDT}" >> "$BCSV"
        fi
        ROWS="$(parse_query_tables "$LOG")"
        if [ -z "$ROWS" ]; then
          echo "    query results: NONE FOUND (see $LOG)"
        else
          while IFS=, read -r K OQ QPS LAT REC; do
            printf "    top%-4s overquery=%-6s qps=%-10s mean_latency_ms=%-8s recall=%s\n" "$K" "$OQ" "$QPS" "$LAT" "$REC"
            echo "${DATASET},${BR},${MODE},${R},${K},${OQ},${QPS},${LAT},${REC}" >> "$QCSV"
          done <<< "$ROWS"
        fi
      done
    done
  done
done

echo
echo "Results: $BCSV and $QCSV"
echo "Plot:    python3 plot_bench_build_latency.py ${OUT}"
