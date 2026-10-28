#!/bin/bash
# Compare index build time (Panama vs native) across build thread counts using BenchYAML.
#
# Usage: ./bench_threads.sh [branches="401 402"] [threads="2 8 16 32"] [datasets="a b c"] [modes="panama native"] [bulk="false true"] [reps=3] [xmx=32G] [skip-build]
#   e.g. ./bench_threads.sh
#        ./bench_threads.sh skip-build datasets="sift-128-euclidean" reps=1
#
# The provider is forced with -Djvector.vectorization_provider=panama|native. This property exists on
# both branches and overrides their differing defaults (native is opt-in on 401, default-on on 402),
# and a forced provider that fails to load throws instead of silently falling back.
#
# Bulk scoring is controlled with -Djvector.bulk_exact_scoring=true|false.
#
# For each branch: checkout, build (once), then for each mode (panama|native), bulk setting (false|true),
# and thread count, run BenchYAML (one dataset per JVM) with -Djvector.physical_core_count=<threads>
# (this sizes PhysicalCoreExecutor, the pool used for graph construction).
#
# Output (in bench-threads-results/):
#   <dataset>-<branch>-<mode>-bulk<bulk>-t<threads>-r<rep>.log   raw logs
#   results.csv     dataset,branch,mode,bulk,threads,rep,build_time_s
#   summary.txt     per-dataset pivot table of mean build times

set -u

BRANCHES="bulk-scoring"
THREADS="1 8 16"
DATASETS="lastfm-64-dot nytimes-256-angular sift-128-euclidean e5-small-v2-100k e5-base-v2-100k"
REPS=2
MODES="panama native"
BULK="false true"
BUILD=true
XMX="32G"

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

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${REPO_ROOT}/bench-threads-results"
CSV="${OUT}/results.csv"
mkdir -p "$OUT"
cd "$REPO_ROOT"

ORIG_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
trap 'git checkout -q "$ORIG_BRANCH"' EXIT
# The JVM handles Ctrl-C itself and exits normally, so bash would otherwise keep looping.
trap 'echo "Interrupted, stopping." >&2; exit 130' INT TERM

HEADER="dataset,branch,mode,bulk,threads,rep,build_time_s"
# Set aside results from an older script version with a different column layout
if [ -f "$CSV" ] && [ "$(head -1 "$CSV")" != "$HEADER" ]; then mv "$CSV" "${OUT}/results.old.$(date +%s).csv"; fi
[ -f "$CSV" ] || echo "$HEADER" > "$CSV"

for BR in $BRANCHES; do
  echo "################ Branch $BR ################"
  git checkout -q "$BR" || { echo "checkout $BR failed" >&2; exit 1; }
  CP_FILE="${OUT}/${BR}.classpath"

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

  for DATASET in $DATASETS; do
    for MODE in $MODES; do
      for B in $BULK; do
        for T in $THREADS; do
          for R in $(seq 1 "$REPS"); do
            LOG="${OUT}/${DATASET}-${BR}-${MODE}-bulk${B}-t${T}-r${R}.log"
            echo "--- dataset=$DATASET branch=$BR mode=$MODE bulk=$B threads=$T rep=$R"
            java -Xmx${XMX} -ea \
              --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED --enable-preview \
              -Djvector.vectorization_provider=${MODE} \
              -Djvector.bulk_exact_scoring=${B} \
              -Djvector.physical_core_count=${T} \
              -cp "$CP" io.github.jbellis.jvector.example.BenchYAML "$DATASET" > "$LOG" 2>&1
            BT="$(grep -m1 'Index build time:' "$LOG" | sed -E 's/.*Index build time: ([0-9.]+) seconds.*/\1/')"
            echo "    build_time_s=${BT:-FAILED}"
            echo "${DATASET},${BR},${MODE},${B},${T},${R},${BT:-NA}" >> "$CSV"
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
    printf "%-22s", "branch/mode/bulk"; for (i=1;i<=nt;i++) printf "%10s", "t="ts[i]; printf "\n"
    m=asorti(keys, ks)
    for (x=1;x<=m;x++) { split(ks[x], a, ","); if (a[1]!=d) continue; k=a[2]; printf "%-22s", k
      for (i=1;i<=nt;i++) { c=d","k","ts[i]; if (n[c]) printf "%10.2f", s[c]/n[c]; else printf "%10s", "-" }
      printf "\n" }
    printf "\n" } }' "$CSV" | tee "${OUT}/summary.txt"
