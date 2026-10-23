#!/bin/bash
# Build and/or run VectorUtilDistanceBenchmark.
#
# Usage: ./run.sh [build] [run] [native=on|off] [extra JMH args...]
#
#   build          compile jvector and package the runnable JMH jar
#   run            run the benchmark; by default runs twice, first with native
#                  vectorization on (Highway/FFM), then off (JDK Vector API)
#   native=on|off  restrict 'run' to just that one implementation
#   anything else  is passed through to JMH, e.g. -p dimension=1024 -wi 1 -i 2
#
# Examples:
#   ./run.sh build run                  # build, then run native=on and native=off
#   ./run.sh run native=on              # native only
#   ./run.sh run native=off VectorUtilDistanceBenchmark.dotProduct -p dimension=1024
#   ./run.sh build

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
JAR="${REPO_ROOT}/benchmarks-jmh/target/compactor-benchmark.jar"

DO_BUILD=false
DO_RUN=false
NATIVE_MODES=(on off)
JMH_ARGS=()

for arg in "$@"; do
  case "$arg" in
    build)      DO_BUILD=true ;;
    run)        DO_RUN=true ;;
    native=on)  NATIVE_MODES=(on) ;;
    native=off) NATIVE_MODES=(off) ;;
    *)          JMH_ARGS+=("$arg") ;;
  esac
done

if [ "$DO_BUILD" = false ] && [ "$DO_RUN" = false ]; then
  sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi

if [ "$DO_BUILD" = true ]; then
  (cd "$REPO_ROOT" && ./mvnw -pl benchmarks-jmh -am package -DskipTests)
fi

if [ "$DO_RUN" = true ]; then
  if [ ! -f "$JAR" ]; then
    echo "ERROR: $JAR not found. Run with 'build' first." >&2
    exit 1
  fi
  # Default to this benchmark unless the caller named one (a non-option first arg).
  if [ ${#JMH_ARGS[@]} -eq 0 ] || [[ "${JMH_ARGS[0]}" == -* ]]; then
    JMH_ARGS=("VectorUtilDistanceBenchmark" "${JMH_ARGS[@]}")
  fi

  for NATIVE in "${NATIVE_MODES[@]}"; do
    if [ "$NATIVE" = on ]; then NATIVE_FLAG=true; else NATIVE_FLAG=false; fi
    echo
    echo "===== Running ${JMH_ARGS[0]} with native vectorization: ${NATIVE} ====="

    JVM_FLAGS="--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED --enable-preview -Djvector.experimental.enable_native_vectorization=${NATIVE_FLAG}"

    # -jvmArgsAppend overrides the @Fork annotation, so the forked JVM gets the chosen native setting.
    java --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED --enable-preview \
      -jar "$JAR" -jvmArgsAppend "$JVM_FLAGS" "${JMH_ARGS[@]}"
  done
fi
