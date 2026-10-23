/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.github.jbellis.jvector.bench;

import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.openjdk.jmh.annotations.*;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * Standalone microbenchmark for the float distance primitives in {@link VectorUtil}
 * (squared L2, cosine, dot product) across a range of vector dimensions.
 *
 * <h2>Compile</h2>
 * From the repository root (builds jvector-base and the other upstream modules, and produces the
 * shaded, runnable JMH jar):
 * <pre>
 * ./mvnw -pl benchmarks-jmh -am package -DskipTests
 * </pre>
 * The runnable jar is {@code benchmarks-jmh/target/compactor-benchmark.jar}. (The plain
 * {@code benchmarks-jmh-*.jar} has no main manifest and cannot be run with {@code java -jar}.)
 *
 * <h2>Run</h2>
 * <pre>
 * java --add-modules=jdk.incubator.vector --enable-preview \
 *   -jar benchmarks-jmh/target/compactor-benchmark.jar VectorUtilDistanceBenchmark
 * </pre>
 *
 * <h2>Useful options</h2>
 * <pre>
 * # a single dimension / a single method
 * ... VectorUtilDistanceBenchmark.dotProduct -p dimension=1024
 *
 * # quick run: 1 warmup, 2 measurement iterations, 1s each
 * ... VectorUtilDistanceBenchmark -wi 1 -i 2 -w 1 -r 1
 *
 * # write results to CSV
 * ... VectorUtilDistanceBenchmark -rf csv -rff results.csv
 * </pre>
 *
 * Note: the forked JVM disables native vectorization
 * ({@code -Djvector.experimental.enable_native_vectorization=false}), so this measures the
 * Panama (JDK Vector API) implementation.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Thread)
@Fork(value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED", "--enable-preview", "-Djvector.experimental.enable_native_vectorization=false"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Threads(1)
public class VectorUtilDistanceBenchmark {
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();

    @Param({"128", "256", "1024", "2048"})
    private int dimension;

    private VectorFloat<?> a;
    private VectorFloat<?> b;

    @Setup
    public void setup() {
        Random random = new Random(42);
        a = randomVector(random, dimension);
        b = randomVector(random, dimension);
    }

    @Benchmark
    public float squareL2Distance() {
        return VectorUtil.squareL2Distance(a, b);
    }

    @Benchmark
    public float cosine() {
        return VectorUtil.cosine(a, b);
    }

    @Benchmark
    public float dotProduct() {
        return VectorUtil.dotProduct(a, b);
    }

    private static VectorFloat<?> randomVector(Random random, int dimension) {
        VectorFloat<?> vector = VECTOR_TYPE_SUPPORT.createFloatVector(dimension);
        for (int i = 0; i < dimension; i++) {
            vector.set(i, random.nextFloat());
        }
        return vector;
    }
}
