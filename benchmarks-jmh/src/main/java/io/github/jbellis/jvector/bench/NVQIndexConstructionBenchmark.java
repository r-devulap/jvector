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

import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Benchmark measuring NVQ index construction time, broken down into:
 * <ul>
 *   <li>Training ({@code NVQuantization.compute}) — computes the global mean and subvector partitioning.</li>
 *   <li>Encoding ({@code encodeAll}) — quantizes all vectors given a pre-trained NVQ model.</li>
 *   <li>Full pipeline — both phases end-to-end.</li>
 * </ul>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-preview", "-Djvector.experimental.enable_native_vectorization=true"})
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Threads(1)
public class NVQIndexConstructionBenchmark {
    private static final Logger log = LoggerFactory.getLogger(NVQIndexConstructionBenchmark.class);
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();

    private RandomAccessVectorValues ravv;
    private NVQuantization nvq;

    @Param({"2", "4", "8"})
    int nSubVectors;

    @Param({"768", "1536"})
    int dimension;

    @Param({"100000"})
    int vectorCount;

    @Setup(Level.Trial)
    public void setup() {
        log.info("Creating dataset: dimension={}, vectorCount={}", dimension, vectorCount);
        List<VectorFloat<?>> vectors = new ArrayList<>(vectorCount);
        for (int i = 0; i < vectorCount; i++) {
            VectorFloat<?> v = VECTOR_TYPE_SUPPORT.createFloatVector(dimension);
            for (int j = 0; j < dimension; j++) {
                v.set(j, (float) Math.random());
            }
            vectors.add(v);
        }
        ravv = new ListRandomAccessVectorValues(vectors, dimension);
        log.info("Dataset created");
    }

    @Setup(Level.Invocation)
    public void setupPerInvocation() {
        // Pre-compute NVQ so nvqEncodeAllBenchmark measures only encodeAll
        nvq = NVQuantization.compute(ravv, nSubVectors);
    }

    /**
     * Benchmarks NVQ training: computing the global mean and subvector structure.
     */
    @Benchmark
    public void nvqComputeBenchmark(Blackhole bh) {
        bh.consume(NVQuantization.compute(ravv, nSubVectors));
    }

    /**
     * Benchmarks NVQ encoding: quantizing all vectors given a pre-computed NVQ model.
     * Uses the common ForkJoinPool for parallelism.
     */
    @Benchmark
    public void nvqEncodeAllBenchmark(Blackhole bh) {
        bh.consume(nvq.encodeAll(ravv));
    }

    /**
     * Benchmarks the full NVQ construction pipeline: compute + encodeAll.
     */
    @Benchmark
    public void nvqFullConstructionBenchmark(Blackhole bh) {
        NVQuantization computed = NVQuantization.compute(ravv, nSubVectors);
        var encoded = computed.encodeAll(ravv);
        bh.consume(computed);
        bh.consume(encoded);
    }
}
