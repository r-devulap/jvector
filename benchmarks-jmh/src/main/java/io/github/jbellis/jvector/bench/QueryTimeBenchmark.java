
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

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import io.github.jbellis.jvector.example.util.SiftLoader;
import java.util.*;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

/**
 * Benchmarks per-query search latency on a pre-built in-memory index with random vectors.
 * Index construction happens once per trial in @Setup; only the search is measured.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Fork(value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-preview", "-Djvector.experimental.enable_native_vectorization=true"})
@Warmup(iterations = 3)
@Measurement(iterations = 5)
@Threads(1)
public class QueryTimeBenchmark {
    private static final Logger log = LoggerFactory.getLogger(QueryTimeBenchmark.class);
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();

    @Param({"1536"})
    private int originalDimension;

    @Param({"16"})
    private int numberOfPQSubspaces;

    @Param({"10"})
    private int topK;

    private RandomAccessVectorValues ravv;
    private ImmutableGraphIndex graphIndex;
    private PQVectors pqVectors;

    private List<VectorFloat<?>> queryVectors;
    private List<VectorFloat<?>> baseVectors;
    private int queryIndex;

    private static final int NUM_QUERY_VECTORS = 1000;
    private static final int M = 32;
    private static final int BEAM_WIDTH = 100;

    @Setup(Level.Trial)
    public void setup() throws IOException {
        // read vectors from a file: 
        baseVectors = SiftLoader.readFvecs("./fvec/wikipedia_squad/100k/ada_002_100000_base_vectors.fvec");
        queryVectors = SiftLoader.readFvecs("./fvec/wikipedia_squad/100k/ada_002_100000_query_vectors_10000.fvec");
        ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);

        // Build index once — not measured
        final BuildScoreProvider buildScoreProvider;
        log.info("Building with PQ ({} subspaces), dim={}", numberOfPQSubspaces, originalDimension);
        ProductQuantization pq = ProductQuantization.compute(ravv, numberOfPQSubspaces, 256, true);
        pqVectors = (PQVectors) pq.encodeAll(ravv);
        buildScoreProvider = BuildScoreProvider.pqBuildScoreProvider(VectorSimilarityFunction.EUCLIDEAN, pqVectors);

        var builder = new GraphIndexBuilder(buildScoreProvider,
                ravv.dimension(),
                M, // graph degree
                BEAM_WIDTH, // construction search depth
                1.2f, // allow degree overflow during construction by this factor
                1.2f, // relax neighbor diversity requirement by this factor
                true); // add the hierarchy
        graphIndex = builder.build(ravv);

        // Pre-generate query vectors so vector creation is not part of the measurement
        queryIndex = 0;
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        // graphIndex is AutoCloseable only if wrapped; nothing to do for ImmutableGraphIndex
    }

    /**
     * Measures the time to execute a single query against the pre-built index.
     * A pool of pre-generated query vectors is cycled through to avoid branch-prediction effects.
     */
    @Benchmark
    public void queryBenchmark(Blackhole blackhole) throws IOException {
        VectorFloat<?> queryVector = queryVectors.get(queryIndex);
        queryIndex = (queryIndex + 1) % NUM_QUERY_VECTORS;

        GraphSearcher searcher = new GraphSearcher(graphIndex);
        final SearchResult result;
        var asf = pqVectors.precomputedScoreFunctionFor(queryVector, VectorSimilarityFunction.EUCLIDEAN);
        var reranker = ravv.rerankerFor(queryVector, VectorSimilarityFunction.EUCLIDEAN);
        var ssp = new DefaultSearchScoreProvider(asf, reranker);
        result = searcher.search(ssp,
                topK,
                topK * 2,
                0.0f,
                0.0f,
                Bits.ALL);
        blackhole.consume(result);
    }
}
