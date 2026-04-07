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

import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.function.IntFunction;

/**
 * Benchmarks two-pass ANN query against a graph index that stores both FUSED_PQ and NVQ features:
 * <ol>
 *   <li>First pass: approximate scoring via FUSED_PQ (Quick ADC over neighbour PQ codes).</li>
 *   <li>Second pass: exact reranking via NVQ.</li>
 * </ol>
 * Index construction (graph build + disk write) is performed once in {@link #setup()}.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-preview", "-Djvector.experimental.enable_native_vectorization=true"})
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Threads(1)
public class FusedPQNVQQueryBenchmark {
    private static final Logger log = LoggerFactory.getLogger(FusedPQNVQQueryBenchmark.class);
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final VectorSimilarityFunction VSF = VectorSimilarityFunction.EUCLIDEAN;

    private static final int GRAPH_DEGREE = 32;
    private static final int EF_CONSTRUCTION = 100;
    private static final float NEIGHBOR_OVERFLOW = 1.2f;
    private static final float ALPHA = 1.2f;

    @Param({"1536"})
    int dimension;

    @Param({"100000"})
    int numBaseVectors;

    @Param({"100"})
    int numQueryVectors;

    /** Number of PQ subspaces. Must divide {@code dimension}. */
    @Param({"64"})
    int pqSubspaces;

    /** Number of NVQ subvectors. Must divide {@code dimension}. */
    @Param({"8"})
    int nvqSubVectors;

    @Param({"10"})
    int topK;

    /** How many approximate candidates to gather before NVQ reranking (rerankK). */
    @Param({"100"})
    int efSearch;

    private List<VectorFloat<?>> queryVectors;
    private OnDiskGraphIndex onDiskIndex;
    private Path indexPath;

    @Setup(Level.Trial)
    public void setup() throws IOException {
        log.info("Building index: dimension={}, numBaseVectors={}, pqSubspaces={}, nvqSubVectors={}",
                dimension, numBaseVectors, pqSubspaces, nvqSubVectors);

        // 1. Generate random base vectors
        List<VectorFloat<?>> baseVectors = new ArrayList<>(numBaseVectors);
        for (int i = 0; i < numBaseVectors; i++) {
            baseVectors.add(createRandomVector(dimension));
        }
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(baseVectors, dimension);

        // 2. Train PQ and encode all vectors
        ProductQuantization pq = ProductQuantization.compute(ravv, pqSubspaces, 256, true);
        PQVectors pqVectors = (PQVectors) pq.encodeAll(ravv);

        // 3. Build in-memory graph using PQ approximate scoring
        ImmutableGraphIndex graph;
        try (var builder = new GraphIndexBuilder(
                BuildScoreProvider.pqBuildScoreProvider(VSF, pqVectors),
                dimension, GRAPH_DEGREE, EF_CONSTRUCTION, NEIGHBOR_OVERFLOW, ALPHA, false)) {
            graph = builder.build(ravv);
        }

        // 4. Train NVQ using the same base vectors
        NVQuantization nvq = NVQuantization.compute(ravv, nvqSubVectors);

        // 5. Write graph to disk with FUSED_PQ (first pass) + NVQ (reranker) features
        indexPath = Files.createTempFile("fused-pq-nvq-bench-", ".odgi");
        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, indexPath)
                .with(new FusedPQ(graph.maxDegree(), pq))
                .with(new NVQ(nvq))
                .withMapper(new OrdinalMapper.IdentityMapper(numBaseVectors - 1))
                .build()) {
            var graphView = graph.getView();
            Map<FeatureId, IntFunction<Feature.State>> suppliers = new EnumMap<>(FeatureId.class);
            suppliers.put(FeatureId.FUSED_PQ, ordinal -> new FusedPQ.State(graphView, pqVectors, ordinal));
            suppliers.put(FeatureId.NVQ_VECTORS, ordinal -> new NVQ.State(nvq.encode(ravv.getVector(ordinal))));
            writer.write(suppliers);
            graphView.close();
        }

        // 6. Load on-disk index
        onDiskIndex = OnDiskGraphIndex.load(ReaderSupplierFactory.open(indexPath));

        // 7. Generate query vectors
        queryVectors = new ArrayList<>(numQueryVectors);
        for (int i = 0; i < numQueryVectors; i++) {
            queryVectors.add(createRandomVector(dimension));
        }

        log.info("Index ready: {} vectors, file size {} bytes", numBaseVectors, Files.size(indexPath));
    }

    @TearDown(Level.Trial)
    public void tearDown() throws IOException {
        if (onDiskIndex != null) {
            onDiskIndex.close();
        }
        Files.deleteIfExists(indexPath);
    }

    /**
     * Two-pass ANN query:
     * <ol>
     *   <li>First pass — {@link FusedPQ} approximate scoring (Quick ADC over fused neighbour PQ codes).</li>
     *   <li>Second pass — {@link NVQ} exact reranking of the top {@code efSearch} approximate candidates.</li>
     * </ol>
     */
    @Benchmark
    @SuppressWarnings("deprecation")
    public void queryFusedPQWithNVQRerank(Blackhole bh) throws IOException {
        for (VectorFloat<?> query : queryVectors) {
            try (var searcher = new GraphSearcher(onDiskIndex)) {
                var view = (ImmutableGraphIndex.ScoringView) searcher.getView();
                // First pass: FUSED_PQ approximate score function (Quick ADC)
                ScoreFunction.ApproximateScoreFunction asf = view.approximateScoreFunctionFor(query, VSF);
                // Second pass: NVQ reranker
                ScoreFunction.ExactScoreFunction reranker = view.rerankerFor(query, VSF);
                var ssp = new DefaultSearchScoreProvider(asf, reranker);
                bh.consume(searcher.search(ssp, topK, efSearch, 0.0f, 0.0f, Bits.ALL));
            }
        }
    }

    private VectorFloat<?> createRandomVector(int dim) {
        VectorFloat<?> v = VECTOR_TYPE_SUPPORT.createFloatVector(dim);
        for (int i = 0; i < dim; i++) {
            v.set(i, (float) Math.random());
        }
        return v;
    }
}
