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

package io.github.jbellis.jvector.example.tutorial;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

// This file provides the entirety of the code used in the bulk scoring tutorial.
// Changes to this file should be applied to `4-bulk-scoring-tutorial.md` as well.
public class BulkScoring {
    public static void main(String[] args) throws IOException {
        VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

        // Short vectors are where bulk scoring helps most: the work per distance computation is small,
        // so the fixed cost of each call into the native library is a large share of the total.
        int dimension = 128;
        int size = 50_000;
        VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.EUCLIDEAN;

        Random random = new Random(42);
        List<VectorFloat<?>> baseVectors = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            float[] values = new float[dimension];
            for (int j = 0; j < dimension; j++) {
                values[j] = random.nextFloat();
            }
            baseVectors.add(vts.createFloatVector(values));
        }
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(baseVectors, dimension);

        // Part 1: scoring a batch of nodes directly.
        // `contiguousRandomAccessScoreProvider` copies all vectors into one array, which is what lets the
        // score function compute many similarities in a single call.
        BuildScoreProvider bulkBsp = BuildScoreProvider.contiguousRandomAccessScoreProvider(ravv, similarityFunction);

        var scoreFunction = bulkBsp.searchProviderFor(0).scoreFunction();

        // Not every score function can do this efficiently, so check first. If it returns false,
        // `similarityToBulk` still works, it just falls back to calling `similarityTo` in a loop.
        System.out.println("supportsBulkSimilarity: " + scoreFunction.supportsBulkSimilarity());

        int[] nodes = {1, 5, 42, 1000, 49_999};
        float[] scores = new float[nodes.length];  // must be at least as long as the count passed in
        scoreFunction.similarityToBulk(nodes, nodes.length, scores);

        for (int i = 0; i < nodes.length; i++) {
            // The bulk result is the same as scoring each node on its own.
            System.out.println("node " + nodes[i] + ": bulk=" + scores[i] + ", single=" + scoreFunction.similarityTo(nodes[i]));
        }

        // Part 2: using it for graph construction.
        // The graph builder uses bulk scoring automatically when its BuildScoreProvider supports it,
        // so the only change is which provider you pass in.
        BuildScoreProvider plainBsp = BuildScoreProvider.randomAccessScoreProvider(ravv, similarityFunction);

        // The first build of each kind includes JIT warm-up, so we run each twice and compare the second.
        for (int round = 1; round <= 2; round++) {
            System.out.printf("round %d: plain %.2fs, bulk %.2fs%n",
                              round,
                              timeBuild(plainBsp, ravv, dimension),
                              timeBuild(bulkBsp, ravv, dimension));
        }
    }

    private static double timeBuild(BuildScoreProvider bsp, RandomAccessVectorValues ravv, int dimension) throws IOException {
        long start = System.nanoTime();
        try (GraphIndexBuilder builder = new GraphIndexBuilder(bsp, dimension, 32, 100, 1.2f, 1.2f, true, true)) {
            builder.build(ravv);
        }
        return (System.nanoTime() - start) / 1_000_000_000.0;
    }
}
