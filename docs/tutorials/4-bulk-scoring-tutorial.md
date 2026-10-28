# JVector Tutorial Part 4: Bulk Scoring

When JVector builds a graph, most of the time goes into computing similarity scores: for every node it inserts, it scores the neighbors of many candidate nodes. By default each score is a separate call.

With the native vectorization provider, each of those calls crosses from Java into native code. For short vectors, such as 128 dimensions, the native work is only about ten nanoseconds, so the cost of making the call is a large share of the total. **Bulk scoring** removes most of that cost by scoring many nodes against the same query in a single call.

This tutorial shows how to use it. It assumes you have read [Part 1](./1-intro-tutorial.md).

> [!NOTE]
> Bulk scoring only speeds things up with the native provider (`-Djvector.experimental.enable_native_vectorization=true`, or `-Djvector.vectorization_provider=native`). With the Panama or default providers it still gives correct results, but it just costs an extra copy of the vectors.

## Creating a bulk-capable score provider

Start with an ordinary `RandomAccessVectorValues`, as in Part 1:

```java
int dimension = 128;
VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.EUCLIDEAN;
RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(baseVectors, dimension);
```

Then, instead of `randomAccessScoreProvider`, ask for the contiguous version:

```java
// `contiguousRandomAccessScoreProvider` copies all vectors into one array, which is what lets the
// score function compute many similarities in a single call.
BuildScoreProvider bulkBsp = BuildScoreProvider.contiguousRandomAccessScoreProvider(ravv, similarityFunction);
```

> [!IMPORTANT]
> The copy means the vectors are held in memory twice (once in your RAVV, once in the provider). The RAVV must be dense and must not change after you create the provider.

> [!NOTE]
> Bulk scoring supports `EUCLIDEAN` and `DOT_PRODUCT`. For `COSINE`, or if the vectors do not fit in a single array (around 2 billion floats), `contiguousRandomAccessScoreProvider` quietly returns the same provider as `randomAccessScoreProvider`.

## Scoring a batch of nodes

Bulk scoring is exposed on `ScoreFunction`. Get one for a query node and score several nodes at once:

```java
var scoreFunction = bulkBsp.searchProviderFor(0).scoreFunction();

// Not every score function can do this efficiently, so check first. If it returns false,
// `similarityToBulk` still works, it just falls back to calling `similarityTo` in a loop.
System.out.println("supportsBulkSimilarity: " + scoreFunction.supportsBulkSimilarity());

int[] nodes = {1, 5, 42, 1000, 49_999};
float[] scores = new float[nodes.length];  // must be at least as long as the count passed in
scoreFunction.similarityToBulk(nodes, nodes.length, scores);
```

`scores[i]` is the similarity between the query and `nodes[i]`, identical to what `scoreFunction.similarityTo(nodes[i])` returns. The `count` argument lets you reuse one pair of arrays for batches of different sizes.

## Using it for graph construction

You do not call `similarityToBulk` yourself when building a graph. The graph searches that run during construction use it automatically whenever the score function supports it, so the only change is the provider you pass to the builder:

```java
BuildScoreProvider plainBsp = BuildScoreProvider.randomAccessScoreProvider(ravv, similarityFunction);

try (GraphIndexBuilder builder = new GraphIndexBuilder(bulkBsp, dimension, 32, 100, 1.2f, 1.2f, true, true)) {
    builder.build(ravv);
}
```

> [!NOTE]
> Only the neighbor scoring during graph search is batched. The diversity check that prunes edges is unchanged.

## When it helps

Bulk scoring reduces per-call overhead, so the benefit depends on how much work each call does:

- **Helps most:** short vectors (tens to a few hundred dimensions) with the native provider.
- **Helps less:** long vectors, where the native computation itself dominates.
- **Does not apply:** Product Quantization builds (`pqBuildScoreProvider`), which use a different score function and do not currently support bulk scoring.

Measure on your own data. The example below builds the same index with both providers and prints the times, so you can see the difference on your machine.

## Full Example

The complete code from this tutorial is available in `BulkScoring.java`. Run it from the root of this repo using:

```sh
mvn compile exec:exec@tutorial -Dtutorial=bulk -pl jvector-examples -am
```

## Disabling it in the benchmark harness

The benchmark harness (`Grid`, used by `BenchYAML`) uses the bulk-capable provider by default for full-resolution builds. To turn it off, for example to avoid the extra memory, pass:

```sh
-Djvector.bulk_exact_scoring=false
```
