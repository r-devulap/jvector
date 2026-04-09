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

#include <immintrin.h>
#include <inttypes.h>
#include <math.h>
#include "jvector_simd.h"

__m512i initialIndexRegister;
__m512i indexIncrement;
__m512i maskSeventhBit;
__m512i maskEighthBit;

__attribute__((constructor))
void initialize_constants() {
    if (check_compatibility()) {
        initialIndexRegister = _mm512_setr_epi32(-16, -15, -14, -13, -12, -11, -10, -9,
                                             -8, -7, -6, -5, -4, -3, -2, -1);
        indexIncrement = _mm512_set1_epi32(16);
        maskSeventhBit = _mm512_set1_epi16(0x0040);
        maskEighthBit = _mm512_set1_epi16(0x0080);
    }
}

float dot_product_f32_64(const float* a, int aoffset, const float* b, int boffset) {

     __m128 va = _mm_castsi128_ps(_mm_loadl_epi64((__m128i *)(a + aoffset)));
     __m128 vb = _mm_castsi128_ps(_mm_loadl_epi64((__m128i *)(b + boffset)));
     __m128 r  = _mm_mul_ps(va, vb); // Perform element-wise multiplication

    // Horizontal sum of the vector to get dot product
    __attribute__((aligned(16))) float result[4];
    _mm_store_ps(result, r);
    return result[0] + result[1];
}

float dot_product_f32_128(const float* a, int aoffset, const float* b, int boffset, int length) {
    float dot = 0.0;
    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;
    int simd_length = length - (length % 4);

    if (length >= 4) {
        __m128 sum = _mm_setzero_ps();

        for(; ao < aoffset + simd_length; ao += 4, bo += 4) {
            // Load float32
            __m128 va = _mm_loadu_ps(a + ao);
            __m128 vb = _mm_loadu_ps(b + bo);

            // Multiply and accumulate
            sum = _mm_fmadd_ps(va, vb, sum);
        }

        // Horizontal sum of the vector to get dot product
        __attribute__((aligned(16))) float result[4];
        _mm_store_ps(result, sum);

        for(int i = 0; i < 4; ++i) {
            dot += result[i];
        }
    }

    for (; ao < alim && bo < blim; ao++, bo++) {
        dot += a[ao] * b[bo];
    }

    return dot;
}

float dot_product_f32_256(const float* a, int aoffset, const float* b, int boffset, int length) {
    float dot = 0.0;
    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;
    int simd_length = length - (length % 8);

    if (length >= 8) {
        __m256 sum = _mm256_setzero_ps();

        for(; ao < aoffset + simd_length; ao += 8, bo += 8) {
            // Load float32
            __m256 va = _mm256_loadu_ps(a + ao);
            __m256 vb = _mm256_loadu_ps(b + bo);

            // Multiply and accumulate
            sum = _mm256_fmadd_ps(va, vb, sum);
        }

        // Horizontal sum of the vector to get dot product
        __attribute__((aligned(32))) float result[8];
        _mm256_store_ps(result, sum);

        for(int i = 0; i < 8; ++i) {
            dot += result[i];
        }
    }

    for (; ao < alim && bo < blim; ao++, bo++) {
        dot += a[ao] * b[bo];
    }

    return dot;
}

float dot_product_f32_512(const float* a, int aoffset, const float* b, int boffset, int length) {
    float dot = 0.0;
    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;
    int simd_length = length - (length % 16);

    if (length >= 16) {
        __m512 sum = _mm512_setzero_ps();
        for(; ao < aoffset + simd_length; ao += 16, bo += 16) {
            // Load float32
            __m512 va = _mm512_loadu_ps(a + ao);
            __m512 vb = _mm512_loadu_ps(b + bo);

            // Multiply and accumulate
            sum = _mm512_fmadd_ps(va, vb, sum);
        }

        // Horizontal sum of the vector to get dot product
        dot = _mm512_reduce_add_ps(sum);
    }

    for (; ao < alim && bo < blim; ao++, bo++) {
        dot += a[ao] * b[bo];
    }

    return dot;
}

float dot_product_f32(int preferred_size, const float* a, int aoffset, const float* b, int boffset, int length) {
    if (length == 2)
        return dot_product_f32_64(a, aoffset, b, boffset);
    if (length <= 7)
        return dot_product_f32_128(a, aoffset, b, boffset, length);

    return (preferred_size == 512 && length >= 16)
           ? dot_product_f32_512(a, aoffset, b, boffset, length)
           : dot_product_f32_256(a, aoffset, b, boffset, length);
}

float euclidean_f32_64(const float* a, int aoffset, const float* b, int boffset) {
     __m128 va = _mm_castsi128_ps(_mm_loadl_epi64((__m128i *)(a + aoffset)));
     __m128 vb = _mm_castsi128_ps(_mm_loadl_epi64((__m128i *)(b + boffset)));
     __m128 r  = _mm_sub_ps(va, vb);
     r = _mm_mul_ps(r, r);

    // Horizontal sum of the vector to get square distance
    __attribute__((aligned(8))) float result[2];
    _mm_store_ps(result, r);
    return result[0] + result[1];
}

float euclidean_f32_128(const float* a, int aoffset, const float* b, int boffset, int length) {
    float squareDistance = 0.0;
    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;
    int simd_length = length - (length % 4);

    if (length >= 4) {
        __m128 sum = _mm_setzero_ps();

        for(; ao < aoffset + simd_length; ao += 4, bo += 4) {
            // Load float32
            __m128 va = _mm_loadu_ps(a + ao);
            __m128 vb = _mm_loadu_ps(b + bo);
            __m128 diff = _mm_sub_ps(va, vb);
            // Multiply and accumulate
            sum = _mm_fmadd_ps(diff, diff, sum);
        }

        // Horizontal sum of the vector to get dot product
        __attribute__((aligned(16))) float result[4];
        _mm_store_ps(result, sum);

        for(int i = 0; i < 4; ++i) {
            squareDistance += result[i];
        }
    }

    for (; ao < alim && bo < blim; ao++, bo++) {
        float diff = a[ao] - b[bo];
        squareDistance += diff * diff;
    }

    return squareDistance;
}

float euclidean_f32_256(const float* a, int aoffset, const float* b, int boffset, int length) {
    float squareDistance = 0.0;
    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;
    int simd_length = length - (length % 8);

    if (length >= 8) {
        __m256 sum = _mm256_setzero_ps();

        for(; ao < aoffset + simd_length; ao += 8, bo += 8) {
            // Load float32
            __m256 va = _mm256_loadu_ps(a + ao);
            __m256 vb = _mm256_loadu_ps(b + bo);
            __m256 diff = _mm256_sub_ps(va, vb);

            // Multiply and accumulate
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        __attribute__((aligned(32))) float result[8];
        _mm256_store_ps(result, sum);

        for(int i = 0; i < 8; ++i) {
            squareDistance += result[i];
        }
    }

    for (; ao < alim && bo < blim; ao++, bo++) {
        float diff = a[ao] - b[bo];
        squareDistance += diff * diff;
    }

    return squareDistance;
}

float euclidean_f32_512(const float* a, int aoffset, const float* b, int boffset, int length) {
    float squareDistance = 0.0;
    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;
    int simd_length = length - (length % 16);

    if (length >= 16) {
        __m512 sum = _mm512_setzero_ps();
        for(; ao < aoffset + simd_length; ao += 16, bo += 16) {
            // Load float32
            __m512 va = _mm512_loadu_ps(a + ao);
            __m512 vb = _mm512_loadu_ps(b + bo);
            __m512 diff = _mm512_sub_ps(va, vb);

            // Multiply and accumulate
            sum = _mm512_fmadd_ps(diff, diff, sum);
        }

        // Horizontal sum of the vector to get dot product
        squareDistance = _mm512_reduce_add_ps(sum);
    }

    for (; ao < alim && bo < blim; ao++, bo++) {
        float diff = a[ao] - b[bo];
        squareDistance += diff * diff;
    }

    return squareDistance;
}

float euclidean_f32(int preferred_size, const float* a, int aoffset, const float* b, int boffset, int length) {
    if (length == 2)
        return euclidean_f32_64(a, aoffset, b, boffset);
    if (length <= 7)
        return euclidean_f32_128(a, aoffset, b, boffset, length);

    return (preferred_size == 512 && length >= 16)
           ? euclidean_f32_512(a, aoffset, b, boffset, length)
           : euclidean_f32_256(a, aoffset, b, boffset, length);
}

float assemble_and_sum_f32_512(const float* data, int dataBase, const unsigned char* baseOffsets, int baseOffsetsOffset, int baseOffsetsLength) {
    __m512 sum = _mm512_setzero_ps();
    int i = 0;
    int limit = baseOffsetsLength - (baseOffsetsLength % 16);
    __m512i indexRegister = initialIndexRegister;
    __m512i dataBaseVec = _mm512_set1_epi32(dataBase);
    baseOffsets = baseOffsets + baseOffsetsOffset;

    for (; i < limit; i += 16) {
        __m128i baseOffsetsRaw = _mm_loadu_si128((__m128i *)(baseOffsets + i));
        __m512i baseOffsetsInt = _mm512_cvtepu8_epi32(baseOffsetsRaw);
        // we have base offsets int, which we need to scale to index into data.
        // first, we want to initialize a vector with the lane number added as an index
        indexRegister = _mm512_add_epi32(indexRegister, indexIncrement);
        // then we want to multiply by dataBase
        __m512i scale = _mm512_mullo_epi32(indexRegister, dataBaseVec);
        // then we want to add the base offsets
        __m512i convOffsets = _mm512_add_epi32(scale, baseOffsetsInt);

        __m512 partials = _mm512_i32gather_ps(convOffsets, data, 4);
        sum = _mm512_add_ps(sum, partials);
    }

    float res = _mm512_reduce_add_ps(sum);
    for (; i < baseOffsetsLength; i++) {
        res += data[dataBase * i + baseOffsets[i]];
    }

    return res;
}

float pq_decoded_cosine_similarity_f32_512(const unsigned char* baseOffsets, int baseOffsetsOffset, int baseOffsetsLength, int clusterCount, const float* partialSums, const float* aMagnitude, float bMagnitude) {
    __m512 sum = _mm512_setzero_ps();
    __m512 vaMagnitude = _mm512_setzero_ps();
    int i = 0;
    int limit = baseOffsetsLength - (baseOffsetsLength % 16);
    __m512i indexRegister = initialIndexRegister;
    __m512i scale = _mm512_set1_epi32(clusterCount);
    baseOffsets = baseOffsets + baseOffsetsOffset;


    for (; i < limit; i += 16) {
        // Load and convert baseOffsets to integers
        __m128i baseOffsetsRaw = _mm_loadu_si128((__m128i *)(baseOffsets + i));
        __m512i baseOffsetsInt = _mm512_cvtepu8_epi32(baseOffsetsRaw);

        indexRegister = _mm512_add_epi32(indexRegister, indexIncrement);
        // Scale the baseOffsets by the cluster count
        __m512i scaledOffsets = _mm512_mullo_epi32(indexRegister, scale);

        // Calculate the final convOffsets by adding the scaled indexes and the base offsets
        __m512i convOffsets = _mm512_add_epi32(scaledOffsets, baseOffsetsInt);

        // Gather and sum values for partial sums and a magnitude
        __m512 partialSumVals = _mm512_i32gather_ps(convOffsets, partialSums, 4);
        sum = _mm512_add_ps(sum, partialSumVals);

        __m512 aMagnitudeVals = _mm512_i32gather_ps(convOffsets, aMagnitude, 4);
        vaMagnitude = _mm512_add_ps(vaMagnitude, aMagnitudeVals);
    }

    // Reduce sums
    float sumResult = _mm512_reduce_add_ps(sum);
    float aMagnitudeResult = _mm512_reduce_add_ps(vaMagnitude);

    // Handle the remaining elements
    for (; i < baseOffsetsLength; i++) {
        int offset = clusterCount * i + baseOffsets[i];
        sumResult += partialSums[offset];
        aMagnitudeResult += aMagnitude[offset];
    }

    return sumResult / sqrtf(aMagnitudeResult * bMagnitude);
}

void calculate_partial_sums_dot_f32_512(const float* codebook, int codebookIndex, int size, int clusterCount, const float* query, int queryOffset, float* partialSums) {
    int codebookBase = codebookIndex * clusterCount;
    for (int i = 0; i < clusterCount; i++) {
      partialSums[codebookBase + i] = dot_product_f32(512, codebook, i * size, query, queryOffset, size);
    }
}

void calculate_partial_sums_euclidean_f32_512(const float* codebook, int codebookIndex, int size, int clusterCount, const float* query, int queryOffset, float* partialSums) {
    int codebookBase = codebookIndex * clusterCount;
    for (int i = 0; i < clusterCount; i++) {
      partialSums[codebookBase + i] = euclidean_f32(512, codebook, i * size, query, queryOffset, size);
    }
}

/* Bulk shuffles for Fused ADC
 * These shuffles take an array of transposed PQ neighbors (in shuffles) and an of quantized partial distances to shuffle.
 * Partial distance quantization depends on the best distance and delta used to quantize.
 * The shuffles for each codebook will be loaded as bytes (supporting up to 256 cluster PQ) and zero-padded to align
 * with 16-bit quantized partial distances. These partial distances will be loaded into SIMD registers, supporting 32 partials
 * per register. Each permutation will take 2 registers, so we need four total permutations to look up against all
 * 256 partial distances. These four permutations will be blended based on the top two bits of each shuffle, allowing 256
 * entry codebook lookup. Quantized partials are quantized based on bounds provided during the search that suggest total
 * distances above the maximum value of an unsigned 16-bit integer will be irrelevant. This allows us to use saturating
 * arithmetic, eliminating the need to widen lanes during accumulation. The total quantized distance is then de-quantized
 * and transformed into the appropriate similarity score.
 *
 * In the case of cosine, we have an additional set of partials used for partial squared magnitudes. These are quantized \
 * with a different pair of delta/base, so they will be aggregated and dequantized separately.
 */


__attribute__((always_inline)) inline __m512i lookup_partial_sums(__m512i shuffle, const char* quantizedPartials, int i) {
    __m512i partialsVecA = _mm512_loadu_epi16(quantizedPartials + i * 512);
    __m512i partialsVecB = _mm512_loadu_epi16(quantizedPartials + i * 512 + 64);
    __m512i partialsVecC = _mm512_loadu_epi16(quantizedPartials + i * 512 + 128);
    __m512i partialsVecD = _mm512_loadu_epi16(quantizedPartials + i * 512 + 192);
    __m512i partialsVecE = _mm512_loadu_epi16(quantizedPartials + i * 512 + 256);
    __m512i partialsVecF = _mm512_loadu_epi16(quantizedPartials + i * 512 + 320);
    __m512i partialsVecG = _mm512_loadu_epi16(quantizedPartials + i * 512 + 384);
    __m512i partialsVecH = _mm512_loadu_epi16(quantizedPartials + i * 512 + 448);

    __m512i partialsVecAB = _mm512_permutex2var_epi16(partialsVecA, shuffle, partialsVecB);
    __m512i partialsVecCD = _mm512_permutex2var_epi16(partialsVecC, shuffle, partialsVecD);
    __m512i partialsVecEF = _mm512_permutex2var_epi16(partialsVecE, shuffle, partialsVecF);
    __m512i partialsVecGH = _mm512_permutex2var_epi16(partialsVecG, shuffle, partialsVecH);

    __mmask32 maskSeven = _mm512_test_epi16_mask(shuffle, maskSeventhBit);
    __mmask32 maskEight = _mm512_test_epi16_mask(shuffle, maskEighthBit);
    __m512i partialsVecABCD = _mm512_mask_blend_epi16(maskSeven, partialsVecAB, partialsVecCD);
    __m512i partialsVecEFGH = _mm512_mask_blend_epi16(maskSeven, partialsVecEF, partialsVecGH);
    __m512i partialSumsVec = _mm512_mask_blend_epi16(maskEight, partialsVecABCD, partialsVecEFGH);

    return partialSumsVec;
}

// dequantize a 256-bit vector containing 16 unsigned 16-bit integers into a 512-bit vector containing 16 32-bit floats
__attribute__((always_inline)) inline __m512 dequantize(__m256i quantizedVec, float delta, float base) {
    __m512i quantizedVecWidened = _mm512_cvtepu16_epi32(quantizedVec);
    __m512 floatVec = _mm512_cvtepi32_ps(quantizedVecWidened);
    __m512 deltaVec = _mm512_set1_ps(delta);
    __m512 baseVec = _mm512_set1_ps(base);
    __m512 dequantizedVec = _mm512_fmadd_ps(floatVec, deltaVec, baseVec);
    return dequantizedVec;
}

void bulk_quantized_shuffle_euclidean_f32_512(const unsigned char* shuffles, int codebookCount, const char* quantizedPartials, float delta, float minDistance, float* results) {
    __m512i sum = _mm512_setzero_epi32();

    for (int i = 0; i < codebookCount; i++) {
         __m256i smallShuffle = _mm256_loadu_epi8(shuffles + i * 32);
         __m512i shuffle = _mm512_cvtepu8_epi16(smallShuffle);
        __m512i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);

        sum = _mm512_adds_epu16(sum, partialsVec);
    }

    __m256i quantizedResultsLeftRaw = _mm512_extracti32x8_epi32(sum, 0);
    __m256i quantizedResultsRightRaw = _mm512_extracti32x8_epi32(sum, 1);
    __m512 resultsLeft = dequantize(quantizedResultsLeftRaw, delta, minDistance);
    __m512 resultsRight = dequantize(quantizedResultsRightRaw, delta, minDistance);

    __m512 ones = _mm512_set1_ps(1.0);
    resultsLeft = _mm512_add_ps(resultsLeft, ones);
    resultsRight = _mm512_add_ps(resultsRight, ones);
    resultsLeft = _mm512_rcp14_ps(resultsLeft);
    resultsRight = _mm512_rcp14_ps(resultsRight);
    _mm512_storeu_ps(results, resultsLeft);
    _mm512_storeu_ps(results + 16, resultsRight);
}

void bulk_quantized_shuffle_dot_f32_512(const unsigned char* shuffles, int codebookCount, const char* quantizedPartials, float delta, float best, float* results) {
    __m512i sum = _mm512_setzero_epi32();

    for (int i = 0; i < codebookCount; i++) {
         __m256i smallShuffle = _mm256_loadu_epi8(shuffles + i * 32);
         __m512i shuffle = _mm512_cvtepu8_epi16(smallShuffle);
        __m512i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
        sum = _mm512_adds_epu16(sum, partialsVec);
    }

    __m256i quantizedResultsLeftRaw = _mm512_extracti32x8_epi32(sum, 0);
    __m256i quantizedResultsRightRaw = _mm512_extracti32x8_epi32(sum, 1);
    __m512 resultsLeft = dequantize(quantizedResultsLeftRaw, delta, best);
    __m512 resultsRight = dequantize(quantizedResultsRightRaw, delta, best);

    __m512 ones = _mm512_set1_ps(1.0);
    resultsLeft = _mm512_add_ps(resultsLeft, ones);
    resultsRight = _mm512_add_ps(resultsRight, ones);
    resultsLeft = _mm512_div_ps(resultsLeft, _mm512_set1_ps(2.0));
    resultsRight = _mm512_div_ps(resultsRight, _mm512_set1_ps(2.0));
    _mm512_storeu_ps(results, resultsLeft);
    _mm512_storeu_ps(results + 16, resultsRight);
}

void bulk_quantized_shuffle_cosine_f32_512(const unsigned char* shuffles, int codebookCount, const char* quantizedPartialSums, float sumDelta, float minDistance, const char* quantizedPartialMagnitudes, float magnitudeDelta, float minMagnitude, float queryMagnitudeSquared, float* results) {
    __m512i sum = _mm512_setzero_epi32();
    __m512i magnitude = _mm512_setzero_epi32();

    for (int i = 0; i < codebookCount; i++) {
        __m256i smallShuffle = _mm256_loadu_epi8((shuffles + i * 32));
        __m512i shuffle = _mm512_cvtepu8_epi16(smallShuffle);
        __m512i partialSumsVec = lookup_partial_sums(shuffle, quantizedPartialSums, i);
        sum = _mm512_adds_epu16(sum, partialSumsVec);

        __m512i partialMagnitudesVec = lookup_partial_sums(shuffle, quantizedPartialMagnitudes, i);
        magnitude = _mm512_adds_epu16(magnitude, partialMagnitudesVec);
    }

    __m256i quantizedSumsLeftRaw = _mm512_extracti32x8_epi32(sum, 0);
    __m256i quantizedSumsRightRaw = _mm512_extracti32x8_epi32(sum, 1);
    __m512 sumsLeft = dequantize(quantizedSumsLeftRaw, sumDelta, minDistance);
    __m512 sumsRight = dequantize(quantizedSumsRightRaw, sumDelta, minDistance);

    __m256i quantizedMagnitudesLeftRaw = _mm512_extracti32x8_epi32(magnitude, 0);
    __m256i quantizedMagnitudesRightRaw = _mm512_extracti32x8_epi32(magnitude, 1);
    __m512 magnitudesLeft = dequantize(quantizedMagnitudesLeftRaw, magnitudeDelta, minMagnitude);
    __m512 magnitudesRight = dequantize(quantizedMagnitudesRightRaw, magnitudeDelta, minMagnitude);

    __m512 queryMagnitudeSquaredVec = _mm512_set1_ps(queryMagnitudeSquared);
    magnitudesLeft = _mm512_mul_ps(magnitudesLeft, queryMagnitudeSquaredVec);
    magnitudesRight = _mm512_mul_ps(magnitudesRight, queryMagnitudeSquaredVec);
    magnitudesLeft = _mm512_sqrt_ps(magnitudesLeft);
    magnitudesRight = _mm512_sqrt_ps(magnitudesRight);
    __m512 resultsLeft = _mm512_div_ps(sumsLeft, magnitudesLeft);
    __m512 resultsRight = _mm512_div_ps(sumsRight, magnitudesRight);

    __m512 ones = _mm512_set1_ps(1.0);
    resultsLeft = _mm512_add_ps(resultsLeft, ones);
    resultsRight = _mm512_add_ps(resultsRight, ones);
    resultsLeft = _mm512_div_ps(resultsLeft, _mm512_set1_ps(2.0));
    resultsRight = _mm512_div_ps(resultsRight, _mm512_set1_ps(2.0));
    _mm512_storeu_ps(results, resultsLeft);
    _mm512_storeu_ps(results + 16, resultsRight);
}

// Partial sum calculations that also record best distances, as this is necessary for Fused ADC quantization
void calculate_partial_sums_best_dot_f32_512(const float* codebook, int codebookIndex, int size, int clusterCount, const float* query, int queryOffset, float* partialSums, float* partialBestDistances) {
    float best = -INFINITY;
    int codebookBase = codebookIndex * clusterCount;
    for (int i = 0; i < clusterCount; i++) {
      float val = dot_product_f32(512, codebook, i * size, query, queryOffset, size);
      partialSums[codebookBase + i] = val;
      if (val > best) {
        best = val;
      }
    }
    partialBestDistances[codebookIndex] = best;
}

void calculate_partial_sums_best_euclidean_f32_512(const float* codebook, int codebookIndex, int size, int clusterCount, const float* query, int queryOffset, float* partialSums, float* partialBestDistances) {
    float best = INFINITY;
    int codebookBase = codebookIndex * clusterCount;
    for (int i = 0; i < clusterCount; i++) {
      float val = euclidean_f32(512, codebook, i * size, query, queryOffset, size);
      partialSums[codebookBase + i] = val;
      if (val < best) {
        best = val;
      }
    }
    partialBestDistances[codebookIndex] = best;
}

/* NVQ 8-bit distance functions using AVX-512
 *
 * These implement the NVQ dequantization pipeline in native AVX-512 intrinsics:
 *   1. Unpack quantized bytes (FastLanes layout: 64 bytes -> 4 groups of 16)
 *   2. Scale into logistic domain: val = byte * logisticScale + logisticBias
 *   3. Apply logit (inverse logistic): dequantized = logit(val) * invAlpha + x0
 *   4. Compute dot product or squared L2 distance against the query float vector
 *
 * All processing is done in AVX-512; the tail (length % 64) is handled by the
 * same loop using masked loads (_mm512_maskz_loadu_epi8 / _mm512_maskz_loadu_ps).
 */

// Forward logistic NQT: sigma(alpha * (value - x0)), vectorized.
// Matches Java's logisticNQT: 2^temp / (2^temp + 1) where
//   temp = alpha * (value - x0),
//   2^temp is approximated as mant * 2^p with p = floor(temp+1),
//   mant = fma(temp - p, 0.5, 1.0) (linear fit on (-1,0]).
// Uses vscalefps (vscalef) to compute mant * 2^p without bit manipulation.
static inline __m512 nvq_logisticNQT_avx512(__m512 value, float alpha, float x0) {
    __m512 one  = _mm512_set1_ps(1.0f);
    __m512 temp = _mm512_fmadd_ps(value, _mm512_set1_ps(alpha),
                                  _mm512_set1_ps(-alpha * x0));
    // p = floor(temp + 1)  [vrndscaleps via _mm512_floor_ps]
    __m512 p    = _mm512_floor_ps(_mm512_add_ps(temp, one));
    // mant = fma(temp - p, 0.5, 1.0): approximates 2^(temp-p) in (0.5, 1]
    __m512 mant = _mm512_fmadd_ps(_mm512_sub_ps(temp, p), _mm512_set1_ps(0.5f), one);
    // result = mant * 2^p  [vscalefps]
    __m512 result = _mm512_scalef_ps(mant, p);
    return _mm512_div_ps(result, _mm512_add_ps(result, one));
}

// Inverse logistic (logit) NQT: logit(value) * inverseAlpha + x0, vectorized.
// Matches Java's logitNQT: result = (m + p) * inverseAlpha + x0 where
//   z = value / (1 - value),  p = floor(log2(z)),  m = mantissa(z) in [1.0, 2.0).
// Uses vgetexpps and vgetmantps instead of integer bit manipulation.
static inline __m512 nvq_logitNQT_avx512(__m512 value, __m512 inverseAlpha, __m512 x0) {
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 z   = _mm512_div_ps(value, _mm512_sub_ps(one, value));
    // p = (biased_exponent >> 23) - 128, matching Java's bit-manipulation which uses
    // bias 128 instead of the IEEE 754 bias 127. vgetexpps returns the standard
    // unbiased exponent (bias 127), so subtract 1 to match.  [vgetexpps - 1]
    __m512 p   = _mm512_sub_ps(_mm512_getexp_ps(z), one);
    // m = mantissa of z normalized to [1.0, 2.0)  [vgetmantps]
    __m512 m   = _mm512_getmant_ps(z, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src);
    return _mm512_fmadd_ps(_mm512_add_ps(m, p), inverseAlpha, x0);
}

// Dequantize 16 bytes from a 512-bit register using the FastLanes layout.
// `part` selects which byte within each int32 lane (0 = lowest byte, 3 = highest).
static inline __m512 nvq_dequantize8bit_avx512(__m512i bytes, int part,
                                               __m512 logisticScale, __m512 logisticBias,
                                               __m512 invScaledAlpha, __m512 scaledX0) {
    __m512i vals = _mm512_and_epi32(_mm512_srli_epi32(bytes, 8 * part),
                                    _mm512_set1_epi32(0xff));
    __m512 arr = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals), logisticScale, logisticBias);
    return nvq_logitNQT_avx512(arr, invScaledAlpha, scaledX0);
}

float nvq_square_l2_distance_8bit_512(const float* vector, int length,
                                      const unsigned char* quantized,
                                      float alpha, float x0,
                                      float minValue, float maxValue) {
    float delta          = maxValue - minValue;
    float scaledAlpha    = alpha / delta;
    float invScaledAlpha = delta / alpha;
    float scaledX0       = x0 * delta;

    // Compute logisticBias and logisticScale using the same AVX-512 logistic approximation.
    __attribute__((aligned(16))) float logistic_params[4];
    _mm_store_ps(logistic_params,
        _mm512_castps512_ps128(
            nvq_logisticNQT_avx512(
                _mm512_set_ps(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, maxValue, minValue),
                scaledAlpha, scaledX0)));
    float logisticBias      = logistic_params[0];
    float logisticScale_val = (logistic_params[1] - logisticBias) / 255.0f;

    __m512 invScaledAlpha_vec = _mm512_set1_ps(invScaledAlpha);
    __m512 scaledX0_vec       = _mm512_set1_ps(scaledX0);
    __m512 logisticScale_vec  = _mm512_set1_ps(logisticScale_val);
    __m512 logisticBias_vec   = _mm512_set1_ps(logisticBias);
    __m512 sum                = _mm512_setzero_ps();

    // FastLanes main loop: each 64-byte block is reinterpreted as 16 int32s;
    // byte j of each int gives one of 4 groups of 16 dequantized values.
    int vectorizedLength = length - (length % 64);
    for (int i = 0; i < vectorizedLength; i += 64) {
        __m512i bytes = _mm512_loadu_si512((const __m512i*)(quantized + i));
        for (int j = 0; j < 4; j++) {
            __m512 v2   = nvq_dequantize8bit_avx512(bytes, j,
                              logisticScale_vec, logisticBias_vec,
                              invScaledAlpha_vec, scaledX0_vec);
            __m512 v1   = _mm512_loadu_ps(vector + i + j * 16);
            __m512 diff = _mm512_sub_ps(v1, v2);
            sum = _mm512_fmadd_ps(diff, diff, sum);
        }
    }

    // Sequential tail: bytes are stored in plain order, not FastLanes transposed.
    // Use cvtepu8_epi32 to zero-extend 16 sequential bytes to 16 int32s per pass.
    for (int i = vectorizedLength; i < length; i += 16) {
        int n = length - i;
        if (n > 16) n = 16;
        __mmask16 mask16 = (n >= 16) ? (__mmask16)0xffff
                                     : (__mmask16)((1U << n) - 1);
        __m512i vals = _mm512_cvtepu8_epi32(
                           _mm_maskz_loadu_epi8(mask16, quantized + i));
        __m512 arr  = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals),
                                      logisticScale_vec, logisticBias_vec);
        __m512 v2   = nvq_logitNQT_avx512(arr, invScaledAlpha_vec, scaledX0_vec);
        __m512 v1   = _mm512_maskz_loadu_ps(mask16, vector + i);
        __m512 diff = _mm512_maskz_sub_ps(mask16, v1, v2);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }

    return _mm512_reduce_add_ps(sum);
}

float nvq_dot_product_8bit_512(const float* vector, int length,
                                const unsigned char* quantized,
                                float alpha, float x0,
                                float minValue, float maxValue) {
    float delta          = maxValue - minValue;
    float scaledAlpha    = alpha / delta;
    float invScaledAlpha = delta / alpha;
    float scaledX0       = x0 * delta;

    __attribute__((aligned(16))) float logistic_params[4];
    _mm_store_ps(logistic_params,
        _mm512_castps512_ps128(
            nvq_logisticNQT_avx512(
                _mm512_set_ps(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, maxValue, minValue),
                scaledAlpha, scaledX0)));
    float logisticBias      = logistic_params[0];
    float logisticScale_val = (logistic_params[1] - logisticBias) / 255.0f;

    __m512 invScaledAlpha_vec = _mm512_set1_ps(invScaledAlpha);
    __m512 scaledX0_vec       = _mm512_set1_ps(scaledX0);
    __m512 logisticScale_vec  = _mm512_set1_ps(logisticScale_val);
    __m512 logisticBias_vec   = _mm512_set1_ps(logisticBias);
    __m512 sum                = _mm512_setzero_ps();

    // FastLanes main loop: each 64-byte block is reinterpreted as 16 int32s;
    // byte j of each int gives one of 4 groups of 16 dequantized values.
    int vectorizedLength = length - (length % 64);
    for (int i = 0; i < vectorizedLength; i += 64) {
        __m512i bytes = _mm512_loadu_si512((const __m512i*)(quantized + i));
        for (int j = 0; j < 4; j++) {
            __m512 v1 = _mm512_loadu_ps(vector + i + j * 16);
            __m512 v2 = nvq_dequantize8bit_avx512(bytes, j,
                            logisticScale_vec, logisticBias_vec,
                            invScaledAlpha_vec, scaledX0_vec);
            sum = _mm512_fmadd_ps(v1, v2, sum);
        }
    }

    // Sequential tail: bytes are stored in plain order, not FastLanes transposed.
    // Use cvtepu8_epi32 to zero-extend 16 sequential bytes to 16 int32s per pass.
    for (int i = vectorizedLength; i < length; i += 16) {
        int n = length - i;
        if (n > 16) n = 16;
        __mmask16 mask16 = (n >= 16) ? (__mmask16)0xffff
                                     : (__mmask16)((1U << n) - 1);
        __m512i vals = _mm512_cvtepu8_epi32(
                           _mm_maskz_loadu_epi8(mask16, quantized + i));
        __m512 arr  = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals),
                                      logisticScale_vec, logisticBias_vec);
        __m512 v2   = nvq_logitNQT_avx512(arr, invScaledAlpha_vec, scaledX0_vec);
        __m512 v1   = _mm512_maskz_loadu_ps(mask16, vector + i);
        sum = _mm512_fmadd_ps(v1, v2, sum);
    }

    return _mm512_reduce_add_ps(sum);
}
