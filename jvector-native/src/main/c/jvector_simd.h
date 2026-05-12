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

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef VECTOR_SIMD_DOT_H
#define VECTOR_SIMD_DOT_H


// APIs exposed to Java via FFI
#ifdef __cplusplus
extern "C" {
#endif
bool check_avx512_compatibility(void);
/* PQ kernels */
float assemble_and_sum_f32(const float* data, int dataBase, const unsigned char* baseOffsets, int baseOffsetsOffset, size_t baseOffsetsLength);
float assemble_and_sum_pq_f32(const float* data, size_t subspaceCount, const unsigned char* baseOffsets1, int baseOffsetsOffset1, const unsigned char* baseOffsets2, int baseOffsetsOffset2, int clusterCount);
float pq_decoded_cosine_similarity_f32(const unsigned char* baseOffsets, int baseOffsetsOffset, int baseOffsetsLength, int clusterCount, const float* partialSums, const float* aMagnitude, float bMagnitude);
void calculate_partial_sums_euclidean_f32(const float* codebook, int codebookBase, int size, int clusterCount, const float* query, int queryOffset, float* partialSums);
void calculate_partial_sums_dot_f32(const float* codebook, int codebookBase, int size, int clusterCount, const float* query, int queryOffset, float* partialSums);

/* Vector similarity kernels */
float dot_product_f32_native(const float* a, size_t aoffset, const float* b, size_t boffset, size_t length);
float cosine_f32_native(const float* a, size_t aoffset, const float* b, size_t boffset, size_t length);
float euclidean_f32_native(const float* a, size_t aoffset, const float* b, size_t boffset, size_t length);

/* NVQ kernels */
void    nvq_quantize_8bit(const float* vector, int length, float alpha, float x0, float minValue, float maxValue, unsigned char* destination);
float   nvq_loss(const float* vector, int length, float alpha, float x0, float minValue, float maxValue, int nBits);
float   nvq_uniform_loss(const float* vector, int length, float minValue, float maxValue, int nBits);
float   nvq_square_l2_distance_8bit(const float* vector, const unsigned char* quantized, int length, float alpha, float x0, float minValue, float maxValue);
float   nvq_dot_product_8bit(const float* vector, const unsigned char* quantized, int length, float alpha, float x0, float minValue, float maxValue);
int64_t nvq_cosine_8bit_packed(const float* vector, const unsigned char* quantized, int length, float alpha, float x0, float minValue, float maxValue, const float* centroid);

/* Element-wise in-place vector arithmetic */
void    add_in_place_f32(float* v1, const float* v2, size_t length);
void    add_scalar_in_place_f32(float* v1, float value, size_t length);
void    sub_in_place_f32(float* v1, const float* v2, size_t length);
void    sub_scalar_in_place_f32(float* v1, float value, size_t length);
float   max_f32(const float* v, size_t length);
void    min_in_place_f32(float* v1, const float* v2, size_t length);
#ifdef __cplusplus
}
#endif // extern "C"
#endif // VECTOR_SIMD_DOT_H
