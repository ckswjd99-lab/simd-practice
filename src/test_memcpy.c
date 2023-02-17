/*
 * Following tutorials at
 * https://velog.io/@kunshim/SIMD-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D
 * 
 * Compile by
 * $ gcc test_memcpy.c -msse4.1 -o test_memcpy
 */

#include <stdio.h>
#include <emmintrin.h>
#include <time.h>

void memcpy_for(char* dest, char* source, size_t count);
void memcpy_simd(char* dest, char* source, size_t count);
int check_memcpy(char *dest, char* source, size_t count);

const size_t TEST_SIZE = 500 * (1 << 10) * (1 << 10);

int main() {
  // Prepare memory
  char* original = (char*)malloc(TEST_SIZE);
  char* temp = (char*)malloc(TEST_SIZE);
  clock_t start, end;

  // Init memory
  for (size_t i = 0; i < TEST_SIZE; i++) {
    original[i] = i;
    temp[i] = 0;
  }

  // Test - memcpy with for loop
  start = clock();
  memcpy_for(temp, original, TEST_SIZE);
  end = clock();
  printf("SISD : %.3fs\n", (float)(end - start)/CLOCKS_PER_SEC);
  check_memcpy(temp, original, TEST_SIZE);

  // Init memory
  for (size_t i = 0; i < TEST_SIZE; i++) 
    temp[i] = 0;
  
  // Test - memcpy with SIMD
  start = clock();
  memcpy_simd(temp, original, TEST_SIZE);
  end = clock();
  printf("SIMD : %.3fs\n", (float)(end - start)/CLOCKS_PER_SEC);
  check_memcpy(temp, original, TEST_SIZE);
  
  // Free memory
  free(original);
  free(temp);
}

void memcpy_for(char* dest, char* source, size_t count) {
  count /= 8;
  for (size_t i = 0; i < count; i++)
    *((long long int*)dest + i) = *((long long int*)source + i);
}

void memcpy_simd(char* dest, char* source, size_t count) {
  count /= 0x10;
  for (size_t i = 0; i< count; i++) {
    __m128i temp = _mm_loadu_si128((__m128i*)source+i);
    _mm_storeu_si128((__m128i*)dest + i, temp);
  }
}

int check_memcpy(char* dest, char* source, size_t count) {
  for (size_t i = 0; i< count; i++) {
    if (source[i] != dest[i])
       return 0;
  }
  puts("OK!");
  return 0;
}