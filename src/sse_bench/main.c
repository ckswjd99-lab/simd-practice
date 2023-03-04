/**
 * @author Chanjeong Park (ckswjd99@snu.ac.kr)
 * @brief benchmark for SSE SIMD operations
 * 
 * Compile by
 * $ gcc main.c test.c operation.c -msse4.1 -o main
 */

#include "test.h"
#include "operation.h"

const size_t TEST_SIZE = 200 * (1 << 20);

int main() {

  display_startrow();

  // Test: memcpy
  test_memcpy(TEST_SIZE);

  // Test: add
  test_add(TEST_SIZE);
  
  // Test: sub
  test_sub(TEST_SIZE);
  
  // Test: mul
  test_mul(TEST_SIZE);
  
  // Test: div
  test_div(TEST_SIZE);

  // Test: max
  test_max(TEST_SIZE);
  
  // Test: min
  test_min(TEST_SIZE);

  // Test: andnot
  test_andnot(TEST_SIZE);

  display_endrow();
}
