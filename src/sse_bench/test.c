#include "test.h"

void test_memcpy(size_t TEST_SIZE) {
  float sisd_elapsed, simd_elapsed;
  
  // Prepare memory
  char* data = (char*)malloc(TEST_SIZE);
  char* dest1 = (char*)malloc(TEST_SIZE);
  char* dest2 = (char*)malloc(TEST_SIZE);

  for (size_t i = 0; i < TEST_SIZE; i++) {
    data[i] = i;
    dest1[i] = 0;
    dest2[i] = 0;
  }

  // Test - memcpy with for loop
  start = clock();
  memcpy_sisd(dest1, data, TEST_SIZE);
  end = clock();
  sisd_elapsed = (float)(end - start)/CLOCKS_PER_SEC;
  
  // Test - memcpy with SIMD
  start = clock();
  memcpy_simd(dest2, data, TEST_SIZE);
  end = clock();
  simd_elapsed = (float)(end - start)/CLOCKS_PER_SEC;

  display_row("memcpy", sisd_elapsed, simd_elapsed, check(dest1, dest2, TEST_SIZE));

  free(data);
  free(dest1);
  free(dest2);
  
}

void test_add(size_t TEST_SIZE) {
  float sisd_elapsed, simd_elapsed;

  // Prepare memory
  char* data1 = (char*)malloc(TEST_SIZE);
  char* data2 = (char*)malloc(TEST_SIZE);
  char* dest1 = (char*)malloc(TEST_SIZE);
  char* dest2 = (char*)malloc(TEST_SIZE);

  for (size_t i = 0; i < TEST_SIZE; i++) {
    data1[i] = i;
    data2[i] = i;
    dest1[i] = 0;
    dest2[i] = 0;
  }

  // Test - add with for loop
  start = clock();
  add_sisd(dest1, data1, data2, TEST_SIZE);
  end = clock();
  sisd_elapsed = (float)(end - start)/CLOCKS_PER_SEC;
  
  // Test - add with SIMD
  start = clock();
  add_simd(dest2, data1, data2, TEST_SIZE);
  end = clock();
  simd_elapsed = (float)(end - start)/CLOCKS_PER_SEC;
  
  display_row("add", sisd_elapsed, simd_elapsed, check(dest1, dest2, TEST_SIZE));

  free(data1);
  free(data2);
  free(dest1);
  free(dest2);
}

void test_sub(size_t TEST_SIZE) {
  float sisd_elapsed, simd_elapsed;

  // Prepare memory
  char* data1 = (char*)malloc(TEST_SIZE);
  char* data2 = (char*)malloc(TEST_SIZE);
  char* dest1 = (char*)malloc(TEST_SIZE);
  char* dest2 = (char*)malloc(TEST_SIZE);

  for (size_t i = 0; i < TEST_SIZE; i++) {
    data1[i] = i;
    data2[i] = i;
    dest1[i] = 0;
    dest2[i] = 0;
  }

  // Test - sub with for loop
  start = clock();
  sub_sisd(dest1, data1, data2, TEST_SIZE);
  end = clock();
  sisd_elapsed = (float)(end - start)/CLOCKS_PER_SEC;
  
  // Test - sub with SIMD
  start = clock();
  sub_simd(dest2, data1, data2, TEST_SIZE);
  end = clock();
  simd_elapsed = (float)(end - start)/CLOCKS_PER_SEC;
  
  display_row("sub", sisd_elapsed, simd_elapsed, check(dest1, dest2, TEST_SIZE));

  free(data1);
  free(data2);
  free(dest1);
  free(dest2);
}

void test_mul(size_t TEST_SIZE) {
  float sisd_elapsed, simd_elapsed;

  // Prepare memory
  char* data1 = (char*)malloc(TEST_SIZE);
  char* data2 = (char*)malloc(TEST_SIZE);
  char* dest1 = (char*)malloc(TEST_SIZE);
  char* dest2 = (char*)malloc(TEST_SIZE);

  for (size_t i = 0; i < TEST_SIZE; i++) {
    data1[i] = TEST_SIZE - i;
    data2[i] = i / TEST_SIZE;
    dest1[i] = 0;
    dest2[i] = 0;
  }

  // Test - add with for loop
  start = clock();
  mul_sisd(dest1, data1, data2, TEST_SIZE);
  end = clock();
  sisd_elapsed = (float)(end - start)/CLOCKS_PER_SEC;
  
  // Test - add with SIMD
  start = clock();
  mul_simd(dest2, data1, data2, TEST_SIZE);
  end = clock();
  simd_elapsed = (float)(end - start)/CLOCKS_PER_SEC;
  
  display_row("mul", sisd_elapsed, simd_elapsed, check(dest1, dest2, TEST_SIZE));

  free(data1);
  free(data2);
  free(dest1);
  free(dest2);
}

void test_div(size_t TEST_SIZE) {
  float sisd_elapsed, simd_elapsed;

  // Prepare memory
  char* data1 = (char*)malloc(TEST_SIZE);
  char* data2 = (char*)malloc(TEST_SIZE);
  char* dest1 = (char*)malloc(TEST_SIZE);
  char* dest2 = (char*)malloc(TEST_SIZE);

  for (size_t i = 0; i < TEST_SIZE; i++) {
    data1[i] = TEST_SIZE - i;
    data2[i] = i / TEST_SIZE + 1;
    dest1[i] = 0;
    dest2[i] = 0;
  }

  // Test - add with for loop
  start = clock();
  div_sisd(dest1, data1, data2, TEST_SIZE);
  end = clock();
  sisd_elapsed = (float)(end - start)/CLOCKS_PER_SEC;
  
  // Test - add with SIMD
  start = clock();
  div_simd(dest2, data1, data2, TEST_SIZE);
  end = clock();
  simd_elapsed = (float)(end - start)/CLOCKS_PER_SEC;
  
  display_row("div", sisd_elapsed, simd_elapsed, check(dest1, dest2, TEST_SIZE));

  free(data1);
  free(data2);
  free(dest1);
  free(dest2);
}

void test_max(size_t TEST_SIZE) {
  float sisd_elapsed, simd_elapsed;
  
  // Prepare memory
  char* data1 = (char*)malloc(TEST_SIZE);
  char* data2 = (char*)malloc(TEST_SIZE);
  char* dest1 = (char*)malloc(TEST_SIZE);
  char* dest2 = (char*)malloc(TEST_SIZE);

  for (size_t i = 0; i < TEST_SIZE; i++) {
    data1[i] = i;
    data2[i] = TEST_SIZE - i;
    dest1[i] = 0;
    dest2[i] = 0;
  }

  // Test - max with for loop
  start = clock();
  max_sisd(dest1, data1, data2, TEST_SIZE);
  end = clock();
  sisd_elapsed = (float)(end - start)/CLOCKS_PER_SEC;
  
  // Test - max with SIMD
  start = clock();
  max_simd(dest2, data1, data2, TEST_SIZE);
  end = clock();
  simd_elapsed = (float)(end - start)/CLOCKS_PER_SEC;

  display_row("max", sisd_elapsed, simd_elapsed, check(dest1, dest2, TEST_SIZE));

  free(data1);
  free(data2);
  free(dest1);
  free(dest2);
}

void test_min(size_t TEST_SIZE) {
  float sisd_elapsed, simd_elapsed;
  
  // Prepare memory
  char* data1 = (char*)malloc(TEST_SIZE);
  char* data2 = (char*)malloc(TEST_SIZE);
  char* dest1 = (char*)malloc(TEST_SIZE);
  char* dest2 = (char*)malloc(TEST_SIZE);

  for (size_t i = 0; i < TEST_SIZE; i++) {
    data1[i] = i;
    data2[i] = TEST_SIZE - i;
    dest1[i] = 0;
    dest2[i] = 0;
  }

  // Test - min with for loop
  start = clock();
  min_sisd(dest1, data1, data2, TEST_SIZE);
  end = clock();
  sisd_elapsed = (float)(end - start)/CLOCKS_PER_SEC;
  
  // Test - min with SIMD
  start = clock();
  min_simd(dest2, data1, data2, TEST_SIZE);
  end = clock();
  simd_elapsed = (float)(end - start)/CLOCKS_PER_SEC;

  display_row("min", sisd_elapsed, simd_elapsed, check(dest1, dest2, TEST_SIZE));

  free(data1);
  free(data2);
  free(dest1);
  free(dest2);
}

int check(char* dest, char* source, size_t count) {
  for (size_t i = 0; i< count; i++) {
    if (source[i] != dest[i]) {
      return 0;
    }
  }
  return 1;
}

void display_startrow() {
  printf("\n");
  printf("                     SISD vs. SIMD BENCHMARK                      \n");
  printf("==================================================================\n");
  printf("|  TEST  |  SISD (sec) |  SIMD (sec) |   FASTER   |  DATA CHECK  |\n");
  printf("+--------+-------------+-------------+------------+--------------+\n");
}

void display_row(char* test_name, float sisd_perf, float simd_perf, unsigned int data_checked) {
  printf("| %-6s |  %.4f     |  %.4f     |  x %.4f  |  %-10s  |\n", test_name, sisd_perf, simd_perf, sisd_perf/simd_perf, data_checked ? "YES" : "NO");
}

void display_endrow() {
  printf("==================================================================\n");
}