#include <time.h>
#include "operation.h"

clock_t start, end;

void test_memcpy(size_t TEST_SIZE);
void test_add(size_t TEST_SIZE);
void test_sub(size_t TEST_SIZE);
void test_mul(size_t TEST_SIZE);
void test_div(size_t TEST_SIZE);
void test_max(size_t TEST_SIZE);
void test_min(size_t TEST_SIZE);
void test_andnot(size_t TEST_SIZE);

int check(char *dest, char* source, size_t count);
int check_float(float *dest, float* source, size_t count);

void display_startrow();
void display_row(char* test_name, float sisd_perf, float simd_perf, unsigned int data_checked);
void display_endrow();