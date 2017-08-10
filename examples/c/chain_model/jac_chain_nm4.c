/* This function was automatically generated by CasADi */
#ifdef __cplusplus
extern "C" {
#endif

#ifdef CODEGEN_PREFIX
#define NAMESPACE_CONCAT(NS, ID) _NAMESPACE_CONCAT(NS, ID)
#define _NAMESPACE_CONCAT(NS, ID) NS##ID
#define CASADI_PREFIX(ID) NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else /* CODEGEN_PREFIX */
#define CASADI_PREFIX(ID) jac_chain_nm4_##ID
#endif /* CODEGEN_PREFIX */

#include <math.h>

#ifndef real_t
#define real_t double
#define to_double(x) (double)x
#define to_int(x) (int)x
#endif /* real_t */

/* Pre-c99 compatibility */
#if __STDC_VERSION__ < 199901L
real_t CASADI_PREFIX(fmin)(real_t x, real_t y) { return x < y ? x : y; }
#define fmin(x, y) CASADI_PREFIX(fmin)(x, y)
real_t CASADI_PREFIX(fmax)(real_t x, real_t y) { return x > y ? x : y; }
#define fmax(x, y) CASADI_PREFIX(fmax)(x, y)
#endif

#define PRINTF printf
real_t CASADI_PREFIX(sq)(real_t x) { return x * x; }
#define sq(x) CASADI_PREFIX(sq)(x)

real_t CASADI_PREFIX(sign)(real_t x) { return x < 0 ? -1 : x > 0 ? 1 : x; }
#define sign(x) CASADI_PREFIX(sign)(x)

static const int CASADI_PREFIX(s0)[] = {
    18, 1, 0, 18, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
#define s0 CASADI_PREFIX(s0)
static const int CASADI_PREFIX(s1)[] = {3, 1, 0, 3, 0, 1, 2};
#define s1 CASADI_PREFIX(s1)
static const int CASADI_PREFIX(s2)[] = {
    18,  18,  0,   18,  36,  54, 72, 90, 108, 126, 144, 162, 180, 198, 216, 234,
    252, 270, 288, 306, 324, 0,  1,  2,  3,   4,   5,   6,   7,   8,   9,   10,
    11,  12,  13,  14,  15,  16, 17, 0,  1,   2,   3,   4,   5,   6,   7,   8,
    9,   10,  11,  12,  13,  14, 15, 16, 17,  0,   1,   2,   3,   4,   5,   6,
    7,   8,   9,   10,  11,  12, 13, 14, 15,  16,  17,  0,   1,   2,   3,   4,
    5,   6,   7,   8,   9,   10, 11, 12, 13,  14,  15,  16,  17,  0,   1,   2,
    3,   4,   5,   6,   7,   8,  9,  10, 11,  12,  13,  14,  15,  16,  17,  0,
    1,   2,   3,   4,   5,   6,  7,  8,  9,   10,  11,  12,  13,  14,  15,  16,
    17,  0,   1,   2,   3,   4,  5,  6,  7,   8,   9,   10,  11,  12,  13,  14,
    15,  16,  17,  0,   1,   2,  3,  4,  5,   6,   7,   8,   9,   10,  11,  12,
    13,  14,  15,  16,  17,  0,  1,  2,  3,   4,   5,   6,   7,   8,   9,   10,
    11,  12,  13,  14,  15,  16, 17, 0,  1,   2,   3,   4,   5,   6,   7,   8,
    9,   10,  11,  12,  13,  14, 15, 16, 17,  0,   1,   2,   3,   4,   5,   6,
    7,   8,   9,   10,  11,  12, 13, 14, 15,  16,  17,  0,   1,   2,   3,   4,
    5,   6,   7,   8,   9,   10, 11, 12, 13,  14,  15,  16,  17,  0,   1,   2,
    3,   4,   5,   6,   7,   8,  9,  10, 11,  12,  13,  14,  15,  16,  17,  0,
    1,   2,   3,   4,   5,   6,  7,  8,  9,   10,  11,  12,  13,  14,  15,  16,
    17,  0,   1,   2,   3,   4,  5,  6,  7,   8,   9,   10,  11,  12,  13,  14,
    15,  16,  17,  0,   1,   2,  3,  4,  5,   6,   7,   8,   9,   10,  11,  12,
    13,  14,  15,  16,  17,  0,  1,  2,  3,   4,   5,   6,   7,   8,   9,   10,
    11,  12,  13,  14,  15,  16, 17, 0,  1,   2,   3,   4,   5,   6,   7,   8,
    9,   10,  11,  12,  13,  14, 15, 16, 17};
#define s2 CASADI_PREFIX(s2)
/* jacFun */
int jac_chain_nm4(void* mem, const real_t** arg, real_t** res, int* iw,
                  real_t* w) {
    mem = 0;
    mem += 0;
    w = 0;
    w += 0;
    iw = 0;
    iw += 0;
    real_t a0 = arg[0] ? arg[0][3] : 0;
    if (res[0] != 0) res[0][0] = a0;
    a0 = arg[0] ? arg[0][4] : 0;
    if (res[0] != 0) res[0][1] = a0;
    a0 = arg[0] ? arg[0][5] : 0;
    if (res[0] != 0) res[0][2] = a0;
    a0 = arg[0] ? arg[0][6] : 0;
    real_t a1 = arg[0] ? arg[0][0] : 0;
    real_t a2 = (a0 - a1);
    real_t a3 = sq(a2);
    real_t a4 = arg[0] ? arg[0][7] : 0;
    real_t a5 = arg[0] ? arg[0][1] : 0;
    real_t a6 = (a4 - a5);
    real_t a7 = sq(a6);
    a3 = (a3 + a7);
    a7 = arg[0] ? arg[0][8] : 0;
    real_t a8 = arg[0] ? arg[0][2] : 0;
    real_t a9 = (a7 - a8);
    real_t a10 = sq(a9);
    a3 = (a3 + a10);
    a3 = sqrt(a3);
    a10 = 3.3000000000000002e-02;
    real_t a11 = (a10 / a3);
    real_t a12 = 1.;
    real_t a13 = (a12 - a11);
    real_t a14 = (a13 * a2);
    real_t a15 = sq(a1);
    real_t a16 = sq(a5);
    a15 = (a15 + a16);
    a16 = sq(a8);
    a15 = (a15 + a16);
    a15 = sqrt(a15);
    a16 = (a10 / a15);
    real_t a17 = (a12 - a16);
    real_t a18 = (a17 * a1);
    a18 = (a14 - a18);
    real_t a19 = 3.3333333333333336e+01;
    a18 = (a19 * a18);
    if (res[0] != 0) res[0][3] = a18;
    a18 = (a13 * a6);
    real_t a20 = (a17 * a5);
    a20 = (a18 - a20);
    a20 = (a19 * a20);
    if (res[0] != 0) res[0][4] = a20;
    a20 = (a13 * a9);
    real_t a21 = (a17 * a8);
    a21 = (a20 - a21);
    a21 = (a19 * a21);
    real_t a22 = 9.8100000000000005e+00;
    a21 = (a21 - a22);
    if (res[0] != 0) res[0][5] = a21;
    a21 = arg[0] ? arg[0][9] : 0;
    if (res[0] != 0) res[0][6] = a21;
    a21 = arg[0] ? arg[0][10] : 0;
    if (res[0] != 0) res[0][7] = a21;
    a21 = arg[0] ? arg[0][11] : 0;
    if (res[0] != 0) res[0][8] = a21;
    a21 = arg[0] ? arg[0][12] : 0;
    a21 = (a21 - a0);
    a0 = sq(a21);
    real_t a23 = arg[0] ? arg[0][13] : 0;
    a23 = (a23 - a4);
    a4 = sq(a23);
    a0 = (a0 + a4);
    a4 = arg[0] ? arg[0][14] : 0;
    a4 = (a4 - a7);
    a7 = sq(a4);
    a0 = (a0 + a7);
    a0 = sqrt(a0);
    a10 = (a10 / a0);
    a7 = (a12 - a10);
    real_t a24 = (a7 * a21);
    a24 = (a24 - a14);
    a24 = (a19 * a24);
    if (res[0] != 0) res[0][9] = a24;
    a24 = (a7 * a23);
    a24 = (a24 - a18);
    a24 = (a19 * a24);
    if (res[0] != 0) res[0][10] = a24;
    a24 = (a7 * a4);
    a24 = (a24 - a20);
    a24 = (a19 * a24);
    a24 = (a24 - a22);
    if (res[0] != 0) res[0][11] = a24;
    a24 = arg[0] ? arg[0][15] : 0;
    if (res[0] != 0) res[0][12] = a24;
    a24 = arg[0] ? arg[0][16] : 0;
    if (res[0] != 0) res[0][13] = a24;
    a24 = arg[0] ? arg[0][17] : 0;
    if (res[0] != 0) res[0][14] = a24;
    a24 = arg[1] ? arg[1][0] : 0;
    if (res[0] != 0) res[0][15] = a24;
    a24 = arg[1] ? arg[1][1] : 0;
    if (res[0] != 0) res[0][16] = a24;
    a24 = arg[1] ? arg[1][2] : 0;
    if (res[0] != 0) res[0][17] = a24;
    a24 = 0.;
    if (res[1] != 0) res[1][0] = a24;
    if (res[1] != 0) res[1][1] = a24;
    if (res[1] != 0) res[1][2] = a24;
    a11 = (a11 / a3);
    a22 = (a2 / a3);
    a22 = (a11 * a22);
    a20 = (a2 * a22);
    a20 = (a20 + a13);
    a16 = (a16 / a15);
    a18 = (a1 / a15);
    a18 = (a16 * a18);
    a14 = (a1 * a18);
    a14 = (a14 + a17);
    a14 = (a20 + a14);
    a14 = (a19 * a14);
    a14 = (-a14);
    if (res[1] != 0) res[1][3] = a14;
    a14 = (a6 * a22);
    real_t a25 = (a5 * a18);
    a25 = (a14 + a25);
    a25 = (a19 * a25);
    a25 = (-a25);
    if (res[1] != 0) res[1][4] = a25;
    a22 = (a9 * a22);
    a18 = (a8 * a18);
    a18 = (a22 + a18);
    a18 = (a19 * a18);
    a18 = (-a18);
    if (res[1] != 0) res[1][5] = a18;
    if (res[1] != 0) res[1][6] = a24;
    if (res[1] != 0) res[1][7] = a24;
    if (res[1] != 0) res[1][8] = a24;
    a20 = (a19 * a20);
    if (res[1] != 0) res[1][9] = a20;
    a14 = (a19 * a14);
    if (res[1] != 0) res[1][10] = a14;
    a22 = (a19 * a22);
    if (res[1] != 0) res[1][11] = a22;
    if (res[1] != 0) res[1][12] = a24;
    if (res[1] != 0) res[1][13] = a24;
    if (res[1] != 0) res[1][14] = a24;
    if (res[1] != 0) res[1][15] = a24;
    if (res[1] != 0) res[1][16] = a24;
    if (res[1] != 0) res[1][17] = a24;
    if (res[1] != 0) res[1][18] = a24;
    if (res[1] != 0) res[1][19] = a24;
    if (res[1] != 0) res[1][20] = a24;
    a22 = (a6 / a3);
    a22 = (a11 * a22);
    a14 = (a2 * a22);
    a20 = (a5 / a15);
    a20 = (a16 * a20);
    a18 = (a1 * a20);
    a18 = (a14 + a18);
    a18 = (a19 * a18);
    a18 = (-a18);
    if (res[1] != 0) res[1][21] = a18;
    a18 = (a6 * a22);
    a18 = (a18 + a13);
    a25 = (a5 * a20);
    a25 = (a25 + a17);
    a25 = (a18 + a25);
    a25 = (a19 * a25);
    a25 = (-a25);
    if (res[1] != 0) res[1][22] = a25;
    a22 = (a9 * a22);
    a20 = (a8 * a20);
    a20 = (a22 + a20);
    a20 = (a19 * a20);
    a20 = (-a20);
    if (res[1] != 0) res[1][23] = a20;
    if (res[1] != 0) res[1][24] = a24;
    if (res[1] != 0) res[1][25] = a24;
    if (res[1] != 0) res[1][26] = a24;
    a14 = (a19 * a14);
    if (res[1] != 0) res[1][27] = a14;
    a18 = (a19 * a18);
    if (res[1] != 0) res[1][28] = a18;
    a22 = (a19 * a22);
    if (res[1] != 0) res[1][29] = a22;
    if (res[1] != 0) res[1][30] = a24;
    if (res[1] != 0) res[1][31] = a24;
    if (res[1] != 0) res[1][32] = a24;
    if (res[1] != 0) res[1][33] = a24;
    if (res[1] != 0) res[1][34] = a24;
    if (res[1] != 0) res[1][35] = a24;
    if (res[1] != 0) res[1][36] = a24;
    if (res[1] != 0) res[1][37] = a24;
    if (res[1] != 0) res[1][38] = a24;
    a22 = (a9 / a3);
    a22 = (a11 * a22);
    a18 = (a2 * a22);
    a15 = (a8 / a15);
    a16 = (a16 * a15);
    a1 = (a1 * a16);
    a1 = (a18 + a1);
    a1 = (a19 * a1);
    a1 = (-a1);
    if (res[1] != 0) res[1][39] = a1;
    a1 = (a6 * a22);
    a5 = (a5 * a16);
    a5 = (a1 + a5);
    a5 = (a19 * a5);
    a5 = (-a5);
    if (res[1] != 0) res[1][40] = a5;
    a22 = (a9 * a22);
    a22 = (a22 + a13);
    a8 = (a8 * a16);
    a8 = (a8 + a17);
    a8 = (a22 + a8);
    a8 = (a19 * a8);
    a8 = (-a8);
    if (res[1] != 0) res[1][41] = a8;
    if (res[1] != 0) res[1][42] = a24;
    if (res[1] != 0) res[1][43] = a24;
    if (res[1] != 0) res[1][44] = a24;
    a18 = (a19 * a18);
    if (res[1] != 0) res[1][45] = a18;
    a1 = (a19 * a1);
    if (res[1] != 0) res[1][46] = a1;
    a22 = (a19 * a22);
    if (res[1] != 0) res[1][47] = a22;
    if (res[1] != 0) res[1][48] = a24;
    if (res[1] != 0) res[1][49] = a24;
    if (res[1] != 0) res[1][50] = a24;
    if (res[1] != 0) res[1][51] = a24;
    if (res[1] != 0) res[1][52] = a24;
    if (res[1] != 0) res[1][53] = a24;
    if (res[1] != 0) res[1][54] = a12;
    if (res[1] != 0) res[1][55] = a24;
    if (res[1] != 0) res[1][56] = a24;
    if (res[1] != 0) res[1][57] = a24;
    if (res[1] != 0) res[1][58] = a24;
    if (res[1] != 0) res[1][59] = a24;
    if (res[1] != 0) res[1][60] = a24;
    if (res[1] != 0) res[1][61] = a24;
    if (res[1] != 0) res[1][62] = a24;
    if (res[1] != 0) res[1][63] = a24;
    if (res[1] != 0) res[1][64] = a24;
    if (res[1] != 0) res[1][65] = a24;
    if (res[1] != 0) res[1][66] = a24;
    if (res[1] != 0) res[1][67] = a24;
    if (res[1] != 0) res[1][68] = a24;
    if (res[1] != 0) res[1][69] = a24;
    if (res[1] != 0) res[1][70] = a24;
    if (res[1] != 0) res[1][71] = a24;
    if (res[1] != 0) res[1][72] = a24;
    if (res[1] != 0) res[1][73] = a12;
    if (res[1] != 0) res[1][74] = a24;
    if (res[1] != 0) res[1][75] = a24;
    if (res[1] != 0) res[1][76] = a24;
    if (res[1] != 0) res[1][77] = a24;
    if (res[1] != 0) res[1][78] = a24;
    if (res[1] != 0) res[1][79] = a24;
    if (res[1] != 0) res[1][80] = a24;
    if (res[1] != 0) res[1][81] = a24;
    if (res[1] != 0) res[1][82] = a24;
    if (res[1] != 0) res[1][83] = a24;
    if (res[1] != 0) res[1][84] = a24;
    if (res[1] != 0) res[1][85] = a24;
    if (res[1] != 0) res[1][86] = a24;
    if (res[1] != 0) res[1][87] = a24;
    if (res[1] != 0) res[1][88] = a24;
    if (res[1] != 0) res[1][89] = a24;
    if (res[1] != 0) res[1][90] = a24;
    if (res[1] != 0) res[1][91] = a24;
    if (res[1] != 0) res[1][92] = a12;
    if (res[1] != 0) res[1][93] = a24;
    if (res[1] != 0) res[1][94] = a24;
    if (res[1] != 0) res[1][95] = a24;
    if (res[1] != 0) res[1][96] = a24;
    if (res[1] != 0) res[1][97] = a24;
    if (res[1] != 0) res[1][98] = a24;
    if (res[1] != 0) res[1][99] = a24;
    if (res[1] != 0) res[1][100] = a24;
    if (res[1] != 0) res[1][101] = a24;
    if (res[1] != 0) res[1][102] = a24;
    if (res[1] != 0) res[1][103] = a24;
    if (res[1] != 0) res[1][104] = a24;
    if (res[1] != 0) res[1][105] = a24;
    if (res[1] != 0) res[1][106] = a24;
    if (res[1] != 0) res[1][107] = a24;
    if (res[1] != 0) res[1][108] = a24;
    if (res[1] != 0) res[1][109] = a24;
    if (res[1] != 0) res[1][110] = a24;
    a22 = (a2 / a3);
    a22 = (a11 * a22);
    a1 = (a2 * a22);
    a1 = (a1 + a13);
    a18 = (a19 * a1);
    if (res[1] != 0) res[1][111] = a18;
    a18 = (a6 * a22);
    a8 = (a19 * a18);
    if (res[1] != 0) res[1][112] = a8;
    a22 = (a9 * a22);
    a8 = (a19 * a22);
    if (res[1] != 0) res[1][113] = a8;
    if (res[1] != 0) res[1][114] = a24;
    if (res[1] != 0) res[1][115] = a24;
    if (res[1] != 0) res[1][116] = a24;
    a10 = (a10 / a0);
    a8 = (a21 / a0);
    a8 = (a10 * a8);
    a17 = (a21 * a8);
    a17 = (a17 + a7);
    a17 = (a17 + a1);
    a17 = (a19 * a17);
    a17 = (-a17);
    if (res[1] != 0) res[1][117] = a17;
    a17 = (a23 * a8);
    a17 = (a17 + a18);
    a17 = (a19 * a17);
    a17 = (-a17);
    if (res[1] != 0) res[1][118] = a17;
    a8 = (a4 * a8);
    a8 = (a8 + a22);
    a8 = (a19 * a8);
    a8 = (-a8);
    if (res[1] != 0) res[1][119] = a8;
    if (res[1] != 0) res[1][120] = a24;
    if (res[1] != 0) res[1][121] = a24;
    if (res[1] != 0) res[1][122] = a24;
    if (res[1] != 0) res[1][123] = a24;
    if (res[1] != 0) res[1][124] = a24;
    if (res[1] != 0) res[1][125] = a24;
    if (res[1] != 0) res[1][126] = a24;
    if (res[1] != 0) res[1][127] = a24;
    if (res[1] != 0) res[1][128] = a24;
    a8 = (a6 / a3);
    a8 = (a11 * a8);
    a22 = (a2 * a8);
    a17 = (a19 * a22);
    if (res[1] != 0) res[1][129] = a17;
    a17 = (a6 * a8);
    a17 = (a17 + a13);
    a18 = (a19 * a17);
    if (res[1] != 0) res[1][130] = a18;
    a8 = (a9 * a8);
    a18 = (a19 * a8);
    if (res[1] != 0) res[1][131] = a18;
    if (res[1] != 0) res[1][132] = a24;
    if (res[1] != 0) res[1][133] = a24;
    if (res[1] != 0) res[1][134] = a24;
    a18 = (a23 / a0);
    a18 = (a10 * a18);
    a1 = (a21 * a18);
    a1 = (a1 + a22);
    a1 = (a19 * a1);
    a1 = (-a1);
    if (res[1] != 0) res[1][135] = a1;
    a1 = (a23 * a18);
    a1 = (a1 + a7);
    a1 = (a1 + a17);
    a1 = (a19 * a1);
    a1 = (-a1);
    if (res[1] != 0) res[1][136] = a1;
    a18 = (a4 * a18);
    a18 = (a18 + a8);
    a18 = (a19 * a18);
    a18 = (-a18);
    if (res[1] != 0) res[1][137] = a18;
    if (res[1] != 0) res[1][138] = a24;
    if (res[1] != 0) res[1][139] = a24;
    if (res[1] != 0) res[1][140] = a24;
    if (res[1] != 0) res[1][141] = a24;
    if (res[1] != 0) res[1][142] = a24;
    if (res[1] != 0) res[1][143] = a24;
    if (res[1] != 0) res[1][144] = a24;
    if (res[1] != 0) res[1][145] = a24;
    if (res[1] != 0) res[1][146] = a24;
    a3 = (a9 / a3);
    a11 = (a11 * a3);
    a2 = (a2 * a11);
    a3 = (a19 * a2);
    if (res[1] != 0) res[1][147] = a3;
    a6 = (a6 * a11);
    a3 = (a19 * a6);
    if (res[1] != 0) res[1][148] = a3;
    a9 = (a9 * a11);
    a9 = (a9 + a13);
    a13 = (a19 * a9);
    if (res[1] != 0) res[1][149] = a13;
    if (res[1] != 0) res[1][150] = a24;
    if (res[1] != 0) res[1][151] = a24;
    if (res[1] != 0) res[1][152] = a24;
    a13 = (a4 / a0);
    a13 = (a10 * a13);
    a11 = (a21 * a13);
    a11 = (a11 + a2);
    a11 = (a19 * a11);
    a11 = (-a11);
    if (res[1] != 0) res[1][153] = a11;
    a11 = (a23 * a13);
    a11 = (a11 + a6);
    a11 = (a19 * a11);
    a11 = (-a11);
    if (res[1] != 0) res[1][154] = a11;
    a13 = (a4 * a13);
    a13 = (a13 + a7);
    a13 = (a13 + a9);
    a13 = (a19 * a13);
    a13 = (-a13);
    if (res[1] != 0) res[1][155] = a13;
    if (res[1] != 0) res[1][156] = a24;
    if (res[1] != 0) res[1][157] = a24;
    if (res[1] != 0) res[1][158] = a24;
    if (res[1] != 0) res[1][159] = a24;
    if (res[1] != 0) res[1][160] = a24;
    if (res[1] != 0) res[1][161] = a24;
    if (res[1] != 0) res[1][162] = a24;
    if (res[1] != 0) res[1][163] = a24;
    if (res[1] != 0) res[1][164] = a24;
    if (res[1] != 0) res[1][165] = a24;
    if (res[1] != 0) res[1][166] = a24;
    if (res[1] != 0) res[1][167] = a24;
    if (res[1] != 0) res[1][168] = a12;
    if (res[1] != 0) res[1][169] = a24;
    if (res[1] != 0) res[1][170] = a24;
    if (res[1] != 0) res[1][171] = a24;
    if (res[1] != 0) res[1][172] = a24;
    if (res[1] != 0) res[1][173] = a24;
    if (res[1] != 0) res[1][174] = a24;
    if (res[1] != 0) res[1][175] = a24;
    if (res[1] != 0) res[1][176] = a24;
    if (res[1] != 0) res[1][177] = a24;
    if (res[1] != 0) res[1][178] = a24;
    if (res[1] != 0) res[1][179] = a24;
    if (res[1] != 0) res[1][180] = a24;
    if (res[1] != 0) res[1][181] = a24;
    if (res[1] != 0) res[1][182] = a24;
    if (res[1] != 0) res[1][183] = a24;
    if (res[1] != 0) res[1][184] = a24;
    if (res[1] != 0) res[1][185] = a24;
    if (res[1] != 0) res[1][186] = a24;
    if (res[1] != 0) res[1][187] = a12;
    if (res[1] != 0) res[1][188] = a24;
    if (res[1] != 0) res[1][189] = a24;
    if (res[1] != 0) res[1][190] = a24;
    if (res[1] != 0) res[1][191] = a24;
    if (res[1] != 0) res[1][192] = a24;
    if (res[1] != 0) res[1][193] = a24;
    if (res[1] != 0) res[1][194] = a24;
    if (res[1] != 0) res[1][195] = a24;
    if (res[1] != 0) res[1][196] = a24;
    if (res[1] != 0) res[1][197] = a24;
    if (res[1] != 0) res[1][198] = a24;
    if (res[1] != 0) res[1][199] = a24;
    if (res[1] != 0) res[1][200] = a24;
    if (res[1] != 0) res[1][201] = a24;
    if (res[1] != 0) res[1][202] = a24;
    if (res[1] != 0) res[1][203] = a24;
    if (res[1] != 0) res[1][204] = a24;
    if (res[1] != 0) res[1][205] = a24;
    if (res[1] != 0) res[1][206] = a12;
    if (res[1] != 0) res[1][207] = a24;
    if (res[1] != 0) res[1][208] = a24;
    if (res[1] != 0) res[1][209] = a24;
    if (res[1] != 0) res[1][210] = a24;
    if (res[1] != 0) res[1][211] = a24;
    if (res[1] != 0) res[1][212] = a24;
    if (res[1] != 0) res[1][213] = a24;
    if (res[1] != 0) res[1][214] = a24;
    if (res[1] != 0) res[1][215] = a24;
    if (res[1] != 0) res[1][216] = a24;
    if (res[1] != 0) res[1][217] = a24;
    if (res[1] != 0) res[1][218] = a24;
    if (res[1] != 0) res[1][219] = a24;
    if (res[1] != 0) res[1][220] = a24;
    if (res[1] != 0) res[1][221] = a24;
    if (res[1] != 0) res[1][222] = a24;
    if (res[1] != 0) res[1][223] = a24;
    if (res[1] != 0) res[1][224] = a24;
    a13 = (a21 / a0);
    a13 = (a10 * a13);
    a9 = (a21 * a13);
    a9 = (a9 + a7);
    a9 = (a19 * a9);
    if (res[1] != 0) res[1][225] = a9;
    a9 = (a23 * a13);
    a9 = (a19 * a9);
    if (res[1] != 0) res[1][226] = a9;
    a13 = (a4 * a13);
    a13 = (a19 * a13);
    if (res[1] != 0) res[1][227] = a13;
    if (res[1] != 0) res[1][228] = a24;
    if (res[1] != 0) res[1][229] = a24;
    if (res[1] != 0) res[1][230] = a24;
    if (res[1] != 0) res[1][231] = a24;
    if (res[1] != 0) res[1][232] = a24;
    if (res[1] != 0) res[1][233] = a24;
    if (res[1] != 0) res[1][234] = a24;
    if (res[1] != 0) res[1][235] = a24;
    if (res[1] != 0) res[1][236] = a24;
    if (res[1] != 0) res[1][237] = a24;
    if (res[1] != 0) res[1][238] = a24;
    if (res[1] != 0) res[1][239] = a24;
    if (res[1] != 0) res[1][240] = a24;
    if (res[1] != 0) res[1][241] = a24;
    if (res[1] != 0) res[1][242] = a24;
    a13 = (a23 / a0);
    a13 = (a10 * a13);
    a9 = (a21 * a13);
    a9 = (a19 * a9);
    if (res[1] != 0) res[1][243] = a9;
    a9 = (a23 * a13);
    a9 = (a9 + a7);
    a9 = (a19 * a9);
    if (res[1] != 0) res[1][244] = a9;
    a13 = (a4 * a13);
    a13 = (a19 * a13);
    if (res[1] != 0) res[1][245] = a13;
    if (res[1] != 0) res[1][246] = a24;
    if (res[1] != 0) res[1][247] = a24;
    if (res[1] != 0) res[1][248] = a24;
    if (res[1] != 0) res[1][249] = a24;
    if (res[1] != 0) res[1][250] = a24;
    if (res[1] != 0) res[1][251] = a24;
    if (res[1] != 0) res[1][252] = a24;
    if (res[1] != 0) res[1][253] = a24;
    if (res[1] != 0) res[1][254] = a24;
    if (res[1] != 0) res[1][255] = a24;
    if (res[1] != 0) res[1][256] = a24;
    if (res[1] != 0) res[1][257] = a24;
    if (res[1] != 0) res[1][258] = a24;
    if (res[1] != 0) res[1][259] = a24;
    if (res[1] != 0) res[1][260] = a24;
    a0 = (a4 / a0);
    a10 = (a10 * a0);
    a21 = (a21 * a10);
    a21 = (a19 * a21);
    if (res[1] != 0) res[1][261] = a21;
    a23 = (a23 * a10);
    a23 = (a19 * a23);
    if (res[1] != 0) res[1][262] = a23;
    a4 = (a4 * a10);
    a4 = (a4 + a7);
    a19 = (a19 * a4);
    if (res[1] != 0) res[1][263] = a19;
    if (res[1] != 0) res[1][264] = a24;
    if (res[1] != 0) res[1][265] = a24;
    if (res[1] != 0) res[1][266] = a24;
    if (res[1] != 0) res[1][267] = a24;
    if (res[1] != 0) res[1][268] = a24;
    if (res[1] != 0) res[1][269] = a24;
    if (res[1] != 0) res[1][270] = a24;
    if (res[1] != 0) res[1][271] = a24;
    if (res[1] != 0) res[1][272] = a24;
    if (res[1] != 0) res[1][273] = a24;
    if (res[1] != 0) res[1][274] = a24;
    if (res[1] != 0) res[1][275] = a24;
    if (res[1] != 0) res[1][276] = a24;
    if (res[1] != 0) res[1][277] = a24;
    if (res[1] != 0) res[1][278] = a24;
    if (res[1] != 0) res[1][279] = a24;
    if (res[1] != 0) res[1][280] = a24;
    if (res[1] != 0) res[1][281] = a24;
    if (res[1] != 0) res[1][282] = a12;
    if (res[1] != 0) res[1][283] = a24;
    if (res[1] != 0) res[1][284] = a24;
    if (res[1] != 0) res[1][285] = a24;
    if (res[1] != 0) res[1][286] = a24;
    if (res[1] != 0) res[1][287] = a24;
    if (res[1] != 0) res[1][288] = a24;
    if (res[1] != 0) res[1][289] = a24;
    if (res[1] != 0) res[1][290] = a24;
    if (res[1] != 0) res[1][291] = a24;
    if (res[1] != 0) res[1][292] = a24;
    if (res[1] != 0) res[1][293] = a24;
    if (res[1] != 0) res[1][294] = a24;
    if (res[1] != 0) res[1][295] = a24;
    if (res[1] != 0) res[1][296] = a24;
    if (res[1] != 0) res[1][297] = a24;
    if (res[1] != 0) res[1][298] = a24;
    if (res[1] != 0) res[1][299] = a24;
    if (res[1] != 0) res[1][300] = a24;
    if (res[1] != 0) res[1][301] = a12;
    if (res[1] != 0) res[1][302] = a24;
    if (res[1] != 0) res[1][303] = a24;
    if (res[1] != 0) res[1][304] = a24;
    if (res[1] != 0) res[1][305] = a24;
    if (res[1] != 0) res[1][306] = a24;
    if (res[1] != 0) res[1][307] = a24;
    if (res[1] != 0) res[1][308] = a24;
    if (res[1] != 0) res[1][309] = a24;
    if (res[1] != 0) res[1][310] = a24;
    if (res[1] != 0) res[1][311] = a24;
    if (res[1] != 0) res[1][312] = a24;
    if (res[1] != 0) res[1][313] = a24;
    if (res[1] != 0) res[1][314] = a24;
    if (res[1] != 0) res[1][315] = a24;
    if (res[1] != 0) res[1][316] = a24;
    if (res[1] != 0) res[1][317] = a24;
    if (res[1] != 0) res[1][318] = a24;
    if (res[1] != 0) res[1][319] = a24;
    if (res[1] != 0) res[1][320] = a12;
    if (res[1] != 0) res[1][321] = a24;
    if (res[1] != 0) res[1][322] = a24;
    if (res[1] != 0) res[1][323] = a24;
    return 0;
}

int jac_chain_nm4_init(int* n_in, int* n_out, int* n_int, int* n_real) {
    if (n_in) *n_in = 2;
    if (n_out) *n_out = 2;
    if (n_int) *n_int = 0;
    if (n_real) *n_real = 0;
    return 0;
}

int jac_chain_nm4_alloc(void** mem, const int* idata, const double* rdata) {
    if (mem) *mem = 0;
    (void)idata;
    (void)rdata;
    return 0;
}

int jac_chain_nm4_free(void* mem) {
    (void)mem;
    return 0;
}

int jac_chain_nm4_sparsity(int i, int* nrow, int* ncol, const int** colind,
                           const int** row) {
    const int* s;
    switch (i) {
        case 0:
        case 2:
            s = s0;
            break;
        case 1:
            s = s1;
            break;
        case 3:
            s = s2;
            break;
        default:
            return 1;
    }

    if (nrow) *nrow = s[0];
    if (ncol) *ncol = s[1];
    if (colind) *colind = s + 2;
    if (row) *row = s + 3 + s[1];
    return 0;
}

int jac_chain_nm4_work(int* sz_arg, int* sz_res, int* sz_iw, int* sz_w) {
    if (sz_arg) *sz_arg = 2;
    if (sz_res) *sz_res = 2;
    if (sz_iw) *sz_iw = 0;
    if (sz_w) *sz_w = 26;
    return 0;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
