/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CODEGEN_PREFIX
  #define NAMESPACE_CONCAT(NS, ID) _NAMESPACE_CONCAT(NS, ID)
  #define _NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) impl_ode_fun_jac_x_xdot_u_chain_nm3_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

static const casadi_int casadi_s0[13] = {9, 1, 0, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8};
static const casadi_int casadi_s1[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s2[93] = {9, 9, 0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8};
static const casadi_int casadi_s3[33] = {9, 3, 0, 9, 18, 27, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8};

casadi_real casadi_sq(casadi_real x) { return x*x;}

/* casadi_impl_ode_fun_jac_x_xdot_u_chain_nm3:(i0[9],i1[9],i2[3])->(o0[9],o1[9x9],o2[9x9],o3[9x3]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, void* mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=arg[1] ? arg[1][0] : 0;
  a1=arg[0] ? arg[0][3] : 0;
  a0=(a0-a1);
  if (res[0]!=0) res[0][0]=a0;
  a0=arg[1] ? arg[1][1] : 0;
  a1=arg[0] ? arg[0][4] : 0;
  a0=(a0-a1);
  if (res[0]!=0) res[0][1]=a0;
  a0=arg[1] ? arg[1][2] : 0;
  a1=arg[0] ? arg[0][5] : 0;
  a0=(a0-a1);
  if (res[0]!=0) res[0][2]=a0;
  a0=arg[1] ? arg[1][3] : 0;
  a1=3.3333333333333336e+01;
  a2=1.;
  a3=3.3000000000000002e-02;
  a4=arg[0] ? arg[0][6] : 0;
  a5=arg[0] ? arg[0][0] : 0;
  a4=(a4-a5);
  a6=casadi_sq(a4);
  a7=arg[0] ? arg[0][7] : 0;
  a8=arg[0] ? arg[0][1] : 0;
  a7=(a7-a8);
  a9=casadi_sq(a7);
  a6=(a6+a9);
  a9=arg[0] ? arg[0][8] : 0;
  a10=arg[0] ? arg[0][2] : 0;
  a9=(a9-a10);
  a11=casadi_sq(a9);
  a6=(a6+a11);
  a6=sqrt(a6);
  a11=(a3/a6);
  a12=(a2-a11);
  a13=(a12*a4);
  a14=casadi_sq(a5);
  a15=casadi_sq(a8);
  a14=(a14+a15);
  a15=casadi_sq(a10);
  a14=(a14+a15);
  a14=sqrt(a14);
  a3=(a3/a14);
  a15=(a2-a3);
  a16=(a15*a5);
  a13=(a13-a16);
  a13=(a1*a13);
  a0=(a0-a13);
  if (res[0]!=0) res[0][3]=a0;
  a0=arg[1] ? arg[1][4] : 0;
  a13=(a12*a7);
  a16=(a15*a8);
  a13=(a13-a16);
  a13=(a1*a13);
  a0=(a0-a13);
  if (res[0]!=0) res[0][4]=a0;
  a0=arg[1] ? arg[1][5] : 0;
  a13=(a12*a9);
  a16=(a15*a10);
  a13=(a13-a16);
  a13=(a1*a13);
  a16=9.8100000000000005e+00;
  a13=(a13-a16);
  a0=(a0-a13);
  if (res[0]!=0) res[0][5]=a0;
  a0=arg[1] ? arg[1][6] : 0;
  a13=arg[2] ? arg[2][0] : 0;
  a0=(a0-a13);
  if (res[0]!=0) res[0][6]=a0;
  a0=arg[1] ? arg[1][7] : 0;
  a13=arg[2] ? arg[2][1] : 0;
  a0=(a0-a13);
  if (res[0]!=0) res[0][7]=a0;
  a0=arg[1] ? arg[1][8] : 0;
  a13=arg[2] ? arg[2][2] : 0;
  a0=(a0-a13);
  if (res[0]!=0) res[0][8]=a0;
  a0=0.;
  if (res[1]!=0) res[1][0]=a0;
  if (res[1]!=0) res[1][1]=a0;
  if (res[1]!=0) res[1][2]=a0;
  a11=(a11/a6);
  a13=(a4/a6);
  a13=(a11*a13);
  a16=(a4*a13);
  a16=(a16+a12);
  a3=(a3/a14);
  a17=(a5/a14);
  a17=(a3*a17);
  a18=(a5*a17);
  a18=(a18+a15);
  a16=(a16+a18);
  a16=(a1*a16);
  if (res[1]!=0) res[1][3]=a16;
  a16=(a7*a13);
  a18=(a8*a17);
  a16=(a16+a18);
  a16=(a1*a16);
  if (res[1]!=0) res[1][4]=a16;
  a13=(a9*a13);
  a17=(a10*a17);
  a13=(a13+a17);
  a13=(a1*a13);
  if (res[1]!=0) res[1][5]=a13;
  if (res[1]!=0) res[1][6]=a0;
  if (res[1]!=0) res[1][7]=a0;
  if (res[1]!=0) res[1][8]=a0;
  if (res[1]!=0) res[1][9]=a0;
  if (res[1]!=0) res[1][10]=a0;
  if (res[1]!=0) res[1][11]=a0;
  a13=(a7/a6);
  a13=(a11*a13);
  a17=(a4*a13);
  a16=(a8/a14);
  a16=(a3*a16);
  a18=(a5*a16);
  a17=(a17+a18);
  a17=(a1*a17);
  if (res[1]!=0) res[1][12]=a17;
  a17=(a7*a13);
  a17=(a17+a12);
  a18=(a8*a16);
  a18=(a18+a15);
  a17=(a17+a18);
  a17=(a1*a17);
  if (res[1]!=0) res[1][13]=a17;
  a13=(a9*a13);
  a16=(a10*a16);
  a13=(a13+a16);
  a13=(a1*a13);
  if (res[1]!=0) res[1][14]=a13;
  if (res[1]!=0) res[1][15]=a0;
  if (res[1]!=0) res[1][16]=a0;
  if (res[1]!=0) res[1][17]=a0;
  if (res[1]!=0) res[1][18]=a0;
  if (res[1]!=0) res[1][19]=a0;
  if (res[1]!=0) res[1][20]=a0;
  a13=(a9/a6);
  a13=(a11*a13);
  a16=(a4*a13);
  a14=(a10/a14);
  a3=(a3*a14);
  a5=(a5*a3);
  a16=(a16+a5);
  a16=(a1*a16);
  if (res[1]!=0) res[1][21]=a16;
  a16=(a7*a13);
  a8=(a8*a3);
  a16=(a16+a8);
  a16=(a1*a16);
  if (res[1]!=0) res[1][22]=a16;
  a13=(a9*a13);
  a13=(a13+a12);
  a10=(a10*a3);
  a10=(a10+a15);
  a13=(a13+a10);
  a13=(a1*a13);
  if (res[1]!=0) res[1][23]=a13;
  if (res[1]!=0) res[1][24]=a0;
  if (res[1]!=0) res[1][25]=a0;
  if (res[1]!=0) res[1][26]=a0;
  a13=-1.;
  if (res[1]!=0) res[1][27]=a13;
  if (res[1]!=0) res[1][28]=a0;
  if (res[1]!=0) res[1][29]=a0;
  if (res[1]!=0) res[1][30]=a0;
  if (res[1]!=0) res[1][31]=a0;
  if (res[1]!=0) res[1][32]=a0;
  if (res[1]!=0) res[1][33]=a0;
  if (res[1]!=0) res[1][34]=a0;
  if (res[1]!=0) res[1][35]=a0;
  if (res[1]!=0) res[1][36]=a0;
  if (res[1]!=0) res[1][37]=a13;
  if (res[1]!=0) res[1][38]=a0;
  if (res[1]!=0) res[1][39]=a0;
  if (res[1]!=0) res[1][40]=a0;
  if (res[1]!=0) res[1][41]=a0;
  if (res[1]!=0) res[1][42]=a0;
  if (res[1]!=0) res[1][43]=a0;
  if (res[1]!=0) res[1][44]=a0;
  if (res[1]!=0) res[1][45]=a0;
  if (res[1]!=0) res[1][46]=a0;
  if (res[1]!=0) res[1][47]=a13;
  if (res[1]!=0) res[1][48]=a0;
  if (res[1]!=0) res[1][49]=a0;
  if (res[1]!=0) res[1][50]=a0;
  if (res[1]!=0) res[1][51]=a0;
  if (res[1]!=0) res[1][52]=a0;
  if (res[1]!=0) res[1][53]=a0;
  if (res[1]!=0) res[1][54]=a0;
  if (res[1]!=0) res[1][55]=a0;
  if (res[1]!=0) res[1][56]=a0;
  a10=(a4/a6);
  a10=(a11*a10);
  a15=(a4*a10);
  a15=(a15+a12);
  a15=(a1*a15);
  a15=(-a15);
  if (res[1]!=0) res[1][57]=a15;
  a15=(a7*a10);
  a15=(a1*a15);
  a15=(-a15);
  if (res[1]!=0) res[1][58]=a15;
  a10=(a9*a10);
  a10=(a1*a10);
  a10=(-a10);
  if (res[1]!=0) res[1][59]=a10;
  if (res[1]!=0) res[1][60]=a0;
  if (res[1]!=0) res[1][61]=a0;
  if (res[1]!=0) res[1][62]=a0;
  if (res[1]!=0) res[1][63]=a0;
  if (res[1]!=0) res[1][64]=a0;
  if (res[1]!=0) res[1][65]=a0;
  a10=(a7/a6);
  a10=(a11*a10);
  a15=(a4*a10);
  a15=(a1*a15);
  a15=(-a15);
  if (res[1]!=0) res[1][66]=a15;
  a15=(a7*a10);
  a15=(a15+a12);
  a15=(a1*a15);
  a15=(-a15);
  if (res[1]!=0) res[1][67]=a15;
  a10=(a9*a10);
  a10=(a1*a10);
  a10=(-a10);
  if (res[1]!=0) res[1][68]=a10;
  if (res[1]!=0) res[1][69]=a0;
  if (res[1]!=0) res[1][70]=a0;
  if (res[1]!=0) res[1][71]=a0;
  if (res[1]!=0) res[1][72]=a0;
  if (res[1]!=0) res[1][73]=a0;
  if (res[1]!=0) res[1][74]=a0;
  a6=(a9/a6);
  a11=(a11*a6);
  a4=(a4*a11);
  a4=(a1*a4);
  a4=(-a4);
  if (res[1]!=0) res[1][75]=a4;
  a7=(a7*a11);
  a7=(a1*a7);
  a7=(-a7);
  if (res[1]!=0) res[1][76]=a7;
  a9=(a9*a11);
  a9=(a9+a12);
  a1=(a1*a9);
  a1=(-a1);
  if (res[1]!=0) res[1][77]=a1;
  if (res[1]!=0) res[1][78]=a0;
  if (res[1]!=0) res[1][79]=a0;
  if (res[1]!=0) res[1][80]=a0;
  if (res[2]!=0) res[2][0]=a2;
  if (res[2]!=0) res[2][1]=a0;
  if (res[2]!=0) res[2][2]=a0;
  if (res[2]!=0) res[2][3]=a0;
  if (res[2]!=0) res[2][4]=a0;
  if (res[2]!=0) res[2][5]=a0;
  if (res[2]!=0) res[2][6]=a0;
  if (res[2]!=0) res[2][7]=a0;
  if (res[2]!=0) res[2][8]=a0;
  if (res[2]!=0) res[2][9]=a0;
  if (res[2]!=0) res[2][10]=a2;
  if (res[2]!=0) res[2][11]=a0;
  if (res[2]!=0) res[2][12]=a0;
  if (res[2]!=0) res[2][13]=a0;
  if (res[2]!=0) res[2][14]=a0;
  if (res[2]!=0) res[2][15]=a0;
  if (res[2]!=0) res[2][16]=a0;
  if (res[2]!=0) res[2][17]=a0;
  if (res[2]!=0) res[2][18]=a0;
  if (res[2]!=0) res[2][19]=a0;
  if (res[2]!=0) res[2][20]=a2;
  if (res[2]!=0) res[2][21]=a0;
  if (res[2]!=0) res[2][22]=a0;
  if (res[2]!=0) res[2][23]=a0;
  if (res[2]!=0) res[2][24]=a0;
  if (res[2]!=0) res[2][25]=a0;
  if (res[2]!=0) res[2][26]=a0;
  if (res[2]!=0) res[2][27]=a0;
  if (res[2]!=0) res[2][28]=a0;
  if (res[2]!=0) res[2][29]=a0;
  if (res[2]!=0) res[2][30]=a2;
  if (res[2]!=0) res[2][31]=a0;
  if (res[2]!=0) res[2][32]=a0;
  if (res[2]!=0) res[2][33]=a0;
  if (res[2]!=0) res[2][34]=a0;
  if (res[2]!=0) res[2][35]=a0;
  if (res[2]!=0) res[2][36]=a0;
  if (res[2]!=0) res[2][37]=a0;
  if (res[2]!=0) res[2][38]=a0;
  if (res[2]!=0) res[2][39]=a0;
  if (res[2]!=0) res[2][40]=a2;
  if (res[2]!=0) res[2][41]=a0;
  if (res[2]!=0) res[2][42]=a0;
  if (res[2]!=0) res[2][43]=a0;
  if (res[2]!=0) res[2][44]=a0;
  if (res[2]!=0) res[2][45]=a0;
  if (res[2]!=0) res[2][46]=a0;
  if (res[2]!=0) res[2][47]=a0;
  if (res[2]!=0) res[2][48]=a0;
  if (res[2]!=0) res[2][49]=a0;
  if (res[2]!=0) res[2][50]=a2;
  if (res[2]!=0) res[2][51]=a0;
  if (res[2]!=0) res[2][52]=a0;
  if (res[2]!=0) res[2][53]=a0;
  if (res[2]!=0) res[2][54]=a0;
  if (res[2]!=0) res[2][55]=a0;
  if (res[2]!=0) res[2][56]=a0;
  if (res[2]!=0) res[2][57]=a0;
  if (res[2]!=0) res[2][58]=a0;
  if (res[2]!=0) res[2][59]=a0;
  if (res[2]!=0) res[2][60]=a2;
  if (res[2]!=0) res[2][61]=a0;
  if (res[2]!=0) res[2][62]=a0;
  if (res[2]!=0) res[2][63]=a0;
  if (res[2]!=0) res[2][64]=a0;
  if (res[2]!=0) res[2][65]=a0;
  if (res[2]!=0) res[2][66]=a0;
  if (res[2]!=0) res[2][67]=a0;
  if (res[2]!=0) res[2][68]=a0;
  if (res[2]!=0) res[2][69]=a0;
  if (res[2]!=0) res[2][70]=a2;
  if (res[2]!=0) res[2][71]=a0;
  if (res[2]!=0) res[2][72]=a0;
  if (res[2]!=0) res[2][73]=a0;
  if (res[2]!=0) res[2][74]=a0;
  if (res[2]!=0) res[2][75]=a0;
  if (res[2]!=0) res[2][76]=a0;
  if (res[2]!=0) res[2][77]=a0;
  if (res[2]!=0) res[2][78]=a0;
  if (res[2]!=0) res[2][79]=a0;
  if (res[2]!=0) res[2][80]=a2;
  if (res[3]!=0) res[3][0]=a0;
  if (res[3]!=0) res[3][1]=a0;
  if (res[3]!=0) res[3][2]=a0;
  if (res[3]!=0) res[3][3]=a0;
  if (res[3]!=0) res[3][4]=a0;
  if (res[3]!=0) res[3][5]=a0;
  if (res[3]!=0) res[3][6]=a13;
  if (res[3]!=0) res[3][7]=a0;
  if (res[3]!=0) res[3][8]=a0;
  if (res[3]!=0) res[3][9]=a0;
  if (res[3]!=0) res[3][10]=a0;
  if (res[3]!=0) res[3][11]=a0;
  if (res[3]!=0) res[3][12]=a0;
  if (res[3]!=0) res[3][13]=a0;
  if (res[3]!=0) res[3][14]=a0;
  if (res[3]!=0) res[3][15]=a0;
  if (res[3]!=0) res[3][16]=a13;
  if (res[3]!=0) res[3][17]=a0;
  if (res[3]!=0) res[3][18]=a0;
  if (res[3]!=0) res[3][19]=a0;
  if (res[3]!=0) res[3][20]=a0;
  if (res[3]!=0) res[3][21]=a0;
  if (res[3]!=0) res[3][22]=a0;
  if (res[3]!=0) res[3][23]=a0;
  if (res[3]!=0) res[3][24]=a0;
  if (res[3]!=0) res[3][25]=a0;
  if (res[3]!=0) res[3][26]=a13;
  return 0;
}

CASADI_SYMBOL_EXPORT int casadi_impl_ode_fun_jac_x_xdot_u_chain_nm3(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, void* mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT void casadi_impl_ode_fun_jac_x_xdot_u_chain_nm3_incref(void) {
}

CASADI_SYMBOL_EXPORT void casadi_impl_ode_fun_jac_x_xdot_u_chain_nm3_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int casadi_impl_ode_fun_jac_x_xdot_u_chain_nm3_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int casadi_impl_ode_fun_jac_x_xdot_u_chain_nm3_n_out(void) { return 4;}

CASADI_SYMBOL_EXPORT const char* casadi_impl_ode_fun_jac_x_xdot_u_chain_nm3_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* casadi_impl_ode_fun_jac_x_xdot_u_chain_nm3_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_impl_ode_fun_jac_x_xdot_u_chain_nm3_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* casadi_impl_ode_fun_jac_x_xdot_u_chain_nm3_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s2;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int casadi_impl_ode_fun_jac_x_xdot_u_chain_nm3_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
