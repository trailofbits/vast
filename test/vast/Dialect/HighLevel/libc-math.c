// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-DAG: hl.func @fabs external
// CHECK-DAG: hl.func @fabsf external
// CHECK-DAG: hl.func @fabsl external

// CHECK-DAG: hl.func @fmod external
// CHECK-DAG: hl.func @fmodf external
// CHECK-DAG: hl.func @fmodl external

// CHECK-DAG: hl.func @remainder external
// CHECK-DAG: hl.func @remainderf external
// CHECK-DAG: hl.func @remainderl external

// CHECK-DAG: hl.func @remquo external
// CHECK-DAG: hl.func @remquof external
// CHECK-DAG: hl.func @remquol external

// CHECK-DAG: hl.func @fma external
// CHECK-DAG: hl.func @fmaf external
// CHECK-DAG: hl.func @fmal external

// CHECK-DAG: hl.func @fmax external
// CHECK-DAG: hl.func @fmaxf external
// CHECK-DAG: hl.func @fmaxl external

// CHECK-DAG: hl.func @fmin external
// CHECK-DAG: hl.func @fminf external
// CHECK-DAG: hl.func @fminl external

// CHECK-DAG: hl.func @fdim external
// CHECK-DAG: hl.func @fdimf external
// CHECK-DAG: hl.func @fdiml external

// CHECK-DAG: hl.func @nan external
// CHECK-DAG: hl.func @nanf external
// CHECK-DAG: hl.func @nanl external

// CHECK-DAG: hl.func @exp external
// CHECK-DAG: hl.func @expf external
// CHECK-DAG: hl.func @expl external

// CHECK-DAG: hl.func @exp2 external
// CHECK-DAG: hl.func @exp2f external
// CHECK-DAG: hl.func @exp2l external

// CHECK-DAG: hl.func @expm1 external
// CHECK-DAG: hl.func @expm1f external
// CHECK-DAG: hl.func @expm1l external

// CHECK-DAG: hl.func @log external
// CHECK-DAG: hl.func @logf external
// CHECK-DAG: hl.func @logl external

// CHECK-DAG: hl.func @log10 external
// CHECK-DAG: hl.func @log10f external
// CHECK-DAG: hl.func @log10l external

// CHECK-DAG: hl.func @log2 external
// CHECK-DAG: hl.func @log2f external
// CHECK-DAG: hl.func @log2l external

// CHECK-DAG: hl.func @log1p external
// CHECK-DAG: hl.func @log1pf external
// CHECK-DAG: hl.func @log1pl external

// CHECK-DAG: hl.func @pow external
// CHECK-DAG: hl.func @powf external
// CHECK-DAG: hl.func @powl external

// CHECK-DAG: hl.func @sqrt external
// CHECK-DAG: hl.func @sqrtf external
// CHECK-DAG: hl.func @sqrtl external

// CHECK-DAG: hl.func @cbrt external
// CHECK-DAG: hl.func @cbrtf external
// CHECK-DAG: hl.func @cbrtl external

// CHECK-DAG: hl.func @hypot external
// CHECK-DAG: hl.func @hypotf external
// CHECK-DAG: hl.func @hypotl external

// CHECK-DAG: hl.func @sin external
// CHECK-DAG: hl.func @sinf external
// CHECK-DAG: hl.func @sinl external

// CHECK-DAG: hl.func @cos external
// CHECK-DAG: hl.func @cosf external
// CHECK-DAG: hl.func @cosl external

// CHECK-DAG: hl.func @tan external
// CHECK-DAG: hl.func @tanf external
// CHECK-DAG: hl.func @tanl external

// CHECK-DAG: hl.func @asin external
// CHECK-DAG: hl.func @asinf external
// CHECK-DAG: hl.func @asinl external

// CHECK-DAG: hl.func @acos external
// CHECK-DAG: hl.func @acosf external
// CHECK-DAG: hl.func @acosl external

// CHECK-DAG: hl.func @atan external
// CHECK-DAG: hl.func @atanf external
// CHECK-DAG: hl.func @atanl external

// CHECK-DAG: hl.func @atan2 external
// CHECK-DAG: hl.func @atan2f external
// CHECK-DAG: hl.func @atan2l external

// CHECK-DAG: hl.func @sinh external
// CHECK-DAG: hl.func @sinhf external
// CHECK-DAG: hl.func @sinhl external

// CHECK-DAG: hl.func @cosh external
// CHECK-DAG: hl.func @coshf external
// CHECK-DAG: hl.func @coshl external

// CHECK-DAG: hl.func @tanh external
// CHECK-DAG: hl.func @tanhf external
// CHECK-DAG: hl.func @tanhl external

// CHECK-DAG: hl.func @asinh external
// CHECK-DAG: hl.func @asinhf external
// CHECK-DAG: hl.func @asinhl external

// CHECK-DAG: hl.func @acosh external
// CHECK-DAG: hl.func @acoshf external
// CHECK-DAG: hl.func @acoshl external

// CHECK-DAG: hl.func @atanh external
// CHECK-DAG: hl.func @atanhf external
// CHECK-DAG: hl.func @atanhl external

// CHECK-DAG: hl.func @erf external
// CHECK-DAG: hl.func @erff external
// CHECK-DAG: hl.func @erfl external

// CHECK-DAG: hl.func @erfc external
// CHECK-DAG: hl.func @erfcf external
// CHECK-DAG: hl.func @erfcl external

// CHECK-DAG: hl.func @tgamma external
// CHECK-DAG: hl.func @tgammaf external
// CHECK-DAG: hl.func @tgammal external

// CHECK-DAG: hl.func @lgamma external
// CHECK-DAG: hl.func @lgammaf external
// CHECK-DAG: hl.func @lgammal external

// CHECK-DAG: hl.func @ceil external
// CHECK-DAG: hl.func @ceilf external
// CHECK-DAG: hl.func @ceill external

// CHECK-DAG: hl.func @floor external
// CHECK-DAG: hl.func @floorf external
// CHECK-DAG: hl.func @floorl external

// CHECK-DAG: hl.func @trunc external
// CHECK-DAG: hl.func @truncf external
// CHECK-DAG: hl.func @truncl external

// CHECK-DAG: hl.func @round external
// CHECK-DAG: hl.func @roundf external
// CHECK-DAG: hl.func @roundl external

// CHECK-DAG: hl.func @lround external
// CHECK-DAG: hl.func @lroundf external
// CHECK-DAG: hl.func @lroundl external

// CHECK-DAG: hl.func @llround external
// CHECK-DAG: hl.func @llroundf external
// CHECK-DAG: hl.func @llroundl external

// CHECK-DAG: hl.func @nearbyint external
// CHECK-DAG: hl.func @nearbyintf external
// CHECK-DAG: hl.func @nearbyintl external

// CHECK-DAG: hl.func @rint external
// CHECK-DAG: hl.func @rintf external
// CHECK-DAG: hl.func @rintl external

// CHECK-DAG: hl.func @lrint external
// CHECK-DAG: hl.func @lrintf external
// CHECK-DAG: hl.func @lrintl external

// CHECK-DAG: hl.func @llrint external
// CHECK-DAG: hl.func @llrintf external
// CHECK-DAG: hl.func @llrintl external

// CHECK-DAG: hl.func @frexp external
// CHECK-DAG: hl.func @frexpf external
// CHECK-DAG: hl.func @frexpl external

// CHECK-DAG: hl.func @ldexp external
// CHECK-DAG: hl.func @ldexpf external
// CHECK-DAG: hl.func @ldexpl external

// CHECK-DAG: hl.func @modf external
// CHECK-DAG: hl.func @modff external
// CHECK-DAG: hl.func @modfl external

// CHECK-DAG: hl.func @scalbn external
// CHECK-DAG: hl.func @scalbnf external
// CHECK-DAG: hl.func @scalbnl external

// CHECK-DAG: hl.func @scalbln external
// CHECK-DAG: hl.func @scalblnf external
// CHECK-DAG: hl.func @scalblnl external

// CHECK-DAG: hl.func @ilogb external
// CHECK-DAG: hl.func @ilogbf external
// CHECK-DAG: hl.func @ilogbl external

// CHECK-DAG: hl.func @logb external
// CHECK-DAG: hl.func @logbf external
// CHECK-DAG: hl.func @logbl external

// CHECK-DAG: hl.func @nextafter external
// CHECK-DAG: hl.func @nextafterf external
// CHECK-DAG: hl.func @nextafterl external

// CHECK-DAG: hl.func @nexttoward external
// CHECK-DAG: hl.func @nexttowardf external
// CHECK-DAG: hl.func @nexttowardl external

// CHECK-DAG: hl.func @copysign external
// CHECK-DAG: hl.func @copysignf external
// CHECK-DAG: hl.func @copysignl external

// CHECK-DAG: hl.typedef @float_t
// CHECK-DAG: hl.typedef @double_t
#include <math.h>
