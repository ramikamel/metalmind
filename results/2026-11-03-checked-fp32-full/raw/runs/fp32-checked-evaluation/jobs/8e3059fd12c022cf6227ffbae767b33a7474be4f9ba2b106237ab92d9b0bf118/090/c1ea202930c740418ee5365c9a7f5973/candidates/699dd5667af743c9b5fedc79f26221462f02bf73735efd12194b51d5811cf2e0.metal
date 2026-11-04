#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x,device const float* p,device float* y,constant uint& B,constant uint& D,constant float& eps,uint i [[thread_position_in_grid]]) {if(i>=B*D)return;float z=float(x[i])+float(p[i%D]);y[i]=float(0.5f*z*(1+tanh(0.7978845608f*(z+0.044715f*z*z*z))));}