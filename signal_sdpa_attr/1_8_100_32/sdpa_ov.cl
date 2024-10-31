typedef char int8_t;
typedef uchar uint8_t;
typedef int int32_t;
typedef uint uint32_t;
typedef half fp16;
#define FP16_MIN (-65504.0f)
#define cl_intel_subgroup_extended_block_read 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define IS_FLOAT_TYPE(x) _Generic((x), float: 1, default: 0)
#define IS_HALF_TYPE(x) _Generic((x), fp16: 1, default: 0)
#define IS_INT_TYPE(x) _Generic((x), int: 1, default: 0)

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void sdpa(
  __global INPUT0_TYPE* qState,
  __global INPUT0_TYPE* kState,
  __global INPUT0_TYPE* vState,
  // __global half* mState,
  __global OUTPUT0_TYPE* qkvOut
  // uint32_t queryLen,
  // uint32_t kvSeqLen,
  // uint32_t batchSize,
  // uint32_t hasMask
  ) {
  // if (get_group_id(0) ==0 && get_group_id(1) == 0 && get_sub_group_id() == 0 && get_sub_group_local_id() == 0)
  // // if (get_group_id(0) ==0 && get_group_id(1) == 0 && get_sub_group_id() == 0)
  // {
  //   printf("Global size: (%d, %d, %d)\n", get_global_size(0), get_global_size(1), get_global_size(2));
  //   printf("attr: q_len %d, kv_len %d, batch_size %d, hasMask %d\n", queryLen, kvSeqLen, batchSize, hasMask);
  //   printf("q[0]: %f, q[1*8*100*32-1]: %f, is_half: %d\n", (float)(qState[0]), (float)(qState[1*8*100*32-1]), IS_HALF_TYPE(qState[0]));
  //   printf("k[0]: %f, k[1*8*100*32-1]: %f, is_half: %d\n", (float)(kState[0]), (float)(kState[1*8*100*32-1]), IS_HALF_TYPE(kState[0]));
  //   printf("v[0]: %f, v[1*8*100*32-1]: %f, is_half: %d\n", (float)(vState[0]), (float)(vState[1*8*100*32-1]), IS_HALF_TYPE(vState[0]));
  // } 

  const float sqrt32 = 0.1767767f; // 1.0f / sqrt(32.0f);
  const uint32_t numberOfHead = 8;
  __local uint8_t slm0[32 * 16 * 2];
  __local uint8_t slm1[32 * 16 * 2];
  __local uint8_t slm2[256 * 16 * 2];
  __local uint8_t slm3[256 * 2];
  __local uint8_t slm4[256 * 2];
  __local uint8_t slm5[256 * 2];
  __global ushort* usVState = (__global ushort*)vState;

  const uint h = get_group_id(1);
  const uint headIdx = get_group_id(0);
  const uint localLinearId = get_sub_group_id();
  const uint lane = get_sub_group_local_id();
  const uint hh0 = localLinearId >> 1;
  const uint vv0 = localLinearId & 0x1;
  const uint hh1 = localLinearId & 0x1;
  const uint vv1 = localLinearId >> 1;
  const uint hh2 = localLinearId & 0x1f;
  const uint vv2 = localLinearId >> 5;
  const uint hh3 = localLinearId & 0xf;
  const uint vv3 = localLinearId >> 4;
  INPUT0_TYPE fp16Q[32];
  INPUT0_TYPE fp16Temp[16];
  ushort8* us8Temp = (ushort8*)fp16Temp;
  ushort* usTemp = (ushort*)fp16Temp;

  ushort fp16Temp2;
  fp16 f16Output[16] = { 0.0f, 0.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
  uint loopCount = (kvSeqLen + 0xf) >> 4;

  uint32_t offsetBase0 = h * 8192 + headIdx * 32 * queryLen + hh0 * 512 + lane * 32;
  uint32_t offsetBase1 = headIdx * kvSeqLen * 32 + vv1 * 32 + hh1 * 16;
  uint32_t offsetBase2 = headIdx * kvSeqLen * 32 + vv2 * 512 + hh2 + lane * 32;
  uint32_t offsetBase3 = h * 8192 + headIdx * 32 * queryLen + hh3 * 512 + vv3 * (32 >> 1) + lane * 32;
  uint32_t boundaryKv = kvSeqLen * 32 * (headIdx + 1);
  uint32_t boundaryQ = queryLen * 32 * (headIdx + 1);

  uint32_t slmOffset0 = localLinearId * 16 * 2;
  uint32_t slmOffset0Load = 0;
  uint32_t slmOffset1 = vv2 * 512 * 2 + hh2 * 16 * 2;
  uint32_t slmOffsetC0 = hh3 * 256 * 2;
  uint32_t slmOffsetC1 = vv3 * (32 >> 1) * 16 * 2;
  offsetBase0 = min(offsetBase0, boundaryQ);
  if (vv0 == 0) {
#pragma unroll
    for (int kk = 0; kk < 32; kk++) {
      fp16Q[kk] = qState[offsetBase0 + kk];
    }

    INPUT0_TYPE historicMaxInit[1] = { FP16_MIN * (INPUT0_TYPE)sqrt32 };
    ushort softMaxDividorInit = 0;

    ushort* usHistoricMaxInit = (ushort*)historicMaxInit;
    intel_sub_group_block_write_us((__local ushort*)(slm3 + hh0 * 16 * 2), usHistoricMaxInit[0]);
    intel_sub_group_block_write_us((__local ushort*)(slm5 + hh0 * 16 * 2), softMaxDividorInit);
  }

  for (int loopIdx = 0; loopIdx < loopCount; loopIdx++) {
    offsetBase1 = min(offsetBase1, boundaryKv);
    offsetBase2 = min(offsetBase2, boundaryKv);
    ushort fp16Temp1 = intel_sub_group_block_read_us((__global ushort*)(kState + offsetBase1));
    fp16Temp2 = usVState[offsetBase2];
    intel_sub_group_block_write_us((__local ushort*)(slm0 + slmOffset0), fp16Temp1);

    fp16 tempResult[16] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    fp16 maskTemp = 0;
    ushort8* us8TempResult = (ushort8*)tempResult;

    // if (hasMask) {
    //   maskTemp = mState[16 * loopIdx + lane];
    //   if (16 * loopIdx + lane >= kvSeqLen) {
    //     maskTemp = (fp16)1.0f;
    //   }
    // }

    if (sub_group_all(maskTemp != 0)) {
      offsetBase2 += 16 * 32;
      offsetBase1 += 16 * 32;
      continue;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (vv0 == 0) {
      us8Temp[0] = intel_sub_group_block_read_us8((__local ushort*)(slm0 + slmOffset0Load));
      us8Temp[1] = intel_sub_group_block_read_us8((__local ushort*)(slm0 + slmOffset0Load + 128 * 2));

#pragma unroll
      for (int l = 0; l < 8; l++) {
#pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          const uint kArrayIdx = 2 * l + ll;
#pragma unroll
          for (int lll = 0; lll < 16; lll++) {
            tempResult[l] = fma(fp16Q[ll * 16 + lll], sub_group_broadcast(fp16Temp[kArrayIdx], lll), tempResult[l]);
          }
        }
      }

      us8Temp[0] = intel_sub_group_block_read_us8((__local ushort*)(slm0 + slmOffset0Load + 256 * 2));
      us8Temp[1] = intel_sub_group_block_read_us8((__local ushort*)(slm0 + slmOffset0Load + (256 + 128) * 2));

#pragma unroll
      for (int l = 0; l < 8; l++) {
#pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          const uint kArrayIdx = 2 * l + ll;
#pragma unroll
          for (int lll = 0; lll < 16; lll++) {
            tempResult[l + 8] = fma(fp16Q[ll * 16 + lll], sub_group_broadcast(fp16Temp[kArrayIdx], lll), tempResult[l + 8]);
          }
        }
      }

      usTemp[0] = intel_sub_group_block_read_us((__local ushort*)(slm3 + hh0 * 16 * 2));
      usTemp[1] = intel_sub_group_block_read_us((__local ushort*)(slm5 + hh0 * 16 * 2));
      fp16Temp[2] = fp16Temp[0];
      if (hasMask) {
#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
          if (sub_group_broadcast(maskTemp, kk) != 0) {
            tempResult[kk] = FP16_MIN;
          }
        }
      } 
      else 
      {
#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
          if (kk + 16 * loopIdx >= kvSeqLen) {
            tempResult[kk] = FP16_MIN;
          }
        }
      }

#pragma unroll
      for (int kk = 0; kk < 16; kk++) {
        tempResult[kk] *= (fp16)sqrt32;
      }

#pragma unroll
      for (int kk = 0; kk < 16; kk++) {
        fp16Temp[2] = fmax(fp16Temp[2], tempResult[kk]);
      }

#pragma unroll
      for (int kk = 0; kk < 16; kk++) {
        tempResult[kk] -= fp16Temp[2];
        tempResult[kk] = native_exp(tempResult[kk]);
      }

      fp16Temp[3] = fp16Temp[0] - fp16Temp[2];
      fp16Temp[3] = native_exp(fp16Temp[3]);

      if (loopIdx != 0) {
        fp16Temp[1] *= fp16Temp[3];
      }

#pragma unroll
      for (int kk = 0; kk < 16; kk++) {
        fp16Temp[1] += tempResult[kk];
      }

      intel_sub_group_block_write_us((__local ushort*)(slm3 + hh0 * 16 * 2), usTemp[2]);
      intel_sub_group_block_write_us((__local ushort*)(slm4 + hh0 * 16 * 2), usTemp[3]);
      intel_sub_group_block_write_us((__local ushort*)(slm5 + hh0 * 16 * 2), usTemp[1]);
      intel_sub_group_block_write_us8((__local ushort*)(slm2 + hh0 * 16 * 16 * 2), us8TempResult[0]);
      intel_sub_group_block_write_us8((__local ushort*)(slm2 + hh0 * 16 * 16 * 2 + 128 * 2), us8TempResult[1]);
    }

    if (lane + 16 * loopIdx >= kvSeqLen) {
      fp16Temp2 = 0;
    }
    intel_sub_group_block_write_us((__local ushort*)(slm1 + slmOffset1), fp16Temp2);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (loopIdx != 0) {
      usTemp[0] = intel_sub_group_block_read_us((__local ushort*)(slm4 + hh3 * 16 * 2));
#pragma unroll
      for (int kk = 0; kk < 16; kk++) {
        f16Output[kk] *= fp16Temp[0];
      }
    }

    us8TempResult[0] = intel_sub_group_block_read_us8((__local ushort*)(slm2 + slmOffsetC0));
    us8TempResult[1] = intel_sub_group_block_read_us8((__local ushort*)(slm2 + slmOffsetC0 + 128 * 2));

    us8Temp[0] = intel_sub_group_block_read_us8((__local ushort*)(slm1 + slmOffsetC1));
    us8Temp[1] = intel_sub_group_block_read_us8((__local ushort*)(slm1 + slmOffsetC1 + 128 * 2));

#pragma unroll
    for (int l = 0; l < 16; l++) {
#pragma unroll
      for (int ll = 0; ll < 16; ll++) {
        f16Output[l] = fma(tempResult[ll], sub_group_broadcast(fp16Temp[l], ll), f16Output[l]);
      }
    }

    offsetBase2 += 512;
    offsetBase1 += 512;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  usTemp[0] = intel_sub_group_block_read_us((__local ushort*)(slm5 + hh3 * 16 * 2));
  fp16Temp[0] = (INPUT0_TYPE)1.0f / fp16Temp[0];
#pragma unroll
  for (int kk = 0; kk < 16; kk++) {
    f16Output[kk] = f16Output[kk] * fp16Temp[0];
  }

  if (h * 256 + hh3 * 16 < queryLen) {
#pragma unroll
    for (int kk = 0; kk < 16; kk++) {
      if (lane + h * 256 + hh3 * 16 < queryLen) {
        qkvOut[offsetBase3 + kk] = f16Output[kk];
      }
    }
  }
}