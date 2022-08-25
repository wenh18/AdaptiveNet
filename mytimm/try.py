from models.mobilenetv3 import MobileNetV3, tf_mobilenetv3_large_100
from models.efficientnet_builder import decode_arch_def
# a =
arch_def = [
    # stage 0, 112x112 in
    ['ds_r1_k3_s1_e1_c16_nre'],  # relu
    # stage 1, 112x112 in
    ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
    # stage 2, 56x56 in
    ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
    # stage 3, 28x28 in
    ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
    # stage 4, 14x14in
    ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
    # stage 5, 14x14in
    ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
    # stage 6, 7x7 in
    ['cn_r1_k1_s1_c960'],  # hard-swish
]
block_args = decode_arch_def(arch_def)
print(block_args)
# model = tf_mobilenetv3_large_100()
# print(model)
# print("[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[")
# print(model.blocks)