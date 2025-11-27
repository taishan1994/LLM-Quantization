import torch
from scipy.linalg import hadamard
import numpy as np

def apply_hadamard(x, group_size=16):
    """对最后一维按 group_size 应用 Hadamard 旋转"""
    original_shape = x.shape
    assert x.shape[-1] % group_size == 0
    H = torch.tensor(hadamard(group_size) / np.sqrt(group_size), dtype=x.dtype, device=x.device)
    x = x.view(-1, group_size)
    x = x @ H  # 正交变换，保持能量不变
    return x.view(original_shape)

def cal_s_fp(x, Qmax, epsilon):
    dim = [-1]
    # 先计算每一组内的最大值
    xmax = x.abs().amax(dim=dim, keepdim=True)  # shape: broadcastable to x along last dim
    # 避免除以0
    mask = (xmax == 0)
    xmax_safe = xmax + epsilon * mask
    s = xmax_safe / Qmax
    return s


e_bit = 2
m_bit = 1
bias = 2 ** (e_bit - 1) - 1  if e_bit > 0 else 1
if e_bit > 0:
    Elow = -bias  # minimum exponent (after considering bias)
    Ehigh = 2 ** (e_bit) - 1 - bias  # maximum exponent (after considering bias)
else:
    Elow = 0
    Ehigh = 0

def nvfp4_quantize(x, group_size=32, epsilon=1e-25):
    """
    将张量量化为NVFP4格式
    :param tensor: 输入张量
    :return: 量化后的张量
    """
    Qmax = 6
    Qmin = -Qmax

    original_shape = x.shape
    new_shape = x.shape[:-1] + (-1, group_size)
    x = x.reshape(new_shape)

    s = cal_s_fp(x, Qmax, epsilon).to(x.dtype)

    # 进一步对scale再进行缩放
     # E4M3 quant for scale + bf16 per-tensor scale, follow NVFP4
    s_of_s = s.abs().max()/ 448
    quant_s = (s/s_of_s).to(torch.float8_e4m3fn)
    s = s_of_s * quant_s.bfloat16()


    s = s.clamp(1e-25, 1e25) 
    x = x / s
    sign, x_abs = x.sign(), x.abs()
    expo = torch.floor(torch.log2(x.abs() + epsilon))
    expo = torch.clamp(expo, min=Elow, max=Ehigh)

    is_subnormal = expo <= Elow
    # normalized number
    mantissa_norm = x_abs / (2 ** expo) - 1  # in [0, 1)
    scale_m = 2 ** m_bit
    m_frac_int = torch.round(mantissa_norm * scale_m)  # in {0, 1, ..., 2^m}
    carry = (m_frac_int >= scale_m)  # == 2^m 
    m_frac_int = torch.where(carry, torch.zeros_like(m_frac_int), m_frac_int)
    mantissa_norm_q = m_frac_int / scale_m
    expo_adj = expo + carry.to(expo.dtype)
    # ============================================================
    # subnormalized number
    expo_sub = 1 - bias
    mantissa_sub = x_abs / (2 ** expo_sub)  # in [0, 1)
    m_sub_int = torch.round(mantissa_sub * scale_m)  # in {0, ..., 2^m}
    mantissa_sub_q = m_sub_int / scale_m
    # compose
    y = torch.where(
        is_subnormal,
        sign * (2 ** expo_sub) * mantissa_sub_q,
        sign * (2 ** expo_adj) * (1 + mantissa_norm_q)
    )
    y = y.clamp(Qmin, Qmax) * s


    if group_size > 0:
        y = y.reshape(original_shape)

    return y


def run_quantization_stats(num_trials=1000):
    """运行多次试验统计量化误差"""
    group_size = 16
    original_errors = []
    hadamard_errors = []
    improvements = []
    
    for i in range(num_trials):
        # 生成随机数据
        x = torch.randn(2, 64)
        
        # 注入异常值
        x_with_outlier = x.clone()
        x_with_outlier[0, -1] = 8.0
        
        # 应用 Hadamard 变换
        x_had = apply_hadamard(x_with_outlier, group_size=group_size)
        
        # 计算量化误差
        original_error = torch.mean((x_with_outlier - nvfp4_quantize(x_with_outlier, group_size=group_size))**2).item()
        hadamard_error = torch.mean((x_had - nvfp4_quantize(x_had, group_size=group_size))**2).item()
        
        original_errors.append(original_error)
        hadamard_errors.append(hadamard_error)
        
        # 计算单次试验的改进比例
        improvement = (original_error - hadamard_error) / original_error * 100
        improvements.append(improvement)
        
        if (i + 1) % 100 == 0:
            print(f"已完成 {i + 1}/{num_trials} 次试验")
    
    # 计算统计信息
    original_mean = np.mean(original_errors)
    original_std = np.std(original_errors)
    hadamard_mean = np.mean(hadamard_errors)
    hadamard_std = np.std(hadamard_errors)
    
    # 改进比例的统计
    improvement_mean = np.mean(improvements)
    improvement_std = np.std(improvements)
    positive_improvements = sum(1 for imp in improvements if imp > 0)
    negative_improvements = sum(1 for imp in improvements if imp < 0)
    
    print("\n===== 量化误差统计结果 =====")
    print(f"试验次数: {num_trials}")
    print(f"原始量化误差 - 均值: {original_mean:.6f}, 标准差: {original_std:.6f}")
    print(f"Hadamard 后误差 - 均值: {hadamard_mean:.6f}, 标准差: {hadamard_std:.6f}")
    print(f"平均误差变化: {improvement_mean:.2f}% ± {improvement_std:.2f}%")
    print(f"误差降低的试验次数: {positive_improvements}/{num_trials} ({positive_improvements/num_trials*100:.1f}%)")
    print(f"误差增加的试验次数: {negative_improvements}/{num_trials} ({negative_improvements/num_trials*100:.1f}%)")
    print(f"总体误差降低比例: {((original_mean - hadamard_mean) / original_mean * 100):.2f}%")
    print("=============================")
    
    return {
        'original_mean': original_mean,
        'original_std': original_std,
        'hadamard_mean': hadamard_mean,
        'hadamard_std': hadamard_std,
        'improvement_mean': improvement_mean,
        'improvement_std': improvement_std,
        'positive_improvements': positive_improvements,
        'negative_improvements': negative_improvements,
        'overall_improvement_percentage': (original_mean - hadamard_mean) / original_mean * 100
    }


def test_different_outlier_values(outlier_values=[1.0, 2.0, 4.0, 8.0, 16.0], num_trials=200):
    """测试不同异常值大小对量化误差的影响"""
    group_size = 16
    
    print(f"\n===== 不同异常值大小的影响测试 =====")
    print(f"每组试验次数: {num_trials}")
    print("=" * 40)
    
    results = {}
    
    for outlier_value in outlier_values:
        original_errors = []
        hadamard_errors = []
        
        for i in range(num_trials):
            # 生成随机数据
            x = torch.randn(2, 64)
            
            # 注入不同大小的异常值
            x_with_outlier = x.clone()
            x_with_outlier[0, -1] = outlier_value
            
            # 应用 Hadamard 变换
            x_had = apply_hadamard(x_with_outlier, group_size=group_size)
            
            # 计算量化误差
            original_error = torch.mean((x_with_outlier - nvfp4_quantize(x_with_outlier, group_size=group_size))**2).item()
            hadamard_error = torch.mean((x_had - nvfp4_quantize(x_had, group_size=group_size))**2).item()
            
            original_errors.append(original_error)
            hadamard_errors.append(hadamard_error)
        
        # 计算统计信息
        original_mean = np.mean(original_errors)
        hadamard_mean = np.mean(hadamard_errors)
        improvement_percentage = (original_mean - hadamard_mean) / original_mean * 100
        
        results[outlier_value] = {
            'original_mean': original_mean,
            'hadamard_mean': hadamard_mean,
            'improvement_percentage': improvement_percentage
        }
        
        print(f"异常值大小: {outlier_value}")
        print(f"  原始量化误差: {original_mean:.6f}")
        print(f"  Hadamard后误差: {hadamard_mean:.6f}")
        print(f"  误差变化: {improvement_percentage:.2f}%")
        print()
    
    return results


if __name__ == "__main__":
    # 单次测试
    x = torch.randn(2, 64)

    x_with_outlier = x.clone()
    x_with_outlier[0, -1] = 8.0  # 注入 outlier

    group_size = 16
    x_had = apply_hadamard(x_with_outlier, group_size=group_size)

    print("单次测试结果:")
    print("原始量化误差:", torch.mean((x_with_outlier - nvfp4_quantize(x_with_outlier, group_size=group_size))**2))
    print("Hadamard 后误差:", torch.mean((x_had - nvfp4_quantize(x_had, group_size=group_size))**2))
    
    # 运行1000次统计
    print("\n开始运行1000次统计...")
    run_quantization_stats(1000)
    
    # 测试不同异常值大小的影响
    print("\n开始测试不同异常值大小的影响...")
    test_different_outlier_values()
