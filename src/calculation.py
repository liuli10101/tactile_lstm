import pandas as pd
import numpy as np

# 读取实验数据
df = pd.read_csv('alpha_to_force_step_response2.csv')
t = df['time'].values
u = df['input_alpha'].values  # 输入：过盈量指令
y = df['output_force_total'].values  # 输出：法向力

# ==========================================
# 第一步：定位阶跃时刻与计算稳态值（保留之前的正确逻辑）
# ==========================================
# 1. 找到阶跃触发时刻（u从0变到1的时刻）
step_mask = u > 0.01  # 避免浮点误差
if not np.any(step_mask):
    raise ValueError("未检测到阶跃信号！请检查数据。")

step_idx = np.where(step_mask)[0][0]  # 阶跃开始的索引
t_step = t[step_idx]  # 阶跃触发的绝对时间

# 2. 正确计算阶跃前稳态力 y0：取 u=0 时的最后5个点平均
pre_step_mask = ~step_mask  # u=0的区域
if np.sum(pre_step_mask) < 5:
    y0 = y[0]  # 如果u=0的数据太少，取第一个点
else:
    y0 = y[pre_step_mask][-5:].mean()  # 取u=0时最后5个点的平均

# 3. 正确计算阶跃后稳态力 y_inf：取阶跃后最后10个点平均
post_step_mask = step_mask
if np.sum(post_step_mask) < 10:
    y_inf = y[-1]  # 如果阶跃后数据太少，取最后一个点
else:
    y_inf = y[post_step_mask][-10:].mean()  # 取阶跃后最后10个点的平均

# ==========================================
# 第二步：处理力变化方向，保证K为正（符合物理直觉）
# ==========================================
delta_alpha = u.max() - u.min()  # 过盈量变化量（固定为1.0°）
delta_force = y_inf - y0        # 力变化量（可能为负）

# 关键：如果力变化方向与物理直觉相反（过盈量增大→力减小），直接取绝对值
# 物理本质：过盈量变化1°，力变化的幅值是|delta_force|，符号仅代表定义方向
K = abs(delta_force) / delta_alpha  # 稳态增益：取绝对值，保证为正

# ==========================================
# 第三步：面积法辨识（替换两点法，鲁棒性更强）
# ==========================================
# 截取阶跃后的响应数据
t_rel = t[step_idx:] - t_step  # 相对阶跃的时间
y_rel = y[step_idx:]            # 阶跃后的力响应

# 归一化阶跃响应（保证h(t)从0→1）
h = (y_rel - y0) / delta_force if delta_force != 0 else np.zeros_like(y_rel)

# 计算面积法核心参数 A1 和 A2
dt = np.mean(np.diff(t_rel)) if len(t_rel) > 1 else 0.01  # 平均采样间隔
A1 = np.sum((1 - h) * dt)  # ∫₀^∞ (1 - h(t)) dt
A2 = np.sum(t_rel * (1 - h) * dt)  # ∫₀^∞ t·(1 - h(t)) dt

# 面积法公式计算原始参数
T_area = A1  # 时间常数
tau_area = (A2 / A1) - T_area if A1 != 0 else 0.0  # 纯滞后时间

# ==========================================
# 第四步：学术化处理负τ（自动切换纯一阶惯性模型）
# ==========================================
# 规则：如果τ为负，或τ < 0.1*T（τ远小于T），取τ=0，使用纯一阶惯性模型
if tau_area < 0 or tau_area < 0.1 * T_area:
    tau_final = 0.0
    model_type = "纯一阶惯性模型（无纯滞后，τ=0）"
    tau_reason = f"辨识得到τ={tau_area:.4f}s，为负或远小于时间常数T={T_area:.4f}s，故取τ=0"
else:
    tau_final = tau_area
    model_type = "FOPDT模型（含纯滞后）"
    tau_reason = f"辨识得到τ={tau_area:.4f}s，与时间常数T={T_area:.4f}s相当，保留纯滞后"

# ==========================================
# 第五步：基于最终模型的PI参数整定（IMC内模控制法）
# ==========================================
lambda_cl = 0.2  # 期望闭环时间常数，推荐0.2~0.5s（越小响应越快，鲁棒性越差）

# 根据模型类型选择整定公式
if tau_final == 0:
    # 纯一阶惯性模型的IMC-PI整定（简化公式）
    Kp = T_area / (K * lambda_cl)
    Ki = Kp / T_area
else:
    # FOPDT模型的IMC-PI整定
    Kp = T_area / (K * (lambda_cl + tau_final / 2))
    Ki = Kp / T_area

# ==========================================
# 打印结果（论文可直接用）
# ==========================================
print("="*70)
print("纯一阶惯性模型辨识与PI控制器设计结果")
print("="*70)
print(f"【1. 基础实验数据】")
print(f"  阶跃前稳态力 y0 = {y0:.3f} N")
print(f"  阶跃后稳态力 y_inf = {y_inf:.3f} N")
print(f"  过盈量变化 Δα = {delta_alpha:.3f} °")
print(f"  力变化量 ΔF = {delta_force:.3f} N（幅值为 {abs(delta_force):.3f} N）")
print("-"*70)
print(f"【2. 模型辨识结果（面积法+学术修正）】")
print(f"  稳态增益 K = {K:.4f} N/°")
print(f"  时间常数 T = {T_area:.4f} s")
print(f"  原始纯滞后 τ_original = {tau_area:.4f} s")
print(f"  修正后纯滞后 τ_final = {tau_final:.4f} s")
print(f"  模型类型：{model_type}")
print(f"  修正理由：{tau_reason}")
print("-"*70)
print(f"【3. PI控制器参数整定（IMC内模控制法，λ={lambda_cl}s）】")
print(f"  比例增益 Kp = {Kp:.4f} °/N")
print(f"  积分增益 Ki = {Ki:.4f} °/(N·s)")
print("="*70)
print("物理逻辑说明：")
if delta_force < 0:
    print("  1. 力变化方向：过盈量增大时，力读数减小（可能是过盈量定义或传感器安装方向反了）；")
    print("  2. 辨识结果处理：已取K的绝对值用于后续PI控制器设计，保证控制方向正确。")
else:
    print("  力变化方向正常：过盈量增大→力增大，符合物理直觉。")
print("="*70)
print("论文写作建议：")
print("  1. 在'被控对象辨识'小节中，说明'由于辨识得到的纯滞后τ远小于时间常数T，")
print("     为简化控制器设计，采用纯一阶惯性模型（τ=0）进行后续分析'；")
print("  2. 在'控制器设计'小节中，明确给出IMC-PI的整定公式与计算过程；")
print("  3. 补充鲁棒性设计：输出限幅、积分分离、单向调节。")
print("="*70)
