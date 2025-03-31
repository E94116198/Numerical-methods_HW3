import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import minimize_scalar, root_scalar

# 原始數據
T = np.array([0, 3, 5, 8, 13])  # 時間 (s)
D = np.array([0, 200, 375, 620, 990])  # 距離 (ft)
V = np.array([75, 77, 80, 74, 72])  # 速度 (ft/s)

# 建立 Hermite 插值函數
hermite_poly = CubicHermiteSpline(T, D, V)

# (a) 計算 t = 10 時的距離和速度
t_pred = 10
d_pred = hermite_poly(t_pred)  # 位置
v_pred = hermite_poly.derivative()(t_pred)  # 速度

print("===問題(a)===")
print(f"At t = {t_pred}s: Predicted Position = {d_pred:.2f} ft, Speed = {v_pred:.2f} ft/s")

# (b) 檢查何時速度超過 55 mi/h (80.67 ft/s)
def speed_exceeds_threshold(t):
    return hermite_poly.derivative()(t) - 80.67

# 設定較小的區間以確保求解成功
try:
    exceed_time = root_scalar(speed_exceeds_threshold, bracket=[3, 3.5], method='brentq').root
    print("\n===問題(b)===")
    print(f"The car first exceeds 55 mi/h at t = {exceed_time:.2f} s")
except ValueError:
    print("\n===問題(b)===")
    print("Error: Unable to find the time when the speed exceeds 55 mi/h")

# (c) 預測最大速度
result = minimize_scalar(lambda t: -hermite_poly.derivative()(t), bounds=(T[0], T[-1]), method='bounded')
max_speed = -result.fun
max_speed_time = result.x
print("\n===問題(c)===")
print(f"Predicted Maximum Speed = {max_speed:.2f} ft/s at t = {max_speed_time:.2f} s")
