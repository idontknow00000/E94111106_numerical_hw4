#第1題    
import numpy as np
''' 定義被積函數'''  
def f(x):    
    return np.exp(x) * np.sin(4 * x)    

''' 給定參數'''  
a = 1.0  # 積分下限    
b = 2.0  # 積分上限    
h = 0.1  # 步長    
n = int((b - a) / h)  # 子區間數量，n = 10    

''' 積分點'''  
x = np.linspace(a, b, n + 1)  # x = [1.0, 1.1, ..., 2.0]    
f_values = f(x)  # 計算 f(x) 在每個點的值    

''' a. 複合梯形法則'''   
def composite_trapezoidal_rule(f_values, h, n):    
    result = (h / 2) * (f_values[0] + 2 * np.sum(f_values[1:n]) + f_values[n])    
    return result    

''' b. 複合辛普森法則'''  
def composite_simpson_rule(f_values, h, n):    
    if n % 2 != 0:    
        raise ValueError("n must be even for Simpson's rule")    
    result = (h / 3) * (f_values[0] + 4 * np.sum(f_values[1:n:2]) + 2 * np.sum(f_values[2:n-1:2]) + f_values[n])    
    return result    

''' c. 複合中點法則'''  
def composite_midpoint_rule(f, x, h, n):    
    midpoints = (x[:-1] + x[1:]) / 2  # 中點：(x_i + x_{i+1})/2    
    result = h * np.sum(f(midpoints))    
    return result    

''' 計算結果'''  
trapezoidal_result = composite_trapezoidal_rule(f_values, h, n)    
simpson_result = composite_simpson_rule(f_values, h, n)    
midpoint_result = composite_midpoint_rule(f, x, h, n)    

''' 輸出結果  '''
print(f"a. Composite Trapezoidal Rule: {trapezoidal_result:.6f}")    
print(f"b. Composite Simpson's Rule: {simpson_result:.6f}")    
print(f"c. Composite Midpoint Rule: {midpoint_result:.6f}")    

''' 為了驗證，計算真實值（可通過數值積分或解析解）'''  
from scipy.integrate import quad        
true_value, _ = quad(f, a, b)    
print(f"\nTrue value (for reference): {true_value:.6f}")    
#第2題    
import numpy as np  
from scipy.special import roots_legendre  
from scipy.integrate import quad  

''' 定義被積函數 '''  
def f(x):  
    return x**2 * np.log(x)  

''' 高斯求積法 '''  
def gaussian_quadrature(f, a, b, n):  
    ''' 獲取高斯-勒讓德節點和權重 '''  
    t, w = roots_legendre(n)  
    
    ''' 變量變換：t -> x '''  
    x = (b - a) / 2 * t + (a + b) / 2  
    ''' 積分公式 '''  
    result = (b - a) / 2 * np.sum(w * f(x))  
    return result  

''' 積分區間 '''  
a = 1.0  
b = 1.5  

''' n = 3 '''  
result_n3 = gaussian_quadrature(f, a, b, 3)  
print(f"Gaussian Quadrature (n=3): {result_n3:.6f}")  

''' n = 4 '''  
result_n4 = gaussian_quadrature(f, a, b, 4)  
print(f"Gaussian Quadrature (n=4): {result_n4:.6f}")  

''' 真實值 '''  
true_value, _ = quad(f, a, b)  
print(f"True value: {true_value:.6f}")  

''' 計算相對誤差 '''  
relative_error_n3 = abs(result_n3 - true_value) / abs(true_value)  
relative_error_n4 = abs(result_n4 - true_value) / abs(true_value)  

''' 轉換為百分比 '''  
relative_error_n3_percent = relative_error_n3 * 100  
relative_error_n4_percent = relative_error_n4 * 100  

print(f"Relative error (n=3, %): {relative_error_n3_percent:.6f}%")  
print(f"Relative error (n=4, %): {relative_error_n4_percent:.6f}%")  
