import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import os
from pyoperon.sklearn import SymbolicRegressor
from pyoperon import R2, MSE, InfixFormatter, FitLeastSquares, Interpreter
import sympy as sp
from sympy import parse_expr
import matplotlib.pyplot as plt
from copy import deepcopy
from sympy.utilities.lambdify import lambdify
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_expression(expr_real, local_X, y_true):
    n_features = local_X.shape[1]
    x_syms = sp.symbols([f"x{i}" for i in range(n_features)])
    f = lambdify(x_syms, expr_real, "numpy")

    y_pred = f(*[local_X[:, i] for i in range(n_features)])
    y_pred = np.array(y_pred)

    # === 修复：如果公式是常数，扩展成和 y_true 一样长 ===
    if y_pred.shape == ():  # 标量
        y_pred = np.full_like(y_true, float(y_pred))
    elif y_pred.shape[0] == 1 and y_true.shape[0] > 1:
        y_pred = np.full_like(y_true, float(y_pred[0]))

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, r2

#通用文件读取器
def load_data_universal(file_path,header=0):
    """
    通用读取函数，支持 .xlsx 和 .csv 文件。

        - 如果您的文件有标题行(例如 '自变量1', '因变量' 等)，请保持 header=0
        - 如果您的文件没有标题行，第一行就是数据，请使用 header=None

    参数:
        file_path (str): 文件路径

    返回:
        pd.DataFrame: 读取后的 DataFrame 数据
    """

    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 错误：文件未找到 -> {file_path}")
        return None

    # 2. 获取文件后缀名
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    df_local = None

    try:
        # 3. 根据后缀名读取
        if file_extension in ['.xlsx', '.xls']:
            print(f"📖 正在读取 Excel 文件: {file_path} ...")
            df_local = pd.read_excel(file_path, header=header)

        elif file_extension == '.csv':
            print(f"📖 正在读取 CSV 文件: {file_path} ...")
            # CSV 编码处理机制：先试 utf-8，失败则试 gb18030 (兼容 gbk 和 gb2312)
            try:
                df_local = pd.read_csv(file_path, encoding='utf-8', header=header)
            except UnicodeDecodeError:
                print("⚠️ UTF-8 解码失败，尝试使用 GB18030 解码...")
                df_local = pd.read_csv(file_path, encoding='gb18030', header=header)

        else:
            print(f"❌ 不支持的文件格式: {file_extension}")
            return None

        # 4. 输出 df 内容概览
        if df_local is not None:
            print("\n✅ 读取成功！数据概览：")
            print("【前 5 行数据】:")
            print(df_local.head())

        return df_local

    except Exception as e:
        print(f"❌ 读取文件时发生未知错误: {e}")
        return None


# 定义特征列和目标列 (请确保这些列名在你的训练/测试数据的CSV表头中存在)
feature_cols = ["Mach", "sin(alpha)"]
target_col = "CN"

# 基础路径设置
BASE_DIR = r"/home/dministrator/projects/symbolic-regression"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

file_name_base = f"{target_col}结果_operon_{feature_cols[1]}"  # 例如: CM结果_有理学习_Alpha

OUTPUT_CSV = os.path.join(BASE_DIR, f"{file_name_base}_模型精调.xlsx")

# 数据集文件路径
file_path_train = os.path.join(DATASET_DIR, "LTV_HF_200mcv.csv")

file_path_test_interp = os.path.join(DATASET_DIR, "LTV_HF_77grid.csv")

#文件读取

df_train = load_data_universal(file_path_train)

df_test = load_data_universal(file_path_test_interp)


# --------------------------------------------------------------------------
# 6. 准备自变量(X)和因变量(y)
#    .iloc[:, :4] 表示选取所有行，以及从第0列到第3列（共4列）作为自变量 X
#    .iloc[:, 4]  表示选取所有行，以及第4列（即第五列）作为因变量 y

# 提取训练集
X_train = df_train[feature_cols].values
y_train = df_train[target_col].values

# 提取测试集
X_test = df_test[feature_cols].values
y_test = df_test[target_col].values


variable_names = ['x0', 'x1']
print(variable_names)

reg = SymbolicRegressor(
        # 1. 扩充符号集 (如果确定不需要超越函数，可删去 exp,log,sin,cos)
        # 注意：建议用 div 或 protected_div 代替 aq，除非你明确喜欢 aq 的特性
        allowed_symbols="add,sub,mul,div,constant,variable", 
        
        # 2. 种群与进化
        population_size=1000,       # ← 增加：数据少，算得快，人多力量大
        generations=1000,           # 保持
        max_evaluations=5000000,    # 配合 population_size 增加
        
        # 3. 核心修正：优化器设置
        optimizer='lm',
        optimizer_iterations=8,    # ← 关键：从 3 增加到 20+，让系数算得更准
        local_search_probability=0.5, # ← 调整：不一定每个都优化，但优化的那个要算准
        lamarckian_probability=0.5,   # ← 同上，把优化后的系数写回基因
        
        # 4. 杂项
        brood_size=10,
        comparison_factor=0,
        crossover_internal_probability=0.9,
        crossover_probability=1.0,
        mutation_probability=0.25,  # ← 稍微提高变异率，防止早熟收敛
        
        # 5. 初始化与限制
        initialization_max_depth=5,
        initialization_max_length=10,
        initialization_method="btc",
        max_depth=10,
        max_length=50,
        
        # 6. 选择策略
        female_selector="tournament",
        male_selector="tournament",
        tournament_size=5,          # ← 稍微增加选择压力
        
        # 7. 线性缩放 (强烈建议开启，除非数据已归一化)
        add_model_intercept_term=True,  # ← 建议开启
        add_model_scale_term=True,      # ← 建议开启
        
        epsilon=1e-05,
        irregularity_bias=0.0,
        max_selection_pressure=100,
        model_selection_criterion="minimum_description_length",
        n_threads=32,
        objectives=['r2', 'length'],
        offspring_generator="os",
        pool_size=1000,
        random_state=None,
        reinserter="keep-best",
        max_time=1800
)

print("Operon开始训练...")
reg.fit(X_train, y_train)
print("训练完成。")
# ######################################################################
# # 续写部分：评估、计算指标并保存到 CSV (已修复)
# ######################################################################

print("开始评估帕累托前沿上的所有模型...")

# 1. 准备一个列表来存储所有结果
results = []

# 2. 遍历帕累托前沿
res = [(s['objective_values'], s['tree'], s['minimum_description_length']) for s in reg.pareto_front_]

print(f"在帕累托前沿上找到 {len(res)} 个模型。正在逐个计算详细指标...")

for obj, expr, mdl in res:
    try:
        # obj[0] = Operon 的 'r2' 目标 (通常是 -R^2)
        # obj[1] = Operon 的 'length' 目标 (复杂度)
        
        formula_str = reg.get_model_string(expr, 10, variable_names)
        formula_str = formula_str.replace("^", "**")
        
        #formula_str = reg.get_model_string(expr, 10)
        print(formula_str)
        complexity = obj[1] # Operon 评估的 'length'
        r2 = -obj[0] # 使用 scikit-learn 重新计算 R2
        
        mse_train, rmse_train, r2_train = evaluate_expression(formula_str, X_train, y_train)
        mse_test, rmse_test, r2_test = evaluate_expression(formula_str, X_test, y_test)
        #mse_extrap, rmse_extrap, r2_extrap = evaluate_expression(formula_str, X_extrap, y_extrap)


        # 7. 按照您要求的格式存入列表
        results.append({
            "Equation": formula_str,
            "Complexity": complexity,
            "R2 (Scikit-Learn)": r2,      # 真实 R2 分数
            "MSE (train)": mse_train,
            "MSE (test)": mse_test,
            #"MSE (extrap)": mse_extrap,
            "RMSE (train)": rmse_train,
            "RMSE (test)": rmse_test,
            #"RMSE (extrap)": rmse_extrap,
            "R2 (train)": r2_train,
            "R2 (test)": r2_test,
            #"R2 (extrap)": r2_extrap,
        })
        
        # 打印即时结果
        print(f"  - 复杂度: {complexity}, R2: {r2:.6f}, 公式: {formula_str[:100]}...") # 截断公式以防过长

    except Exception as e:
        # 捕获可能的其他错误（例如预测失败）
        print(f"评估公式时出错: {e}")
        # 打印失败的公式以供调试
        try:
            failed_formula = reg.get_model_string(expr, 10, variable_names)
            print(f"  - 失败的公式: {failed_formula}")
        except:
            print("  - 无法获取失败的公式字符串。")


# 8. 循环结束后，将结果列表转换为 DataFrame
results_df = pd.DataFrame(results)

# 9. 检查 DataFrame 是否为空
if results_df.empty:
    print("\n错误：评估了所有模型，但 'results' 列表仍然是空的。")
    print("请检查上面的错误日志。")
else:
    # # 10. 对 DataFrame 按 R2 降序排序，查看最佳结果
    # results_df = results_df.sort_values(by="R2 (Scikit-Learn)", ascending=False)

    # 11. 保存到 CSV 文件
    output_filename = f"{target_col}结果_operon_{feature_cols[1]}_2.csv"
    try:
        # 使用 utf-8-sig 编码以确保 Excel 能正确打开中文
        results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n成功将 {len(results_df)} 条结果写入到 {output_filename}")
        print("\n最终结果预览 (R2 最佳的前 5 个模型):")
        print(results_df.head())
    except Exception as e:
        print(f"\n保存到CSV文件时出错: {e}")





