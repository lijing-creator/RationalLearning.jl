from pysr import PySRRegressor, TensorBoardLoggerSpec
import sympy as sp
import pandas as pd
from sympy.utilities.lambdify import lambdify
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os

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

def evaluate_all_expressions_not_scaled(local_model, local_x_train, local_y_train, local_x_test, local_y_test):
    results = []
    equations = local_model.equations_  # DataFrame: 包含所有 Pareto 表达式

    for idx, row in equations.iterrows():
        expr_real = row["sympy_format"]  # 符号化表达式
        # expr_real = denormalize_formula_minmax(expr_scaled, local_scaler_x, local_scaler_y)

        mse_train, rmse_train, r2_train = evaluate_expression(expr_real, local_x_train, local_y_train)
        mse_test, rmse_test, r2_test = evaluate_expression(expr_real, local_x_test, local_y_test)
        #mse_extrap, rmse_extrap, r2_extrap = evaluate_expression(expr_real, local_x_extrap, local_y_extrap)

        results.append({
            "Equation": str(expr_real),
            "Complexity": row["complexity"],
            "Loss ": row["loss"],
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

    return pd.DataFrame(results)

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
BASE_DIR = r"D:\BaiduSyncdisk\飞行器数据集-AIAA case2"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

file_name_base = f"{target_col}结果_pysr_{feature_cols[1]}_1"  # 例如: CM结果_有理学习_Alpha

OUTPUT_CSV = os.path.join(BASE_DIR, f"{file_name_base}.xlsx")

# 数据集文件路径
file_path_train = os.path.join(DATASET_DIR, "LTV_HF_200mcv.csv")

file_path_test_interp = os.path.join(DATASET_DIR, "LTV_HF_77grid.csv")

# file_path_test_extrap_Ma = "D:\BaiduSyncdisk\飞行器数据集-法向力增量\dataset\DCN_d2_extrap.xlsx"

#文件读取

df_train = load_data_universal(file_path_train)

df_test = load_data_universal(file_path_test_interp)

# df_extrap = load_data_universal(file_path_test_extrap_Ma)

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

# X_extrap = df_extrap.iloc[:, :4].values
# y_extrap = df_extrap.iloc[:, 4].values


# Create a logger that writes to "logs/run*":
logger_spec = TensorBoardLoggerSpec(
    log_dir="logs/run",
    log_interval=10,  # Log every 10 iterations
)

model = PySRRegressor(

        procs=20,  # 使用的CPU核心数，可以根据您的电脑配置修改
        populations=40,  # 种群数量，数量越多，搜索越广泛
        niterations=500,  # 进化迭代次数。可以先设小一点（如20）来快速测试，再增加以获得更好结果
        batching=True,
        # 定义您认为可能出现在方程中的运算符
        binary_operators=["+", "-", "*", "/"],

        unary_operators=[ "sqrt","square"],

        # 更多高级配置
        model_selection="best",  # 自动从最终结果中选择最佳方程
        #elementwise_loss="(x, y) -> (x - y)^2",  # 损失函数，默认为均方误差
        output_directory="pysr_output",
        #开启加速
        turbo=True,
        logger_spec=logger_spec,
     )

print("\n开始使用PySR进行符号回归计算，这可能需要较长时间，请耐心等待...")

# 运行核心计算步骤
model.fit(X_train, y_train)

df_results = evaluate_all_expressions_not_scaled(model, X_train, y_train, X_test, y_test)

print(df_results.head())
df_results.to_excel(OUTPUT_CSV, index=True)





