import pandas as pd
import numpy as np

def calculate_accuracy_by_subject(file_path: str):
    """
    读取指定的Excel文件，按'subject'列分组，计算每个学科的预测准确率。

    Args:
        file_path (str): Excel文件的路径。

    Returns:
        None: 直接打印结果到控制台。
    """
    try:
        # 1. 使用 pandas 读取 Excel 文件
        df = pd.read_excel(file_path)
        print(f"成功读取文件: {file_path}")
        print("-" * 30)
    except FileNotFoundError:
        print(f"错误：文件未找到，请检查路径 '{file_path}' 是否正确。")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    # 2. 数据清洗与标准化 (非常重要的一步)
    # 确保 'answer' 和 'prediction' 列存在
    required_columns = ['answer', 'prediction', 'subject']
    if not all(col in df.columns for col in required_columns):
        print(f"错误：文件中缺少必要的列。需要 {required_columns}，但只找到 {list(df.columns)}")
        return
        
    # 将 answer 和 prediction 列转换为字符串，去除首尾空格，并转为大写，以便进行不区分大小写的比较
    df['answer_clean'] = df['answer'].astype(str).str.strip().str.upper()
    df['prediction_clean'] = df['prediction'].astype(str).str.strip().str.upper()

    # 3. 计算每一行是否预测正确
    df['is_correct'] = (df['answer_clean'] == df['prediction_clean'])

    # 4. 按 'subject' 分组并计算指标
    # 使用 groupby 和 agg 一次性计算每个组的 '总数(count)' 和 '正确数(sum)'
    # 因为 is_correct 列中 True=1, False=0，所以求和(sum)即为正确数
    subject_results = df.groupby('subject')['is_correct'].agg(
        total='count',
        correct='sum'
    ).reset_index()

    # 5. 计算准确率
    # 避免除以零的错误
    subject_results['accuracy'] = (subject_results['correct'] / subject_results['total']) * 100

    # 6. 计算总体准确率
    overall_correct = df['is_correct'].sum()
    overall_total = len(df)
    overall_accuracy = (overall_correct / overall_total) * 100 if overall_total > 0 else 0

    # 7. 打印结果
    print("各学科预测准确率:")
    # 格式化输出，使表格更美观
    subject_results['accuracy'] = subject_results['accuracy'].map('{:.2f}%'.format)
    print(subject_results.to_string(index=False))
    
    print("\n" + "=" * 30)
    print(f"总体情况:")
    print(f"  总题目数: {overall_total}")
    print(f"  总正确数: {overall_correct}")
    print(f"  总准确率: {overall_accuracy:.2f}%")
    print("=" * 30)


if __name__ == '__main__':
    # --- 请在这里修改为你的 Excel 文件路径 ---
    # 例如: 'vlm_eval_results/Qwen-VL-Chat_eval_on_MMBench_DEV_EN.xlsx'
    file_path = './vlm0.4_result.xlsx'
    
    calculate_accuracy_by_subject(file_path)