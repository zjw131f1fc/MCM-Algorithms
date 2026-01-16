"""运行所有测试脚本"""
import subprocess
import sys

test_scripts = [
    'test_evaluation.py',
    'test_prediction.py',
    'test_optimization.py',
    'test_graph.py',
    'test_mechanism.py',
    'test_ml.py'
]

print("=" * 60)
print("开始运行所有测试")
print("=" * 60)

for script in test_scripts:
    print(f"\n{'=' * 60}")
    print(f"运行 {script}")
    print("=" * 60)
    try:
        result = subprocess.run([sys.executable, script],
                              capture_output=False,
                              text=True,
                              check=True)
    except subprocess.CalledProcessError as e:
        print(f"错误: {script} 运行失败")
        print(f"返回码: {e.returncode}")
    except Exception as e:
        print(f"错误: 无法运行 {script}")
        print(f"异常: {e}")

print("\n" + "=" * 60)
print("所有测试完成")
print("=" * 60)
