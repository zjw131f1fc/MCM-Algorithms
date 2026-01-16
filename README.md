# MCM-Algorithms

美赛常用算法代码库

## 评价与决策

### topsis
```python
from evaluation.topsis import topsis
scores, ranks = topsis(data, weights=None, criteria_types=None, normalize='vector')
```
- `data`: 决策矩阵 (m方案 × n指标)
- `weights`: 权重向量，None则等权
- `criteria_types`: 1正向/-1负向，None全正向
- `normalize`: 'vector' | 'minmax'
- 返回: (scores, ranks) 得分和排名
