import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------
# 1. 한글 폰트 설정 (사용하시는 OS에 맞게 설정해주세요)
# -----------------------------------------------------------
import platform
system_name = platform.system()

if system_name == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
elif system_name == 'Windows': # Windows
    plt.rc('font', family='Malgun Gothic')
else: # Linux (Colab 등)
    plt.rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# -----------------------------------------------------------
# 2. 데이터 입력
# -----------------------------------------------------------
labels = ['baseline', '+knowledge graph', '+LLM', '+ESM']

# Pearson Delta 데이터
pearson_k562_mean = [39.33, 50.89, 52.07, 51.65]
pearson_k562_std  = [0.48, 0.48, 0.68, 0.85]
pearson_hepg2_mean = [38.79, 47.24, 47.94, 46.49]
pearson_hepg2_std  = [0.77, 0.63, 1.26, 0.54]

# AUROC 데이터
auroc_k562_mean = [52.40, 62.98, 63.97, 67.49]
auroc_k562_std  = [0.47, 0.55, 0.90, 0.74]
auroc_hepg2_mean = [49.93, 59.52, 60.50, 63.53]
auroc_hepg2_std  = [2.07, 0.69, 0.85, 0.72]

# -----------------------------------------------------------
# 3. 그래프 그리기
# -----------------------------------------------------------
x = np.arange(len(labels))  # 라벨 위치
width = 0.35  # 막대 너비

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 함수: 각 서브플롯(ax)에 막대 그래프 생성
def draw_bars(ax, k562_m, k562_s, hepg2_m, hepg2_s, title, ylabel):
    rects1 = ax.bar(x - width/2, k562_m, width, yerr=k562_s, label='K562', capsize=5, color='skyblue', alpha=0.9)
    rects2 = ax.bar(x + width/2, hepg2_m, width, yerr=hepg2_s, label='HePG2', capsize=5, color='salmon', alpha=0.9)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 막대 위에 값 표시 (옵션)
    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

# 첫 번째 그래프: Pearson Delta
draw_bars(ax1, pearson_k562_mean, pearson_k562_std, 
          pearson_hepg2_mean, pearson_hepg2_std, 
          'Pearson Delta Comparison', 'Pearson Delta')

# 두 번째 그래프: AUROC
draw_bars(ax2, auroc_k562_mean, auroc_k562_std, 
          auroc_hepg2_mean, auroc_hepg2_std, 
          'AUROC Comparison', 'AUROC')

plt.tight_layout()
plt.savefig('plot.png', dpi=300)