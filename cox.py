#!/usr/bin/env python3
"""
鼻咽癌多因素 Cox 回归分析 - 全变量处理版
输入：Excel 文件（包含所有原始列）
输出：每个结局的单因素/多因素结果，及重要特征排序
功能：
  1. 自动识别数值、分类、日期、文本列
  2. 从文本影像学结论中提取关键特征（淋巴结大小、坏死、颅底侵犯、EBV值）
  3. 处理分类变量（独热编码）
  4. 填充缺失值
  5. 支持逐步回归和 Lasso-Cox 特征选择
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ==================== 用户配置 ====================
EXCEL_FILE = '9000data.xlsx'          # 数据文件路径
SHEET_NAME = 0                         # 工作表名

# 结局变量定义（列名请与Excel完全一致）
outcomes = {
    'OS': {'time': 'OS time', 'event': 'OS'},        # 注意列名末尾可能有空格
    'DFS': {'time': 'DFS time', 'event': 'DFS'},
    'DMFS': {'time': 'DMFS time', 'event': 'DMFS'},
    'LRRFS': {'time': 'LRRFS time', 'event': 'LRRFS'}
}

# 分析选项
METHOD = 'lasso'          # 可选：'stepwise' 或 'lasso'
P_ENTER = 0.05            # 逐步回归进入阈值
USE_LASSO_CV = True       # Lasso-Cox 是否使用交叉验证选择 alpha
MAX_FEATURES = 50         # 若候选特征过多，Lasso 会自行筛选

# 输出文件
OUTPUT_FILE = 'cox_results_full.xlsx'

# ==================== 数据加载 ====================
print("正在加载数据...")
df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME, dtype=str)  # 先全部作为字符串读取，便于统一处理
print(f"原始数据形状: {df.shape}")

# ==================== 通用预处理 ====================
# 将日期列转换为数值（如需要，此处暂不处理，后续特征工程中可能用到）
date_cols = [col for col in df.columns if '日期' in col or 'Date' in col or '时间' in col or '随访' in col]
# 暂时保留为字符串，后续可能会提取时间差，但本例中已有 Months_From_Tx_End，故忽略

FORCE_NUMERIC = ['preHGB', 'HGB', 'preALB', 'ALB', 'preCRP', 'CRP', 
                 'preLDH', 'LDH', 'preEBV', 'EBV', 'EBV_DNA_load']

# 识别数值列：尝试转换为数字，成功则为数值列
numeric_cols = []
for col in df.columns:
    try:
        pd.to_numeric(df[col], errors='raise')
        numeric_cols.append(col)
    except:
        pass
    if col in FORCE_NUMERIC:
        numeric_cols.append(col)
        continue

# 分类列：非数值且非日期/文本的列（这里简单定义为 object 类型且不是太长的文本）
cat_cols = []
text_cols = []
for col in df.columns:
    if col in numeric_cols:
        continue
    # 检查是否为长文本（包含空格、汉字较多）
    sample = df[col].dropna().astype(str).iloc[0] if not df[col].dropna().empty else ''
    if len(sample) > 10 or '。' in sample or '；' in sample or ' ' in sample:
        text_cols.append(col)
    else:
        cat_cols.append(col)

# 强制将 preHBV 视为数值列（兼容大小写与首尾空格）
forced_numeric_cols = [c for c in df.columns if c.strip().lower() == 'prehbv']
for c in forced_numeric_cols:
    if c not in numeric_cols:
        numeric_cols.append(c)
cat_cols = [c for c in cat_cols if c not in forced_numeric_cols]
text_cols = [c for c in text_cols if c not in forced_numeric_cols]


print(f"数值列: {len(numeric_cols)} 个")
print(f"分类列: {len(cat_cols)} 个")
print(f"文本列: {len(text_cols)} 个")

# ==================== 文本特征提取 ====================
# 从影像学结论和发现中提取关键预后特征
def extract_imaging_features_enhanced(row, text_cols):
    """
    增强版影像特征提取函数
    针对鼻咽癌MRI/CT/超声/PET等报告的全面特征提取
    """
    features = {}
    
    # 合并所有文本列，统一搜索
    all_text = ' '.join([str(row[col]) for col in text_cols if pd.notna(row[col])])
    all_text_lower = all_text.lower()
    
    # ==================== 1. 淋巴结特征（全面增强） ====================
    
    # 1.1 提取所有淋巴结大小（支持多种格式）
    # 格式：13mm（右）、16mm×15mm、约12mm、短径约3mm~4mm、3至4mm、10～12mm
    size_patterns = [
        r'(\d+)\s*mm\s*（\s*([左右])\s*）',  # 13mm（右）
        r'(\d+)\s*mm\s*[×x]\s*(\d+)\s*mm',   # 16mm×15mm 或 16mmx15mm
        r'[大小约]*(\d+)\s*mm',               # 约12mm、大小约12mm
        r'短径[约为]*(\d+)\s*mm',             # 短径约3mm
        r'短径[约为]*(\d+)\s*mm\s*[~～至]\s*(\d+)\s*mm',  # 短径3~4mm、3至4mm
        r'直径[约为]*(\d+)\s*mm\s*[~～]\s*(\d+)\s*mm',    # 直径10～12mm
        r'大者[约为]*(\d+)\s*mm\s*[×x]?\s*(\d*)\s*mm?',   # 大者约12mm×15mm
    ]
    
    all_sizes = []
    for pattern in size_patterns:
        matches = re.findall(pattern, all_text)
        for match in matches:
            if isinstance(match, tuple):
                if len(match) == 2 and match[1] in ['左', '右']:
                    all_sizes.append(float(match[0]))  # 带左右的，只取数值
                elif len(match) == 2 and match[1].isdigit():
                    all_sizes.append((float(match[0]) + float(match[1])) / 2)  # 范围取平均
                else:
                    all_sizes.append(float(match[0]))
            else:
                all_sizes.append(float(match))
    
    # 淋巴结大小统计特征
    if all_sizes:
        features['node_max_size'] = max(all_sizes)
        features['node_mean_size'] = np.mean(all_sizes)
        features['node_count'] = len(all_sizes)
        features['node_has_large'] = 1 if max(all_sizes) >= 10 else 0  # ≥10mm认为肿大
        features['node_has_very_large'] = 1 if max(all_sizes) >= 20 else 0  # ≥20mm显著肿大
    else:
        features['node_max_size'] = np.nan
        features['node_mean_size'] = np.nan
        features['node_count'] = 0
        features['node_has_large'] = 0
        features['node_has_very_large'] = 0
    
    # 1.2 淋巴结位置（分区）
    node_regions = {
        'node_rl': r'咽后[间隙]*[见有]*[肿大]*淋巴结',  # 咽后淋巴结
        'node_ii': r'[双左右]*颈\s*[Ii]{2}\s*区',  # II区
        'node_iii': r'[双左右]*颈\s*[Ii]{3}\s*区',  # III区  
        'node_iv': r'[双左右]*颈\s*[Ii][Vv]\s*区',  # IV区
        'node_ib': r'[双左右]*颈\s*[Ii][Bb]\s*区',  # Ib区
        'node_ia': r'[双左右]*颈\s*[Ii][Aa]\s*区',  # Ia区
        'node_va': r'[双左右]*颈\s*[Vv][Aa]\s*区',  # Va区
    }
    for feat_name, pattern in node_regions.items():
        features[feat_name] = 1 if re.search(pattern, all_text) else 0
    
    # 1.3 淋巴结性质
    features['node_necrosis'] = 1 if '坏死' in all_text else 0
    features['node_fusion'] = 1 if '融合' in all_text else 0
    features['node_multiple'] = 1 if '多发' in all_text and '淋巴结' in all_text else 0
    features['node_bilateral'] = 1 if '双侧' in all_text and '淋巴结' in all_text else 0
    features['node_unilateral_left'] = 1 if ('左侧' in all_text or '左颈' in all_text) and '淋巴结' in all_text else 0
    features['node_unilateral_right'] = 1 if ('右侧' in all_text or '右颈' in all_text) and '淋巴结' in all_text else 0
    
    # 1.4 淋巴结强化特征
    features['node_strong_enhance'] = 1 if '明显强化' in all_text and '淋巴结' in all_text else 0
    features['node_mild_enhance'] = 1 if ('轻度强化' in all_text or '轻中度强化' in all_text) and '淋巴结' in all_text else 0
    features['node_heterogeneous'] = 1 if '不均匀强化' in all_text and '淋巴结' in all_text else 0
    features['node_uniform'] = 1 if '均匀强化' in all_text and '淋巴结' in all_text else 0
    
    # ==================== 2. 原发肿瘤特征（鼻咽部） ====================
    
    # 2.1 肿瘤位置与侵犯范围
    tumor_locations = {
        'tumor_nasopharynx': r'鼻咽[腔部顶壁后壁侧壁]*[见有]*[肿物结节占位增厚]',
        'tumor_left_wall': r'鼻咽左侧壁[见有]*[肿物结节占位增厚]',
        'tumor_right_wall': r'鼻咽右侧壁[见有]*[肿物结节占位增厚]',
        'tumor_posterior_wall': r'鼻咽后壁[见有]*[肿物结节占位增厚]',
        'tumor_roof': r'鼻咽顶[壁后壁]*[见有]*[肿物结节占位增厚]',
        'tumor_fossa_rosenmuller': r'咽隐窝[消失变浅]',  # 罗斯森氏窝（咽隐窝）
    }
    for feat_name, pattern in tumor_locations.items():
        features[feat_name] = 1 if re.search(pattern, all_text) else 0
    
    # 2.2 周围结构侵犯
    invasion_structures = {
        'invade_parapharyngeal': r'咽旁[脂肪间隙]*[受侵推移变窄]',
        'invade_pterygopalatine': r'翼腭窝[受侵侵犯]',
        'invade_infratemporal': r'颞下窝[受侵侵犯]',
        'invade_skull_base': r'颅底[骨质]*[受侵破坏]',
        'invade_clivus': r'斜坡[骨质]*[受侵破坏信号]',
        'invade_pterygoid': r'翼突[基底部]*[骨质]*[受侵破坏信号]',
        'invade_sphenoid': r'蝶窦[受侵侵犯]',
        'invade_oropharynx': r'口咽[侧壁]*[受侵受累增厚]',
        'invade_postnasal': r'后鼻孔[受侵]',
        'invade_muscle': r'[腭帆提肌腭帆张肌头长肌翼内肌翼外肌][受侵肿胀]',
        'invade_cavernous_sinus': r'海绵窦[受侵增宽]',
    }
    for feat_name, pattern in invasion_structures.items():
        features[feat_name] = 1 if re.search(pattern, all_text) else 0
    
    # 2.3 肿瘤信号特征（MRI）
    features['tumor_t1_isointense'] = 1 if 'T1WI呈等信号' in all_text or 'T1W呈等信号' in all_text else 0
    features['tumor_t2_hyperintense'] = 1 if 'T2WI呈稍高信号' in all_text or 'T2W呈稍高信号' in all_text else 0
    features['tumor_enhancement'] = 1 if '增强后明显强化' in all_text or '增强扫描明显强化' in all_text else 0
    
    # ==================== 3. 骨质破坏特征（详细分类） ====================
    
    # 3.1 骨质破坏位置
    bone_sites = {
        'bone_skull_base': r'颅底骨质[破坏信号]',
        'bone_clivus': r'斜坡骨质[破坏信号减低]',
        'bone_pterygoid': r'翼突[基底部]*骨质[破坏信号]',
        'bone_sphenoid': r'蝶骨[基底部]*骨质[破坏信号]',
    }
    for feat_name, pattern in bone_sites.items():
        features[feat_name] = 1 if re.search(pattern, all_text) else 0
    
    # 3.2 骨质破坏程度/变化
    features['bone_destruction_new'] = 1 if '骨质破坏' in all_text and '修复' not in all_text else 0
    features['bone_destruction_healing'] = 1 if '骨质破坏.*修复' in all_text or '较前修复' in all_text else 0
    features['bone_signal_abnormal'] = 1 if '骨质信号[减低异常]' in all_text else 0
    
    # ==================== 4. 治疗反应与随访（动态变化） ====================
    
    # 4.1 治疗状态识别
    features['status_post_rt'] = 1 if '放疗后' in all_text or '放化疗后' in all_text else 0
    features['status_post_ct'] = 1 if '化疗后' in all_text or '放化疗后' in all_text else 0
    
    # 4.2 肿瘤反应评估（RECIST类似）
    response_patterns = {
        'response_complete': r'未见[明确]*[肿瘤肿物复发]',
        'response_partial': r'较前[明显]*[缩小好转减轻]',
        'response_stable': r'较前[相仿未见明显变化]',
        'response_progression': r'较前[增大进展]',
        'response_healing': r'较前修复',
    }
    for feat_name, pattern in response_patterns.items():
        features[feat_name] = 1 if re.search(pattern, all_text) else 0
    
    # 4.3 当前状态描述
    features['current_no_tumor'] = 1 if '未见明确复发' in all_text or '未见肿瘤复发' in all_text else 0
    features['current_mucosal_thickening'] = 1 if '粘膜增厚' in all_text else 0
    features['current_post_radiation_change'] = 1 if '放疗后改变' in all_text else 0
    
    # ==================== 5. 远处转移与并发症 ====================
    
    # 5.1 远处转移（基于常见鼻咽癌转移部位）
    metastasis_sites = {
        'meta_liver': r'肝脏.*[占位性病变肿物]',
        'meta_lung': r'[双]*肺.*[占位结节实质性病变]',
        'meta_bone': r'[颅骨胸骨肋骨椎骨骨盆四肢骨].*[破坏代谢活跃]',
        'meta_adrenal': r'肾上腺[区]*.*[占位性病变]',
    }
    for feat_name, pattern in metastasis_sites.items():
        features[feat_name] = 1 if re.search(pattern, all_text) else 0
    
    # 5.2 良性病变（鉴别诊断）
    features['benign_liver_hemangioma'] = 1 if '肝血管瘤' in all_text else 0
    features['benign_reactive_node'] = 1 if '反应性淋巴结' in all_text else 0
    features['benign_cyst'] = 1 if '囊肿' in all_text else 0
    
    # ==================== 6. 炎症与并发症 ====================
    
    # 6.1 鼻窦炎（非常常见）
    sinus_patterns = {
        'sinus_maxillary': r'上颌窦[粘膜增厚炎症]',
        'sinus_ethmoid': r'筛窦[粘膜增厚炎症]',
        'sinus_sphenoid': r'蝶窦[粘膜增厚炎症]',
        'sinus_frontal': r'额窦[粘膜增厚炎症]',
        'sinus_general': r'鼻窦炎',
    }
    for feat_name, pattern in sinus_patterns.items():
        features[feat_name] = 1 if re.search(pattern, all_text) else 0
    
    # 6.2 乳突炎
    features['mastoiditis_left'] = 1 if '左侧乳突炎' in all_text else 0
    features['mastoiditis_right'] = 1 if '右侧乳突炎' in all_text else 0
    features['mastoiditis_bilateral'] = 1 if '双侧乳突炎' in all_text else 0
    
    # ==================== 7. EBV 相关（鼻咽癌特异性） ====================
    
    # 从特定列提取，如果存在
    ebv_text = str(row.get('EBV_exam_conclusion', ''))
    ebv_nums = re.findall(r'(\d+\.?\d*)', ebv_text)
    if ebv_nums:
        try:
            features['EBV_DNA_load'] = float(ebv_nums[0])
            features['EBV_positive'] = 1 if float(ebv_nums[0]) > 0 else 0
        except:
            features['EBV_DNA_load'] = np.nan
            features['EBV_positive'] = 0
    else:
        features['EBV_DNA_load'] = np.nan
        features['EBV_positive'] = 0
    
    # 文本中提及EBV
    features['EBV_mentioned'] = 1 if 'EBV' in all_text or 'EB病毒' in all_text else 0
    
    # ==================== 8. PET-CT 代谢特征（如果有） ====================
    
    # 提取SUV值
    suv_matches = re.findall(r'SUV[约为]*(\d+\.?\d*)', all_text)
    if suv_matches:
        suv_values = [float(x) for x in suv_matches]
        features['SUV_max'] = max(suv_values)
        features['SUV_mean'] = np.mean(suv_values)
        features['has_high_metabolism'] = 1 if max(suv_values) > 10 else 0  # SUV>10认为高代谢
    else:
        features['SUV_max'] = np.nan
        features['SUV_mean'] = np.nan
        features['has_high_metabolism'] = 0
    
    # ==================== 9. 时间特征（随访间隔） ====================
    
    # 提取随访时间间隔（月）
    followup_matches = re.findall(r'与(\d{4})[-/](\d{1,2})[-/](\d{1,2})[日]*片对比', all_text)
    if followup_matches:
        features['has_prior_comparison'] = 1
        # 可以计算与当前检查的时间差，如果需要
    else:
        features['has_prior_comparison'] = 0
    
    # ==================== 10. 影像质量与完整性 ====================
    
    features['image_quality_good'] = 1 if '图像清晰' in all_text else 0
    features['contrast_enhancement_used'] = 1 if '增强' in all_text else 0
    
    return pd.Series(features)


# ==================== 使用示例（整合到原代码中） ====================

def process_data_enhanced(df, text_cols):
    """
    使用增强特征提取处理数据框
    """
    print(f"\n正在从 {len(text_cols)} 个文本列提取增强特征...")
    
    # 应用特征提取
    text_features = df.apply(lambda row: extract_imaging_features_enhanced(row, text_cols), axis=1)
    
    # 合并到原数据
    df = pd.concat([df, text_features], axis=1)
    
    # 获取新生成的特征列
    new_features = list(text_features.columns)
    print(f"提取了 {len(new_features)} 个新特征")
    print(f"特征示例: {new_features[:10]}...")
    
    # 删除原始文本列
    df = df.drop(columns=text_cols, errors='ignore')
    
    return df, new_features


# ==================== 特征分类（用于后续分析） ====================

def categorize_features(feature_names):
    """
    将提取的特征按类别分组，便于后续分析和解释
    """
    categories = {
        '淋巴结特征': [f for f in feature_names if f.startswith('node_')],
        '原发肿瘤特征': [f for f in feature_names if f.startswith('tumor_')],
        '侵犯特征': [f for f in feature_names if f.startswith('invade_')],
        '骨质特征': [f for f in feature_names if f.startswith('bone_')],
        '治疗反应': [f for f in feature_names if f.startswith('response_') or f.startswith('status_') or f.startswith('current_')],
        '转移特征': [f for f in feature_names if f.startswith('meta_')],
        '良性病变': [f for f in feature_names if f.startswith('benign_')],
        '炎症特征': [f for f in feature_names if f.startswith('sinus_') or f.startswith('mastoiditis_')],
        'EBV特征': [f for f in feature_names if f.startswith('EBV_')],
        'PET代谢特征': [f for f in feature_names if f.startswith('SUV_') or f.startswith('has_')],
        '其他': [f for f in feature_names if f not in 
                 [item for sublist in [
                     [f for f in feature_names if f.startswith('node_')],
                     [f for f in feature_names if f.startswith('tumor_')],
                     [f for f in feature_names if f.startswith('invade_')],
                     [f for f in feature_names if f.startswith('bone_')],
                     [f for f in feature_names if f.startswith('response_') or f.startswith('status_') or f.startswith('current_')],
                     [f for f in feature_names if f.startswith('meta_')],
                     [f for f in feature_names if f.startswith('benign_')],
                     [f for f in feature_names if f.startswith('sinus_') or f.startswith('mastoiditis_')],
                     [f for f in feature_names if f.startswith('EBV_')],
                     [f for f in feature_names if f.startswith('SUV_') or f.startswith('has_')],
                 ] for item in sublist]]
    }
    
    # 打印分类结果
    for cat, feats in categories.items():
        if feats:
            print(f"\n{cat} ({len(feats)}个):")
            print(f"  {feats}")
    
    return categories


print("\n正在从文本列提取特征...")
df, new_features = process_data_enhanced(df, text_cols)
numeric_cols.extend(new_features)

# 将新提取的特征加入数值列
numeric_cols.extend(['node_max_size', 'node_necrosis', 'skull_invasion', 'trend', 'EBV_post'])
numeric_cols = list(set(numeric_cols))

# ==================== 分类变量编码 ====================
# 对分类列进行独热编码（注意排除那些已作为数值处理的列）
cat_cols_to_encode = [c for c in cat_cols if c not in outcomes['OS']['time'] and c not in outcomes['OS']['event']]
# 同时排除时间列和事件列
for out in outcomes.values():
    cat_cols_to_encode = [c for c in cat_cols_to_encode if c not in [out['time'], out['event']]]

# 对分类变量进行独热编码（丢弃第一个避免多重共线性）
if cat_cols_to_encode:
    df = pd.get_dummies(df, columns=cat_cols_to_encode, drop_first=True)
    print(f"编码后新增特征: {len(df.columns) - len(numeric_cols) - len(outcomes)*2} 个")

# ==================== 数值列类型转换与缺失处理 ====================
# 将数值列转换为 float
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 填充数值列的缺失值（用中位数）
for col in numeric_cols:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

# ==================== 构建最终候选特征列表 ====================
# 所有非结局的列都可能作为特征
exclude_cols = []
for out in outcomes.values():
    exclude_cols.extend([out['time'], out['event']])
exclude_cols.extend(['patient_sn', '研究号', '开始治疗时间', '复查时间', '末次随访时间', 
                     'Visit_Date.x', '死亡时间', 'Recurrence_Date', '转移时间', 
                     '鼻咽复发时间', '颈部复发时间', '复发时间'])  # 根据实际列名调整
candidate_features = [col for col in df.columns if col not in exclude_cols]
print(f"\n候选特征总数: {len(candidate_features)}")

# ==================== 生存分析函数 ====================
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

def univariate_cox(df, time_col, event_col, features):
    results = []
    for feat in features:
        try:
            cph = CoxPHFitter()
            data = df[[time_col, event_col, feat]].dropna()
            if data.shape[0] < 10 or data[event_col].sum() < 5:
                continue
            cph.fit(data, duration_col=time_col, event_col=event_col, formula=feat)
            summary = cph.summary.loc[feat]
            results.append({
                'feature': feat,
                'coef': summary['coef'],
                'exp(coef)': summary['exp(coef)'],
                'se(coef)': summary['se(coef)'],
                'lower_0.95': summary['exp(coef) lower 95%'],
                'upper_0.95': summary['exp(coef) upper 95%'],
                'p': summary['p']
            })
        except Exception as e:
            continue
    return pd.DataFrame(results)

def multivariate_cox(df, time_col, event_col, features):
    if len(features) == 0:
        return None
    formula = ' + '.join(features)
    cph = CoxPHFitter()
    data = df[[time_col, event_col] + features].dropna()
    if data.shape[0] < 10 or data[event_col].sum() < 5:
        return None
    cph.fit(data, duration_col=time_col, event_col=event_col, formula=formula)
    return cph

def stepwise_selection(df, time_col, event_col, feature_candidates, p_enter=0.05):
    selected = []
    remaining = feature_candidates.copy()
    best_p = 0.0
    while remaining and best_p < p_enter:
        scores = []
        for feat in remaining:
            try:
                formula = ' + '.join(selected + [feat])
                cph = CoxPHFitter()
                data = df[[time_col, event_col] + selected + [feat]].dropna()
                if data.shape[0] < 10 or data[event_col].sum() < 5:
                    continue
                cph.fit(data, duration_col=time_col, event_col=event_col, formula=formula)
                p_val = cph.summary.loc[feat, 'p']
                scores.append((p_val, feat))
            except:
                continue
        if not scores:
            break
        scores.sort()
        best_p, best_feat = scores[0]
        if best_p < p_enter:
            selected.append(best_feat)
            remaining.remove(best_feat)
            print(f"  加入 {best_feat}, p={best_p:.4f}")
        else:
            break
    return selected

# ==================== 对每个结局进行分析 ====================
all_results = {}

for outcome_name, cols in outcomes.items():
    print(f"\n{'='*60}")
    print(f"正在分析 {outcome_name} ...")
    time_col = cols['time']
    event_col = cols['event']

    try:
        if time_col not in df.columns or event_col not in df.columns:
            warn = f"缺少列: {time_col if time_col not in df.columns else ''} {event_col if event_col not in df.columns else ''}".strip()
            print(f"{outcome_name} 跳过: {warn}")
            all_results[outcome_name] = pd.DataFrame({'warning': [warn], 'outcome': [outcome_name]})
            continue

        # 确保时间列为数值
        df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
        df[event_col] = pd.to_numeric(df[event_col], errors='coerce')

        # 删除该结局中时间或事件缺失的样本
        data_clean = df[[time_col, event_col] + candidate_features].dropna(subset=[time_col, event_col])
        if len(data_clean) == 0:
            print(f"无有效数据，跳过 {outcome_name}")
            all_results[outcome_name] = pd.DataFrame({'warning': [f'No valid data for {outcome_name}'], 'outcome': [outcome_name]})
            continue

        print(f"有效样本量: {len(data_clean)}, 事件数: {data_clean[event_col].sum()}")

        if METHOD == 'stepwise':
            # 逐步回归
            selected = stepwise_selection(data_clean, time_col, event_col, candidate_features, p_enter=P_ENTER)
            print(f"逐步回归选入特征: {selected}")
            if selected:
                final_model = multivariate_cox(data_clean, time_col, event_col, selected)
                if final_model:
                    res = final_model.summary
                    res['outcome'] = outcome_name
                    all_results[outcome_name] = res
                else:
                    print("多因素模型拟合失败")
            else:
                all_results[outcome_name] = pd.DataFrame({'warning': ['No features selected in stepwise'], 'outcome': [outcome_name]})

        elif METHOD == 'lasso':
            try:
                from sksurv.linear_model import CoxnetSurvivalAnalysis
                from sksurv.metrics import concordance_index_censored
            except ImportError:
                msg = "请安装 scikit-survival: pip install scikit-survival"
                print(msg)
                all_results[outcome_name] = pd.DataFrame({'warning': [msg], 'outcome': [outcome_name]})
                continue
        
            # 准备特征 X 和生存对象 y
            X = data_clean[candidate_features].values
            y = np.array([(bool(e), t) for e, t in zip(data_clean[event_col], data_clean[time_col])],
                         dtype=[('event', '?'), ('time', '<f8')])
        
            # ========== 数据清洗：处理无穷大和缺失值 ==========
            print(f"\n数据清洗...")

            # 先将 X 转换为 DataFrame
            X_df = pd.DataFrame(X, columns=candidate_features)

            # ===== 关键修复：确保所有列为数值类型 =====
            print("转换数据类型...")
            for col in X_df.columns:
                # 强制转换为数值，无法转换的设为 NaN
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

            # 检查转换后的数据类型
            print(f"数据类型检查:\n{X_df.dtypes.value_counts()}")

            # 替换无穷大为 NaN（现在可以安全操作了）
            X_df = X_df.replace([np.inf, -np.inf], np.nan)

            # 检查问题列（现在应该都是数值类型了）
            problem_cols = []
            for col in X_df.columns:
                if X_df[col].isnull().any():
                    problem_cols.append(col)
                    print(f"列 '{col}': NaN数量={X_df[col].isnull().sum()}, "
                        f"min={X_df[col].min():.2f}, max={X_df[col].max():.2e}")

            # 填充缺失值
            print(f"\n填充缺失值...")
            for col in X_df.columns:
                if X_df[col].isnull().all():
                    # 如果整列都是 NaN，填充 0
                    X_df[col] = 0.0
                    print(f"  {col}: 全为NaN，填充0")
                else:
                    median_val = X_df[col].median()
                    X_df[col] = X_df[col].fillna(median_val)

            # 最终检查（使用 pandas 的 isna() 而不是 numpy 的 isnan()）
            X_clean = X_df.values.astype(float)  # 确保为 float 类型
            assert not pd.isna(X_clean).any(), "仍有NaN存在"
            assert not np.isinf(X_clean).any(), "仍有Inf存在"

            print(f"清洗完成，数据形状: {X_clean.shape}")

            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)

        
            if USE_LASSO_CV:
                print("\n使用改进的交叉验证选择 alpha...")

                # 更合理的 alpha 范围（从很小开始）
                alphas = np.logspace(-4, -1, 20)  # 0.0001 到 0.1

                from sklearn.model_selection import KFold
                kf = KFold(n_splits=3, shuffle=True, random_state=42)

                cv_results = []

                for alpha in alphas:
                    fold_scores = []
                    fold_nonzero = []

                    for train_idx, val_idx in kf.split(X_scaled):
                        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]

                        try:
                            model = CoxnetSurvivalAnalysis(
                                l1_ratio=1.0,
                                alphas=[alpha],
                                fit_baseline_model=True,
                                max_iter=1000
                            )
                            model.fit(X_train, y_train)

                            # 预测风险评分
                            pred = model.predict(X_val)

                            # 计算 C-index
                            c_index = concordance_index_censored(
                                y_val['event'], y_val['time'], pred
                            )[0]

                            # 记录选中特征数
                            n_nonzero = np.sum(np.abs(model.coef_) > 1e-6)

                            fold_scores.append(c_index)
                            fold_nonzero.append(n_nonzero)

                        except Exception as e:
                            print(f"    alpha={alpha:.6f} 失败: {e}")
                            fold_scores.append(0.5)
                            fold_nonzero.append(0)

                    mean_score = np.mean(fold_scores)
                    mean_nonzero = np.mean(fold_nonzero)
                    std_score = np.std(fold_scores)

                    cv_results.append({
                        'alpha': alpha,
                        'c_index': mean_score,
                        'c_index_std': std_score,
                        'n_features': mean_nonzero
                    })

                    print(f"  alpha={alpha:.6f} | C-index={mean_score:.4f}±{std_score:.4f} | "
                        f"特征数={mean_nonzero:.0f}")

                    # 早停：如果特征数为0且C-index=0.5，跳过后续更大的alpha
                    if mean_nonzero == 0 and mean_score == 0.5 and alpha > 0.01:
                        print("    -> 后续alpha将无特征，提前终止搜索")
                        break

                # 选择最佳 alpha（有特征且C-index最高的）
                valid_results = [r for r in cv_results if r['n_features'] > 0]

                if valid_results:
                    best = max(valid_results, key=lambda x: x['c_index'])
                    best_alpha = best['alpha']
                    print(f"\n最佳 alpha: {best_alpha:.6f}")
                    print(f"  C-index: {best['c_index']:.4f}±{best['c_index_std']:.4f}")
                    print(f"  选中特征数: {best['n_features']:.0f}")
                else:
                    # 如果没有有效的，选最小的alpha
                    best_alpha = alphas[0]
                    print(f"\n警告：所有alpha都未选中特征，使用最小alpha: {best_alpha}")

                # 用最佳参数拟合最终模型
                cox_lasso = CoxnetSurvivalAnalysis(
                    l1_ratio=1.0,
                    alphas=[best_alpha],
                    fit_baseline_model=True
                )
                cox_lasso.fit(X_scaled, y)

            else:
                cox_lasso = CoxnetSurvivalAnalysis(
                    l1_ratio=1.0, 
                    alphas=[0.001],  # 使用更小的默认alpha
                    fit_baseline_model=True
                )
                cox_lasso.fit(X_scaled, y)
        
            # ========== 提取结果 ==========
            coef = cox_lasso.coef_.flatten()
            selected_idx = np.where(np.abs(coef) > 1e-6)[0]
            selected_features = [candidate_features[i] for i in selected_idx]
        
            print(f"\n最终模型选中特征: {len(selected_features)} 个")
            if len(selected_features) > 0:
                print(f"选中特征: {selected_features[:10]}...")  # 打印前10个
            else:
                print("警告：未选中任何特征！")
        
            # 构建结果 DataFrame（确保有数据）
            res_df = pd.DataFrame({
                'feature': candidate_features,
                'coef': coef,
                'exp(coef)': np.exp(coef),
                'abs_coef': np.abs(coef),
                'selected': np.abs(coef) > 1e-6
            })
            res_df = res_df.sort_values('abs_coef', ascending=False)  # 按重要性排序
            res_df['outcome'] = outcome_name
        
            # 如果没有选中特征，添加警告信息
            if len(selected_idx) == 0:
                res_df['warning'] = 'No features selected - consider smaller alpha'

            all_results[outcome_name] = res_df

    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        print(f"{outcome_name} 分析失败: {err_msg}")
        all_results[outcome_name] = pd.DataFrame({'warning': [err_msg], 'outcome': [outcome_name]})

# ==================== 保存结果 ====================
with pd.ExcelWriter(OUTPUT_FILE) as writer:
    for outcome_name, res in all_results.items():
        res.to_excel(writer, sheet_name=outcome_name[:31], index=False)  # Excel sheet name max 31 chars
    # 汇总所有选中特征（仅 Lasso 方法时有意义）
    if METHOD == 'lasso' and all_results:
        summary = pd.concat(all_results.values())
        summary.to_excel(writer, sheet_name='Summary', index=False)

print(f"\n分析完成，结果已保存至 {OUTPUT_FILE}")