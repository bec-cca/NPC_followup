#!/usr/bin/env python3
import pandas as pd
import json
import pathlib
import tqdm
import re
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# ==================== 用户配置 ====================
EXCEL_FILE   = '9000data.xlsx'                # 输入 Excel 文件
IMAGE_ROOT   = pathlib.Path('images')         # 图像文件根目录（若无图像可忽略）
OUT_DIR      = pathlib.Path('/data1/sshan24/Janus/medical_processor/data_risk_v2_split')   # 输出目录
SPLIT_SEED   = 42                              # 随机种子

# 数据集划分比例 (修改为 8:2)
TRAIN_RATIO  = 0.8
VAL_RATIO    = 0.2    # 取消验证集，直接8:2划分
# TEST_RATIO   = 0.0

# 时间点定义（单位：月）
TIME_POINTS = [12, 36, 60]  # 1年、3年、5年
TIME_LABELS = ['1y', '3y', '5y']

# ========== 平衡参数 ==========
# 正负样本平衡（在每个时间点内）
ENABLE_POS_NEG_BALANCE = True        # 是否启用正负样本平衡
POS_NEG_RATIO = 0.33                   # 目标正负比例（阳性:阴性 = 1.0 表示 1:1）

# 时间点平衡
ENABLE_TIME_BALANCE = True            # 是否启用时间点平衡
TIME_BALANCE_STRATEGY = 'uniform'     # 平衡策略：'uniform'（各时间点样本数相同）、'proportional'（按比例）、'weighted'（加权）
TIME_TARGET_RATIO = {                  # 各时间点目标比例（仅当TIME_BALANCE_STRATEGY='weighted'时生效）
    '1y': 1/3,
    '3y': 1/3, 
    '5y': 1/3
}

# 总样本数限制（每个结局的总训练样本数）
MAX_TOTAL_SAMPLES_PER_OUTCOME = 30000  # 每个结局训练集总样本数（设为 None 则无限制）

# ==================== 准备工作 ====================
OUT_DIR.mkdir(exist_ok=True)
df = pd.read_excel(EXCEL_FILE, na_filter=True)

# ==================== 定义关键特征（基于您的多因素分析结果）====================
KEY_FEATURES = {
    'OS': [
        'Months_From_Tx_End', 'age', 'Stage', 'T', 'EBV_positive', 'sex', 
        'preLDH', 'HGB', 'response_progression', '累积顺铂剂量', 'invade_skull_base',
        'response_complete', 'node_necrosis', 'preALB', 'N', 'node_fusion', 'LDH',
        '化疗药_1+8', 'node_mean_size', '同期化疗', 'ALP', 'hist', '方案_1+5+8',
        'smoke', 'meta_liver', 'response_stable', 'node_heterogeneous', '诱导疗程数',
        '方案_3+2', 'preCRP', 'Visit_ID', '化疗药_1+9', 'invade_cavernous_sinus',
        'PLR', 'NEUT', 'LYMPH', '方案_3', '化疗药_4', 'invade_infratemporal',
        '化疗药_5+8', 'EBV_exam_conclusion', '方案_1+2+3', '方案_4', 'preEBV',
        'node_multiple', 'EBV_DNA_load'
    ],
    'DFS': [
        'Months_From_Tx_End', 'Phase_Follow-up', 'age', 'EBV_positive', 'Stage',
        '累积顺铂剂量', 'T', 'sex', 'Phase_During-Tx', 'response_progression',
        'node_mean_size', '同期疗程数', 'node_max_size', 'meta_bone', 'meta_liver',
        'N', '方案_1+3', 'node_necrosis', 'current_no_tumor', 'preLDH',
        'node_heterogeneous', 'response_stable', '化疗药_6', '同期化疗',
        'node_unilateral_right', '化疗药_7', 'response_complete', 'TSH', 'HGB',
        'hist', '化疗药_2+8', 'preALB', 'invade_skull_base', '方案_3', 'node_multiple',
        'tumor_t1_isointense', 'smoke', '化疗药_8', 'FT3', '诱导化疗', '方案_2+3',
        'SUV_max', 'node_bilateral', 'node_fusion', '辅助化疗', 'LDH', '方案_2+4',
        'node_rl', 'Ca', 'current_post_radiation_change', 'invade_cavernous_sinus',
        '方案_3+2', 'sinus_ethmoid', 'NEUT', 'node_uniform', 'benign_liver_hemangioma',
        'sinus_general', 'node_has_large', '化疗药_6+8', 'meta_adrenal', 'node_ib',
        'sinus_maxillary', '方案_2', '方案_1+5+8', '化疗药_10', 'sinus_sphenoid',
        'preCRP', '化疗药_1+9', '方案_1+4', 'LYMPH', '方案_11', 'benign_reactive_node',
        'node_strong_enhance', '方案_2+1', 'image_quality_good', '化疗药_9',
        '化疗药_1+8', 'SUV_mean', '方案_4', '方案_4+8', '化疗药_1+6', 'benign_cyst',
        'meta_lung', 'status_post_ct', 'PLT', '诱导疗程数', '化疗药_4', '化疗药_2+5',
        'ALP', 'alcohol', '化疗药_7+8', '化疗药_5', '方案_3+1', '方案_9',
        'bone_skull_base', 'invade_pterygoid', '化疗药_5+8', 'node_iii', 'NLR',
        'family', 'response_partial', 'invade_infratemporal', 'tumor_posterior_wall',
        'has_high_metabolism', 'bone_destruction_healing', 'node_mild_enhance', 'AST',
        '方案_1+2+3', 'tumor_fossa_rosenmuller', 'EBV_mentioned', 'tumor_roof',
        'node_iv', 'Visit_ID', 'tumor_right_wall', 'has_prior_comparison',
        'invade_clivus', 'node_has_very_large', '化疗情况含靶向', 'preEBV', 'node_count',
        'node_va', '方案_1+8', 'mastoiditis_right', 'bone_sphenoid', 'EBV_DNA_load',
        'node_ia', 'tumor_left_wall', 'mastoiditis_bilateral', 'invade_postnasal',
        'invade_sphenoid', '方案_3+4', 'bone_destruction_new', 'invade_pterygopalatine',
        '化疗药_10+8', 'ALT', 'current_mucosal_thickening', '方案_3+8', '方案_5',
        'response_healing', 'tumor_nasopharynx', '化疗药_3', 'status_post_rt', 'node_ii',
        'EBV_exam_conclusion'
    ],
    'DMFS': [
        'Months_From_Tx_End', 'EBV_positive', 'N', 'Stage', 'sex', 'preLDH',
        'LYMPH', 'age', 'Visit_ID', 'preALB', 'node_fusion', 'LDH',
        'response_complete', 'tumor_t1_isointense', 'smoke', '方案_3+2',
        '方案_1+5+8', '化疗药_5', 'node_heterogeneous', 'HGB', 'hist',
        'invade_skull_base', '化疗药_2+8', 'node_multiple', 'node_necrosis',
        'meta_liver', 'T', '化疗药_1+9', 'node_mean_size', 'response_partial',
        'EBV_exam_conclusion', '方案_4'
    ],
    'LRRFS': [
        'Months_From_Tx_End', 'Phase_Follow-up', 'EBV_positive', 'T', 'Phase_During-Tx',
        'response_progression', 'Stage', 'current_no_tumor', 'node_unilateral_right',
        'invade_clivus', 'node_heterogeneous', '同期化疗', 'node_bilateral', 'bone_clivus',
        '同期疗程数', 'node_rl', 'age', 'current_post_radiation_change', 'node_necrosis',
        'node_max_size', 'node_uniform', 'node_has_large', '方案_2', '诱导疗程数',
        'node_mean_size', 'response_stable', 'hist', 'AST', '化疗药_7', 'TSH',
        'response_partial', 'smoke', '累积顺铂剂量', '方案_1+3', 'meta_bone', 'ALT',
        'alcohol', 'SUV_max', '方案_1', 'meta_liver', 'status_post_ct', 'sex', 'Ca',
        '化疗药_5+8', 'benign_cyst', 'node_unilateral_left', '化疗药_10', 'preALB',
        '方案_3+8', 'response_complete', '方案_3+4', 'tumor_roof', '方案_2+3',
        '化疗药_9', 'Visit_ID', 'invade_cavernous_sinus', '方案_4', 'bone_skull_base',
        'node_ii', '方案_2+4', 'PLT', 'node_ib', '方案_1+5+8', 'benign_reactive_node',
        '方案_1+2+3', 'node_strong_enhance', '方案_11', '化疗药_1+8', '化疗药_1',
        'node_count', 'ALP', 'has_high_metabolism', 'status_post_rt', 'tumor_enhancement',
        '方案_3', 'EBV_exam_conclusion', 'tumor_t2_hyperintense', '化疗药_8',
        'image_quality_good', 'NEUT', 'node_iii', 'mastoiditis_left', '方案_3+2',
        '方案_2+1', '方案_1+4', 'mastoiditis_bilateral', 'preLDH', 'FT4', 'PLR',
        '化疗药_4', 'meta_lung', '化疗药_6+8', 'node_has_very_large', 'sinus_maxillary',
        '化疗药_7+8', 'NLR', '辅助化疗', 'tumor_posterior_wall', 'has_prior_comparison',
        '方案_6', '化疗药_2+5', '诱导化疗', 'bone_sphenoid', 'EBV_mentioned',
        '方案_1+5', 'tumor_right_wall', '化疗情况含靶向', '方案_9', '方案_4+8',
        '方案_3+1', '化疗药_1+9', 'tumor_fossa_rosenmuller', '化疗药_1+6',
        'node_multiple', '化疗药_2+8', 'invade_muscle', '化疗药_5', 'family',
        'node_mild_enhance', 'invade_postnasal', 'sinus_general', 'preHGB',
        'invade_pterygopalatine', 'bone_destruction_new', 'node_fusion', '化疗药_3',
        '化疗药_11', 'node_iv', 'tumor_t1_isointense', 'benign_liver_hemangioma',
        'bone_destruction_healing', 'invade_oropharynx', '化疗药_2', 'tumor_left_wall',
        'invade_sphenoid', '化疗药_3+8', 'meta_adrenal', 'response_healing', 'ALB',
        'LDH', 'invade_infratemporal', 'LYMPH', 'contrast_enhancement_used',
        'sinus_ethmoid', 'tumor_nasopharynx', 'invade_parapharyngeal', 'node_va',
        'current_mucosal_thickening', 'EBV_DNA_load'
    ]
}

# 获取所有文本列（用于特征提取）
text_cols = [
    'MRI_exam_conclusion', 'US_exam_conclusion', 'X_exam_conclusion',
    'PET_exam_conclusion', 'CT_exam_conclusion', 'ECT_exam_conclusion',
    'MRI_exam_finding', 'US_exam_finding', 'X_exam_finding',
    'CT_exam_finding', 'PET_exam_finding', 'ECT_exam_finding',
    'EBV_exam_conclusion'
]

# ==================== 增强版文本特征提取 ====================
def extract_imaging_features_enhanced(row):
    """
    增强版影像特征提取函数
    针对鼻咽癌MRI/CT/超声/PET等报告的全面特征提取
    """
    features = {}
    
    # 合并所有文本列，统一搜索
    all_text = ' '.join([str(row[col]) for col in text_cols if pd.notna(row[col])])
    all_text_lower = all_text.lower()
    
    # ==================== 1. 淋巴结特征 ====================
    
    # 1.1 提取所有淋巴结大小（支持多种格式）
    size_patterns = [
        r'(\d+)\s*mm\s*（\s*([左右])\s*）',
        r'(\d+)\s*mm\s*[×x]\s*(\d+)\s*mm',
        r'[大小约]*(\d+)\s*mm',
        r'短径[约为]*(\d+)\s*mm',
        r'短径[约为]*(\d+)\s*mm\s*[~～至]\s*(\d+)\s*mm',
        r'直径[约为]*(\d+)\s*mm\s*[~～]\s*(\d+)\s*mm',
        r'大者[约为]*(\d+)\s*mm\s*[×x]?\s*(\d*)\s*mm?',
    ]
    
    all_sizes = []
    for pattern in size_patterns:
        matches = re.findall(pattern, all_text)
        for match in matches:
            if isinstance(match, tuple):
                if len(match) == 2 and match[1] in ['左', '右']:
                    all_sizes.append(float(match[0]))
                elif len(match) == 2 and match[1].isdigit():
                    all_sizes.append((float(match[0]) + float(match[1])) / 2)
                else:
                    all_sizes.append(float(match[0]))
            else:
                all_sizes.append(float(match))
    
    if all_sizes:
        features['node_max_size'] = max(all_sizes)
        features['node_mean_size'] = np.mean(all_sizes)
        features['node_count'] = len(all_sizes)
        features['node_has_large'] = 1 if max(all_sizes) >= 10 else 0
        features['node_has_very_large'] = 1 if max(all_sizes) >= 20 else 0
    else:
        features['node_max_size'] = np.nan
        features['node_mean_size'] = np.nan
        features['node_count'] = 0
        features['node_has_large'] = 0
        features['node_has_very_large'] = 0
    
    # 1.2 淋巴结位置
    node_regions = {
        'node_rl': r'咽后[间隙]*[见有]*[肿大]*淋巴结',
        'node_ii': r'[双左右]*颈\s*[Ii]{2}\s*区',
        'node_iii': r'[双左右]*颈\s*[Ii]{3}\s*区',
        'node_iv': r'[双左右]*颈\s*[Ii][Vv]\s*区',
        'node_ib': r'[双左右]*颈\s*[Ii][Bb]\s*区',
        'node_ia': r'[双左右]*颈\s*[Ii][Aa]\s*区',
        'node_va': r'[双左右]*颈\s*[Vv][Aa]\s*区',
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
    
    # ==================== 2. 原发肿瘤特征 ====================
    
    # 2.1 肿瘤位置与侵犯范围
    tumor_locations = {
        'tumor_nasopharynx': r'鼻咽[腔部顶壁后壁侧壁]*[见有]*[肿物结节占位增厚]',
        'tumor_left_wall': r'鼻咽左侧壁[见有]*[肿物结节占位增厚]',
        'tumor_right_wall': r'鼻咽右侧壁[见有]*[肿物结节占位增厚]',
        'tumor_posterior_wall': r'鼻咽后壁[见有]*[肿物结节占位增厚]',
        'tumor_roof': r'鼻咽顶[壁后壁]*[见有]*[肿物结节占位增厚]',
        'tumor_fossa_rosenmuller': r'咽隐窝[消失变浅]',
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
    
    # 2.3 肿瘤信号特征
    features['tumor_t1_isointense'] = 1 if 'T1WI呈等信号' in all_text or 'T1W呈等信号' in all_text else 0
    features['tumor_t2_hyperintense'] = 1 if 'T2WI呈稍高信号' in all_text or 'T2W呈稍高信号' in all_text else 0
    features['tumor_enhancement'] = 1 if '增强后明显强化' in all_text or '增强扫描明显强化' in all_text else 0
    
    # ==================== 3. 骨质破坏特征 ====================
    
    bone_sites = {
        'bone_skull_base': r'颅底骨质[破坏信号]',
        'bone_clivus': r'斜坡骨质[破坏信号减低]',
        'bone_pterygoid': r'翼突[基底部]*骨质[破坏信号]',
        'bone_sphenoid': r'蝶骨[基底部]*骨质[破坏信号]',
    }
    for feat_name, pattern in bone_sites.items():
        features[feat_name] = 1 if re.search(pattern, all_text) else 0
    
    features['bone_destruction_new'] = 1 if '骨质破坏' in all_text and '修复' not in all_text else 0
    features['bone_destruction_healing'] = 1 if '骨质破坏.*修复' in all_text or '较前修复' in all_text else 0
    features['bone_signal_abnormal'] = 1 if '骨质信号[减低异常]' in all_text else 0
    
    # ==================== 4. 治疗反应与随访 ====================
    
    features['status_post_rt'] = 1 if '放疗后' in all_text or '放化疗后' in all_text else 0
    features['status_post_ct'] = 1 if '化疗后' in all_text or '放化疗后' in all_text else 0
    
    response_patterns = {
        'response_complete': r'未见[明确]*[肿瘤肿物复发]',
        'response_partial': r'较前[明显]*[缩小好转减轻]',
        'response_stable': r'较前[相仿未见明显变化]',
        'response_progression': r'较前[增大进展]',
        'response_healing': r'较前修复',
    }
    for feat_name, pattern in response_patterns.items():
        features[feat_name] = 1 if re.search(pattern, all_text) else 0
    
    features['current_no_tumor'] = 1 if '未见明确复发' in all_text or '未见肿瘤复发' in all_text else 0
    features['current_mucosal_thickening'] = 1 if '粘膜增厚' in all_text else 0
    features['current_post_radiation_change'] = 1 if '放疗后改变' in all_text else 0
    
    # ==================== 5. 远处转移 ====================
    
    metastasis_sites = {
        'meta_liver': r'肝脏.*[占位性病变肿物]',
        'meta_lung': r'[双]*肺.*[占位结节实质性病变]',
        'meta_bone': r'[颅骨胸骨肋骨椎骨骨盆四肢骨].*[破坏代谢活跃]',
        'meta_adrenal': r'肾上腺[区]*.*[占位性病变]',
    }
    for feat_name, pattern in metastasis_sites.items():
        features[feat_name] = 1 if re.search(pattern, all_text) else 0
    
    features['benign_liver_hemangioma'] = 1 if '肝血管瘤' in all_text else 0
    features['benign_reactive_node'] = 1 if '反应性淋巴结' in all_text else 0
    features['benign_cyst'] = 1 if '囊肿' in all_text else 0
    
    # ==================== 6. 炎症 ====================
    
    sinus_patterns = {
        'sinus_maxillary': r'上颌窦[粘膜增厚炎症]',
        'sinus_ethmoid': r'筛窦[粘膜增厚炎症]',
        'sinus_sphenoid': r'蝶窦[粘膜增厚炎症]',
        'sinus_frontal': r'额窦[粘膜增厚炎症]',
        'sinus_general': r'鼻窦炎',
    }
    for feat_name, pattern in sinus_patterns.items():
        features[feat_name] = 1 if re.search(pattern, all_text) else 0
    
    features['mastoiditis_left'] = 1 if '左侧乳突炎' in all_text else 0
    features['mastoiditis_right'] = 1 if '右侧乳突炎' in all_text else 0
    features['mastoiditis_bilateral'] = 1 if '双侧乳突炎' in all_text else 0
    
    # ==================== 7. EBV ====================
    
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
    
    features['EBV_mentioned'] = 1 if 'EBV' in all_text or 'EB病毒' in all_text else 0
    
    # ==================== 8. PET-CT ====================
    
    suv_matches = re.findall(r'SUV[约为]*(\d+\.?\d*)', all_text)
    if suv_matches:
        suv_values = [float(x) for x in suv_matches]
        features['SUV_max'] = max(suv_values)
        features['SUV_mean'] = np.mean(suv_values)
        features['has_high_metabolism'] = 1 if max(suv_values) > 10 else 0
    else:
        features['SUV_max'] = np.nan
        features['SUV_mean'] = np.nan
        features['has_high_metabolism'] = 0
    
    # ==================== 9. 其他 ====================
    
    followup_matches = re.findall(r'与(\d{4})[-/](\d{1,2})[-/](\d{1,2})[日]*片对比', all_text)
    features['has_prior_comparison'] = 1 if followup_matches else 0
    
    features['image_quality_good'] = 1 if '图像清晰' in all_text else 0
    features['contrast_enhancement_used'] = 1 if '增强' in all_text else 0
    
    return pd.Series(features)

# ==================== 应用特征提取 ====================
print(f"\n正在从文本列提取特征...")
text_features = df.apply(extract_imaging_features_enhanced, axis=1)
df = pd.concat([df, text_features], axis=1)
print(f"提取了 {len(text_features.columns)} 个新特征")

# ==================== 结局定义 ====================
def get_outcome_definition(outcome_type):
    """获取结局定义"""
    definitions = {
        'OS': '总生存期（Overall Survival）：从治疗开始到死亡或末次随访的时间',
        'DFS': '无病生存期（Disease-Free Survival）：从治疗开始到疾病复发或死亡的时间',
        'DMFS': '无远处转移生存期（Distant Metastasis-Free Survival）：从治疗开始到发生远处转移的时间',
        'LRRFS': '无局部区域复发生存期（Locoregional Recurrence-Free Survival）：从治疗开始到原发灶或颈部复发的时间'
    }
    return definitions.get(outcome_type, '')

# ==================== 特征名中文映射 ====================
FEAT_NAME_MAP = {
    'Months_From_Tx_End': '随访时间',
    'age': '年龄',
    'Stage': '临床分期',
    'T': 'T分期',
    'N': 'N分期',
    'EBV_positive': 'EBV阳性',
    'EBV_DNA_load': 'EBV载量',
    'sex': '性别',
    'preLDH': '治疗前LDH',
    'HGB': '血红蛋白',
    'preALB': '治疗前白蛋白',
    'response_progression': '疾病进展',
    'response_complete': '完全缓解',
    'response_partial': '部分缓解',
    'response_stable': '疾病稳定',
    '累积顺铂剂量': '顺铂累积量',
    'node_necrosis': '淋巴结坏死',
    'node_fusion': '淋巴结融合',
    'node_multiple': '多发淋巴结',
    'node_bilateral': '双侧淋巴结',
    'node_heterogeneous': '不均匀强化',
    'node_mean_size': '淋巴结平均径',
    'node_max_size': '淋巴结最大径',
    'invade_skull_base': '颅底侵犯',
    'invade_cavernous_sinus': '海绵窦侵犯',
    'invade_clivus': '斜坡侵犯',
    'meta_liver': '肝转移',
    'meta_bone': '骨转移',
    'meta_lung': '肺转移',
    'ALP': '碱性磷酸酶',
    'LDH': '乳酸脱氢酶',
    'NLR': '中性粒/淋巴细胞比',
    'PLR': '血小板/淋巴细胞比',
    'Phase_Follow-up': '随访阶段',
    'Phase_During-Tx': '治疗中阶段',
    'current_no_tumor': '无瘤状态',
    'smoke': '吸烟',
    'alcohol': '饮酒',
}

# ==================== 计算高危标签 ====================
def calculate_high_risk_labels(row, outcome_type, time_point, time_label):
    """
    根据给定逻辑计算高危标签
    
    逻辑：
    1. 如果发生事件且事件时间 <= 时间点 → 高风险 (1)
    2. 如果发生事件但事件时间 > 时间点 → 低风险 (0)
    3. 如果未发生事件但随访时间 >= 时间点 → 低风险 (0)
    4. 如果未发生事件但随访时间 < 时间点 → 无法判断，设为缺失
    5. 如果事件时间缺失 → 无法判断，设为缺失
    """
    # 事件状态和时间列名
    event_col = outcome_type
    time_col = f"{outcome_type} time"
    follow_time_col = 'Months_From_Tx_End'
    
    # 获取事件状态和时间
    event = row.get(event_col, 0)
    event_time = row.get(time_col, np.nan)
    follow_time = row.get(follow_time_col, np.nan)
    
    # 确保数值类型
    try:
        event = int(event) if pd.notna(event) else 0
    except:
        event = 0
        
    try:
        event_time = float(event_time) if pd.notna(event_time) else np.nan
    except:
        event_time = np.nan
    
    # 计算高风险标签
    col_name = f"{outcome_type}_{time_label}_risk"
    
    # 逻辑判断
    if pd.isna(event_time) or pd.isna(follow_time):
        return col_name, np.nan
    elif event == 1 and event_time - follow_time <= time_point:
        return col_name, 1  # 高风险
    elif event == 1 and event_time - follow_time > time_point:
        return col_name, 0  # 低风险
    # elif event == 0 and event_time - follow_time >= time_point:
    #     return col_name, 0  # 低风险
    # elif event == 0 and event_time - follow_time < time_point:
    #     return col_name, np.nan  # 无法判断
    elif event == 0:
        return col_name, 0  # 低风险（未发生事件且随访时间足够长）
    else:
        return col_name, np.nan
    

# 为所有结局和时间点计算高危标签
print(f"\n正在计算高危标签...")
for outcome in ['OS', 'DFS', 'DMFS', 'LRRFS']:
    for tp, label in zip(TIME_POINTS, TIME_LABELS):
        col_name = f"{outcome}_{label}_risk"
        df[col_name] = df.apply(
            lambda row: calculate_high_risk_labels(row, outcome, tp, label)[1], 
            axis=1
        )
        pos_count = df[col_name].sum()
        valid_count = df[col_name].notna().sum()
        print(f"  {col_name}: 高风险 {pos_count}/{valid_count} ({pos_count/valid_count*100:.1f}%)")

# ==================== 构建样本 ====================
def build_sample(row, eval_format=False, outcome_type='OS', time_point=60, time_label='1y'):
    """
    构建单任务样本，预测特定时间点的高危/低危
    """
    pid = row.patient_sn
    
    # 获取图像文件
    imgs = [str(p) for p in IMAGE_ROOT.glob(f'{pid}*.jpg')] + \
           [str(p) for p in IMAGE_ROOT.glob(f'{pid}*.png')]
    has_img = bool(imgs)
    image_text = "<image>\n\n" if has_img else ""
    
    # 获取高危标签
    risk_col = f"{outcome_type}_{time_label}_risk"
    risk_value = row.get(risk_col, np.nan)
    
    # 如果标签缺失，返回None（后续会过滤）
    if pd.isna(risk_value):
        return None
    
    # 生成标签文本
    risk_text = "是" if risk_value == 1 else "否"
    
    # 关键修复：根据outcome_type选择对应的关键特征
    key_feats = KEY_FEATURES.get(outcome_type, [])
    
    # 生成该结局的关键特征摘要
    summary_parts = []
    for feat in key_feats[:30]:
        if feat in row and pd.notna(row[feat]):
            val = row[feat]
            if isinstance(val, float):
                val_display = f"{val:.2e}" if abs(val) < 0.01 else f"{val:.2f}"
            else:
                val_display = str(val)
            feat_display = FEAT_NAME_MAP.get(feat, feat)
            summary_parts.append(f"{feat_display}: {val_display}")
    
    key_features_text = "；".join(summary_parts) if summary_parts else ""
    
    # 时间点中文描述
    time_desc = {12: '1年', 24: '2年', 36: '3年', 60: '5年'}[time_point]
    
    if eval_format:
        # 验证/测试集：query/response格式
        query_text = (
            f"你是一名鼻咽癌预后预测专家。请预测该患者的 **{outcome_type}** 在{time_desc}内是否为高危。\n\n"
            f"【预测目标】\n{get_outcome_definition(outcome_type)}\n\n"
            f"【关键预后指标】\n{key_features_text}\n\n"
            f"请根据上述关键指标，给出该患者在{time_desc}内{outcome_type}的高危风险预测结果："
            f"是否高危（是/否）。"
        )
        
        sample = {
            "query": query_text,
            "response": f"{risk_text}",
            # 添加time_label用于后续统计
            "time_label": time_label,
            "risk_value": risk_value  # 添加risk_value用于统计
        }
        if has_img:
            sample["images"] = imgs
            
    else:
        # 训练集：messages格式
        system_content = (
            f"你是一名鼻咽癌{outcome_type}预测专家。{get_outcome_definition(outcome_type)}\n"
            f"请根据提供的关键预后指标，预测患者在{time_desc}内是否为高危。"
        )
        
        user_content = (
            f"请预测该患者的 {outcome_type} 在{time_desc}内的高危风险。\n\n"
            f"【关键预后指标】\n{key_features_text}\n\n"
            f"{image_text}"
            f"请给出预测结果：是否高危（是/否）。"
        )
        
        sample = {
            "messages": [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": user_content
                },
                {
                    "role": "assistant",
                    "content": f"{risk_text}"
                }
            ],
            # 添加time_label用于后续统计
            "time_label": time_label,
            "risk_value": risk_value  # 添加risk_value用于统计
        }
        if has_img:
            sample["images"] = imgs
    
    return sample

# ==================== 辅助函数：获取样本的标签值 ====================
def get_sample_label(sample):
    """
    从样本中获取标签值（是/否）
    """
    if 'response' in sample:
        return sample['response']
    else:
        return sample['messages'][-1]['content']

# ==================== 辅助函数：获取样本的风险值 ====================
def get_sample_risk_value(sample):
    """
    从样本中获取风险值（1/0）
    """
    if 'risk_value' in sample:
        return sample['risk_value']
    else:
        label = get_sample_label(sample)
        return 1 if label == '是' else 0

# ==================== 平衡函数（在每个结局内部使用）====================
def balance_samples_by_time(samples_by_time, strategy='uniform', target_ratio=None, 
                           enable_pos_neg_balance=True, pos_neg_ratio=1.0, max_total=None):
    """
    对单个结局内不同时间点的样本进行平衡
    
    Parameters:
    -----------
    samples_by_time: dict
        key为time_label，value为该时间点的样本列表
    strategy: str
        'uniform': 各时间点样本数相同
        'proportional': 按原始比例
        'weighted': 按指定权重
    target_ratio: dict
        各时间点的目标比例
    enable_pos_neg_balance: bool
        是否在每个时间点内进行正负样本平衡
    pos_neg_ratio: float
        正负样本比例
    max_total: int
        最大总样本数
    
    Returns:
    --------
    balanced_samples: list
        平衡后的样本列表
    stats: dict
        统计信息
    """
    # 计算各时间点原始样本数
    time_counts = {label: len(samples) for label, samples in samples_by_time.items()}
    total_samples = sum(time_counts.values())
    
    print(f"\n    时间点平衡前分布:")
    for label in TIME_LABELS:
        samples = samples_by_time[label]
        pos = sum(1 for s in samples if get_sample_label(s) == '是')
        neg = len(samples) - pos
        print(f"      {label}: {len(samples)} (正: {pos}, 负: {neg}, 正比例: {pos/len(samples)*100:.2f}%)")
    
    # 确定各时间点目标样本数
    if strategy == 'uniform':
        # 各时间点样本数相同（以最小的为准）
        min_count = min(time_counts.values())
        target_counts = {label: min_count for label in TIME_LABELS}
        print(f"    均匀采样策略，各时间点目标样本数: {min_count}")
        
    elif strategy == 'proportional':
        # 按原始比例，但可调整总量
        if max_total is not None:
            target_total = min(max_total, total_samples)
        else:
            target_total = total_samples
        
        target_counts = {}
        for label, count in time_counts.items():
            ratio = count / total_samples
            target_counts[label] = int(target_total * ratio)
        print(f"    比例采样策略，目标总数: {target_total}")
        
    elif strategy == 'weighted':
        # 按指定权重
        if target_ratio is None:
            target_ratio = {label: 1/len(TIME_LABELS) for label in TIME_LABELS}
        
        if max_total is not None:
            target_total = max_total
        else:
            target_total = total_samples
        
        target_counts = {}
        for label, ratio in target_ratio.items():
            target_counts[label] = int(target_total * ratio)
        print(f"    加权采样策略，目标总数: {target_total}, 权重: {target_ratio}")
    
    # 对每个时间点进行采样
    balanced_all = []
    time_stats = {}
    
    for label in TIME_LABELS:
        samples = samples_by_time[label]
        target_count = target_counts[label]
        
        if enable_pos_neg_balance and len(samples) > 0:
            # 分离正负样本
            pos_samples = [s for s in samples if get_sample_label(s) == '是']
            neg_samples = [s for s in samples if get_sample_label(s) == '否']
            
            n_pos = len(pos_samples)
            n_neg = len(neg_samples)
            
            # 计算目标正负样本数
            target_pos = int(target_count * pos_neg_ratio / (1 + pos_neg_ratio))
            target_neg = target_count - target_pos
            
            # 采样
            if n_pos > 0:
                if n_pos < target_pos:
                    pos_balanced = np.random.choice(pos_samples, size=target_pos, replace=True)
                elif n_pos > target_pos:
                    pos_balanced = np.random.choice(pos_samples, size=target_pos, replace=False)
                else:
                    pos_balanced = pos_samples
            else:
                pos_balanced = []
            
            if n_neg > 0:
                if n_neg < target_neg:
                    neg_balanced = np.random.choice(neg_samples, size=target_neg, replace=True)
                elif n_neg > target_neg:
                    neg_balanced = np.random.choice(neg_samples, size=target_neg, replace=False)
                else:
                    neg_balanced = neg_samples
            else:
                neg_balanced = []
            
            # 合并
            balanced_time = list(pos_balanced) + list(neg_balanced)
            
            time_stats[label] = {
                'original_pos': n_pos,
                'original_neg': n_neg,
                'original_total': len(samples),
                'balanced_pos': len(pos_balanced),
                'balanced_neg': len(neg_balanced),
                'balanced_total': len(balanced_time)
            }
        else:
            # 不进行正负平衡，直接随机采样
            if len(samples) > target_count:
                balanced_time = np.random.choice(samples, size=target_count, replace=False)
            elif len(samples) < target_count:
                balanced_time = np.random.choice(samples, size=target_count, replace=True)
            else:
                balanced_time = samples
            balanced_time = list(balanced_time)
            
            pos_count = sum(1 for s in balanced_time if get_sample_label(s) == '是')
            time_stats[label] = {
                'original_pos': sum(1 for s in samples if get_sample_label(s) == '是'),
                'original_neg': len(samples) - sum(1 for s in samples if get_sample_label(s) == '是'),
                'original_total': len(samples),
                'balanced_pos': pos_count,
                'balanced_neg': len(balanced_time) - pos_count,
                'balanced_total': len(balanced_time)
            }
        
        balanced_all.extend(balanced_time)
    
    # 打乱顺序
    np.random.shuffle(balanced_all)
    
    print(f"    时间点平衡后分布:")
    for label in TIME_LABELS:
        stats = time_stats[label]
        print(f"      {label}: {stats['balanced_total']} (正: {stats['balanced_pos']}, 负: {stats['balanced_neg']}, "
              f"正比例: {stats['balanced_pos']/stats['balanced_total']*100:.2f}%)")
    
    return balanced_all, time_stats

# ==================== 按任务划分数据集 ====================
print(f"\n{'='*70}")
print(f"按任务（OS/DFS/DMFS/LRRFS）划分数据集")
print(f"{'='*70}\n")

# 获取所有患者
patient_list = df['patient_sn'].unique()

# 8:2划分患者（所有任务共享相同的患者划分）
train_pids, eval_pids = train_test_split(
    patient_list, 
    test_size=VAL_RATIO,  # 0.2
    random_state=SPLIT_SEED
)

print(f"患者数统计:")
print(f"  总患者数: {len(patient_list)}")
print(f"  训练集患者数: {len(train_pids)} ({len(train_pids)/len(patient_list)*100:.1f}%)")
print(f"  测试集患者数: {len(eval_pids)} ({len(eval_pids)/len(patient_list)*100:.1f}%)")

# 按患者划分数据集
train_df_raw = df[df['patient_sn'].isin(train_pids)]
eval_df_raw = df[df['patient_sn'].isin(eval_pids)]

print(f"\n记录数统计:")
print(f"  总记录数: {len(df)}")
print(f"  训练集记录数: {len(train_df_raw)} ({len(train_df_raw)/len(df)*100:.1f}%)")
print(f"  测试集记录数: {len(eval_df_raw)} ({len(eval_df_raw)/len(df)*100:.1f}%)")

# 存储所有分布统计
all_distributions = {}

# ==================== 对每个任务分别处理 ====================
outcomes = ['OS', 'DFS', 'DMFS', 'LRRFS']

for outcome in outcomes:
    print(f"\n{'='*70}")
    print(f"处理任务: {outcome}")
    print(f"{'='*70}")
    
    # ==================== 收集该任务的所有样本（按时间点）====================
    
    # 训练集样本（按时间点收集）
    train_samples_by_time = {label: [] for label in TIME_LABELS}
    
    for _, row in tqdm.tqdm(train_df_raw.iterrows(), total=len(train_df_raw), desc=f'{outcome} train collecting'):
        for tp, label in zip(TIME_POINTS, TIME_LABELS):
            sample = build_sample(row, eval_format=False, 
                                 outcome_type=outcome, 
                                 time_point=tp, 
                                 time_label=label)
            if sample is not None:
                train_samples_by_time[label].append(sample)
    
    # 打印原始分布
    print(f"\n{outcome} 训练集原始分布:")
    total_train_before = 0
    for label in TIME_LABELS:
        samples = train_samples_by_time[label]
        pos = sum(1 for s in samples if get_sample_label(s) == '是')
        neg = len(samples) - pos
        total_train_before += len(samples)
        print(f"  {label}: {len(samples)} (正样本: {pos}, 负样本: {neg}, 正样本比例: {pos/len(samples)*100:.2f}%)")
    
    # ==================== 在结局内部进行平衡 ====================
    
    if ENABLE_TIME_BALANCE or ENABLE_POS_NEG_BALANCE:
        print(f"\n  开始平衡处理:")
        print(f"    时间点平衡: {'启用' if ENABLE_TIME_BALANCE else '禁用'} ({TIME_BALANCE_STRATEGY if ENABLE_TIME_BALANCE else ''})")
        print(f"    正负样本平衡: {'启用' if ENABLE_POS_NEG_BALANCE else '禁用'} (比例: {POS_NEG_RATIO})")
        
        train_samples, time_stats = balance_samples_by_time(
            train_samples_by_time,
            strategy=TIME_BALANCE_STRATEGY if ENABLE_TIME_BALANCE else 'proportional',
            target_ratio=TIME_TARGET_RATIO if ENABLE_TIME_BALANCE else None,
            enable_pos_neg_balance=ENABLE_POS_NEG_BALANCE,
            pos_neg_ratio=POS_NEG_RATIO,
            max_total=MAX_TOTAL_SAMPLES_PER_OUTCOME
        )
    else:
        # 不进行平衡，直接合并
        train_samples = []
        for label in TIME_LABELS:
            train_samples.extend(train_samples_by_time[label])
        np.random.shuffle(train_samples)
    
    # 统计平衡后的分布
    time_counts_train = {label: 0 for label in TIME_LABELS}
    time_pos_neg_train = {label: {'positive': 0, 'negative': 0} for label in TIME_LABELS}
    
    for sample in train_samples:
        label = sample['time_label']
        time_counts_train[label] += 1
        if get_sample_label(sample) == '是':
            time_pos_neg_train[label]['positive'] += 1
        else:
            time_pos_neg_train[label]['negative'] += 1
    
    # 写入训练集文件
    train_file = OUT_DIR / f'{outcome}_train.jsonl'
    with train_file.open('w', encoding='utf-8') as f:
        for sample in train_samples:
            # 移除统计用的字段
            sample_copy = sample.copy()
            sample_copy.pop('time_label', None)
            sample_copy.pop('risk_value', None)
            f.write(json.dumps(sample_copy, ensure_ascii=False) + '\n')
    
    # 保存训练集分布统计
    train_distribution = {}
    for label in TIME_LABELS:
        pos = time_pos_neg_train[label]['positive']
        neg = time_pos_neg_train[label]['negative']
        total = time_counts_train[label]
        train_distribution[label] = {
            'positive': pos,
            'negative': neg,
            'total': total,
            'pos_ratio': pos / total * 100 if total > 0 else 0
        }
    all_distributions[f"{outcome}_train"] = train_distribution
    
    print(f"\n{outcome} 训练集最终统计:")
    print(f"  总样本数: {len(train_samples)}")
    for label in TIME_LABELS:
        count = time_counts_train[label]
        pos = time_pos_neg_train[label]['positive']
        neg = time_pos_neg_train[label]['negative']
        ratio = pos / count * 100 if count > 0 else 0
        print(f"    {label}: {count} (正样本: {pos}, 负样本: {neg}, 正样本比例: {ratio:.2f}%)")
    
    # ==================== 测试集（不进行平衡，只统计）====================
    eval_samples_by_time = {label: [] for label in TIME_LABELS}
    
    for _, row in tqdm.tqdm(eval_df_raw.iterrows(), total=len(eval_df_raw), desc=f'{outcome} eval collecting'):
        for tp, label in zip(TIME_POINTS, TIME_LABELS):
            sample = build_sample(row, eval_format=True, 
                                 outcome_type=outcome, 
                                 time_point=tp, 
                                 time_label=label)
            if sample is not None:
                eval_samples_by_time[label].append(sample)
    
    # 测试集不进行平衡，直接合并
    eval_samples = []
    for label in TIME_LABELS:
        eval_samples.extend(eval_samples_by_time[label])
    
    # 打乱顺序
    np.random.shuffle(eval_samples)
    
    # 统计测试集分布
    time_counts_eval = {label: 0 for label in TIME_LABELS}
    time_pos_neg_eval = {label: {'positive': 0, 'negative': 0} for label in TIME_LABELS}
    
    for sample in eval_samples:
        label = sample['time_label']
        time_counts_eval[label] += 1
        if get_sample_label(sample) == '是':
            time_pos_neg_eval[label]['positive'] += 1
        else:
            time_pos_neg_eval[label]['negative'] += 1
    
    # 写入测试集文件
    eval_file = OUT_DIR / f'{outcome}_eval.jsonl'
    with eval_file.open('w', encoding='utf-8') as f:
        for sample in eval_samples:
            # 移除统计用的字段
            sample_copy = sample.copy()
            sample_copy.pop('time_label', None)
            sample_copy.pop('risk_value', None)
            f.write(json.dumps(sample_copy, ensure_ascii=False) + '\n')
    
    # 保存测试集分布统计
    eval_distribution = {}
    for label in TIME_LABELS:
        pos = time_pos_neg_eval[label]['positive']
        neg = time_pos_neg_eval[label]['negative']
        total = time_counts_eval[label]
        eval_distribution[label] = {
            'positive': pos,
            'negative': neg,
            'total': total,
            'pos_ratio': pos / total * 100 if total > 0 else 0
        }
    all_distributions[f"{outcome}_eval"] = eval_distribution
    
    print(f"\n{outcome} 测试集统计:")
    print(f"  总样本数: {len(eval_samples)}")
    for label in TIME_LABELS:
        count = time_counts_eval[label]
        pos = time_pos_neg_eval[label]['positive']
        neg = time_pos_neg_eval[label]['negative']
        ratio = pos / count * 100 if count > 0 else 0
        print(f"    {label}: {count} (正样本: {pos}, 负样本: {neg}, 正样本比例: {ratio:.2f}%)")

# ==================== 输出所有数据集的正负样本比例汇总表 ====================
print(f"\n{'='*90}")
print("所有数据集正负样本比例汇总")
print(f"{'='*90}")
print(f"{'数据集':<12} {'时间点':<6} {'正样本':<8} {'负样本':<8} {'总计':<8} {'正样本比例':<12} {'类型':<10}")
print("-" * 90)

for outcome in outcomes:
    # 训练集
    train_dist = all_distributions[f"{outcome}_train"]
    for label in TIME_LABELS:
        stats = train_dist[label]
        print(f"{outcome+'_train':<12} {label:<6} {stats['positive']:<8} {stats['negative']:<8} "
              f"{stats['total']:<8} {stats['pos_ratio']:>6.2f}%    {'训练集':<10}")
    
    # 测试集
    eval_dist = all_distributions[f"{outcome}_eval"]
    for label in TIME_LABELS:
        stats = eval_dist[label]
        print(f"{outcome+'_eval':<12} {label:<6} {stats['positive']:<8} {stats['negative']:<8} "
              f"{stats['total']:<8} {stats['pos_ratio']:>6.2f}%    {'测试集':<10}")

print(f"{'='*90}")

# 额外输出每个任务的整体统计（跨时间点汇总）
print(f"\n{'='*70}")
print("各任务整体统计（跨时间点汇总）")
print(f"{'='*70}")
print(f"{'任务':<8} {'数据集':<10} {'总正样本':<10} {'总负样本':<10} {'总计':<10} {'正样本比例':<12}")
print("-" * 70)

for outcome in outcomes:
    # 训练集整体
    train_dist = all_distributions[f"{outcome}_train"]
    total_pos_train = sum(train_dist[label]['positive'] for label in TIME_LABELS)
    total_neg_train = sum(train_dist[label]['negative'] for label in TIME_LABELS)
    total_train = total_pos_train + total_neg_train
    ratio_train = total_pos_train / total_train * 100 if total_train > 0 else 0
    print(f"{outcome:<8} {'训练集':<10} {total_pos_train:<10} {total_neg_train:<10} "
          f"{total_train:<10} {ratio_train:>6.2f}%")
    
    # 测试集整体
    eval_dist = all_distributions[f"{outcome}_eval"]
    total_pos_eval = sum(eval_dist[label]['positive'] for label in TIME_LABELS)
    total_neg_eval = sum(eval_dist[label]['negative'] for label in TIME_LABELS)
    total_eval = total_pos_eval + total_neg_eval
    ratio_eval = total_pos_eval / total_eval * 100 if total_eval > 0 else 0
    print(f"{outcome:<8} {'测试集':<10} {total_pos_eval:<10} {total_neg_eval:<10} "
          f"{total_eval:<10} {ratio_eval:>6.2f}%")
    print("-" * 70)

print(f"{'='*70}")

# ==================== 生成 dataset_info.json ====================
info = []

for outcome in outcomes:
    train_file = OUT_DIR / f'{outcome}_train.jsonl'
    eval_file = OUT_DIR / f'{outcome}_eval.jsonl'
    
    info.append({
        "dataset_path": str(train_file.resolve()),
        "dataset_name": f"medical_{outcome}_train",
        "subsets": ["train"],
        "split": ["train"],
        "columns": {}
    })
    
    info.append({
        "dataset_path": str(eval_file.resolve()),
        "dataset_name": f"medical_{outcome}_eval",
        "subsets": ["eval"],
        "split": ["eval"],
        "columns": {}
    })

(OUT_DIR / 'dataset_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2))

print(f'\n{"="*70}')
print('完成！按任务划分的数据集已生成')
print(f'{"="*70}')
print(f'\n输出文件:')
for outcome in outcomes:
    print(f'  {outcome}_train.jsonl: {outcome}任务的训练集（messages格式）')
    print(f'  {outcome}_eval.jsonl: {outcome}任务的测试集（query/response格式）')
print(f'  dataset_info.json: 数据集元信息')
print(f'\n关键特性:')
print(f'  - 训练:验证 = 8:2（所有任务共享相同的患者划分）')
print(f'  - 每个任务独立的数据集文件')
print(f'  - 在每个任务内部进行平衡:')
print(f'    * 时间点平衡: {ENABLE_TIME_BALANCE} ({TIME_BALANCE_STRATEGY})')
print(f'    * 正负样本平衡: {ENABLE_POS_NEG_BALANCE} (比例: {POS_NEG_RATIO})')
print(f'  - 每个样本预测特定任务在特定时间点是否高危')
print(f'  - 输出格式: "是/否"')
print(f'  - 高危判断逻辑基于事件时间和随访时间')