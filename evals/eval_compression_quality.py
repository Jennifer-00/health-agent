"""评估 search_rag 压缩质量 —— 按用户的信息保留标准逐条标注"""
import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from agent.tool_result_compressor import compress_tool_result, _split_sentences

# ═══════════════════════════════════════════════════════════════════════════
# 3 组测试样本
# ═══════════════════════════════════════════════════════════════════════════

MUST_KEEP_WORDS = ['过敏', '禁忌', '不建议', '禁用', '慎用', '避免', '警告']
NUMERIC_PAT = re.compile(r'(\d+\.?\d*\s*(mg|g|ml|μg|单位|片|粒|次/日|bid|tid|qd|mmHg|%)?)')
EMERGENCY_WORDS = ['立即就医', '紧急就医', '应停药', '拨打120', '急诊', '危及生命', '病死率']
CONDITION_WORDS = ['eGFR', '合并', '≥', '≤', '<', '>', '以下', '以上', '时应', '者应', '禁用', '指征']
CONCLUSION_WORDS = ['推荐', '应', '需', '必须', '首选', '一线', '可', '建议', '目标', '结论']
BACKGROUND_WORDS = ['机制', '研究', '试验', '显示', '证实', '通路', '激活', '信号', '报道']
FILLER_WORDS = ['此外', '另外', '值得', '注意', '值得关注', '不仅如此']


def classify(sentence):
    ci = any(w in sentence for w in MUST_KEEP_WORDS)
    nu = bool(NUMERIC_PAT.search(sentence))
    em = any(w in sentence for w in EMERGENCY_WORDS)
    co = any(w in sentence for w in CONDITION_WORDS)
    cl = any(w in sentence for w in CONCLUSION_WORDS)
    bg = any(w in sentence for w in BACKGROUND_WORDS)
    fi = any(w in sentence for w in FILLER_WORDS)
    return cl, ci, nu, co, em, bg, fi


def tag_str(cl, ci, nu, co, em, bg, fi):
    tags = []
    if ci: tags.append('[禁忌/过敏]')
    if em: tags.append('[就医警示]')
    if nu: tags.append('[数值]')
    if co: tags.append('[适用条件]')
    if cl: tags.append('[结论]')
    if bg: tags.append('[背景]')
    if fi: tags.append('[铺垫]')
    return ' '.join(tags) if tags else '[无标签]'


def must_keep(cl, ci, nu, co, em, bg, fi):
    """是否包含必须保留的信息"""
    return ci or em or co


def is_background(cl, ci, nu, co, em, bg, fi):
    return bg or fi


def mk_rag_sample(chunks):
    return '\n\n'.join([
        '【结果 %d】来源：%s\n章节：%s\n内容：%s' % (i+1, src, sec, content)
        for i, (src, sec, content) in enumerate(chunks)
    ])


# ── Case A: 药物禁忌/过敏 ─────────────────────────────────────────────────
case_a_chunks = [
    ('药品说明书-格华止', '禁忌症',
     '对盐酸二甲双胍或任何辅料过敏者绝对禁用。'
     '中重度肾功能不全（eGFR低于45ml/min/1.73m²）者禁用。'
     '急性或慢性代谢性酸中毒患者禁用。严重肝功能不全、急性酒精中毒者禁用。'
     '碘化造影剂检查前48小时至检查后48小时应暂停服用二甲双胍，复查肾功能正常后方可恢复。'
     '相对禁忌：eGFR在45-59ml/min/1.73m²之间的患者需评估风险获益比后减量使用。'
     '妊娠期及哺乳期妇女不推荐使用。不推荐用于10岁以下儿童。'),
    ('临床用药指南', '内分泌 > 口服降糖药 > 二甲双胍 禁忌',
     '二甲双胍最严重的风险是乳酸性酸中毒，虽然发生率极低（约3-9例/10万患者年），但病死率高达30%-50%。'
     '乳酸性酸中毒的高危因素包括：严重肾功能不全（eGFR<30）、肝功能衰竭、'
     '急性失代偿性心力衰竭、严重感染和脓毒症、组织缺氧状态。'
     '一旦出现疑似症状如嗜睡、肌痛、呼吸困难、腹痛伴严重乏力等，应立即停药并紧急就医。'),
    ('临床药理学-代谢性疾病篇', '双胍类药物 > 作用机制',
     '二甲双胍的分子机制涉及AMPK信号通路的激活，通过抑制线粒体呼吸链复合体I减少ATP合成，进而激活AMPK。'
     'AMPK活化后磷酸化下游靶蛋白，抑制肝脏糖异生关键酶PEPCK和G6Pase的表达。'
     '此外二甲双胍还通过增加GLP-1分泌和调节肠道菌群组成发挥降糖外获益。'
     '这些机制共同构成了二甲双胍降糖、减重、改善胰岛素抵抗的药理学基础。'),
    ('医院处方集-内分泌科', '口服降糖药处方规范',
     '二甲双胍片500mg po bid（起始剂量500mg qd，根据胃肠道耐受情况在1-2周内逐渐增加）。'
     '最大推荐剂量：普通片2000mg/日，分2-3次服用；缓释片2000mg/日，每日一次。'
     '服用方法：随餐或餐后立即服用以减少胃肠道刺激。'
     '处方前需评估肾功能，eGFR<45者应停用，eGFR 45-59者最大剂量不超过1000mg/日。'),
    ('二甲双胍药物相互作用手册', '联合用药安全性',
     '二甲双胍与碘造影剂联合使用会增加急性肾损伤风险，必须在造影前后48小时暂停二甲双胍。'
     '与西咪替丁合用时，西咪替丁可竞争性抑制肾小管分泌二甲双胍，导致二甲双胍血药浓度升高约40%。'
     '酒精可增强二甲双胍对乳酸代谢的抑制作用，显著增加乳酸性酸中毒的风险，应避免过量饮酒。'
     '二甲双胍与华法林、地高辛等药物无明显相互作用，联合使用时无需调整剂量。'),
]

# ── Case B: 剂量/数值类 ──────────────────────────────────────────────────
case_b_chunks = [
    ('高血压合理用药指南2023', '钙通道阻滞剂 > 氨氯地平',
     '氨氯地平为长效二氢吡啶类钙通道阻滞剂，半衰期长达35-50小时，每日一次即可平稳控制24小时血压。'
     '起始剂量为5mg每日一次，根据血压控制情况在2-4周后可调整至10mg每日一次。'
     '老年患者或肝功能不全者起始剂量应减至2.5mg每日一次。'
     '一般治疗目标为血压<140/90mmHg，合并糖尿病或慢性肾病者应控制在<130/80mmHg。'
     '氨氯地平降低收缩压幅度约为15-20mmHg。'),
    ('氨氯地平药品说明书-络活喜', '用法用量',
     '成人：治疗高血压起始剂量5mg，每日一次，最大剂量10mg每日一次。'
     '治疗慢性稳定性心绞痛的推荐剂量为5-10mg，每日一次。'
     '老年患者：建议起始剂量2.5mg每日一次。肝功能不全患者：起始剂量2.5mg每日一次。'
     '肾功能不全患者无需调整剂量。本品可单独使用或与其他抗高血压药物联合使用。'),
    ('钙通道阻滞剂的药理学基础', '二氢吡啶类 > 作用机制',
     '氨氯地平通过选择性抑制血管平滑肌细胞膜上的L型钙通道，阻断钙离子内流，'
     '从而扩张外周小动脉和冠状动脉，降低外周血管阻力，产生降压和抗心绞痛作用。'
     '不同于短效CCB，氨氯地平起效缓慢且平稳，避免了快速血管扩张引起的反射性交感神经激活，'
     '因此在降压过程中心率变化不明显，尤其适合心率偏慢的老年高血压患者。'
     '氨氯地平还具有一定的抗氧化和抗动脉粥样硬化作用，可能与其改善内皮功能有关。'),
    ('中国老年高血压管理专家共识', '特殊人群降压治疗',
     '老年高血压患者降压治疗应遵循个体化原则，起始小剂量，缓慢加量，'
     '避免血压快速下降引起脑灌注不足。'
     '65-79岁患者血压控制目标为<140/90mmHg，能够耐受者可进一步降至<130/80mmHg。'
     '80岁以上高龄老人收缩压控制目标可适当放宽至<150mmHg。'
     '氨氯地平在老年患者中表现出良好的安全性和耐受性，'
     '长期使用不增加电解质紊乱和糖脂代谢异常的风险。'),
    ('难治性高血压治疗策略', '联合用药方案',
     '血压未达标的患者推荐联合用药，A+C（ACEI/ARB + CCB）是中国高血压患者最常用的联合方案。'
     '对于氨氯地平5mg单药治疗4周后血压仍未达标的患者，建议加用ACEI或ARB类药物。'
     '氨氯地平10mg+缬沙坦160mg的固定复方制剂在中国人群中显示出良好的降压效果。'
     '三联疗法的选择应基于患者的合并症、靶器官损害和药物耐受情况进行个体化制定。'),
]

# ── Case C: 不良反应/就医警示 ─────────────────────────────────────────────
case_c_chunks = [
    ('SGLT2抑制剂安全性专家共识', '不良反应管理 > 酮症酸中毒',
     'SGLT2抑制剂相关的酮症酸中毒（euglycemic DKA）是临床需要高度警惕的严重不良反应。'
     '与非SGLT2相关的DKA不同，此类DKA发生时血糖可能并不显著升高（常<250mg/dL），容易被漏诊。'
     '高危因素包括：手术、严重感染、长时间禁食、大量饮酒、胰岛素减量或停用、胰腺炎等。'
     '一旦出现恶心、呕吐、腹痛、乏力、呼吸困难等症状，患者应立即停药并紧急就医。'
     '就医时应主动告知接诊医生正在使用达格列净，以便医生排查euglycemic DKA。'),
    ('达格列净药品说明书-安达唐', '不良反应',
     '常见不良反应：生殖器真菌感染（发生率约5-8%，女性多于男性）、泌尿系感染（发生率约4-6%）。'
     '低血压和脱水：由于渗透性利尿作用，可能导致血容量不足，表现为低血压、头晕、口干等，'
     '尤其在老年患者、肾功能不全患者、正在使用利尿剂的患者中更易发生。'
     '罕见但严重的不良反应包括：酮症酸中毒（含血糖正常的酮症酸中毒）、急性肾损伤、'
     '会阴部坏死性筋膜炎（Fournier坏疽，虽极罕见但可能危及生命）。'
     '出现生殖器或会阴部疼痛、压痛、红斑、肿胀且伴有发热或不适感应立即就医。'),
    ('达格列净的心血管结局研究', 'DECLARE-TIMI 58',
     'DECLARE-TIMI 58研究是一项多中心随机双盲安慰剂对照临床试验，纳入17160例2型糖尿病患者。'
     '结果显示达格列净组较安慰剂组心血管死亡或心衰住院的复合终点降低17%'
     '（HR 0.83, 95%CI 0.73-0.95）。'
     '肾脏复合终点（eGFR持续下降≥40%、终末期肾病、肾脏死亡）降低47%（HR 0.53, 0.43-0.66）。'
     '亚组分析显示，无论患者基线有无心血管病史或心力衰竭史，'
     '达格列净均显示出一致的心肾保护效应。'),
    ('SGLT2抑制剂药物警戒年度报告2024', '中国不良反应监测',
     '截至2024年底，我国SGLT2抑制剂相关不良反应报告累计超过15000份。'
     '其中生殖器感染占比最高（约35%），其次为泌尿系感染（约18%）、低血压/脱水（约12%）。'
     '酮症酸中毒报告约400例，其中约15%为血糖正常的酮症酸中毒，'
     '提示临床对此类DKA的认知仍需加强。'
     'Fournier坏疽报告不足10例，但病死率极高，需强调早期识别和紧急处理的重要性。'),
    ('慢性肾病合并糖尿病用药指南', '肾功能调整',
     'eGFR≥45ml/min/1.73m²时，达格列净可按标准剂量10mg每日一次使用，无需调整。'
     'eGFR 25-44时，不建议新起始达格列净，但已在使用中的患者可继续服用，'
     '降糖效果可能减弱但心肾保护效应依然存在。'
     'eGFR<25时不应使用达格列净。使用期间如eGFR持续下降应评估原因，'
     '但轻度一过性eGFR下降（用药初期）是血流动力学效应所致，通常无需停药，反而提示药物起效。'
     '应每3-6个月监测肾功能。'),
]

CASES = [
    ('A: 药物禁忌/过敏', '二甲双胍 eGFR 禁忌 造影剂 过敏', case_a_chunks),
    ('B: 剂量/数值', '氨氯地平 降压 剂量 血压目标 老年', case_b_chunks),
    ('C: 不良反应/就医警示', '达格列净 SGLT2 副作用 酮症酸中毒 就医', case_c_chunks),
]


def evaluate_case(name, query, chunks):
    sample = mk_rag_sample(chunks)
    compressed = compress_tool_result('search_rag', sample, query=query)

    print('=' * 100)
    print('Case %s  |  query: %s' % (name, query))
    print('=' * 100)

    # 解析压缩结果中的 kept chunks
    kept_info = []
    for m in re.finditer(
        r'(\d+)\. source: (.+?)\n   section: (.+?)\n   key_points:\n((?:   - .+\n?)+)',
        compressed
    ):
        idx = int(m.group(1)) - 1
        kps = re.findall(r'   - (.+)', m.group(4))
        kept_info.append((idx, kps))

    kept_indices = {k[0] for k in kept_info}

    stats = {'must_dropped': 0, 'must_total': 0, 'bg_dropped': 0, 'bg_total': 0}

    for i, (src, sec, content) in enumerate(chunks):
        sentences = _split_sentences(content)
        kept_kps = []
        for ki, kps in kept_info:
            if ki == i:
                kept_kps = kps
                break

        kept = i in kept_indices
        symbol = '+' if kept else '-'
        status = 'KEPT  ' if kept else 'DROPPED'

        print()
        print('  Chunk %d [%s] %s  src=%s' % (i+1, symbol, status, src[:50]))
        print('        section: %s' % sec[:70])

        if not kept:
            # 检查被丢弃的 chunk 是否有信息丢失
            for sent in sentences:
                cl, ci, nu, co, em, bg, fi = classify(sent)
                if must_keep(cl, ci, nu, co, em, bg, fi):
                    print('        ** LOST must-keep [%s]: %s' % (tag_str(cl, ci, nu, co, em, bg, fi), sent[:100]))
                    stats['must_dropped'] += 1
                stats['must_total'] += 1
                if is_background(cl, ci, nu, co, em, bg, fi):
                    stats['bg_dropped'] += 1
                    stats['bg_total'] += 1
        else:
            kept_set = set(kept_kps)
            print('        key_points (%d/%d sentences kept):' % (len(kept_kps), len(sentences)))
            for kp in kept_kps:
                cl, ci, nu, co, em, bg, fi = classify(kp)
                print('          %-55s  %s' % (tag_str(cl, ci, nu, co, em, bg, fi)[:55], kp[:80]))
            # 检查被丢弃的句子中是否有必须保留的信息
            for sent in sentences:
                if sent not in kept_set:
                    cl, ci, nu, co, em, bg, fi = classify(sent)
                    if must_keep(cl, ci, nu, co, em, bg, fi):
                        print('        ** LOST must-keep [%s]: %s' % (tag_str(cl, ci, nu, co, em, bg, fi), sent[:100]))
                        stats['must_dropped'] += 1
                    stats['must_total'] += 1
                    if is_background(cl, ci, nu, co, em, bg, fi):
                        stats['bg_dropped'] += 1
                        stats['bg_total'] += 1
                else:
                    cl, ci, nu, co, em, bg, fi = classify(sent)
                    stats['must_total'] += (1 if must_keep(cl, ci, nu, co, em, bg, fi) else 0)

    # 压缩输出
    print()
    print('  --- Compressed output ---')
    for line in compressed.split('\n')[:40]:
        print('  | %s' % line)

    must_retain = 100 - stats['must_dropped'] / max(stats['must_total'], 1) * 100
    bg_drop = stats['bg_dropped'] / max(stats['bg_total'], 1) * 100

    print()
    print('  >>> Must-keep retention: %.0f%% (%d lost / %d total must-keep sentences)'
          % (must_retain, stats['must_dropped'], stats['must_total']))
    print('  >>> Background/filler dropped: %.0f%% (%d dropped / %d total bg sentences)'
          % (bg_drop, stats['bg_dropped'], stats['bg_total']))
    return must_retain, bg_drop


if __name__ == '__main__':
    for name, query, chunks in CASES:
        mr, bd = evaluate_case(name, query, chunks)
        print()

    print('=' * 100)
    print('Evaluation complete.')
    print('=' * 100)
