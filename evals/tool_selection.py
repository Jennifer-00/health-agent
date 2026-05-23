"""
evals/tool_selection.py — Agent 工具选择质量评测。

评测维度：
  1. Exact Match     : 实际调用工具集合 == 期望集合
  2. Under-call      : 该调没调（漏调）
  3. Over-call       : 不该调却调了（误调）
  4. Wrong-tool      : 调了工具但调错了
  5. Param relevance : query 参数与期望关键词的重叠率

结果默认保存至 evals/results/tool_selection_<timestamp>.json

运行：
  python evals/tool_selection.py
  python evals/tool_selection.py --concurrency 3
  python evals/tool_selection.py --output evals/results/my_run.json
"""
import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
load_dotenv(override=True)


# ── 测试用例定义 ───────────────────────────────────────────────────────────────

@dataclass
class Case:
    id: int
    message: str
    expected_tools: list[str]          # [] 表示不应调任何工具
    param_keywords: list[str] = field(default_factory=list)  # 期望出现在 query 里的关键词
    history: list[dict] = field(default_factory=list)         # 多轮上下文


CASES: list[Case] = [

    # ════════════════════════════════════════════════════════════════════════
    # search_memory（50 条）
    # 触发条件：用户提到"上次/之前/又/还是/老是/经常/一直"等过去时态词
    # ════════════════════════════════════════════════════════════════════════

    # 明确过去引用 (1-15)
    Case(1,  "我上次说的头痛有没有好转", ["search_memory"], ["头痛"]),
    Case(2,  "之前提到的失眠问题还有记录吗", ["search_memory"], ["失眠"]),
    Case(3,  "我上次量的血压数值是多少", ["search_memory"], ["血压"]),
    Case(4,  "之前你说我缺维生素D，现在补了有没有改善", ["search_memory"], ["维生素D"]),
    Case(5,  "我上次就诊说的是什么病", ["search_memory"], ["就诊"]),
    Case(6,  "之前我说过对哪些药过敏", ["search_memory"], ["过敏", "药"]),
    Case(7,  "我上次记录的体重是多少", ["search_memory"], ["体重"]),
    Case(8,  "之前提到的那个皮疹后来怎么了", ["search_memory"], ["皮疹"]),
    Case(9,  "我上次说肚子痛，有记录吗", ["search_memory"], ["腹痛"]),
    Case(10, "之前我提到过心跳加速的问题", ["search_memory"], ["心跳"]),
    Case(11, "我上次说的那个医生的建议你记得吗", ["search_memory"], ["医生建议"]),
    Case(12, "之前记录过我的血糖情况吗", ["search_memory"], ["血糖"]),
    Case(13, "我上次说有点头晕，后来有没有好", ["search_memory"], ["头晕"]),
    Case(14, "之前你帮我记录了用药时间吗", ["search_memory"], ["用药"]),
    Case(15, "我上次提到的那个过敏症状是什么", ["search_memory"], ["过敏"]),

    # 反复/持续症状 (16-30)
    Case(16, "我又开始头痛了，跟上次一样", ["search_memory"], ["头痛"]),
    Case(17, "我还是睡不好，这已经是第几次了", ["search_memory"], ["睡眠", "失眠"]),
    Case(18, "老是胃痛，你觉得是什么原因", ["search_memory"], ["胃痛"]),
    Case(19, "我经常头晕，之前有类似记录吗", ["search_memory"], ["头晕"]),
    Case(20, "我一直有反酸的问题，有记录吗", ["search_memory"], ["反酸"]),
    Case(21, "我又便秘了，跟之前一样", ["search_memory"], ["便秘"]),
    Case(22, "总是感觉疲劳，之前说过吗", ["search_memory"], ["疲劳"]),
    Case(23, "我还是腰痛，这次更严重了", ["search_memory"], ["腰痛"]),
    Case(24, "我反复出现口腔溃疡，有没有规律", ["search_memory"], ["口腔溃疡"]),
    Case(25, "老是觉得心跳快，之前有记录过吗", ["search_memory"], ["心跳"]),
    Case(26, "我一直有点干咳，你记得我提过吗", ["search_memory"], ["咳嗽"]),
    Case(27, "又开始过敏了，上次是什么季节", ["search_memory"], ["过敏"]),
    Case(28, "我还是有点焦虑，最近有没有加重", ["search_memory"], ["焦虑"]),
    Case(29, "经常手脚冰凉，这个之前提过吗", ["search_memory"], ["手脚冰凉"]),
    Case(30, "我又发烧了，上次是几度", ["search_memory"], ["发烧"]),

    # 用药历史 (31-40)
    Case(31, "我一直在吃的那个降压药叫什么名字", ["search_memory"], ["降压药"]),
    Case(32, "之前开的止痛药还有多少剂量", ["search_memory"], ["止痛药"]),
    Case(33, "我上次吃的抗生素是哪种", ["search_memory"], ["抗生素"]),
    Case(34, "之前记录的用药剂量对吗", ["search_memory"], ["用药", "剂量"]),
    Case(35, "我一直在补的那个保健品叫什么", ["search_memory"], ["保健品"]),
    Case(36, "上次医生给我开的药我忘记名字了", ["search_memory"], ["药", "医生"]),
    Case(37, "之前我说对哪种抗生素过敏来着", ["search_memory"], ["抗生素", "过敏"]),
    Case(38, "我一直在吃的二甲双胍是多少毫克", ["search_memory"], ["二甲双胍"]),
    Case(39, "之前记录了我什么时候开始服药吗", ["search_memory"], ["服药"]),
    Case(40, "我上次说换了药之后有没有副作用记录", ["search_memory"], ["换药", "副作用"]),

    # 跟进建议与对比 (41-50)
    Case(41, "上次你建议我多喝水，我照做了，有没有改善", ["search_memory"], ["建议"]),
    Case(42, "之前说的那个饮食方案你还记得吗", ["search_memory"], ["饮食"]),
    Case(43, "我之前的睡眠质量比现在好，是什么时候开始变差的", ["search_memory"], ["睡眠"]),
    Case(44, "上次运动后肌肉酸痛，这次也一样", ["search_memory"], ["肌肉酸痛", "运动"]),
    Case(45, "之前我的体检指标有异常吗", ["search_memory"], ["体检"]),
    Case(46, "我以前血压控制得挺好的，最近怎么又高了", ["search_memory"], ["血压"]),
    Case(47, "上次你建议我减少盐分摄入，我坚持了两周了", ["search_memory"], ["饮食", "盐"]),
    Case(48, "之前记录的我的心率正常范围是多少", ["search_memory"], ["心率"]),
    Case(49, "我以前有过低血糖的记录吗", ["search_memory"], ["低血糖"]),
    Case(50, "上次感冒之后咳嗽了多久才好的", ["search_memory"], ["感冒", "咳嗽"]),


    # ════════════════════════════════════════════════════════════════════════
    # search_rag（55 条）
    # 触发条件：药物说明/副作用/相互作用、疾病描述/诊断/治疗、临床指南
    # ════════════════════════════════════════════════════════════════════════

    # 药物副作用 (51-62)
    Case(51, "布洛芬有哪些常见副作用", ["search_rag"], ["布洛芬", "副作用"]),
    Case(52, "二甲双胍长期服用有什么风险", ["search_rag"], ["二甲双胍", "副作用"]),
    Case(53, "阿莫西林的不良反应有哪些", ["search_rag"], ["阿莫西林", "副作用"]),
    Case(54, "氨氯地平副作用严重吗", ["search_rag"], ["氨氯地平", "副作用"]),
    Case(55, "阿托伐他汀会导致肌肉酸痛吗", ["search_rag"], ["阿托伐他汀", "副作用"]),
    Case(56, "奥美拉唑长期吃有什么影响", ["search_rag"], ["奥美拉唑", "副作用"]),
    Case(57, "左旋甲状腺素副作用是什么", ["search_rag"], ["左旋甲状腺素"]),
    Case(58, "氯雷他定会让人犯困吗", ["search_rag"], ["氯雷他定", "副作用"]),
    Case(59, "沙丁胺醇的副作用有哪些", ["search_rag"], ["沙丁胺醇"]),
    Case(60, "华法林有什么出血风险", ["search_rag"], ["华法林", "出血"]),
    Case(61, "对乙酰氨基酚过量会怎样", ["search_rag"], ["对乙酰氨基酚", "过量"]),
    Case(62, "甲硝唑为什么不能喝酒", ["search_rag"], ["甲硝唑", "酒精"]),

    # 药物相互作用 (63-72)
    Case(63, "布洛芬和阿司匹林可以一起吃吗", ["search_rag"], ["布洛芬", "阿司匹林"]),
    Case(64, "华法林和阿司匹林同时用安全吗", ["search_rag"], ["华法林", "阿司匹林"]),
    Case(65, "二甲双胍和酒精有相互作用吗", ["search_rag"], ["二甲双胍", "酒精"]),
    Case(66, "辛伐他汀和葡萄柚有什么冲突", ["search_rag"], ["辛伐他汀", "葡萄柚"]),
    Case(67, "氨氯地平和地高辛一起用有风险吗", ["search_rag"], ["氨氯地平", "地高辛"]),
    Case(68, "抗抑郁药和布洛芬可以同服吗", ["search_rag"], ["抗抑郁药", "布洛芬"]),
    Case(69, "左旋甲状腺素和钙片要隔多久吃", ["search_rag"], ["左旋甲状腺素", "钙片"]),
    Case(70, "止咳药和抗过敏药能一起吃吗", ["search_rag"], ["止咳药", "抗过敏"]),
    Case(71, "头孢和牛奶有相互作用吗", ["search_rag"], ["头孢", "牛奶"]),
    Case(72, "降压药和咖啡因有影响吗", ["search_rag"], ["降压药", "咖啡因"]),

    # 药物用法/剂量 (73-82)
    Case(73, "布洛芬成人一次吃多少合适", ["search_rag"], ["布洛芬", "剂量"]),
    Case(74, "阿莫西林一天吃几次", ["search_rag"], ["阿莫西林", "用法"]),
    Case(75, "二甲双胍应该饭前还是饭后吃", ["search_rag"], ["二甲双胍", "用法"]),
    Case(76, "头孢克肟儿童剂量怎么算", ["search_rag"], ["头孢克肟", "剂量", "儿童"]),
    Case(77, "奥美拉唑什么时候吃效果最好", ["search_rag"], ["奥美拉唑", "用法"]),
    Case(78, "维生素C一天最多吃多少", ["search_rag"], ["维生素C", "剂量"]),
    Case(79, "退烧药多久吃一次", ["search_rag"], ["退烧药", "用法"]),
    Case(80, "阿司匹林肠溶片要嚼碎吗", ["search_rag"], ["阿司匹林", "用法"]),
    Case(81, "胰岛素注射部位怎么选", ["search_rag"], ["胰岛素", "注射"]),
    Case(82, "孕妇可以用扑热息痛吗", ["search_rag"], ["对乙酰氨基酚", "孕妇"]),

    # 疾病描述/诊断标准 (83-95)
    Case(83, "高血压的诊断标准是什么", ["search_rag"], ["高血压", "诊断"]),
    Case(84, "2型糖尿病和1型有什么区别", ["search_rag"], ["糖尿病", "分型"]),
    Case(85, "甲亢的主要症状是什么", ["search_rag"], ["甲亢", "症状"]),
    Case(86, "痛风的诊断需要做什么检查", ["search_rag"], ["痛风", "诊断"]),
    Case(87, "哮喘发作时有哪些表现", ["search_rag"], ["哮喘", "症状"]),
    Case(88, "幽门螺杆菌感染有什么症状", ["search_rag"], ["幽门螺杆菌", "症状"]),
    Case(89, "脂肪肝分几个等级", ["search_rag"], ["脂肪肝", "分级"]),
    Case(90, "类风湿关节炎怎么诊断", ["search_rag"], ["类风湿", "诊断"]),
    Case(91, "冠心病的早期信号有哪些", ["search_rag"], ["冠心病", "症状"]),
    Case(92, "肾结石有什么典型症状", ["search_rag"], ["肾结石", "症状"]),
    Case(93, "贫血的诊断标准血红蛋白是多少", ["search_rag"], ["贫血", "诊断"]),
    Case(94, "骨质疏松如何早期发现", ["search_rag"], ["骨质疏松", "诊断"]),
    Case(95, "甲减有什么表现", ["search_rag"], ["甲减", "症状"]),

    # 治疗方案/临床指南 (96-105)
    Case(96,  "高血压患者饮食上要注意什么", ["search_rag"], ["高血压", "饮食"]),
    Case(97,  "糖尿病的治疗原则是什么", ["search_rag"], ["糖尿病", "治疗"]),
    Case(98,  "痛风急性发作怎么处理", ["search_rag"], ["痛风", "治疗"]),
    Case(99,  "慢性胃炎需要根除幽门螺杆菌吗", ["search_rag"], ["胃炎", "幽门螺杆菌"]),
    Case(100, "高血脂的一线治疗药物是什么", ["search_rag"], ["高血脂", "治疗"]),
    Case(101, "轻度抑郁症需要吃药吗", ["search_rag"], ["抑郁症", "治疗"]),
    Case(102, "急性扭伤怎么处理", ["search_rag"], ["扭伤", "处理"]),
    Case(103, "反流性食管炎的治疗方法", ["search_rag"], ["反流性食管炎", "治疗"]),
    Case(104, "睡眠呼吸暂停怎么治疗", ["search_rag"], ["睡眠呼吸暂停", "治疗"]),
    Case(105, "慢性支气管炎的日常管理", ["search_rag"], ["慢性支气管炎", "管理"]),


    # ════════════════════════════════════════════════════════════════════════
    # web_search（25 条）
    # 触发条件：近期/最新/今年/新型等时效性内容
    # ════════════════════════════════════════════════════════════════════════

    # 最新研究 (106-115)
    Case(106, "最新研究说阿司匹林预防心脏病还有用吗", ["web_search"], ["阿司匹林", "最新"]),
    Case(107, "最近有没有新的降糖药上市", ["web_search"], ["降糖药", "新药"]),
    Case(108, "2025年流感疫苗的推荐情况怎么样", ["web_search"], ["流感疫苗", "2025"]),
    Case(109, "最新新冠变种的症状有哪些", ["web_search"], ["新冠", "最新"]),
    Case(110, "今年的诺如病毒感染情况严重吗", ["web_search"], ["诺如病毒", "今年"]),
    Case(111, "最新研究说咖啡对心脏病有什么影响", ["web_search"], ["咖啡", "心脏", "研究"]),
    Case(112, "近期有没有关于肠道菌群的新研究", ["web_search"], ["肠道菌群", "研究"]),
    Case(113, "最新版高血压指南有什么更新", ["web_search"], ["高血压", "指南", "最新"]),
    Case(114, "最近降脂药有新的研究结果吗", ["web_search"], ["降脂药", "研究"]),
    Case(115, "今年花粉季节过敏情况怎么样", ["web_search"], ["花粉", "过敏", "今年"]),

    # 新药/新疗法 (116-122)
    Case(116, "最近有没有治疗阿尔茨海默症的突破", ["web_search"], ["阿尔茨海默", "新药"]),
    Case(117, "GLP-1减肥药最新临床数据怎么样", ["web_search"], ["GLP-1", "最新"]),
    Case(118, "新型口服胰岛素研发到哪个阶段了", ["web_search"], ["口服胰岛素", "研发"]),
    Case(119, "最近批准的抗癌新药有哪些", ["web_search"], ["抗癌", "新药"]),
    Case(120, "CAR-T细胞疗法最新进展", ["web_search"], ["CAR-T", "最新"]),
    Case(121, "最新基因编辑治疗遗传病有进展吗", ["web_search"], ["基因编辑", "最新"]),
    Case(122, "新冠口服药最近有什么更新", ["web_search"], ["新冠", "口服药", "最新"]),

    # 当前疫情/健康事件 (123-130)
    Case(123, "最近国内流感流行严重吗", ["web_search"], ["流感", "流行"]),
    Case(124, "现在猴痘疫情情况怎么样", ["web_search"], ["猴痘", "疫情"]),
    Case(125, "近期有没有食品安全预警", ["web_search"], ["食品安全", "预警"]),
    Case(126, "最近有没有药品召回通知", ["web_search"], ["药品召回"]),
    Case(127, "今年夏天中暑病例多吗", ["web_search"], ["中暑", "今年"]),
    Case(128, "最近麻疹有没有在流行", ["web_search"], ["麻疹", "流行"]),
    Case(129, "近期有没有关于饮用水安全的预警", ["web_search"], ["饮用水", "安全"]),
    Case(130, "最新的儿童疫苗接种指南有更新吗", ["web_search"], ["儿童疫苗", "最新"]),


    # ════════════════════════════════════════════════════════════════════════
    # generate_report（15 条）
    # ════════════════════════════════════════════════════════════════════════

    Case(131, "/report", ["generate_report"], []),
    Case(132, "帮我生成健康报告", ["generate_report"], []),
    Case(133, "我想看我的健康摘要", ["generate_report"], []),
    Case(134, "生成一份我的健康总结", ["generate_report"], []),
    Case(135, "查看健康报告", ["generate_report"], []),
    Case(136, "给我出一份健康分析报告", ["generate_report"], []),
    Case(137, "帮我汇总一下我的健康情况", ["generate_report"], []),
    Case(138, "我的健康记录总结一下", ["generate_report"], []),
    Case(139, "生成报告", ["generate_report"], []),
    Case(140, "我想要一份完整的健康档案", ["generate_report"], []),
    Case(141, "把我的症状和用药情况汇总一下", ["generate_report"], []),
    Case(142, "帮我做一个健康分析", ["generate_report"], []),
    Case(143, "出一份我的健康数据报告", ["generate_report"], []),
    Case(144, "我的健康历史给我总结一下", ["generate_report"], []),
    Case(145, "做一份我的健康摘要报告", ["generate_report"], []),


    # ════════════════════════════════════════════════════════════════════════
    # 不调任何工具（35 条）
    # 场景：纯问候、感谢、通用常识、简单首次症状陈述、闲聊
    # ════════════════════════════════════════════════════════════════════════

    # 问候与寒暄 (146-155)
    Case(146, "你好", [], []),
    Case(147, "早上好", [], []),
    Case(148, "晚上好", [], []),
    Case(149, "谢谢你的帮助", [], []),
    Case(150, "再见", [], []),
    Case(151, "你好啊，今天感觉怎么样", [], []),
    Case(152, "嗯嗯，我知道了，谢谢", [], []),
    Case(153, "好的，我明白了", [], []),
    Case(154, "辛苦了", [], []),
    Case(155, "太感谢了", [], []),

    # 关于助手本身 (156-161)
    Case(156, "你能做什么", [], []),
    Case(157, "你是谁", [], []),
    Case(158, "你能帮我记录什么", [], []),
    Case(159, "怎么使用你", [], []),
    Case(160, "你和普通医生有什么不同", [], []),
    Case(161, "你有什么功能", [], []),

    # 基础医学常识（LLM 应可直接回答，不需要权威文献）(162-170)
    Case(162, "正常体温是多少度", [], []),
    Case(163, "成年人正常心率是多少", [], []),
    Case(164, "人一天应该喝多少水", [], []),
    Case(165, "什么是血压", [], []),
    Case(166, "BMI怎么计算", [], []),
    Case(167, "什么是心率", [], []),
    Case(168, "什么叫收缩压和舒张压", [], []),
    Case(169, "血型有哪几种", [], []),
    Case(170, "什么是免疫系统", [], []),

    # 首次陈述症状——健康助手应主动查历史，expected 为 search_memory (171-178)
    Case(171, "我今天有点头痛", ["search_memory"], ["头痛"]),
    Case(172, "我喉咙有点不舒服", ["search_memory"], ["喉咙"]),
    Case(173, "我今天有点累", ["search_memory"], ["疲劳"]),
    Case(174, "我最近消化不太好", ["search_memory"], ["消化"]),
    Case(175, "我今天有点低烧", ["search_memory"], ["发烧"]),
    Case(176, "我肩膀有点酸", ["search_memory"], ["肩膀"]),
    Case(177, "我今天眼睛有点干", ["search_memory"], ["眼睛"]),
    Case(178, "我最近食欲不太好", ["search_memory"], ["食欲"]),

    # 非健康话题 / 无关问题 (179-180)
    Case(179, "今天天气怎么样", [], []),
    Case(180, "你会说英语吗", [], []),


    # ════════════════════════════════════════════════════════════════════════
    # 多工具（20 条）
    # memory + rag (181-192) / memory + web (193-196) / rag + web (197-200)
    # ════════════════════════════════════════════════════════════════════════

    # memory + rag (181-192)
    Case(181, "我上次用的布洛芬，它长期吃有什么副作用", ["search_memory", "search_rag"], ["布洛芬", "副作用"]),
    Case(182, "之前医生让我吃的那个降压药，和阿司匹林一起吃安全吗", ["search_memory", "search_rag"], ["降压药", "阿司匹林"]),
    Case(183, "我一直在吃二甲双胍，它和我上次提到的那个感冒药有没有冲突", ["search_memory", "search_rag"], ["二甲双胍"]),
    Case(184, "我上次说有高血压，这个病需要终身服药吗", ["search_memory", "search_rag"], ["高血压", "治疗"]),
    Case(185, "之前记录的那个胃痛，幽门螺杆菌感染的症状是这样吗", ["search_memory", "search_rag"], ["胃痛", "幽门螺杆菌"]),
    Case(186, "我老是腰痛，腰椎间盘突出的诊断标准是什么", ["search_memory", "search_rag"], ["腰痛", "腰椎间盘"]),
    Case(187, "我上次提过对青霉素过敏，头孢也会过敏吗", ["search_memory", "search_rag"], ["青霉素", "过敏", "头孢"]),
    Case(188, "我经常头晕，低血压的诊断是什么", ["search_memory", "search_rag"], ["头晕", "低血压"]),
    Case(189, "之前说过有痛风，秋水仙碱的用法是什么", ["search_memory", "search_rag"], ["痛风", "秋水仙碱"]),
    Case(190, "我还是失眠，苯二氮䓬类安眠药有什么依赖风险", ["search_memory", "search_rag"], ["失眠", "安眠药"]),
    Case(191, "我上次血脂高，他汀类药物长期服用有什么注意事项", ["search_memory", "search_rag"], ["血脂", "他汀"]),
    Case(192, "我一直有反酸，奥美拉唑长期吃有风险吗", ["search_memory", "search_rag"], ["反酸", "奥美拉唑"]),

    # memory + web (193-196)
    Case(193, "我上次提到对新冠很担心，最新变种情况怎么样", ["search_memory", "web_search"], ["新冠", "最新"]),
    Case(194, "我之前说在用GLP-1减肥，最新的安全性数据怎么样", ["search_memory", "web_search"], ["GLP-1", "最新"]),
    Case(195, "我一直关注的那个流感疫苗，今年推荐版本更新了吗", ["search_memory", "web_search"], ["流感疫苗", "今年"]),
    Case(196, "之前记录我有过敏体质，今年花粉季严重吗", ["search_memory", "web_search"], ["过敏", "花粉", "今年"]),

    # rag + web (197-200)
    Case(197, "最新指南推荐的高血压一线用药是什么", ["search_rag", "web_search"], ["高血压", "指南", "最新"]),
    Case(198, "最新研究说阿司匹林对预防中风的作用有变化吗", ["search_rag", "web_search"], ["阿司匹林", "中风", "最新"]),
    Case(199, "GLP-1药物的作用机制和最近的临床结果", ["search_rag", "web_search"], ["GLP-1", "最新"]),
    Case(200, "最新版糖尿病指南对二甲双胍的推荐有变化吗", ["search_rag", "web_search"], ["二甲双胍", "指南", "最新"]),
]


# ── 评测逻辑 ───────────────────────────────────────────────────────────────────

async def evaluate_case(case: Case, llm_with_tools) -> dict:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from agent.prompts import SYSTEM_PROMPT
    from datetime import date

    today = date.today().strftime("%Y年%m月%d日")
    system = SystemMessage(content=[{
        "type": "text",
        "text": f"今天是 {today}。\n\n{SYSTEM_PROMPT}",
        "cache_control": {"type": "ephemeral"},
    }])

    history_lc = []
    for m in case.history:
        if m["role"] == "user":
            history_lc.append(HumanMessage(content=m["content"]))
        else:
            history_lc.append(AIMessage(content=m["content"]))

    messages = history_lc + [HumanMessage(content=case.message)]

    t0 = time.monotonic()
    response = await llm_with_tools.ainvoke([system] + messages)
    ms = (time.monotonic() - t0) * 1000

    actual_calls = response.tool_calls or []
    actual_tools = [tc["name"] for tc in actual_calls]

    expected_set = set(case.expected_tools)
    actual_set   = set(actual_tools)

    exact_match = expected_set == actual_set
    under_call  = sorted(expected_set - actual_set)
    over_call   = sorted(actual_set - expected_set)
    wrong_tool  = bool(actual_set and expected_set and not (actual_set & expected_set))

    # 参数相关性
    param_scores = []
    for tc in actual_calls:
        if tc["name"] in ("search_memory", "search_rag", "web_search") and case.param_keywords:
            query = (tc.get("args") or {}).get("query", "").lower()
            hits  = sum(1 for kw in case.param_keywords if kw.lower() in query)
            param_scores.append(hits / len(case.param_keywords))

    return {
        "id":            case.id,
        "message":       case.message,
        "expected":      sorted(case.expected_tools),
        "actual":        actual_tools,
        "exact_match":   exact_match,
        "under_call":    under_call,
        "over_call":     over_call,
        "wrong_tool":    wrong_tool,
        "param_scores":  param_scores,
        "param_avg":     sum(param_scores) / len(param_scores) if param_scores else None,
        "tool_details":  [{"name": tc["name"], "query": (tc.get("args") or {}).get("query", "")}
                          for tc in actual_calls],
        "ms":            ms,
    }


async def run_all(concurrency: int) -> list[dict]:
    import os
    from langchain_openai import ChatOpenAI
    from agent.tools import TOOL_SPECS

    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(TOOL_SPECS)
    sem = asyncio.Semaphore(concurrency)

    async def safe_eval(case):
        async with sem:
            try:
                return await evaluate_case(case, llm_with_tools)
            except Exception as exc:
                return {
                    "id": case.id, "message": case.message,
                    "expected": case.expected_tools, "actual": [],
                    "exact_match": False, "under_call": case.expected_tools,
                    "over_call": [], "wrong_tool": False,
                    "param_scores": [], "param_avg": None,
                    "tool_details": [], "ms": 0,
                    "error": str(exc),
                }

    print(f"[info] 共 {len(CASES)} 条，并发={concurrency}，开始评测…")
    t0 = time.monotonic()
    results = await asyncio.gather(*[safe_eval(c) for c in CASES])
    total_s = time.monotonic() - t0
    print(f"[info] 完成，总耗时 {total_s:.1f}s")
    return list(results)


# ── 报告打印 ───────────────────────────────────────────────────────────────────

def print_report(results: list[dict]) -> None:
    total   = len(results)
    exact   = sum(1 for r in results if r["exact_match"])
    errors  = [r for r in results if "error" in r]

    print("\n" + "=" * 65)
    print("  工具选择评测报告")
    print("=" * 65)
    print(f"总案例      : {total}")
    print(f"完全正确    : {exact}/{total} = {exact/total*100:.1f}%")
    if errors:
        print(f"执行出错    : {len(errors)} 条（已计入失败）")

    # ── 错误类型 ──────────────────────────────────────────────────────────
    under   = [r for r in results if r["under_call"]]
    over    = [r for r in results if r["over_call"]]
    wrong   = [r for r in results if r["wrong_tool"]]
    print(f"\n错误类型分布")
    print(f"  漏调 (under-call) : {len(under)} 条")
    print(f"  误调 (over-call)  : {len(over)} 条")
    print(f"  调错 (wrong-tool) : {len(wrong)} 条")

    # ── 工具级 Precision / Recall ─────────────────────────────────────────
    all_tools = ["search_memory", "search_rag", "web_search", "generate_report", "(无工具)"]
    print(f"\n{'工具':<20} {'precision':>10} {'recall':>8} {'F1':>6}")
    print("─" * 50)
    for tool in all_tools:
        if tool == "(无工具)":
            tp = sum(1 for r in results if not r["expected"] and not r["actual"])
            fp = sum(1 for r in results if not r["expected"] and r["actual"])
            fn = sum(1 for r in results if r["expected"] and not r["actual"])
        else:
            tp = sum(1 for r in results if tool in r["expected"] and tool in r["actual"])
            fp = sum(1 for r in results if tool not in r["expected"] and tool in r["actual"])
            fn = sum(1 for r in results if tool in r["expected"] and tool not in r["actual"])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"  {tool:<18} {precision:>10.2f} {recall:>8.2f} {f1:>6.2f}")

    # ── 参数相关性 ────────────────────────────────────────────────────────
    param_results = [r for r in results if r["param_avg"] is not None]
    if param_results:
        avg_param = sum(r["param_avg"] for r in param_results) / len(param_results)
        min_param = min(r["param_avg"] for r in param_results)
        print(f"\n参数相关性  avg={avg_param:.2f}  min={min_param:.2f}  (基于 {len(param_results)} 条)")

    # ── 问题案例详情 ──────────────────────────────────────────────────────
    failed = [r for r in results if not r["exact_match"]]
    if failed:
        print(f"\n问题案例 ({len(failed)} 条)")
        print("─" * 65)
        for r in failed[:30]:   # 最多展示 30 条
            tag = []
            if r["under_call"]: tag.append(f"漏调{r['under_call']}")
            if r["over_call"]:  tag.append(f"误调{r['over_call']}")
            if r["wrong_tool"]: tag.append("调错")
            if "error" in r:    tag.append(f"执行错误:{r['error'][:40]}")
            msg_short = r["message"][:35] + ("…" if len(r["message"]) > 35 else "")
            print(f"  #{r['id']:03d} [{', '.join(tag)}]")
            print(f"       消息: {msg_short}")
            print(f"       期望: {r['expected']}  实际: {r['actual']}")
            if r["tool_details"]:
                for td in r["tool_details"]:
                    if td["query"]:
                        print(f"       参数: {td['name']} query={td['query'][:50]!r}")
        if len(failed) > 30:
            print(f"  … 还有 {len(failed) - 30} 条，见输出 JSON")

    # ── 参数低分案例 ──────────────────────────────────────────────────────
    low_param = [r for r in param_results if r["param_avg"] < 0.5]
    if low_param:
        print(f"\n参数相关性低分案例 (score<0.5, 共 {len(low_param)} 条)")
        print("─" * 65)
        for r in low_param[:10]:
            msg_short = r["message"][:35] + ("…" if len(r["message"]) > 35 else "")
            print(f"  #{r['id']:03d} score={r['param_avg']:.2f}  {msg_short}")
            for td in r["tool_details"]:
                if td["query"]:
                    print(f"       {td['name']} query={td['query'][:50]!r}")

    print("\n" + "=" * 65)


# ── Markdown 报告生成 ─────────────────────────────────────────────────────────

def build_md_report(results: list[dict]) -> str:
    total  = len(results)
    exact  = sum(1 for r in results if r["exact_match"])
    errors = [r for r in results if "error" in r]
    under  = [r for r in results if r["under_call"]]
    over   = [r for r in results if r["over_call"]]
    wrong  = [r for r in results if r["wrong_tool"]]

    valid  = total - len(errors)
    valid_exact = sum(1 for r in results if r["exact_match"] and "error" not in r)

    lines = []
    lines.append(f"# 工具选择评测报告\n")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## 总体结果\n")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 总案例 | {total} |")
    lines.append(f"| 完全正确 | {exact} / {total} = **{exact/total*100:.1f}%** |")
    if errors:
        lines.append(f"| 执行出错（429 限速，已计入失败） | {len(errors)} 条 |")
        lines.append(f"| 排除错误后有效准确率 | {valid_exact} / {valid} = **{valid_exact/valid*100:.1f}%** |" if valid else "")
    lines.append("")

    lines.append("## 错误类型分布\n")
    lines.append("| 类型 | 数量 |")
    lines.append("|------|------|")
    lines.append(f"| 漏调 under-call | {len(under)} 条 |")
    lines.append(f"| 误调 over-call  | {len(over)} 条 |")
    lines.append(f"| 调错 wrong-tool | {len(wrong)} 条 |")
    lines.append("")

    lines.append("## 工具级别 Precision / Recall / F1\n")
    lines.append("| 工具 | Precision | Recall | F1 |")
    lines.append("|------|----------:|-------:|---:|")
    all_tools = ["search_memory", "search_rag", "web_search", "generate_report", "（无工具）"]
    for tool in all_tools:
        if tool == "（无工具）":
            tp = sum(1 for r in results if not r["expected"] and not r["actual"])
            fp = sum(1 for r in results if not r["expected"] and r["actual"])
            fn = sum(1 for r in results if r["expected"] and not r["actual"])
        else:
            tp = sum(1 for r in results if tool in r["expected"] and tool in r["actual"])
            fp = sum(1 for r in results if tool not in r["expected"] and tool in r["actual"])
            fn = sum(1 for r in results if tool in r["expected"] and tool not in r["actual"])
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        lines.append(f"| {tool} | {p:.2f} | {r:.2f} | {f1:.2f} |")

    param_results = [r for r in results if r["param_avg"] is not None]
    if param_results:
        avg_p = sum(r["param_avg"] for r in param_results) / len(param_results)
        min_p = min(r["param_avg"] for r in param_results)
        lines.append(f"\n**参数相关性**：avg = {avg_p:.2f}，min = {min_p:.2f}（基于 {len(param_results)} 条）\n")

    # 问题案例
    failed = [r for r in results if not r["exact_match"]]
    if failed:
        lines.append(f"## 问题案例（共 {len(failed)} 条）\n")
        # 按错误类型分组
        by_type: dict[str, list] = {"under_call": [], "over_call": [], "wrong_tool": []}
        for r in failed:
            if r["under_call"]:   by_type["under_call"].append(r)
            elif r["over_call"]:  by_type["over_call"].append(r)
            elif r["wrong_tool"]: by_type["wrong_tool"].append(r)

        label_map = {"under_call": "漏调", "over_call": "误调", "wrong_tool": "调错"}
        for key, label in label_map.items():
            group = by_type[key]
            if not group:
                continue
            lines.append(f"### {label}（{len(group)} 条）\n")
            lines.append("| # | 消息 | 期望 | 实际 |")
            lines.append("|---|------|------|------|")
            for r in group:
                msg   = r["message"][:40] + ("…" if len(r["message"]) > 40 else "")
                exp   = ", ".join(r["expected"]) or "—"
                act   = ", ".join(r["actual"])   or "—"
                lines.append(f"| {r['id']:03d} | {msg} | {exp} | {act} |")
            lines.append("")

    # 参数低分案例
    low_param = [r for r in param_results if r["param_avg"] < 0.5]
    if low_param:
        lines.append(f"## 参数相关性低分案例（score < 0.5，共 {len(low_param)} 条）\n")
        lines.append("| # | Score | 消息 | 工具 | 实际 query |")
        lines.append("|---|------:|------|------|-----------|")
        for r in low_param:
            msg = r["message"][:35] + ("…" if len(r["message"]) > 35 else "")
            for td in r["tool_details"]:
                if td.get("query"):
                    lines.append(f"| {r['id']:03d} | {r['param_avg']:.2f} | {msg} | {td['name']} | `{td['query'][:40]}` |")

    return "\n".join(lines) + "\n"


# ── 主入口 ────────────────────────────────────────────────────────────────────

async def main(concurrency: int, output: str | None) -> None:
    results = await run_all(concurrency)
    print_report(results)

    # 默认保存到 evals/results/tool_selection_<timestamp>.json
    if output is None:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = str(results_dir / f"tool_selection_{ts}.json")

    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[info] JSON 已写入 {output}")

    md_path = output.replace(".json", ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_md_report(results))
    print(f"[info] 报告已写入 {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=5, help="并发数（默认5）")
    parser.add_argument("--output", type=str, default=None, help="结果输出 JSON 路径（默认自动生成）")
    args = parser.parse_args()
    asyncio.run(main(args.concurrency, args.output))
