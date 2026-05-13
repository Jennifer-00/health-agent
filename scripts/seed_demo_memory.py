"""
预置演示用记忆，录 demo 前跑一次。
用法（在项目根目录）：python scripts/seed_demo_memory.py [--user demo@health.ai]
"""
import argparse
import asyncio
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from dotenv import load_dotenv
load_dotenv(override=True)

from memory.mem0_client import Mem0Client

parser = argparse.ArgumentParser()
parser.add_argument("--user", default="demo@health.ai", help="目标用户 user_id（默认 demo@health.ai）")
parser.add_argument("--clear", action="store_true", help="写入前先清空该用户的全部记忆")
args = parser.parse_args()

DEMO_USER = args.user

# 多主题、多时间层次的记忆，覆盖检索场景
MEMORIES = [
    # ── 眼科 ──────────────────────────────────────────────────────────────────
    "用户去年10月做了近视激光手术（LASIK），术后视力恢复到1.0，医生嘱咐术后一年内避免揉眼",
    "用户最近一个月反复感到眼睛干涩，早上起床时更明显，用过玻璃酸钠滴眼液有所缓解",
    "用户提到睡眠不足时右眼容易出现轻微外斜，通常休息后自行恢复",

    # ── 睡眠 ──────────────────────────────────────────────────────────────────
    "用户过去三周持续入睡困难，基本凌晨两点后才能睡着，白天精神差、注意力难以集中",
    "用户睡前习惯刷手机1小时以上，曾尝试褪黑素0.5mg，效果不稳定",

    # ── 头颈 ──────────────────────────────────────────────────────────────────
    "用户最近三周反复出现后枕部头痛，晨起时最重，下午有所缓解，与久坐和颈部僵硬相关",
    "用户颈部长期僵硬，尤其久坐办公后加重，偶尔向右侧转头时有轻微弹响",

    # ── 消化 ──────────────────────────────────────────────────────────────────
    "用户约三个月前因胃部不适就诊，检查为浅表性胃炎，医生建议规律饮食、避免辛辣，开了奥美拉唑20mg，按需服用",
    "用户有时饭后1小时出现上腹隐痛，空腹时更明显，最近一个月发作频率约每周两次",

    # ── 过敏与用药 ────────────────────────────────────────────────────────────
    "用户对青霉素过敏，曾出现皮疹；对花粉轻度过敏，春季鼻塞流涕，自行服用氯雷他定10mg缓解",

    # ── 慢性病与检查 ──────────────────────────────────────────────────────────
    "用户有轻度高血压家族史（父亲患有高血压）；自身血压偏高，上月测量135/88 mmHg，暂未用药",
    "用户上个月空腹血糖5.9 mmol/L，医生告知处于正常偏高范围，建议控制精制碳水摄入",
    "用户体检结果显示轻度脂肪肝，BMI 26.4，医生建议减重5公斤并增加有氧运动",

    # ── 生活习惯 ──────────────────────────────────────────────────────────────
    "用户每天久坐超过8小时，基本不运动，偶尔周末步行",
    "用户每天咖啡摄入约2杯（共400ml），下午3点后不喝；不吸烟，偶尔饮酒（每月1-2次）",

    # ── 皮肤 ──────────────────────────────────────────────────────────────────
    "用户近两个月面部出现零星痘痘，主要在下巴和额头，月经前后加重，判断与内分泌波动相关",

    # ── 过敏与抗过敏药 ────────────────────────────────────────────────────────
    "用户对青霉素类抗生素过敏（已确认），曾注射后出现荨麻疹和面部红肿，就医后肌注肾上腺素处理",
    "用户对花粉过敏（树花粉为主），每年3-5月症状明显：打喷嚏、流清涕、眼痒，自行服用氯雷他定10mg/天，效果一般",
    "用户尝试过西替利嗪10mg治疗花粉症，效果优于氯雷他定但有明显嗜睡感，白天工作时不敢用",
    "用户对海鲜（虾、蟹）轻度不耐受，进食后偶有腹胀和皮肤瘙痒，无严重过敏反应记录",
    "用户曾因急性荨麻疹就诊，医生开了地氯雷他定5mg连服7天，第三天症状消退",
    "用户鼻炎发作时曾使用布地奈德鼻喷剂（辅舒良），连用两周后症状明显改善，但停药后复发",
    "用户提到对磺胺类药物有轻度皮疹反应史，具体药物为复方新诺明，已告知医生记入病历",
]


# 直接写入 Neo4j 的结构化档案（跳过 LLM 提取，置信度直接给 0.9）
PROFILE_FACTS = [
    {"category": "过敏",  "name": "青霉素",     "detail": "注射后出现荨麻疹和面部红肿，就医肌注肾上腺素处理"},
    {"category": "过敏",  "name": "磺胺类药物",  "detail": "复方新诺明，曾出现轻度皮疹，已记入病历"},
    {"category": "过敏",  "name": "花粉",        "detail": "树花粉为主，每年3-5月鼻塞流涕眼痒"},
    {"category": "过敏",  "name": "海鲜",        "detail": "虾蟹轻度不耐受，进食后腹胀皮肤瘙痒，无严重过敏"},
    {"category": "用药",  "name": "氯雷他定",    "detail": "10mg/天，花粉季按需服用，效果一般"},
    {"category": "用药",  "name": "西替利嗪",    "detail": "10mg，花粉症备用，效果优于氯雷他定但有嗜睡副作用"},
    {"category": "用药",  "name": "地氯雷他定",  "detail": "5mg，急性荨麻疹发作时连服7天"},
    {"category": "用药",  "name": "布地奈德鼻喷剂", "detail": "辅舒良，鼻炎发作时使用，停药后易复发"},
    {"category": "用药",  "name": "奥美拉唑",    "detail": "20mg，浅表性胃炎按需服用"},
    {"category": "用药",  "name": "玻璃酸钠滴眼液", "detail": "眼干时按需使用，术后医嘱"},
    {"category": "诊断",  "name": "浅表性胃炎",  "detail": "约三个月前确诊，医嘱规律饮食避免辛辣"},
    {"category": "病史",  "name": "近视激光手术", "detail": "LASIK，去年10月，术后视力1.0，一年内禁止揉眼"},
    {"category": "病史",  "name": "急性荨麻疹",  "detail": "曾发作一次，地氯雷他定治疗后消退"},
    {"category": "家族史", "name": "高血压",     "detail": "父亲患有高血压"},
]


async def seed():
    from memory import profile_graph
    await profile_graph.init_schema()

    client = Mem0Client(user_id=DEMO_USER)

    if args.clear:
        print(f"清空用户 {DEMO_USER!r} 的全部 Mem0 记忆...")
        existing = await client.get_all()
        for m in existing:
            mid = m.get("id")
            if mid:
                await client.delete(mid)
        print(f"已删除 {len(existing)} 条 Mem0 记忆\n")

    # ── Qdrant（情景类）────────────────────────────────────────────────────────
    print(f"=== [1/2] 写入 Mem0 / Qdrant（情景类记忆，共 {len(MEMORIES)} 条）===")
    ok = 0
    for i, text in enumerate(MEMORIES, 1):
        try:
            await client.add(text)
            print(f"  [{i:02d}/{len(MEMORIES)}] OK  {text[:40]}...")
            ok += 1
        except Exception as exc:
            print(f"  [{i:02d}/{len(MEMORIES)}] ERR {exc}", file=sys.stderr)
    print(f"  完成：{ok}/{len(MEMORIES)} 条\n")

    # ── Neo4j（档案类）────────────────────────────────────────────────────────
    print(f"=== [2/2] 写入 Neo4j（结构化档案，共 {len(PROFILE_FACTS)} 条）===")
    ok2 = 0
    for i, fact in enumerate(PROFILE_FACTS, 1):
        try:
            await profile_graph.upsert_from_extraction(DEMO_USER, [fact])
            print(f"  [{i:02d}/{len(PROFILE_FACTS)}] OK  [{fact['category']}] {fact['name']} — {fact['detail'][:30]}...")
            ok2 += 1
        except Exception as exc:
            print(f"  [{i:02d}/{len(PROFILE_FACTS)}] ERR {exc}", file=sys.stderr)
    print(f"  完成：{ok2}/{len(PROFILE_FACTS)} 条\n")

    print(f"全部完成，user_id={DEMO_USER!r}")


asyncio.run(seed())
