#!/usr/bin/env python3
"""netops-duty 独立演示 —— 不依赖 DeerFlow，只用本地 ollama 跑通完整值班流程。

用途：
  1. 验证 skill 的提示词设计是否有效（换模型、换数据，直接对比输出）
  2. 给客户 / 面试官做演示：一条命令，把一坨告警变成一份值班报告
  3. 作为把 skill 接进 DeerFlow 之前的离线调试台

用法：
  python demo_duty.py                          # 用内置示例数据
  python demo_duty.py --alarms my_alarms.json  # 用自己的告警数据
  python demo_duty.py --model qwen3:8b --base-url http://127.0.0.1:11434

依赖：仅 Python 3.9+ 标准库，无需 pip install。
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ALARMS = SKILL_DIR / "assets" / "sample_alarms.json"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_system_prompt() -> str:
    """把 SKILL.md 与两份 reference 拼成系统提示词。"""
    parts = [
        "# 你现在扮演的角色\n",
        "你是一名资深网络运维值班工程师。严格按下面的 skill 定义执行，"
        "尤其要遵守『三条铁律』和『证据等级』要求。\n",
        "---\n## SKILL 定义\n",
        read(SKILL_DIR / "SKILL.md"),
        "\n---\n## 参考：告警收敛规则\n",
        read(SKILL_DIR / "references" / "alarm-correlation.md"),
        "\n---\n## 参考：根因假设树\n",
        read(SKILL_DIR / "references" / "rca-hypothesis-tree.md"),
        "\n---\n## 输出要求\n",
        "严格按以下结构输出 Markdown 报告，不要增删章节：\n",
        "1. 当前状态（含故障等级、影响面）\n",
        "2. 时间线（表格：时刻 / 对象 / 事件 / 来源）\n",
        "3. 告警收敛结果（表格：事件组 / 涉及对象 / 首条时刻 / 告警条数 / 角色）\n",
        "4. 根因判断（当前最可能根因 + 证据等级 + 2~4 个假设的五要素表）\n",
        "5. 处置建议（按优先级排序，每条给出可执行动作）\n",
        "6. 交班必做\n",
        "7. 待补信息\n",
        "\n注意：输入中没有的信息一律不要编造；缺失就写进『待补信息』。",
    ]
    return "\n".join(parts)


def build_user_prompt(data: dict) -> str:
    lines = [
        "## 采集窗口",
        data.get("meta", {}).get("collect_window", "未提供"),
        "\n## 告警原始数据",
        "```json",
        json.dumps(data.get("alarms", []), ensure_ascii=False, indent=1),
        "```",
        "\n## 设备台账",
        "```json",
        json.dumps(data.get("devices", {}), ensure_ascii=False, indent=1),
        "```",
        "\n## 历史记录（同类事件的过往处置）",
        "```json",
        json.dumps(data.get("history", {}), ensure_ascii=False, indent=1),
        "```",
        "\n## 变更记录",
        "```json",
        json.dumps(data.get("changes", []), ensure_ascii=False, indent=1),
        "```",
        "\n## 处置预案库",
        "```json",
        json.dumps(data.get("runbook", []), ensure_ascii=False, indent=1),
        "```",
        "\n请按输出要求给出值班报告。",
    ]
    return "\n".join(lines)


def chat(model: str, base_url: str, messages: list, timeout: int) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 32768},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")).get("message", {})


def main() -> int:
    ap = argparse.ArgumentParser(description="netops-duty 独立演示")
    ap.add_argument("--alarms", default=str(DEFAULT_ALARMS))
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out-dir", default=str(SKILL_DIR / "out"))
    args = ap.parse_args()

    src = Path(args.alarms)
    if not src.exists():
        print(f"[x] 告警数据不存在：{src}", file=sys.stderr)
        return 2
    data = json.loads(read(src))

    print(f"[*] 告警数据：{src}")
    print(f"[*] 告警条数：{len(data.get('alarms', []))}")
    print(f"[*] 模型：{args.model} @ {args.base_url}")
    print("[*] 正在分析……（本地推理，通常 1~5 分钟，取决于机器）")

    t0 = time.time()
    try:
        msg = chat(
            args.model,
            args.base_url,
            [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": build_user_prompt(data)},
            ],
            args.timeout,
        )
    except urllib.error.URLError as exc:
        print(f"[x] 连不上模型服务，请先执行 `ollama serve`。原始错误：{exc}", file=sys.stderr)
        return 3

    report = (msg.get("content") or "").strip()
    if not report:
        print("[!] 模型只返回了思考过程，未给出正文，已回退使用 thinking 字段。", file=sys.stderr)
        report = (msg.get("thinking") or "").strip()
    if not report:
        print("[x] 模型没有返回任何内容。", file=sys.stderr)
        return 4

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"duty-report-{time.strftime('%Y%m%d-%H%M%S')}.md"
    out.write_text(report, encoding="utf-8")

    elapsed = time.time() - t0
    print(f"[√] 耗时 {elapsed:.0f}s，报告已写入：{out}")
    print("-" * 60)
    print(report[:1200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
