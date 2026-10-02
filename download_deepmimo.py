"""下载 DeepMIMO 真实场景数据。

关键约束(踩过的坑):
1. token 一次性: `/api/download/secure` 返回的 redirectUrl 只能用一次,
   用完即失效。所以必须"现取现用",不能先探测再下载。
2. 每日配额: 服务端有 daily limit,403 "Download would exceed daily limit"。
   因此按优先级顺序逐个尝试,失败不中断,而不是一次性全下。
3. 历史上 f005.backblazeb2.com 被网络阻断;连上学校 VPN 后已恢复。
   代码会自动跟随重定向,所以不需要关心最终落到哪个 CDN 域名。

用法:
    .venv-dm/bin/python download_deepmimo.py                # 按优先级下载第一个成功的
    .venv-dm/bin/python download_deepmimo.py --all          # 尝试所有候选(注意配额)
    .venv-dm/bin/python download_deepmimo.py --only asu_campus_3p5
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import requests

API = "https://deepmimo.net"
HEADERS = {"User-Agent": "DeepMIMO-Python/4.0", "Accept": "*/*"}
TIMEOUT = 60
MAX_MB = 400  # 超过此大小直接跳过,避免浪费配额下巨文件

# 优先级说明:
# - 3p5 = 3.5GHz, 最贴近 5G Sub-6 实际工作频段, 优先选它
# - city_* 为城市宏蜂窝, 角弥散/多径更丰富, 适合研究 beam prediction
# - asu_campus 为经典校园场景, 文献里用得多, 便于对照引用
# - i1_2p5 是 2.3GB 大文件, 放最后且会被 MAX_MB 拦掉
PRIORITY = [
    "asu_campus_3p5",
    "city_6_miami_3p5",
    "city_7_sandiego_3p5",
    "boston5g_3p5",
    "city_37_seoul_3p5",
    "city_30_singapore_3p5",
    "city_33_hong_kong_3p5",
]


class DownloadError(RuntimeError):
    """下载过程中的可预期错误。"""


def get_token(scenario: str) -> str:
    """向 secure 端点申请一次性下载链接。"""
    url = f"{API}/api/download/secure?filename={scenario}.zip"
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise DownloadError(data.get("error", "unknown error"))
    redirect = data.get("redirectUrl", "")
    if not redirect:
        raise DownloadError("missing redirectUrl")
    if not redirect.startswith("http"):
        redirect = f"{API}{redirect}"
    return redirect


def download_one(scenario: str, out_dir: Path) -> Path:
    """下载单个场景并解压,返回解压后的场景目录。

    注意: 这里绝不能先用 HEAD 探大小 —— 那会消耗掉一次性 token
    (HEAD 跟随重定向后 token 即失效)。改为流式下载时按已写字节数判断。
    """
    # token 必须现取现用,不能提前获取
    redirect = get_token(scenario)

    zip_path = out_dir / f"{scenario}.zip"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 此时 redirect 仍有效,立刻下载
    with requests.get(redirect, stream=True, headers=HEADERS, timeout=TIMEOUT) as resp:
        if resp.status_code != 200:
            raise DownloadError(f"HTTP {resp.status_code}: {resp.text[:120]}")
        total = int(resp.headers.get("Content-Length", 0))
        written = 0
        with zip_path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                written += len(chunk)
                if written > MAX_MB * 1024**2:
                    fh.close()
                    zip_path.unlink(missing_ok=True)
                    raise DownloadError(f"aborted: exceeded {MAX_MB} MB")
                if total:
                    pct = 100 * written / total
                    print(f"\r  下载中 {written / 1024**2:.1f}/{total / 1024**2:.1f} MB ({pct:.0f}%)", end="")
    print()

    if written < 1024:
        raise DownloadError(f"downloaded file too small ({written} B), likely an error page")

    extract_dir = out_dir / scenario
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    zip_path.unlink()  # 解压后删掉 zip,省空间
    return extract_dir


def describe(path: Path) -> None:
    """打印解压后目录结构,确认数据真的到位。"""
    files = sorted(p for p in path.rglob("*") if p.is_file())
    print(f"  解压出 {len(files)} 个文件:")
    for p in files[:12]:
        print(f"    {p.relative_to(path)}  ({p.stat().st_size / 1024:.0f} KB)")
    if len(files) > 12:
        print(f"    ... 还有 {len(files) - 12} 个")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="尝试所有候选场景")
    parser.add_argument("--only", type=str, default="", help="只下载指定场景")
    parser.add_argument("--out", type=str, default="deepmimo_scenarios", help="输出目录")
    args = parser.parse_args()

    out_dir = Path(__file__).parent / args.out
    candidates = [args.only] if args.only else PRIORITY
    if not args.only and not args.all:
        candidates = PRIORITY  # 默认按优先级,成功即停

    print(f"输出目录: {out_dir}\n")
    for scenario in candidates:
        print(f"=== 尝试 {scenario} ===")
        try:
            path = download_one(scenario, out_dir)
        except (DownloadError, requests.RequestException, zipfile.BadZipFile) as exc:
            print(f"  失败: {type(exc).__name__}: {exc}\n")
            continue
        print(f"  成功 -> {path}")
        describe(path)
        if not args.all:
            return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
