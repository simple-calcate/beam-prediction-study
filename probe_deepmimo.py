"""探测 DeepMIMO 哪些场景真正可下载,以及文件大小。

因为 201 个场景里只有一部分被官方批准开放下载,
所以这里先批量试探 token 接口,筛出可用的再下载。
"""

import requests

API = "https://deepmimo.net"
HEADERS = {"User-Agent": "DeepMIMO-Python/4.0", "Accept": "*/*"}

# 优先尝试的场景: 3.5GHz 城市场景(与5G Sub-6 最相关) + 经典校园场景
CANDIDATES = [
    "city_6_miami_3p5",
    "city_7_sandiego_3p5",
    "city_30_singapore_3p5",
    "city_37_seoul_3p5",
    "asu_campus_3p5",
    "boston5g_3p5",
    "city_33_hong_kong_3p5",
    "i1_2p5",
    "o1_3p5",
]


def search_all() -> list[str]:
    resp = requests.post(
        f"{API}/api/search/scenarios",
        json={},
        headers={**HEADERS, "Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["scenarios"]


def probe_token(scenario: str) -> dict:
    """向 secure 端点要一个下载 token,看是否被批准。"""
    url = f"{API}/api/download/secure?filename={scenario}.zip"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def head_size(url: str) -> str:
    """用 HEAD 请求看真实文件大小,避免直接下到一半才发现超大。"""
    try:
        resp = requests.head(url, headers=HEADERS, timeout=30, allow_redirects=True)
        size = resp.headers.get("Content-Length")
        if size:
            return f"{int(size) / 1024 ** 2:.1f} MB"
        return "unknown"
    except Exception as exc:  # noqa: BLE001
        return f"HEAD failed: {type(exc).__name__}"


def main() -> None:
    all_scenarios = search_all()
    print(f"场景总数: {len(all_scenarios)}\n")

    approved = []
    for name in CANDIDATES:
        if name not in all_scenarios:
            print(f"[跳过] {name}: 不在场景列表中")
            continue
        token = probe_token(name)
        if "error" in token:
            print(f"[不可用] {name}: {token.get('error')}")
            continue
        redirect = token.get("redirectUrl", "")
        if not redirect.startswith("http"):
            redirect = f"{API}{redirect}"
        size = head_size(redirect)
        print(f"[可用] {name}: {size}")
        print(f"         host = {redirect.split('/')[2]}")
        approved.append((name, redirect, size))

    print(f"\n=== 可用场景: {len(approved)}/{len(CANDIDATES)} ===")
    for name, redirect, size in approved:
        print(f"{name}\t{size}\t{redirect}")


if __name__ == "__main__":
    main()
