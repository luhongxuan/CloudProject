import requests
import time

def fetch_fresh_proxies():
    """
    從多個來源自動抓取最新 Proxy
    """
    urls = [
        "https://api.proxyscrape.com/v3/free-proxy-list/get?request=displayproxies&proxy_format=ipport&timeout=2000&country=all",
        "https://www.proxy-list.download/api/v1/get?type=http",
        "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/main/http.txt",
    ]
    proxies = []
    for url in urls:
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                new_proxies = [p.strip() for p in res.text.split("\n") if p.strip()]
                proxies.extend(new_proxies)
        except Exception:
            continue
    proxies = list(set(proxies))
    print(f"✅ 總共取得 {len(proxies)} 個 Proxy")
    return proxies


def test_proxy(proxy, timeout=3):
    """
    測試單個 Proxy 是否可用
    """
    try:
        start = time.time()
        res = requests.get("https://httpbin.org/ip", 
                           proxies={"http": f"http://{proxy}", "https": f"http://{proxy}"}, 
                           timeout=timeout)
        delay = round(time.time() - start, 2)
        if res.status_code == 200:
            ip = res.json().get("origin", "")
            print(f"✅ 可用代理: {proxy:<22} | 延遲: {delay}s | 回傳IP: {ip}")
            return True
    except Exception as e:
        print(f"❌ 失敗: {proxy:<22} | {e}")
    return False


if __name__ == "__main__":
    proxy_list = fetch_fresh_proxies()
    print("\n🚀 開始測試代理...\n")

    valid_proxies = []
    for i, proxy in enumerate(proxy_list):  # 限制測試前 20 個即可
        if test_proxy(proxy):
            valid_proxies.append(proxy)
        time.sleep(0.5)  # 避免太快被封

    print("\n==============================")
    print(f"✅ 測試完成！共找到 {len(valid_proxies)} 個可用代理：")
    for p in valid_proxies:
        print("   -", p)
