import requests
from datetime import datetime, timezone

def get_bash_question_count():
    url = "https://api.stackexchange.com/2.3/search/advanced"
    params = {
        "tagged": "bash",
        "site": "stackoverflow",
        "pagesize": 1,          # 只需 1 条，目的是拿 total
        "filter": "!nNPvSNVZJS" # 官方支持返回 total 的 filter
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    total = data.get("total", None)
    has_more = data.get("has_more", None)

    query_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    return total, has_more, query_time


if __name__ == "__main__":
    total, has_more, query_time = get_bash_question_count()

    print(f"Query time (UTC): {query_time}")
    print(f"Total 'bash' questions on Stack Overflow: {total}")
    print(f"Has more (API internal flag): {has_more}")

