import requests

url = "https://www.btcc.com/v1/account/openposition"

payload = {
    "accountid": "1590742",
    "direction": "1",
    "multiple": "50",
    "symbol": "GWFX/USDT/GTS/MM/SVIP5/0/A/USELESSUSDT50x",
    "symbolid": "3618867",
    "request_volume": "10000",
    "request_price": "0.06900",
    "posway": "1",
    "type": "1",
    "order_channel": "1",
    "token": "00015645871780045069914"
}

headers = {
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent": "Mozilla/5.0"
}

try:
    response = requests.post(
        url,
        data=payload,
        headers=headers,
        timeout=10
    )

    print("Status Code:", response.status_code)
    print("Response Text:", response.text)

    # 如果回傳是 JSON，可以用這個解析
    try:
        print("JSON:", response.json())
    except ValueError:
        print("Response is not JSON")

except requests.exceptions.RequestException as e:
    print("Request failed:", e)