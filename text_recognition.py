import requests

url = "https://microsoft-computer-vision3.p.rapidapi.com/ocr"

querystring = {"detectOrientation": "true", "language": "en"}

payload = {
    "url": "https://s38vlx.storage.yandex.net/rdisk/3f7558cde24c86d46d11c1f58258c8a3e464e0c04f0cdc10c28018d8ccc5ed5b/63935bc5/6bPiGVqMZcLAWxUdU3ZTXkkXo6k8N5f6qog7I0Wopiwkb-Hdl0lx7gY-pr_efCsD9kI2pWyDP1slBVDRt7owXQ==?uid=0&filename=FhcO-GdXoAA4VwT.jpeg&disposition=inline&hash=&limit=0&content_type=image%2Fjpeg&owner_uid=0&fsize=593242&hid=2dc7ea68444ef434368b8b008566c05e&media_type=image&tknv=v2&etag=4c874a777a8951853723f48f5c5c8567&rtoken=jvVLLOJUDwCO&force_default=no&ycrid=na-6645036cd90bddb1699a4c593deb4fac-downloader18e&ts=5ef67430abb40&s=7bb9482a674f4188fb61cc8ce6011a39e652168f1b1f181bdc1f5ab66c5bacdc&pb=U2FsdGVkX19is13VKj83sK1AKYCcIMsDxv2RtRV6oGMbziB91uMFuDdLbdvDkU3W4RPIFdxWhp8WxAeSsy5w4byVVCflQE5msB7GYJ1dGaQ"}
headers = {
    "content-type": "application/json",
    "X-RapidAPI-Key": "91988b0f9amsha033953c156cc3cp1a3435jsn5ab27c2114eb",
    "X-RapidAPI-Host": "microsoft-computer-vision3.p.rapidapi.com"
}

response = requests.request("POST", url, json=payload, headers=headers, params=querystring)

print(response.text)
