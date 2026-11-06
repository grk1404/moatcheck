"""One-time Telegram setup for the MoatCheck alert service.

1. In Telegram, message @BotFather, send /newbot, and copy the token it gives you.
2. Open your new bot in Telegram and press Start (or send it any message).
3. Run:  python telegram_setup.py <token>
   It finds your chat id, saves both values to data/alert_secrets.json (git-ignored),
   and sends a test message.

Run it on the machine where the alert service runs (the secrets file stays local).
"""
import json
import sys
from pathlib import Path

import requests

SECRETS = Path("data/alert_secrets.json")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    token = sys.argv[1].strip()
    api = f"https://api.telegram.org/bot{token}"

    me = requests.get(f"{api}/getMe", timeout=20).json()
    if not me.get("ok"):
        print("Telegram rejected the token:", me.get("description"))
        return 1
    print("Bot found: @" + me["result"]["username"])

    updates = requests.get(f"{api}/getUpdates", timeout=20).json().get("result", [])
    chats = {}
    for u in updates:
        chat = (u.get("message") or u.get("channel_post") or {}).get("chat")
        if chat:
            chats[chat["id"]] = chat.get("first_name") or chat.get("title") or ""
    if not chats:
        print("No messages found. Open your bot in Telegram, press Start, then run this again.")
        return 1
    chat_id, name = list(chats.items())[-1]
    print(f"Using chat {chat_id} ({name})")

    secrets = {}
    if SECRETS.exists():
        try:
            secrets = json.loads(SECRETS.read_text(encoding="utf-8"))
        except Exception:
            secrets = {}
    secrets["TELEGRAM_BOT_TOKEN"] = token
    secrets["TELEGRAM_CHAT_ID"] = str(chat_id)
    SECRETS.parent.mkdir(parents=True, exist_ok=True)
    SECRETS.write_text(json.dumps(secrets, indent=2), encoding="utf-8")
    print(f"Saved credentials to {SECRETS}")

    r = requests.post(f"{api}/sendMessage",
                      data={"chat_id": chat_id, "text": "✅ MoatCheck alerts are connected to Telegram."}, timeout=20)
    print("Test message:", "sent, check Telegram" if r.json().get("ok") else f"failed: {r.text[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
