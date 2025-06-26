"""Test script to verify API conversation management."""

import requests
import json

BASE_URL = "http://localhost:8000"

print("🧪 Testing API Conversation Management")
print("=" * 50)

# 1. Start conversation
print("\n1. Starting conversation...")
start_resp = requests.post(
    f"{BASE_URL}/conversation/start",
    json={}  # Empty body for optional thread_id
)
if start_resp.status_code != 200:
    print(f"❌ Error: {start_resp.status_code} - {start_resp.text}")
    exit(1)
    
start_data = start_resp.json()
print(f"✅ Thread ID: {start_data['thread_id']}")
print(f"🤖 Iris: {start_data['message']}")

thread_id = start_data['thread_id']

# 2. Send first message
print("\n2. Sending first message...")
msg1_resp = requests.post(
    f"{BASE_URL}/conversation/message",
    json={
        "thread_id": thread_id,
        "message": "Hi Iris, I'm the CEO of TechCorp and we need help with an acquisition"
    }
)
msg1_data = msg1_resp.json()
print(f"🤖 Iris: {msg1_data['message']}")
print(f"📋 Legal Area: {msg1_data.get('legal_area', 'Not classified yet')}")
print(f"💬 Conversation Complete: {msg1_data['conversation_complete']}")

# 3. Send second message
print("\n3. Sending second message...")
msg2_resp = requests.post(
    f"{BASE_URL}/conversation/message",
    json={
        "thread_id": thread_id,
        "message": "The deal size is about $50 million and we need to close in 3 months"
    }
)
msg2_data = msg2_resp.json()
print(f"🤖 Iris: {msg2_data['message']}")
print(f"💬 Conversation Complete: {msg2_data['conversation_complete']}")

# 4. Check status
print("\n4. Checking conversation status...")
status_resp = requests.post(
    f"{BASE_URL}/conversation/status",
    json={"thread_id": thread_id}
)
status_data = status_resp.json()
print(f"📊 Status: {json.dumps(status_data, indent=2)}")

# 5. Get history
print("\n5. Getting conversation history...")
history_resp = requests.get(f"{BASE_URL}/conversation/{thread_id}/history")
history_data = history_resp.json()
print(f"📜 Message Count: {history_data['message_count']}")
print(f"🔍 Last 3 messages:")
for msg in history_data['history'][-3:]:
    print(f"   [{msg['role']}]: {msg['content'][:100]}...")

print("\n✅ API test complete!")
print("🎯 The conversation is managed across requests without CLI prompts!")