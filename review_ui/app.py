"""
Human-in-the-Loop Review Dashboard — Streamlit UI

Run with:
    streamlit run review_ui/app.py

Requires:
    AGENTOPS_API_URL=http://localhost:8080  (or set in .env)
"""
import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("AGENTOPS_API_URL", "http://localhost:8080")

st.set_page_config(
    page_title="AgentOps Review Dashboard",
    page_icon="🤖",
    layout="wide",
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def fetch_pending():
    try:
        r = requests.get(f"{API_URL}/tasks/pending", timeout=5)
        return r.json().get("pending", [])
    except Exception:
        return []


def fetch_all():
    try:
        r = requests.get(f"{API_URL}/tasks/all", timeout=5)
        return r.json().get("reviews", [])
    except Exception:
        return []


def fetch_chat(task_id: str):
    try:
        r = requests.get(f"{API_URL}/tasks/{task_id}/chat", timeout=5)
        return r.json().get("chat", [])
    except Exception:
        return []


def send_chat(task_id: str, message: str):
    try:
        requests.post(f"{API_URL}/tasks/{task_id}/chat", json={"message": message}, timeout=5)
    except Exception:
        pass


def submit_decision(task_id: str, decision: str, feedback: str,
                    modified_plan=None, human_output=None):
    payload = {"decision": decision, "feedback": feedback}
    if modified_plan:
        payload["modified_plan"] = modified_plan
    if human_output:
        payload["human_output"] = human_output
    try:
        r = requests.post(f"{API_URL}/tasks/{task_id}/review", json=payload, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def level_badge(level: str) -> str:
    colours = {
        "notify": "🟢 Notify",
        "approve_action": "🟡 Approve Action",
        "approve_plan": "🔴 Approve Plan",
        "take_over": "⛔ Take Over",
    }
    return colours.get(level, level)


def status_badge(status: str) -> str:
    return {"pending": "⏳ Pending", "decided": "✅ Decided", "notified": "📣 Notified"}.get(status, status)


# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.title("🤖 AgentOps")
st.sidebar.caption(f"API: `{API_URL}`")

view = st.sidebar.radio("View", ["📥 Pending Reviews", "📋 All Reviews"])
auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True)

if auto_refresh:
    time.sleep(0.1)
    st.rerun() if st.sidebar.button("↻ Refresh now") else None

# ── Main content ───────────────────────────────────────────────────────────────

st.title("🔍 Human-in-the-Loop Review Dashboard")

if view == "📥 Pending Reviews":
    items = fetch_pending()
    if not items:
        st.success("✅ No tasks pending review.")
    else:
        st.warning(f"**{len(items)} task(s) awaiting your decision.**")

    for item in items:
        task_id = item["task_id"]
        level = item.get("approval_level", "approve_plan")

        with st.expander(f"{level_badge(level)}  |  Task `{task_id[:8]}…`  |  {item.get('reason', '')}", expanded=True):

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("📋 Task Context")
                st.markdown(f"**Original Request:**\n> {item.get('original_request', 'N/A')}")

                plan = item.get("execution_plan", {})
                if plan:
                    st.markdown(f"**Plan Summary:** {plan.get('task_summary', 'N/A')}")
                    st.markdown(
                        f"**Confidence:** `{plan.get('confidence_score', 'N/A')}`  |  "
                        f"**Risk:** `{plan.get('risk_level', 'N/A')}`"
                    )
                    subtasks = plan.get("subtasks", [])
                    if subtasks:
                        st.markdown("**Subtasks:**")
                        for st_item in subtasks:
                            st.markdown(
                                f"- `{st_item['id']}` → **{st_item['assigned_agent']}**: {st_item['description']}"
                            )

                completed = item.get("completed_steps", [])
                if completed:
                    st.markdown(f"**Completed steps:** {', '.join(completed)}")

                current = item.get("current_step")
                if current:
                    st.info(f"⏸ **Paused at:** {current}")

                proposed = item.get("proposed_action")
                if proposed:
                    st.markdown(f"**Proposed action:** `{proposed}`")

            with col2:
                st.subheader("💬 Chat with Agent")
                chat = fetch_chat(task_id)
                chat_container = st.container()
                with chat_container:
                    for msg in chat:
                        role = msg["role"]
                        icon = "🧑" if role == "human" else "🤖"
                        st.markdown(f"{icon} **{role.capitalize()}:** {msg['message']}")

                chat_input = st.text_input("Ask the agent a question…", key=f"chat_{task_id}")
                if st.button("Send", key=f"send_{task_id}") and chat_input:
                    send_chat(task_id, chat_input)
                    st.rerun()

            st.divider()
            st.subheader("🎯 Your Decision")

            feedback = st.text_area("Feedback / reason (required for reject / modify):", key=f"fb_{task_id}")

            dcol1, dcol2, dcol3, dcol4 = st.columns(4)

            with dcol1:
                if st.button("✅ Approve", key=f"approve_{task_id}", use_container_width=True):
                    if submit_decision(task_id, "approve", feedback):
                        st.success("Approved — task will resume.")
                        time.sleep(1)
                        st.rerun()

            with dcol2:
                if st.button("❌ Reject", key=f"reject_{task_id}", use_container_width=True):
                    if submit_decision(task_id, "reject", feedback):
                        st.error("Rejected — supervisor will re-plan.")
                        time.sleep(1)
                        st.rerun()

            with dcol3:
                with st.popover("✏️ Modify Plan"):
                    modified_json = st.text_area(
                        "Paste modified plan JSON:", height=200, key=f"mod_{task_id}"
                    )
                    if st.button("Submit modification", key=f"modsubmit_{task_id}"):
                        import json
                        try:
                            mod_plan = json.loads(modified_json)
                            if submit_decision(task_id, "modify", feedback, modified_plan=mod_plan):
                                st.success("Modification submitted.")
                                time.sleep(1)
                                st.rerun()
                        except json.JSONDecodeError:
                            st.error("Invalid JSON — please check the plan format.")

            with dcol4:
                with st.popover("⛔ Take Over"):
                    human_ans = st.text_area(
                        "Provide the final answer directly:", height=150, key=f"to_{task_id}"
                    )
                    if st.button("Submit answer", key=f"tosubmit_{task_id}"):
                        if human_ans and submit_decision(task_id, "take_over", feedback, human_output=human_ans):
                            st.success("Answer submitted — agents will use your output.")
                            time.sleep(1)
                            st.rerun()


elif view == "📋 All Reviews":
    items = fetch_all()
    if not items:
        st.info("No review history yet.")
    else:
        st.markdown(f"**{len(items)} total review(s)**")
        for item in items:
            task_id = item["task_id"]
            level = item.get("approval_level", "?")
            status = item.get("status", "?")
            decision = item.get("decision", "—")

            with st.expander(
                f"{status_badge(status)}  |  {level_badge(level)}  |  `{task_id[:8]}…`  |  Decision: **{decision}**"
            ):
                st.markdown(f"**Request:** {item.get('original_request', 'N/A')}")
                st.markdown(f"**Reason:** {item.get('reason', 'N/A')}")
                st.markdown(f"**Feedback:** {item.get('feedback') or '—'}")
                if item.get("human_output"):
                    st.markdown(f"**Human output:** {item['human_output']}")

# ── Auto refresh ───────────────────────────────────────────────────────────────

if auto_refresh:
    time.sleep(5)
    st.rerun()
