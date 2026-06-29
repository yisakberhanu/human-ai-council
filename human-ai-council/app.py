import os
import re
import json
import time
import uuid
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import requests
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

# ============================================================
# Model Council V13
# Real meeting round flow:
# 1) Generate one idea per agent
# 2) Human listens to one agent at a time
# 3) Human votes after each agent
# 4) Agents vote on other agents' ideas
# 5) Round finishes with weighted decision
# 6) Human accepts/rejects before next round
# ============================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MAX_AGENT_TOKENS = int(os.getenv("MAX_AGENT_TOKENS", "260"))
MAX_JUDGE_TOKENS = int(os.getenv("MAX_JUDGE_TOKENS", "220"))
MAX_REPAIR_TOKENS = int(os.getenv("MAX_REPAIR_TOKENS", "120"))
MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", "55"))

HUMAN_VOTE_WEIGHT = float(os.getenv("HUMAN_VOTE_WEIGHT", "30"))
AGENT_TOTAL_WEIGHT = float(os.getenv("AGENT_TOTAL_WEIGHT", "70"))

REPAIR_MODEL = os.getenv("REPAIR_MODEL", "inclusionai/ling-2.6-flash")
FAST_JUDGE_MODEL = os.getenv("FAST_JUDGE_MODEL", os.getenv("EVALUATOR_MODEL", "inclusionai/ling-2.6-flash"))
STRONG_JUDGE_MODEL = os.getenv("STRONG_JUDGE_MODEL", FAST_JUDGE_MODEL)

DATA_DIR = Path(os.getenv("DATA_DIR", "meeting_data_v13"))
DATA_DIR.mkdir(exist_ok=True)
STATE_FILE = DATA_DIR / "state.json"
SKILLS_DIR = DATA_DIR / "skills"
SKILLS_DIR.mkdir(exist_ok=True)

STATE_LOCK = threading.RLock()

AGENTS = {
    "product": {
        "name": "Product Agent",
        "emoji": "🟢",
        "model": os.getenv("PRODUCT_MODEL", "google/gemma-3-27b-it"),
        "base": "Defend user trust, product value, positioning, adoption, UX clarity, and what customers will accept.",
        "color": "product",
    },
    "builder": {
        "name": "Builder Agent",
        "emoji": "🔵",
        "model": os.getenv("BUILDER_MODEL", "qwen/qwen3-30b-a3b-instruct-2507"),
        "base": "Defend implementation feasibility: APIs, database, workflow, latency, reliability, observability, and testability.",
        "color": "builder",
    },
    "strategy": {
        "name": "Strategy Agent",
        "emoji": "🟣",
        "model": os.getenv("STRATEGY_MODEL", "openai/gpt-oss-20b"),
        "base": "Defend market opportunity, roadmap, pricing, pilot design, priorities, and business direction.",
        "color": "strategy",
    },
    "critic": {
        "name": "Critic Agent",
        "emoji": "🔴",
        "model": os.getenv("CRITIC_MODEL", "inclusionai/ling-2.6-flash"),
        "base": "Defend risk review: bias, fairness, legal exposure, weak assumptions, accountability, misuse, and failure modes.",
        "color": "critic",
    },
}

CORE_SKILL_RULES = [
    "Before speaking, check accepted decisions and do not repeat resolved points.",
    "Give one concise idea that adds new value, resolves a conflict, or moves the decision forward.",
    "Use your own base perspective; do not give generic agreement.",
    "Convert abstract opinions into one concrete artifact: KPI, workflow step, API, audit rule, risk test, pricing/pilot gate, or UX element.",
    "When a decision is being made, vote for one concrete idea from another agent when possible.",
    "End with a decision-moving statement: accept, reject, measure, build, test, or decide next.",
]

# -----------------------------
# State helpers
# -----------------------------

def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_agent_state(agent_key):
    agent = AGENTS[agent_key]
    return {
        "key": agent_key,
        "name": agent["name"],
        "model": agent["model"],
        "base": agent["base"],
        "current": "No accepted position yet.",
        "speak_count": 0,
        "scores": [],
        "updated_at": now_iso(),
        "skill_rules": list(CORE_SKILL_RULES),
        "rejected_rules": [],
    }


def default_state():
    return {
        "version": "V13_real_meeting_round_vote",
        "meeting_id": str(uuid.uuid4()),
        "created_at": now_iso(),
        "round_number": 0,
        "active_topic": "",
        "next_focus": "Start the first round.",
        "current_round_id": None,
        "rounds": [],
        "decisions": [],
        "agreements": [],
        "transcript": [],
        "agent_states": {k: default_agent_state(k) for k in AGENTS},
        "session_memory": {
            "summary": "No decisions yet.",
            "unresolved": [],
            "accepted_decisions": [],
        },
    }


def load_state():
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            state = default_state()
    else:
        state = default_state()

    state.setdefault("version", "V13_real_meeting_round_vote")
    state.setdefault("rounds", [])
    state.setdefault("transcript", [])
    state.setdefault("decisions", [])
    state.setdefault("agreements", [])
    state.setdefault("session_memory", {"summary": "No decisions yet.", "unresolved": [], "accepted_decisions": []})
    state.setdefault("agent_states", {})
    for key in AGENTS:
        if key not in state["agent_states"]:
            state["agent_states"][key] = default_agent_state(key)
        else:
            # Keep new core fields if user upgrades from older version.
            st = state["agent_states"][key]
            st.setdefault("key", key)
            st.setdefault("name", AGENTS[key]["name"])
            st.setdefault("model", AGENTS[key]["model"])
            st.setdefault("base", AGENTS[key]["base"])
            st.setdefault("current", "No accepted position yet.")
            st.setdefault("speak_count", 0)
            st.setdefault("scores", [])
            st.setdefault("skill_rules", list(CORE_SKILL_RULES))
            st.setdefault("rejected_rules", [])
            st.setdefault("updated_at", now_iso())
    return state


STATE = load_state()


def save_state():
    STATE_FILE.write_text(json.dumps(STATE, ensure_ascii=False, indent=2), encoding="utf-8")
    save_skill_files()


def save_skill_files():
    for key, st in STATE.get("agent_states", {}).items():
        path = SKILLS_DIR / f"{key}_skill.md"
        rules = "\n".join([f"- {r}" for r in st.get("skill_rules", [])[:10]])
        rejected = "\n".join([f"- {r}" for r in st.get("rejected_rules", [])[-5:]]) or "- None"
        text = f"""# {st.get('name', key)} Skill\n\nModel: {st.get('model','')}\nUpdated: {st.get('updated_at','')}\n\n## Base Perspective\n{st.get('base','')}\n\n## Current Position\n{st.get('current','')}\n\n## Active Skill Rules\n{rules}\n\n## Recent Rejected / Avoid Rules\n{rejected}\n"""
        path.write_text(text, encoding="utf-8")


def public_state():
    # Keep UI state compact and serializable.
    return {
        "version": STATE.get("version"),
        "meeting_id": STATE.get("meeting_id"),
        "round_number": STATE.get("round_number"),
        "active_topic": STATE.get("active_topic"),
        "next_focus": STATE.get("next_focus"),
        "current_round_id": STATE.get("current_round_id"),
        "rounds": STATE.get("rounds", [])[-8:],
        "current_round": get_current_round(),
        "decisions": STATE.get("decisions", [])[-12:],
        "agreements": STATE.get("agreements", [])[-12:],
        "transcript": STATE.get("transcript", [])[-80:],
        "agent_states": STATE.get("agent_states", {}),
        "session_memory": STATE.get("session_memory", {}),
        "weights": {
            "human": HUMAN_VOTE_WEIGHT,
            "agents_total": AGENT_TOTAL_WEIGHT,
            "per_agent": AGENT_TOTAL_WEIGHT / max(1, len(AGENTS)),
        },
    }


def get_current_round():
    rid = STATE.get("current_round_id")
    if not rid:
        return None
    for r in STATE.get("rounds", []):
        if r.get("round_id") == rid:
            return r
    return None


def add_transcript(role, name, text, agent_key=None, kind="message", extra=None):
    item = {
        "id": str(uuid.uuid4()),
        "time": now_iso(),
        "role": role,
        "name": name,
        "agent_key": agent_key,
        "kind": kind,
        "text": text,
    }
    if extra:
        item.update(extra)
    STATE.setdefault("transcript", []).append(item)
    STATE["transcript"] = STATE["transcript"][-300:]
    return item

# -----------------------------
# Model helpers
# -----------------------------

def call_openrouter(model, messages, max_tokens=250, temperature=0.25):
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to .env and restart Flask.")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://127.0.0.1:5000",
        "X-Title": "Model Council V13",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=MODEL_TIMEOUT)
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenRouter {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    return data["choices"][0]["message"].get("content", "").strip()


def extract_json_object(text):
    if not text:
        return {}
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    # Pull the first likely JSON object.
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        chunk = s[start:end + 1]
        try:
            return json.loads(chunk)
        except Exception:
            # Salvage simple string fields.
            obj = {}
            for key in ["idea_title", "idea", "stance", "position_update", "vote_for", "reason"]:
                m = re.search(rf'"{key}"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)', chunk)
                if m:
                    obj[key] = m.group(1)
            if obj:
                return obj
    return {}


def short(text, n=500):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= n else text[: n - 3] + "..."


def local_agent_idea(agent_key, topic, round_number):
    agent = AGENTS[agent_key]
    if agent_key == "product":
        title = "Trust-first product framing"
        idea = "Frame the product as an assistant that makes hiring decisions easier to review, not an autonomous replacement. Ship visible explanations and human controls first."
    elif agent_key == "builder":
        title = "Auditable MVP workflow"
        idea = "Build a narrow workflow: input job description and resume, generate structured candidate summary, show scoring rationale, require human approval, and log every decision."
    elif agent_key == "strategy":
        title = "Pilot before scale"
        idea = "Start with a small pilot for one hiring team, measure time saved, adoption, trust, and risk before expanding pricing or automation scope."
    else:
        title = "Guardrails before automation"
        idea = "Do not automate rejection or ranking without bias checks, audit sampling, appeal path, and clear accountability for final hiring decisions."
    return {
        "idea_id": f"R{round_number}-{agent_key}",
        "agent_key": agent_key,
        "agent_name": agent["name"],
        "model": "local-fallback",
        "idea_title": title,
        "idea": idea,
        "stance": "propose",
        "position_update": f"I focus on {agent['base']}",
        "status": "waiting",
        "human_vote": None,
        "agent_votes_received": [],
        "quality": 70,
    }


def make_agent_idea(agent_key, topic, round_number):
    agent = AGENTS[agent_key]
    st = STATE["agent_states"][agent_key]
    decisions = "\n".join([f"- {d.get('decision','')}" for d in STATE.get("decisions", [])[-6:]]) or "- None"
    agreements = "\n".join([f"- {a}" for a in STATE.get("agreements", [])[-6:]]) or "- None"
    rules = "\n".join([f"- {r}" for r in st.get("skill_rules", [])[:8]])
    prompt = f"""
You are {agent['name']} in a human-led AI meeting.
Base perspective: {agent['base']}
Current position: {st.get('current','No position yet')}
Active topic: {topic}
Current next focus: {STATE.get('next_focus','')}
Accepted decisions:
{decisions}
Agreements:
{agreements}
Your active skill rules:
{rules}

Task: give exactly ONE clear idea for this round. Do not repeat accepted decisions. Make the idea vote-able.
Return compact valid JSON only. No markdown.
Schema:
{{
  "idea_title": "max 6 words",
  "idea": "one concise idea under 65 words",
  "stance": "propose|support|object|alternative",
  "position_update": "one sentence about how your position changes or stays focused"
}}
""".strip()
    try:
        content = call_openrouter(
            agent["model"],
            [
                {"role": "system", "content": "Return compact valid JSON only. Do not include markdown."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_AGENT_TOKENS,
            temperature=0.35,
        )
        obj = extract_json_object(content)
        if not obj or not obj.get("idea"):
            # Useful fallback: convert raw text into idea.
            obj = {
                "idea_title": "Agent proposal",
                "idea": short(content, 420) or local_agent_idea(agent_key, topic, round_number)["idea"],
                "stance": "propose",
                "position_update": "I keep my role perspective and add one concrete proposal.",
            }
        idea = {
            "idea_id": f"R{round_number}-{agent_key}",
            "agent_key": agent_key,
            "agent_name": agent["name"],
            "model": agent["model"],
            "idea_title": short(obj.get("idea_title", "Agent idea"), 80),
            "idea": short(obj.get("idea", ""), 520),
            "stance": short(obj.get("stance", "propose"), 30),
            "position_update": short(obj.get("position_update", ""), 220),
            "status": "waiting",
            "human_vote": None,
            "agent_votes_received": [],
            "quality": 75,
        }
        return idea
    except Exception as e:
        idea = local_agent_idea(agent_key, topic, round_number)
        idea["error"] = str(e)[:300]
        return idea


def make_agent_vote(voter_key, round_obj):
    agent = AGENTS[voter_key]
    st = STATE["agent_states"][voter_key]
    choices = []
    for idea in round_obj.get("ideas", []):
        if idea["agent_key"] != voter_key:
            choices.append({
                "idea_id": idea["idea_id"],
                "agent": idea["agent_name"],
                "title": idea["idea_title"],
                "idea": idea["idea"],
            })
    if not choices:
        return {"voter_key": voter_key, "voter_name": agent["name"], "idea_id": None, "reason": "No other idea available."}

    prompt = f"""
You are {agent['name']}.
Base perspective: {agent['base']}
Current position: {st.get('current','')}
Round topic: {round_obj.get('topic','')}
You must vote for ONE idea from another agent. Do not vote for yourself.
Choose the idea that best moves the meeting toward a concrete decision, even if you have conditions.
Ideas:
{json.dumps(choices, ensure_ascii=False, indent=2)}
Return compact JSON only:
{{"vote_for":"idea_id", "reason":"under 18 words", "support_level":"support|conditional|reject_others"}}
""".strip()
    try:
        content = call_openrouter(
            agent["model"],
            [
                {"role": "system", "content": "Return compact valid JSON only. No markdown."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.2,
        )
        obj = extract_json_object(content)
        vote_for = obj.get("vote_for") or obj.get("idea_id")
        valid_ids = {c["idea_id"] for c in choices}
        if vote_for not in valid_ids:
            vote_for = heuristic_vote(voter_key, choices)
        return {
            "voter_key": voter_key,
            "voter_name": agent["name"],
            "idea_id": vote_for,
            "reason": short(obj.get("reason", "Best moves the decision forward."), 130),
            "support_level": short(obj.get("support_level", "support"), 40),
        }
    except Exception as e:
        return {
            "voter_key": voter_key,
            "voter_name": agent["name"],
            "idea_id": heuristic_vote(voter_key, choices),
            "reason": f"Fallback vote after model error: {str(e)[:80]}",
            "support_level": "conditional",
        }


def heuristic_vote(voter_key, choices):
    # A deterministic fallback that pushes role diversity.
    preferred_order = {
        "product": ["strategy", "builder", "critic"],
        "builder": ["critic", "strategy", "product"],
        "strategy": ["builder", "product", "critic"],
        "critic": ["builder", "product", "strategy"],
    }
    for pref in preferred_order.get(voter_key, []):
        for c in choices:
            if c["idea_id"].endswith(pref):
                return c["idea_id"]
    return choices[0]["idea_id"]

# -----------------------------
# Voting and decision logic
# -----------------------------

def compute_scores(round_obj):
    ideas = round_obj.get("ideas", [])
    score_map = {i["idea_id"]: {"total": 0.0, "human": 0.0, "agents": 0.0, "agent_voters": [], "human_label": None} for i in ideas}

    # Human vote: total 30%, normalized over preferred ideas first, then supported ideas.
    preferred = [i for i in ideas if i.get("human_vote") == "prefer"]
    supported = [i for i in ideas if i.get("human_vote") == "support"]
    rejected = [i for i in ideas if i.get("human_vote") == "reject"]

    if preferred:
        share = HUMAN_VOTE_WEIGHT / len(preferred)
        for i in preferred:
            score_map[i["idea_id"]]["human"] = share
            score_map[i["idea_id"]]["human_label"] = "Prefer"
    elif supported:
        share = HUMAN_VOTE_WEIGHT / len(supported)
        for i in supported:
            score_map[i["idea_id"]]["human"] = share
            score_map[i["idea_id"]]["human_label"] = "Support"

    for i in rejected:
        score_map[i["idea_id"]]["human_label"] = "Reject"

    # Agent votes: 70% divided across four agents.
    per_agent_weight = AGENT_TOTAL_WEIGHT / max(1, len(AGENTS))
    for vote in round_obj.get("agent_votes", []):
        idea_id = vote.get("idea_id")
        if idea_id in score_map:
            score_map[idea_id]["agents"] += per_agent_weight
            score_map[idea_id]["agent_voters"].append({
                "voter_key": vote.get("voter_key"),
                "voter_name": vote.get("voter_name"),
                "reason": vote.get("reason"),
            })

    for idea_id, sc in score_map.items():
        sc["total"] = round(sc["human"] + sc["agents"], 2)
        sc["human"] = round(sc["human"], 2)
        sc["agents"] = round(sc["agents"], 2)

    ranked = sorted(score_map.items(), key=lambda kv: kv[1]["total"], reverse=True)
    winner_id, winner_score = (ranked[0] if ranked else (None, None))
    winner_idea = None
    for i in ideas:
        if i["idea_id"] == winner_id:
            winner_idea = i
            break
    return score_map, winner_idea, winner_score


def update_agent_positions_from_decision(round_obj, decision_text):
    winner = round_obj.get("winner") or {}
    winner_title = winner.get("idea_title", "accepted idea")
    for key, st in STATE.get("agent_states", {}).items():
        if key == winner.get("agent_key"):
            st["current"] = f"My idea won this round: {winner_title}. Now I should help operationalize it."
        else:
            st["current"] = f"Accepted round decision: {winner_title}. I should build on it from my base perspective."
        st["updated_at"] = now_iso()
        # Keep skill rules general, not topic-specific.
        ensure_rule(st, "After a round decision is accepted, build on the winning idea instead of re-opening the same debate.")
        compress_skill_rules(st)


def ensure_rule(st, rule):
    rules = st.setdefault("skill_rules", [])
    if rule not in rules:
        rules.append(rule)


def compress_skill_rules(st):
    # Deduplicate and keep at most 10 general reusable rules.
    seen = set()
    clean = []
    for r in st.get("skill_rules", []):
        r = short(r, 220)
        # Avoid saving topic-specific decisions as permanent skill.
        if r.lower().startswith("do not re-debate accepted decision"):
            st.setdefault("rejected_rules", []).append(r)
            continue
        key = re.sub(r"\W+", " ", r.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            clean.append(r)
    st["skill_rules"] = clean[:10]

# -----------------------------
# Routes
# -----------------------------

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/state", methods=["GET"])
def api_state():
    with STATE_LOCK:
        return jsonify(public_state())


@app.route("/api/reset", methods=["POST"])
def api_reset():
    global STATE
    with STATE_LOCK:
        STATE = default_state()
        save_state()
        return jsonify(public_state())


@app.route("/api/repair_text", methods=["POST"])
def api_repair_text():
    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"repaired": ""})
    prompt = f"Fix speech-to-text errors, preserve meaning, keep it concise. Return only fixed text.\n\n{text}"
    try:
        fixed = call_openrouter(
            REPAIR_MODEL,
            [{"role": "user", "content": prompt}],
            max_tokens=MAX_REPAIR_TOKENS,
            temperature=0.1,
        )
        return jsonify({"repaired": fixed.strip()})
    except Exception as e:
        return jsonify({"repaired": text, "error": str(e)[:300]})


@app.route("/api/start_round", methods=["POST"])
def api_start_round():
    data = request.get_json(force=True) or {}
    topic = (data.get("topic") or "").strip()
    if not topic:
        topic = STATE.get("next_focus") or "Continue the meeting from the latest accepted decision."

    with STATE_LOCK:
        # Do not allow starting a new round if current round is unfinished.
        existing = get_current_round()
        if existing and existing.get("phase") not in ["accepted", "rejected", "closed"]:
            return jsonify({"error": "Current round is not finished. Vote and accept/reject before starting the next round.", "state": public_state()}), 409

        STATE["round_number"] = int(STATE.get("round_number", 0)) + 1
        round_number = STATE["round_number"]
        STATE["active_topic"] = topic
        add_transcript("human", "Human Host", topic, kind="topic")

    # Generate outside lock to avoid blocking state reads.
    ideas = []
    with ThreadPoolExecutor(max_workers=len(AGENTS)) as ex:
        futs = {ex.submit(make_agent_idea, key, topic, round_number): key for key in AGENTS}
        for fut in as_completed(futs):
            ideas.append(fut.result())
    # Keep consistent room order.
    order = {k: i for i, k in enumerate(AGENTS.keys())}
    ideas.sort(key=lambda x: order.get(x.get("agent_key"), 999))

    with STATE_LOCK:
        round_obj = {
            "round_id": str(uuid.uuid4()),
            "round_number": round_number,
            "topic": topic,
            "started_at": now_iso(),
            "phase": "listening",
            "current_index": 0,
            "ideas": ideas,
            "human_votes_complete": False,
            "agent_votes": [],
            "scores": {},
            "winner": None,
            "decision": None,
            "judge_note": None,
            "next_focus": "Listen to every agent idea, vote after each one, then finish the round.",
        }
        STATE["current_round_id"] = round_obj["round_id"]
        STATE.setdefault("rounds", []).append(round_obj)
        for idea in ideas:
            add_transcript("agent", idea["agent_name"], idea["idea"], agent_key=idea["agent_key"], kind="idea", extra={"idea_id": idea["idea_id"], "idea_title": idea["idea_title"], "model": idea["model"]})
            st = STATE["agent_states"][idea["agent_key"]]
            st["speak_count"] = int(st.get("speak_count", 0)) + 1
            st["current"] = idea.get("position_update") or st.get("current", "")
            st["updated_at"] = now_iso()
        save_state()
        return jsonify(public_state())


@app.route("/api/human_vote", methods=["POST"])
def api_human_vote():
    data = request.get_json(force=True) or {}
    idea_id = data.get("idea_id")
    vote = data.get("vote")  # support/prefer/reject
    if vote not in ["support", "prefer", "reject"]:
        return jsonify({"error": "vote must be support, prefer, or reject"}), 400
    with STATE_LOCK:
        r = get_current_round()
        if not r:
            return jsonify({"error": "No active round."}), 400
        found = False
        for idea in r.get("ideas", []):
            if idea.get("idea_id") == idea_id:
                idea["human_vote"] = vote
                idea["status"] = "voted"
                found = True
                add_transcript("human", "Human Vote", f"{vote.upper()} — {idea.get('idea_title')}: {idea.get('idea')}", kind="vote", extra={"idea_id": idea_id, "vote": vote})
                break
        if not found:
            return jsonify({"error": "Idea not found."}), 404
        r["human_votes_complete"] = all(i.get("human_vote") in ["support", "prefer", "reject"] for i in r.get("ideas", []))
        score_map, winner, winner_score = compute_scores(r)
        r["scores"] = score_map
        if winner:
            r["winner"] = {**winner, "score": winner_score}
        save_state()
        return jsonify(public_state())


@app.route("/api/finish_round", methods=["POST"])
def api_finish_round():
    with STATE_LOCK:
        r = get_current_round()
        if not r:
            return jsonify({"error": "No active round."}), 400
        if not all(i.get("human_vote") in ["support", "prefer", "reject"] for i in r.get("ideas", [])):
            return jsonify({"error": "Human must vote on every idea before finishing the round.", "state": public_state()}), 409
        if r.get("phase") in ["agent_voted", "decision_ready", "accepted"]:
            return jsonify(public_state())
        r["phase"] = "agent_voting"
        save_state()

    # Agent votes outside lock.
    votes = []
    with ThreadPoolExecutor(max_workers=len(AGENTS)) as ex:
        futs = {ex.submit(make_agent_vote, key, r): key for key in AGENTS}
        for fut in as_completed(futs):
            votes.append(fut.result())
    votes.sort(key=lambda v: list(AGENTS.keys()).index(v.get("voter_key")) if v.get("voter_key") in AGENTS else 999)

    with STATE_LOCK:
        r = get_current_round()
        r["agent_votes"] = votes
        for idea in r.get("ideas", []):
            idea["agent_votes_received"] = [v for v in votes if v.get("idea_id") == idea.get("idea_id")]
        score_map, winner, winner_score = compute_scores(r)
        r["scores"] = score_map
        if winner:
            r["winner"] = {**winner, "score": winner_score}
            r["next_focus"] = f"Decide whether to accept winning idea: {winner.get('idea_title')}"
        r["phase"] = "decision_ready"
        add_transcript("system", "Decision Board", build_decision_text(r), kind="decision")
        # Optional judge note, non-voting.
        r["judge_note"] = make_judge_note(r)
        save_state()
        return jsonify(public_state())


def build_decision_text(r):
    winner = r.get("winner") or {}
    lines = [f"Round {r.get('round_number')} decision is ready."]
    if winner:
        lines.append(f"Winning idea: {winner.get('idea_title')} ({winner.get('score',{}).get('total', winner.get('score'))}%).")
        lines.append(f"From: {winner.get('agent_name')}")
        lines.append(winner.get("idea", ""))
    return "\n".join(lines)


def make_judge_note(r):
    # Judge is advisory only, not a vote. If model fails, use local note.
    compact = []
    for idea in r.get("ideas", []):
        compact.append({
            "id": idea["idea_id"],
            "agent": idea["agent_name"],
            "title": idea["idea_title"],
            "idea": idea["idea"],
            "human_vote": idea.get("human_vote"),
            "score": r.get("scores", {}).get(idea["idea_id"], {}).get("total", 0),
        })
    prompt = f"""
You are a non-voting meeting judge. Do not choose a different winner. The winner is based on human 30% + agents 70%.
Summarize in under 80 words:
1. why the winning idea won
2. strongest remaining objection
3. next round focus
Ideas and scores:
{json.dumps(compact, ensure_ascii=False, indent=2)}
Return plain text only.
""".strip()
    try:
        return short(call_openrouter(FAST_JUDGE_MODEL, [{"role": "user", "content": prompt}], max_tokens=MAX_JUDGE_TOKENS, temperature=0.2), 700)
    except Exception as e:
        winner = r.get("winner") or {}
        return f"Judge unavailable. Winner is {winner.get('idea_title','unknown')} by weighted vote. Next: turn it into a concrete decision. Error: {str(e)[:120]}"


@app.route("/api/accept_decision", methods=["POST"])
def api_accept_decision():
    with STATE_LOCK:
        r = get_current_round()
        if not r:
            return jsonify({"error": "No active round."}), 400
        if r.get("phase") != "decision_ready":
            return jsonify({"error": "Round is not decision-ready. Finish voting first."}), 409
        winner = r.get("winner")
        if not winner:
            return jsonify({"error": "No winning idea yet."}), 400
        decision_text = f"Accept round {r.get('round_number')} winning idea: {winner.get('idea_title')} — {winner.get('idea')}"
        decision = {
            "id": str(uuid.uuid4()),
            "time": now_iso(),
            "round_number": r.get("round_number"),
            "decision": decision_text,
            "winner_idea_id": winner.get("idea_id"),
            "winner_agent": winner.get("agent_name"),
            "score": winner.get("score", {}).get("total") if isinstance(winner.get("score"), dict) else winner.get("score"),
        }
        STATE.setdefault("decisions", []).append(decision)
        STATE.setdefault("agreements", []).append(f"Round {r.get('round_number')}: {winner.get('idea_title')}")
        STATE["session_memory"]["accepted_decisions"] = STATE.get("decisions", [])[-8:]
        STATE["session_memory"]["summary"] = normalize_session_summary()
        STATE["next_focus"] = next_focus_from_winner(winner)
        r["phase"] = "accepted"
        r["decision"] = decision
        r["next_focus"] = STATE["next_focus"]
        update_agent_positions_from_decision(r, decision_text)
        add_transcript("system", "Accepted Decision", decision_text + f"\nNext focus: {STATE['next_focus']}", kind="accepted")
        save_state()
        return jsonify(public_state())


@app.route("/api/reject_decision", methods=["POST"])
def api_reject_decision():
    data = request.get_json(force=True) or {}
    reason = short(data.get("reason", "Human rejected the winning decision and requested one more debate round."), 400)
    with STATE_LOCK:
        r = get_current_round()
        if not r:
            return jsonify({"error": "No active round."}), 400
        r["phase"] = "rejected"
        r["decision"] = {"rejected_at": now_iso(), "reason": reason}
        STATE["next_focus"] = f"Reopen debate because: {reason}"
        add_transcript("human", "Human Rejection", reason, kind="rejected")
        save_state()
        return jsonify(public_state())


def normalize_session_summary():
    decisions = STATE.get("decisions", [])[-6:]
    if not decisions:
        return "No accepted decisions yet."
    lines = [f"R{d.get('round_number')}: {d.get('decision','')}" for d in decisions]
    return "Accepted decisions so far:\n" + "\n".join(lines)


def next_focus_from_winner(winner):
    title = (winner.get("idea_title") or "winning idea").strip()
    agent_key = winner.get("agent_key")
    if agent_key == "builder":
        return f"Turn '{title}' into exact MVP workflow, APIs, data fields, latency targets, and tests."
    if agent_key == "critic":
        return f"Turn '{title}' into concrete safeguards, KPIs, audit rules, and failure tests."
    if agent_key == "product":
        return f"Turn '{title}' into user-facing UX, trust messaging, and product acceptance criteria."
    if agent_key == "strategy":
        return f"Turn '{title}' into pilot roadmap, pricing assumption, customer segment, and success gate."
    return f"Operationalize the accepted idea: {title}."


@app.route("/api/next_round", methods=["POST"])
def api_next_round():
    with STATE_LOCK:
        r = get_current_round()
        if r and r.get("phase") not in ["accepted", "rejected", "closed"]:
            return jsonify({"error": "Finish the current round before starting the next one.", "state": public_state()}), 409
        topic = STATE.get("next_focus") or "Continue from the last accepted decision."
    # reuse start_round logic by direct call semantics isn't easy; duplicate minimal via test_request_context no. Just call helper route not possible.
    return api_start_round_with_topic(topic)


def api_start_round_with_topic(topic):
    # Internal helper called by /api/next_round.
    with STATE_LOCK:
        STATE["round_number"] = int(STATE.get("round_number", 0)) + 1
        round_number = STATE["round_number"]
        STATE["active_topic"] = topic
        add_transcript("human", "Human Host", f"Next round focus: {topic}", kind="topic")

    ideas = []
    with ThreadPoolExecutor(max_workers=len(AGENTS)) as ex:
        futs = {ex.submit(make_agent_idea, key, topic, round_number): key for key in AGENTS}
        for fut in as_completed(futs):
            ideas.append(fut.result())
    order = {k: i for i, k in enumerate(AGENTS.keys())}
    ideas.sort(key=lambda x: order.get(x.get("agent_key"), 999))

    with STATE_LOCK:
        round_obj = {
            "round_id": str(uuid.uuid4()),
            "round_number": round_number,
            "topic": topic,
            "started_at": now_iso(),
            "phase": "listening",
            "current_index": 0,
            "ideas": ideas,
            "human_votes_complete": False,
            "agent_votes": [],
            "scores": {},
            "winner": None,
            "decision": None,
            "judge_note": None,
            "next_focus": "Listen to every agent idea, vote after each one, then finish the round.",
        }
        STATE["current_round_id"] = round_obj["round_id"]
        STATE.setdefault("rounds", []).append(round_obj)
        for idea in ideas:
            add_transcript("agent", idea["agent_name"], idea["idea"], agent_key=idea["agent_key"], kind="idea", extra={"idea_id": idea["idea_id"], "idea_title": idea["idea_title"], "model": idea["model"]})
            st = STATE["agent_states"][idea["agent_key"]]
            st["speak_count"] = int(st.get("speak_count", 0)) + 1
            st["current"] = idea.get("position_update") or st.get("current", "")
            st["updated_at"] = now_iso()
        save_state()
        return jsonify(public_state())


if __name__ == "__main__":
    print("Starting Model Council V13 — Real Meeting Round + Weighted Voting")
    print("Open http://127.0.0.1:5000")
    print(f"OpenRouter enabled: {bool(OPENROUTER_API_KEY)}")
    print(f"Human vote weight: {HUMAN_VOTE_WEIGHT}% | Agent total: {AGENT_TOTAL_WEIGHT}%")
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=True)
