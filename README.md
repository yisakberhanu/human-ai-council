# Human AI Council

> A human-led multi-agent AI meeting platform where specialized AI agents debate, critique, vote, learn from experience, and improve over time.

---

# Overview

Human AI Council is an experimental framework for building realistic AI meetings.

Instead of interacting with a single chatbot, the human acts as the meeting host while multiple specialized AI agents participate in structured discussions.

Each agent has its own expertise, memory, goals, and evolving perspective.

The council debates ideas, challenges assumptions, votes on proposals, reaches consensus, and gradually improves its behavior across meetings.

The project is designed for research, product design, decision support, English speaking practice, interview preparation, and human-AI collaboration.

---

# Features

## Human-Led Meetings

* Human controls the meeting.
* AI agents never replace the host.
* Human can interrupt, guide, or redirect the discussion.

---

## Specialized AI Agents

Current agents include:

* Product Agent
* Builder Agent
* Strategy Agent
* Critic Agent

Each agent has:

* Base perspective
* Current evolving position
* Private memory
* Voting ability
* Ability to agree or object

---

## Adaptive Discussions

Unlike static chatbots, agents:

* Listen before speaking
* Compare their opinion with previous speakers
* Agree or object
* Update their position after accepted decisions
* Avoid repeating solved discussions

Each round builds on previous consensus.

---

## Decision Engine

Every discussion round ends with:

* Agent voting
* Human voting
* Decision board
* Winning proposal
* Accepted or rejected decision
* Next meeting focus

Accepted decisions become permanent meeting memory.

---

## Continuous Learning

After every meeting the system records:

* Accepted ideas
* Rejected ideas
* Human feedback
* Agent votes
* Skill updates
* Memory updates

Agents improve through structured experience rather than isolated prompts.

---

## Meeting Memory

The council maintains:

* Active topic
* Current focus
* Accepted decisions
* Agreements
* Consensus history
* Agent memories

This allows long-running discussions without losing context.

---

## English Speaking Mode (Planned)

A second operating mode focuses on spoken English improvement.

Specialized agents will evaluate:

* Grammar
* Fluency
* Vocabulary
* Pronunciation
* Conversation quality
* Critical feedback

The council will recommend one improvement target after each speaking round.

---

## Future Roadmap

### AI Evaluation Judge

A dedicated evaluator model that:

* scores arguments
* detects repetition
* measures evidence quality
* checks logical consistency
* identifies weak reasoning
* explains why one proposal wins

---

### Tool Calling

Agents will use external tools including:

* Web search
* Knowledge bases
* Document retrieval
* APIs
* Databases

---

### Voice Meetings

* Speech recognition
* Text-to-speech
* Live conversation
* Speaker highlighting
* Audio replay

---

### Agent Skill Learning

Future versions will allow agents to improve:

* reasoning
* planning
* communication
* trust
* reliability
* domain expertise

using structured feedback from previous meetings.

---

# Current Architecture

Human

↓

Meeting Controller

↓

Shared Meeting Memory

↓

Product Agent

Builder Agent

Strategy Agent

Critic Agent

↓

Voting Engine

↓

Decision Board

↓

Consensus Memory

↓

Skill Update System

↓

Next Discussion Round

---

# Technology

* Python
* Flask
* OpenRouter API
* Google Gemma
* Qwen
* GPT OSS
* HTML
* CSS
* JavaScript

---

# Vision

The goal is not to build another chatbot.

The goal is to build a realistic AI council where specialized agents collaborate, challenge each other, make decisions, learn from experience, and continuously improve alongside a human host.

---

# License

MIT License
