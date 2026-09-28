"""
MAIN OLAMA CALLER AND SYSTEM PROMPT, TWEAK NO - 85 - 17/09
# TWEAK FROM? - SELF
thin wrapper around a local ollama server + the grounding prompt 
models tried: 1. qwen2.5:7b-instruct-q4_K_M current
2. qwen2.5:3b-instruct
3.qwen2.5:1.5b-instruct is a

    ollama pull qwen2.5:3b-instruct
    ollama serve TO USE for tst

#85 - ADDED DOCUMENT YEAR TO EACH CHUNK + MANDATORY RECENCY RULE + BANNED
      "which of these" / "check the website" / "please clarify" RESPONSES.
      THIS IS THE FIX FOR THE "ANSWERS FROM 2018/2019 WHEN 2026 EXISTS" BUG.
"""
#MAINA Ollama caller+grounding prompt+persona+time-aware reasoning.
import json
import re
from datetime import date  
import httpx
from bkend import cfgstore
ollamaurl = "http://localhost:11434/api/chat"
chunkchar_cap = 1500

# pulls "2024" out of URLs like /includes/noticeboard/2024/...
_URL_Y = re.compile(r'/(20\d{2})/')
def _url_year(url: str) -> str:
    m = _URL_Y.search(url or "")
    return m.group(1) if m else "unknown"

PERSONA = {
    "friendly":     "Be warm and helpful, like a senior student helping a junior. Natural conversational tone.",
    "professional": "Be formal and matter-of-fact. No small talk.",
    "concise":      "Be extremely brief. 1-2 sentences when possible. Cut filler.",
    "chill":        "Be casual and relaxed. Light tone, minimal formality.",
}
SYS = """You are the DCSA (Department of Computer Science & Applications, Panjab University) Helpdesk Assistant.
{persona_line}
Today's date is {today}.
STRICT RULES — follow exactly:
1. THE CONTEXT IS YOUR ONLY SOURCE OF TRUTH. Do not use any knowledge from training. If a fact is not written in the CONTEXT below, you MUST NOT state it — even if you are certain it is true in the real world. This includes programme names, course variants, people's names, dates, fees, and URLs.
2. If the context does not contain the answer, say exactly: "I couldn't find this in the official DCSA sources." and stop. Do not guess.
3. NEVER invent names, emails, phone numbers, fees, dates, or roles. If a name appears in the context but a ROLE (chairperson, coordinator, teacher) is not explicitly attached to that name in the SAME chunk, do NOT claim the role.
4. NEVER stitch facts across different chunks. A name in chunk A and a title in chunk B does NOT mean that name holds that title.
5. RECENCY IS MANDATORY. Each chunk has a "Document year" field. When the question asks about current / now / latest / "is X still" / "has X ended" — OR when two chunks disagree — you MUST prefer the chunk with the HIGHER document year. When listing facts, use the newest year. If the newest source in the context is more than 2 years older than today ({today}), say clearly: "the newest official source I have is from [year], so this may be outdated."
6. NEVER tell the user to "check the website", "check the noticeboard", "refer to the class schedule", or "visit X page". Either answer from the context, or say you couldn't find it.
7. NEVER respond with "which of these does your question relate to", "could you clarify", "I'm unable to understand", or any variant of these. Just answer the question.
8. If asked "how many types" or "list all", enumerate EVERY variant that appears in the context (e.g. morning, evening, self-financing, honours). Do not collapse them into one.
9. CONTACT NUMBERS AND EMAILS RULE: Any phone number or email that appears INSIDE an official DCSA notice, PDF, or webpage (e.g. "contact Mr. Manpreet 76960-62863") IS an official departmental contact. YOU MUST give it when asked. Do NOT refuse. Only refuse if the number is NOT present anywhere in the context.
10. When the context contains a person's name followed by a phone number or email on the same line, treat that as their contact.
11. MULTI-PART QUESTIONS: If the question asks multiple things ("who is X and does Y exist?"), answer EACH part separately. If one part isn't in the context, say so for that part only and answer the others.
12. Be direct. No preamble. No "Based on the context...".
"""
USER = """CONTEXT (from official DCSA sources):
{ctx}
STUDENT QUESTION: {q}
{style_hint}
Answer now, following the rules."""
STYLE = {
    "short":    "Keep the answer to 1-3 sentences.",
    "balanced": "Keep the answer to one short paragraph (3-6 sentences).",
    "detailed": "Give a structured answer, using bullets if listing things.",
}
def fmtctx(chunks) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        text = c.text[:chunkchar_cap]
        parts.append(
            f"[CHUNK {i}] Source: {c.title or c.url}\n"
            f"URL: {c.url}\n"
            f"Document year: {_url_year(c.url)}\n"
            f"Last checked: {c.fetchedatiso}\n"
            f"Content:\n{text}\n"
        )
    return "\n---\n".join(parts)
def _call(model, prompt_text, temp, max_tok, timeout):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": True,
        "options": {"temperature": temp, "num_predict": max_tok},
    }
    parts = []
    with httpx.Client(timeout=httpx.Timeout(timeout, read=timeout)) as client:
        with client.stream("POST", ollamaurl, json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece = obj.get("message", {}).get("content", "")
                if piece:
                    parts.append(piece)
                if obj.get("done"):
                    break
    return "".join(parts).strip()
def genanswer(query: str, chunks, timeout: float = 180.0) -> str:
    cfg = cfgstore.loadcfg()
    model   = cfg["model"]["llm"]
    fb      = cfg["model"].get("llm_fallback")
    style   = cfg["generation"]["response_mode"]
    persona = cfg["generation"].get("persona", "friendly")
    temp    = cfg["generation"]["temperature"]
    maxt    = cfg["generation"]["max_tokens"]
    sys_prompt = SYS.format(
        persona_line=PERSONA.get(persona, PERSONA["friendly"]),
        today=date.today().isoformat(),
    )
    usr_prompt = USER.format(
        ctx=fmtctx(chunks), q=query,
        style_hint=STYLE.get(style, STYLE["balanced"]),
    )
    full = f"{sys_prompt}\n\n{usr_prompt}"
    try:
        return _call(model, full, temp, maxt, timeout)
    except Exception as e:
        if fb and fb != model:
            try:
                return _call(fb, full, temp, maxt, timeout)
            except Exception as e2:
                return f"(LLM error: {e2})"
        return f"(LLM error: {e})"
def genclarifyq(query: str, candsections: list[str]) -> str:
    # kept for backwards compat, app.py no longer calls it
    pretty = ", ".join(candsections)
    return (f"Quick check — which of these does your question relate to: {pretty}? "
            f"Or add a detail (your programme/semester) so I can point you precisely.")
nomatchmsg = (
    "I couldn't find this in the official DCSA sources I've indexed, so I won't guess. "
    "DCSA office: 1800-180-2065 / +91 172 2534818, 2534866 "
    "(9:30 am–1:00 pm & 2:30 pm–5:00 pm, working days) — https://dcsa.puchd.ac.in/show-contactus.php"
)