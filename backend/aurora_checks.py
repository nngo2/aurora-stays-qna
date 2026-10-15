# Answer checks module: a grounding check (LLM) and hotel policy rules (regular expressions) on the draft
# answer. When a check fails, the answer is rewritten once with the issues found (at most MAX_REDRAFTS times).
import re
from pydantic import BaseModel, Field
from aurora_generation import generate

MAX_REDRAFTS = 1

class Claim(BaseModel):
    claim: str
    supported: bool = Field(description="True if the policy passages state this claim")

class Grounding(BaseModel):
    claims: list[Claim]

GROUNDING_PROMPT = """Break the answer into its factual claims (fees, times, limits, rules, conditions,
approvals, procedures) and decide for each one whether the retrieved policy passages state it.
Do not list as claims: statements that the policies do not include or state some information (never turn
"the policies do not state whether X" into the claim X), recommendations to check the live property profile,
the reservation confirmation, or the property team, or the "Source:" line.

Answer:
{answer}

Retrieved policy passages:
{evidence}"""

# Sentences that make no policy claim are removed before the check, because the LLM checker sometimes flags them:
# "the policies do not state whether X" (it turns this into the claim X), the verification step required by
# generation rule 8 ("verify the reservation confirmation"), and the "Source:" line
NO_CLAIM = re.compile(
    r"\b(do|does) not (state|include|specify|mention|cover|say)|\bnot (stated|included|specified|covered)\b"
    r"|\b(verify|check|confirm|contact|consult)\b.{0,80}\b(live property profile|property profile|reservation"
    r" confirmation|confirmation|property team|folio)\b|^\s*source:",
    re.IGNORECASE)

def check_grounding(answer, evidence, model):
    # One LLM call: the claims of the answer that the evidence does not support
    claims_text = " ".join(s for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if not NO_CLAIM.search(s))
    if not claims_text.strip():
        return []
    claims = model.with_structured_output(Grounding).with_retry(stop_after_attempt=3).invoke(
        GROUNDING_PROMPT.format(answer=claims_text, evidence="\n\n".join(evidence))).claims
    return [c.claim for c in claims if not c.supported]

# Statements an answer to staff must never make (generation rules 7 and 9, and the Refunds guide's card rule).
# A sentence that contains a negation ("do not ask for a CVV", "is not guaranteed") is allowed.
POLICY_REGEX = {
    "promises a refund": r"\b(will|shall) (be|get|receive)( a| an)?( full| partial)? refund|\bwill be refunded\b|\bguarantee[sd]? (a |an |the )?refund",
    "promises a waiver": r"\b(will|shall) (be )?waive[sd]?\b|\bwill be waived\b",
    "promises an upgrade or availability": r"\b(guarantee[sd]?|will get|will receive) (a |an |the )?(free )?(upgrade|room|availability)|\bfree upgrade\b",
    "asks for card or password data": r"\b(send|provide|share|give|email|confirm)\b[^.]{0,40}\b(card number|cvv|cvc|security code|password)",
    "asks about a disability or animal paperwork": r"\b(ask|request|require|demand)\b[^.]{0,60}\b(diagnosis|medical record|certificate|vest|proof of disability|nature of (the|their|his|her) disability)",
}
NEGATION = re.compile(r"\b(not|never|don't|cannot|can't)\b", re.IGNORECASE)

def check_policy(answer):
    # The policy rules broken by any sentence of the answer
    issues = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
        if NEGATION.search(sentence):
            continue
        issues += [name for name, pattern in POLICY_REGEX.items()
                   if re.search(pattern, sentence, re.IGNORECASE) and name not in issues]
    return issues

def run_checks(question, evidence, draft, model, checker, guidance="", max_redrafts=MAX_REDRAFTS):
    # Check the draft; on failure, rewrite it once with the issues listed, then check the rewrite
    steps, redrafts = [], 0
    while True:
        unsupported = check_grounding(draft, evidence, checker)
        policy_issues = check_policy(draft)
        steps.append(f"checks: {len(unsupported)} unsupported claim(s), {len(policy_issues)} policy issue(s)")
        if (not unsupported and not policy_issues) or redrafts == max_redrafts:
            break
        feedback = "\n".join(
            ["Your previous draft had these problems. Remove or correct each flagged statement so that it says "
             "only what the policies state; do not just reword it."]
            + [f"- Not supported by the policies: {c}" for c in unsupported]
            + [f"- Breaks a policy rule ({p})." for p in policy_issues]
            + ["Previous draft:", draft]
        )
        draft = generate(question, evidence, model, guidance + "\n" + feedback).content
        redrafts += 1
        steps.append("redraft with the check feedback")
    return {"answer": draft, "unsupported": unsupported, "policy_issues": policy_issues,
            "checks_passed": not unsupported and not policy_issues, "redrafts": redrafts, "steps": steps}
