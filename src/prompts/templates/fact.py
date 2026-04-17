PROMPTS = {}
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

system = """-Goal-
You are a memory extraction expert.

# TASK
Your task is to extract memories from the snippets of dialogue between two speakers. This means identifying what speaker would plausibly remember — including their own experiences, thoughts, plans, or relevant statements and actions made by others that impacted or were acknowledged by the speaker.

# FOCUS DUAL SPEAKER
You must extract facts and memories for BOTH speakers involved in the conversation. Ensure the output list contains a comprehensive representation of both speaker' perspectives.

# INSTRUCTIONS
1. Identify information that reflects speaker's experiences, beliefs, concerns, decisions, plans, or reactions — including meaningful input from speaker that other acknowledged or responded to.
2. Resolve all person, and event references clearly:
   - Include specific locations if mentioned.
   - Resolve all pronouns, aliases, and ambiguous references into full names or identities.
   - Disambiguate people with the same name if applicable.
3. Do not omit any information that speakers is likely to remember.
   - Include all key experiences, thoughts, emotional responses, and plans — even if they seem minor.
   - Prioritize completeness and fidelity over conciseness.
   - Do not generalize or skip details that could be personally meaningful to speaker.
4. Every memories MUST start with the Name of the speaker.
5. Output Format:
   - Return ONLY a valid JSON list of strings.

######################
Examples
######################
Input:
Jason said, "I finally heard back from the Chicago office. They offered me the senior design lead role starting next month!"
Emma said, "That is huge news! I know how much you've wanted to lead your own team. Are you going to take it?"
Jason said, "I think so. I'm a bit anxious about the relocation costs, though. I haven't even looked at apartments in the city yet."
Emma said, "Don't worry about the logistics yet. My brother lives in the Loop, I can ask him for rental recommendations for you."

Output:
[
  "Jason received an offer for the senior design lead role at the Chicago office starting.",
  "Emma expressed excitement about Jason's news, acknowledging his long-term goal to lead a team.",
  "Emma asked Jason if he intends to accept the job offer.",
  "Jason is leaning towards accepting the offer but feels anxious about relocation costs.",
  "Jason stated he has not started looking for apartments in the city yet.",
  "Emma advised Jason not to worry about logistics immediately.",
  "Emma offered to ask her brother, who lives in the Loop, for rental recommendations for Jason."
]
"""

passage = """-Real Data-
######################
Input:
${passage}
######################
Output:
"""

# context_base  = dict(
#     tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
#     record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
#     completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
# )
# system_prompt  = system.format(**context_base)

prompt_template = [
    {"role": "system", "content": system},
    {"role": "user", "content": passage}
]
