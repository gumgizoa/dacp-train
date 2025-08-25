MCQA_SYSTEM_PROMPT="""You are a knowledgeable assistant that answers multiple-choice questions. Choose the correct option based on the question and options provided. Explain your reasoning if possible."""
MCQA_INSTRUCTION_TEMPLATE="""Question: {query}
Options: 
{options}

Provide a detailed reasoning to solve the question if possible. End your response with the correct option in the format "Answer: $OPTION"."""
MCQA_ANSWER_TEMPLATE="""Anwser: {answer}"""
MCQA_ANSWER_COT_TEMPLATE="""{trace}

Answer: {answer}"""

OEQA_SYSTEM_PROMPT = """You are an expert assistant that provides clear, thorough, and accurate answers to open-ended questions.
Explain your reasoning step by step if possible.
Be concise but complete, and structure your response logically."""
OEQA_INSTRUCTION_TEMPLATE="""Question: {query}

Provide detailed reasoning to solve the question if possible. Conclude your response with the exact format: "Answer: $ANSWER", where $ANSWER is your final answer."""
OEQA_ANSWER_TEMPLATE="""Answer: {answer}"""
OEQA_ANSWER_COT_TEMPLATE="""{trace}

Answer: {answer}"""