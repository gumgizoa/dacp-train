MCQA_SYSTEM_PROMPT="""You are a knowledgeable assistant that answers multiple-choice questions. Choose the correct option based on the question and options provided. Explain your reasoning if possible."""
MCQA_INSTRUCTION_TEMPLATE="""Provide a detailed explanation to solve the question if possible. End your response with the correct option number in the format "Answer: $OPTION (an integer number)".
Question: {query}
Options: 
{options}"""
MCQA_ANSWER_TEMPLATE="""Anwser: {answer}"""
MCQA_ANSWER_COT_TEMPLATE="""<think>\n{trace}\n</think>\n\nAnswer: {answer}"""

OEQA_SYSTEM_PROMPT = """You are an expert assistant that provides clear, thorough, and accurate answers to open-ended questions."""
OEQA_INSTRUCTION_TEMPLATE="""Provide detailed explanation to solve the question if possible.
Question: {query}"""

OEQA_ANSWER_TEMPLATE="{answer}"
OEQA_ANSWER_COT_TEMPLATE="<think>\n{trace}\n</think>\n\n{answer}"