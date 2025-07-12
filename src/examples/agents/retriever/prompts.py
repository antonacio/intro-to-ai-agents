GENERATE_QUERIES_SYSTEM_PROMPT = """\
You will be given a research task.
Your job is to generate three distinct and relevant search queries that could help find information to complete the research task.
These search queries will be used to retrieve specific information from documents in a vector store.
Ensure each query is unique and covers different aspects or approaches to the research task.
Avoid redundancy or overlap between the queries.


Here is the research task:
<research_task>
{research_task}
</research_task>
"""
